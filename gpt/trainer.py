# very much based on deepspeed-examples.
# https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/main.py

import json
import math
import random

from typing import Optional

import deepspeed
import numpy as np
import torch
import torch.distributed
import typer
from datasets import load_dataset
from deepspeed import get_accelerator
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.utils import logger
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, default_data_collator, get_scheduler

import wandb
from model import Transformer, ModelArgs


app = typer.Typer()


class WikiTextDataset(Dataset):
    def __init__(self, tokenizer, type_path="train", max_length=1024):
        self.vernum = 103
        self.dataset = load_dataset(
            "wikitext", f"wikitext-{self.vernum}-raw-v1", split=type_path
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return int(len(self.dataset) * 0.1)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        # logger.info(text)
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.squeeze()}


def train(ds_engine, train_loader, device):
    ds_engine.train()
    total_loss = 0
    for _, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)

        outputs = ds_engine(input_ids)
        loss = outputs["loss"]
        total_loss += loss.item()
        if torch.distributed.get_rank() == 0:
            logger.info(f"loss : {loss.item()}")
            wandb.log({"trainloss": loss.item()})

        ds_engine.backward(loss)
        ds_engine.step()
        get_accelerator().empty_cache()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids)
            loss = outputs["loss"]
            total_loss += loss.float()

    losses = total_loss / len(val_loader)

    try:
        losses = get_all_reduce_mean(losses)
    except:  # noqa: E722
        pass
    try:
        perplexity = torch.exp(losses).item()
    except OverflowError:
        perplexity = float("inf")

    model.train()

    return losses, perplexity


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@app.command()
def main(
    model_type: str,
    num_warmup_steps: Optional[int] = 0,
    seed: Optional[int] = 42,
    gradient_checkpointing: Optional[bool] = True,
    gradient_checkpointing_offload: Optional[bool] = True,
    gradient_checkpointing_num_checkpoints: Optional[int] = 3,
    zero_stage: Optional[int] = 3,
    output_dir: Optional[str] = "output",
    offload: Optional[bool] = False,
    run_name: Optional[str] = None,
    local_rank: int = typer.Option(-1, "--local_rank"),
    train_micro_batch_size_per_gpu: Optional[int] = 32,
    train_batch_size: Optional[int] = 2048,
    learning_rate: Optional[float] = 1e-3,
    weight_decay: Optional[float] = 0.1,
    num_train_epochs: Optional[int] = 1,
    lr_scheduler_type: Optional[str] = "linear",
    profile: Optional[bool] = False,
):
    set_seed(seed)
    assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
    # n_layers, n_heads and n_embd are determined from model_type
    config = {
        'gpt2':         dict(n_layers=12, n_heads=12, n_embd=768),  # 124M params
        'gpt2-medium':  dict(n_layers=24, n_heads=16, n_embd=1024), # 350M params
        'gpt2-large':   dict(n_layers=36, n_heads=20, n_embd=1280), # 774M params
        'gpt2-xl':      dict(n_layers=48, n_heads=25, n_embd=1600), # 1558M params
    }[model_type]

    n_layers, n_heads, n_embd = config.values()
    head_dim = n_embd // n_heads

    if run_name is None:
        run_name = f"LR:{learning_rate}_HeadDim:{head_dim}_TotalBS:{train_batch_size}_Nheads:{n_heads}_NLayers:{n_layers}"

    if local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(local_rank)
        device = torch.device(get_accelerator().device_name(), local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    offload_device = "cpu" if offload else "none"

    ds_config = {
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "train_batch_size": train_batch_size,
        "zero_optimization": {
            "stage": zero_stage,
            "offload_param": {"device": offload_device},
            "offload_optimizer": {"device": offload_device},
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 3e7,
            "stage3_prefetch_bucket_size": 3e7,
            "memory_efficient_linear": False,
        },
        "bfloat16": {"enabled": True},
        "gradient_clipping": 1.0,
    }  # wall_clock_breakdown -> profile breakdown of time spent in different parts of the training loop
    # ref: https://www.deepspeed.ai/docs/config-json/#bfloat16-training-options
    if profile:
        flops_profiler = {
            "enabled": True,
            "profile_step": 1,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
            "output_file": output_dir + "/flops_profiler.txt",
        }
        ds_config["flops_profiler"] = flops_profiler
    # ref: https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html
    if gradient_checkpointing:
        gradient_checkpointing_dict = {
            "partition_activations": False,
            "cpu_checkpointing": gradient_checkpointing_offload,
            "contiguous_memory_optimization": False,
            "number_checkpoints": gradient_checkpointing_num_checkpoints,
            "synchronize_checkpoint_boundary": False,
            "profile": True,
        }
        ds_config["activation_checkpointing"] = gradient_checkpointing_dict

    torch.distributed.barrier()
    global_rank = torch.distributed.get_rank()

    if global_rank == 0:
        wandb.init(
            name=run_name,
            config={
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "num_train_epochs": num_train_epochs,
                "lr_scheduler_type": lr_scheduler_type,
                "num_warmup_steps": num_warmup_steps,
                "seed": seed,
                "gradient_checkpointing": gradient_checkpointing,
                "zero_stage": zero_stage,
                "output_dir": output_dir,
                "offload": offload,
                "head_dim": head_dim,
                "n_heads": n_heads,
                "n_layers": n_layers,
            },
        )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    config = ModelArgs(**config)
    with deepspeed.zero.Init():
        model = Transformer(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enabled(ds_config)

    model.train()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_memory_gb = total_params * 4 / (1024**3)
    print(f"model params: {total_params}")
    logger.info(f"Model memory size: {model_memory_gb} GB")

    train_dataset = WikiTextDataset(tokenizer, "train")
    val_dataset = WikiTextDataset(tokenizer, "validation")

    train_sampler = (
        RandomSampler(train_dataset)
        if local_rank == -1
        else DistributedSampler(train_dataset, seed=seed)
    )
    eval_sampler = (
        SequentialSampler(val_dataset)
        if local_rank == -1
        else DistributedSampler(val_dataset, seed=seed)
    )

    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        sampler=train_sampler,
        batch_size=train_micro_batch_size_per_gpu,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=default_data_collator,
        sampler=eval_sampler,
        batch_size=train_micro_batch_size_per_gpu * 2,
    )

    # weight decay config
    no_decay_name_list = [
        "bias",
        "ln_",
        "ln_f.weight",
    ]

    optimizer_grouped_parameters = []
    final_optimizer_settings = {}

    for n, p in model.named_parameters():
        group_parameters = {}
        if p.requires_grad:
            if any(ndnl in n for ndnl in no_decay_name_list):
                group_parameters["weight_decay"] = 0.0
            else:
                group_parameters["weight_decay"] = weight_decay
            # Define learning rate for specific types of params
            is_embed = "embed" in n
            if "embed" in n or any(ndnl in n for ndnl in no_decay_name_list):
                group_parameters["lr"] = learning_rate * (3.3 if is_embed else 1.0)
            else:
                group_parameters["lr"] = learning_rate * (1 / head_dim)

            group_parameters["params"] = [p]
            final_optimizer_settings[n] = {
                "lr": group_parameters["lr"],
                "wd": group_parameters["weight_decay"],
            }
            optimizer_grouped_parameters.append(group_parameters)

    # View the settings, see if anything is wrong.
    with open("./opt_config.json", "w") as json_file:
        json.dump(final_optimizer_settings, json_file, indent=4)

    AdamOptimizer = DeepSpeedCPUAdam if offload else FusedAdam

    optimizer = AdamOptimizer(
        optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.95)
    )

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_epochs * math.ceil(len(train_loader)),
    )

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model, config=ds_config, lr_scheduler=lr_scheduler, optimizer=optimizer
    )

    for epoch in range(num_train_epochs):
        if local_rank == -1:
            pass
        else:
            train_sampler.set_epoch(epoch)

        avg_train_loss = train(model_engine, train_loader, model_engine.device)

        logger.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}")
        eval_loss, perp = validate(model_engine, val_loader, device=device)
        if global_rank == 0:
            logger.info(f"Eval loss : {eval_loss}")
            wandb.log({"ppl": perp, "loss": eval_loss, "epoch": epoch})


if __name__ == "__main__":
    typer.run(main)
