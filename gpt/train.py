from pathlib import Path
from time import time
from typing import Optional

import tiktoken
import torch
import torch._inductor.config
import torch.nn as nn
import typer

from vanilla_model import ModelArgs, Transformer
from torchao.sparsity.training import (
    SemiSparseLinear,
    swap_linear_with_semi_sparse_linear,
)

app = typer.Typer()
# global variables
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
TINY_SHAKESPEARE_FN = DATASET_DIR / "input.txt"

# torch inductor config
torch._inductor.config.max_autotune = True
torch._inductor.config.max_autotune_gemm = True
torch._inductor.config.max_autotune_gemm_search_space = "DEFAULT"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open(TINY_SHAKESPEARE_FN.as_posix(), "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


def swap_ffn_linear_layers(model):
    """enable sparsity only in the feedfoward layers"""
    sparse_config = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            sub_name in name for sub_name in ["mlp.c_fc", "mlp.c_proj"]
        ):
            sparse_config[name] = SemiSparseLinear
    swap_linear_with_semi_sparse_linear(model, sparse_config)

    return model


@app.command()
def train(
    num_warmup_steps: Optional[int] = 0,
    seed: Optional[int] = 1337,
    gradient_checkpointing: Optional[bool] = False,
    gradient_checkpointing_offload: Optional[bool] = False,
    zero_stage: Optional[int] = 0,
    output_dir: Optional[str] = "output",
    offload: Optional[bool] = False,
    debug: Optional[bool] = False,
    run_name: Optional[str] = None,
    local_rank: Optional[int] = -1,
    profile: Optional[bool] = False,
    sequence_length: Optional[int] = 1024,
    micro_batch_size: Optional[int] = 16,
    global_batch_size: Optional[int] = 524288,
    learning_rate: Optional[float] = 6e-4,
    weight_decay: Optional[float] = 0.1,
    epochs: Optional[int] = 1,
    lr_scheduler_type: Optional[str] = "linear",
    sparse_ffn: Optional[bool] = False,
    use_compile: Optional[bool] = False,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"device: {device}")
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # ToDo: cli transformer model args
    run_name = f"LR:{3e-4}__Nhead:{ModelArgs().n_heads}_NLayer:{ModelArgs().n_layers}_EmbDim:{ModelArgs().n_embd}"
    print(f"run_name: {run_name}")

    train_loader = DataLoaderLite(B=micro_batch_size, T=sequence_length)

    model = Transformer(ModelArgs())
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params: {total_params}")

    if sparse_ffn:
        model = model.to(device).to(torch.float16)
        model = swap_ffn_linear_layers(model)
    # import code; code.interact(local=locals())
    if use_compile:
        model = torch.compile(model)

    # optimize!
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for i in range(50):
        st = time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # import code; code.interact(local=locals())
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        et = time() - st
        tokens_per_sec = (train_loader.B * train_loader.T) / (et)
        print(
            f"step {i}, loss: {loss.item()}, et: {et*1e3:.2f}ms, tok/sec: {tokens_per_sec:.2f}"
        )
        # print(f"step {i}, loss: {loss.item()}") # loss is a tensor with a single element(cuda) -> item convert to float


if __name__ == "__main__":
    app()
