import math
from time import time
from pathlib import Path
from dataclasses import dataclass
import torch
import torch._inductor.config
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode
from torch.amp.grad_scaler import GradScaler

from torchao.sparsity.training import (
    SemiSparseLinear,
    swap_linear_with_semi_sparse_linear,
)

import tiktoken

from typing import Optional
import lovely_tensors as lt
lt.monkey_patch()


# global variables
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
TINY_SHAKESPEARE_FN = DATASET_DIR / "input.txt"

@dataclass
class ModelArgs:
    max_seq_len: int = 1024 # max sequence length (context window)
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layers: int = 12 # number of layers
    n_heads: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None  # hidden layer size in feedforward network is ffn_dim_multiplier times n_embd



class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelArgs, alpha=0.5, flash=False):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads
        self.flash = flash
        assert (
            self.head_dim * self.n_heads == self.n_embd
        ), "embed_dim must be divisible by num_heads"
        assert self.n_embd % self.n_heads == 0

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)

        # Custom initialization for linear layers (muP-initialization)
        for name, param in self.c_attn.named_parameters():
            if "weight" in name:
                torch.nn.init.normal_(param, mean=0, std=alpha * (1 / config.n_embd) ** 0.5)
        for name, param in self.c_proj.named_parameters():
            if "weight" in name:
                torch.nn.init.normal_(param, mean=0, std=alpha * (1 / config.n_embd) ** 0.5)

        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                                    .view(1, 1, config.max_seq_len, config.max_seq_len))


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            """
            mup
                attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=True, scale=1 / self.head_dim)
            """
        if not self.flash:
            # attention (materializes the large (T,T) matrix for all the queries and keys)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class FeedForward(nn.Module):
    def __init__(
            self,
            n_embd: int,
            hidden_dim: int,
            ):
        super().__init__()
        self.n_embd = n_embd
        self.hidden_dim = hidden_dim
        self.c_fc    = nn.Linear(self.n_embd, self.hidden_dim)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(self.hidden_dim, self.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads
        self.ln_1 = nn.LayerNorm(self.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(self.n_embd)
        self.mlp = FeedForward(
            n_embd=config.n_embd,
            hidden_dim=4 * config.n_embd
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.max_seq_len, config.n_embd),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        """
        ref: Using the Output Embedding to Improve Language Models
        https://arxiv.org/abs/1608.05859

        we avoid tie weights in this implementation for esaier distributed training
        """
        # weight sharing scheme
        # self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.max_seq_len, f"Cannot forward sequence of length {T}, block size is only {self.config.max_seq_len}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layers, n_heads and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layers=12, n_heads=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layers=24, n_heads=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layers=36, n_heads=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layers=48, n_heads=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257 # always 50257 for GPT model checkpoints
        config_args["max_seq_len"] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized GPT model
        config = ModelArgs(**config_args)
        model = Transformer(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open(TINY_SHAKESPEARE_FN.as_posix(), 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
    

if __name__ == "__main__":   
    """
    model = Transformer.from_pretrained("gpt2")
    print("didn't crash yay!")
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"device: {device}")

    seed = 1337
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    train_loader = DataLoaderLite(B=16, T=1024)

    torch.set_float32_matmul_precision("high")

    model = Transformer(ModelArgs())

    # print params: should be 0 if zero-3
    total_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params: {total_params}")

    # enable sparsity only in the feedfoward layers
    def swap_ffn_linear_layers(model):
        sparse_config = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(sub_name in name for sub_name in ['mlp.c_fc', 'mlp.c_proj']):
                sparse_config[name] = SemiSparseLinear
        swap_linear_with_semi_sparse_linear(model, sparse_config)

        return model

    model = model.to(device).to(torch.float16)
    model = swap_ffn_linear_layers(model)
    #import code; code.interact(local=locals())
    torch._inductor.config.max_autotune = True
    torch._inductor.config.max_autotune_gemm = True
    torch._inductor.config.max_autotune_gemm_search_space = "DEFAULT"
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_check_all_directions = True
    model = torch.compile(model)

    # optimize!
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    run_name = f"LR:{3e-4}__Nhead:{ModelArgs().n_heads}_NLayer:{ModelArgs().n_layers}_EmbDim:{ModelArgs().n_embd}"
    print(f"run_name: {run_name}")

    for i in range(50):
        st = time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, loss = model(x, y)
        # import code; code.interact(local=locals())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        et = time() - st
        tokens_per_sec = (train_loader.B * train_loader.T) / (et)
        print(f"step {i}, loss: {loss.item()}, et: {et*1e3:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
        # print(f"step {i}, loss: {loss.item()}") # loss is a tensor with a single element(cuda) -> item convert to float