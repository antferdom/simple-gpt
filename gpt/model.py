from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F

from typing import Optional


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
    def __init__(self, config: ModelArgs, alpha=0.5):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads
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

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1 / self.head_dim) # (B, nh, T, hs)

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

        for name, param in self.mlp.named_parameters():
            if "weight" in name:
                init.normal_(param, mean=0, std=0.5 * (1 / config.n_embd) ** 0.5)
            else:
                init.zeros_(param)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs, alpha=0.5):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.max_seq_len, config.n_embd),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.loss_fn = nn.CrossEntropyLoss()

        init.normal_(self.lm_head.weight, mean=0, std=alpha * (1 / config.n_embd))
        init.normal_(self.transformer.wte.weight, mean=0, std=alpha * 3.3)
    
    def gradient_checkpointing_enabled(self, ds_config):
        from deepspeed.runtime.activation_checkpointing import checkpointing
        checkpointing.configure(mpu_=None, deepspeed_config=ds_config)
        self._gradient_checkpointing_func = checkpointing.checkpoint
        self.num_checkpoints = ds_config['activation_checkpointing']['number_checkpoints']

    def forward(self, idx):
        # idx is of shape (B, T)
        B, T = idx.size()
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

        outputs = {"logits": logits}
        if idx is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = idx[..., 1:].contiguous()

            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            outputs["loss"] = loss
        return outputs