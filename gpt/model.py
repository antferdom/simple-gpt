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

        # scale=1 / self.head_dim -> # mup
        # Flash Attention custom cuda kernels
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, nh, T, hs)

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