"""Improved small causal language model for MP1.

The submitted model replaces learned absolute positions with rotary position
embeddings (RoPE), LayerNorm with RMSNorm, and the GELU MLP with a
parameter-efficient SwiGLU feed-forward network. Setting ``rope`` to false in
the config gives a matched learned-position ablation.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned per-channel scale."""

    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        scale = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * scale.to(dtype=x.dtype)) * self.weight


def rotate_half(x):
    """Rotate adjacent feature pairs for rotary position embeddings."""
    pairs = x.unflatten(-1, (-1, 2))
    first, second = pairs.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_width, context, base=10_000.0):
        super().__init__()
        if head_width % 2:
            raise ValueError('RoPE requires an even attention-head width.')
        inverse = 1.0 / (base ** (torch.arange(0, head_width, 2).float() / head_width))
        angles = torch.outer(torch.arange(context).float(), inverse).repeat_interleave(2, dim=-1)
        self.register_buffer('cos', angles.cos()[None, None], persistent=False)
        self.register_buffer('sin', angles.sin()[None, None], persistent=False)

    def forward(self, x):
        length = x.shape[-2]
        return x * self.cos[:, :, :length] + rotate_half(x) * self.sin[:, :, :length]


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.out = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.out(F.silu(self.gate(x)) * self.value(x))


class Block(nn.Module):
    def __init__(self, width, heads, context, hidden, rope=True):
        super().__init__()
        if width % heads:
            raise ValueError('Model width must be divisible by the number of heads.')
        self.heads = heads
        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.mlp = SwiGLU(width, hidden)
        self.rotary = RotaryEmbedding(width // heads, context) if rope else None

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        if self.rotary is not None:
            q, k = self.rotary(q), self.rotary(k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        heads = config['heads']
        depth = config['depth']
        self.use_rope = config.get('rope', True)
        # 8/3 * width gives SwiGLU roughly the parameter count of a 4*width MLP.
        hidden = config.get('ffn_hidden', math.ceil((8 * width / 3) / 16) * 16)
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = None if self.use_rope else nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList([
            Block(width, heads, self.context, hidden, self.use_rope) for _ in range(depth)
        ])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight
        # Scale residual-output initialization as depth grows.
        residual_std = 0.02 / math.sqrt(2 * depth)
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=residual_std)
            nn.init.normal_(block.mlp.out.weight, std=residual_std)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return StudentGPT(config)
