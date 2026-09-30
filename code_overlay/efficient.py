"""Training dropout and BF16-compatible RoPE; same eval architecture as StudentGPT."""
import torch
from torch.nn import functional as F
from student import StudentGPT


class EfficientGPT(StudentGPT):
    def features(self, ids):
        x = self.token(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        dropout = self.config.get('dropout', 0.1)
        x = F.dropout(x, p=dropout, training=self.training)
        for block in self.blocks:
            batch, length, width = x.shape
            q, k, v = block.qkv(block.norm1(x)).view(
                batch, length, 3, block.heads, width // block.heads
            ).permute(2, 0, 3, 1, 4)
            if block.rotary is not None:
                q = block.rotary(q).to(v.dtype)
                k = block.rotary(k).to(v.dtype)
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            branch = block.proj(attended.transpose(1, 2).reshape(batch, length, width))
            x = x + F.dropout(branch, p=dropout, training=self.training)
            x = x + F.dropout(block.mlp(block.norm2(x)), p=dropout, training=self.training)
        return self.norm(x)


def build_model(config):
    return EfficientGPT(config)
