"""Optional zero-padding of attention head width; same FP32 attention scale."""
import math
import torch
from torch.nn import functional as F
from push145_cpu_fast import FastLearnedCache,pair_rotary


class PaddedAttentionCache(FastLearnedCache):
    def _features_part(self,ids):
        x=self.token(ids)
        if self.pos is not None:
            x=x+self.pos(torch.arange(ids.shape[1],device=ids.device))
        for block in self.blocks:
            batch,length,width=x.shape
            head_width=width//block.heads
            q,k,v=block.qkv(block.norm1(x)).view(batch,length,3,block.heads,head_width).permute(2,0,3,1,4)
            if block.rotary is not None:
                q=pair_rotary(q,block.rotary)
                k=pair_rotary(k,block.rotary)
            pad_to=self.config.get('cpu_attention_pad_to',64)
            if pad_to>head_width:
                padding=(0,pad_to-head_width)
                q,k,v=F.pad(q,padding),F.pad(k,padding),F.pad(v,padding)
            attended=F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=1/math.sqrt(head_width))
            attended=attended[...,:head_width]
            x=x+block.proj(attended.transpose(1,2).reshape(batch,length,width))
            x=x+block.mlp(block.norm2(x))
        return self.norm(x)


def build_model(config):
    return PaddedAttentionCache(config)
