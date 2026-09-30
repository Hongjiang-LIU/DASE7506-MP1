"""Inference algebra with fused SwiGLU input projection, same trained weights."""
import torch
from torch.nn import functional as F
from refined import RefinedGPT


def fast_features(self,ids):
    x=self.token(ids)
    if hasattr(self,'short_embeddings'):
        for history,(size,table,projection) in enumerate(zip(self.ngram_buckets,self.short_embeddings,self.short_projections),start=2):
            if ids.shape[1] < history:
                continue
            code=torch.zeros_like(ids[:,history-1:])
            for offset in range(history):
                code=(code*1000003+ids[:,offset:offset+code.shape[1]]+1)%size
            x=x+F.pad(projection(table(code)),(0,0,history-1,0))
    if self.pos is not None:
        x=x+self.pos(torch.arange(ids.shape[1],device=ids.device))
    for block in self.blocks:
        batch,length,width=x.shape
        normed=F.rms_norm(x,(width,),block.norm1.weight,eps=1e-6)
        q,k,v=block.qkv(normed).view(batch,length,3,block.heads,width//block.heads).permute(2,0,3,1,4)
        if block.rotary is not None:
            q=block.rotary(q).to(v.dtype);k=block.rotary(k).to(v.dtype)
        attention=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x=x+block.proj(attention.transpose(1,2).reshape(batch,length,width))
        normed=F.rms_norm(x,(width,),block.norm2.weight,eps=1e-6)
        gate,value=F.linear(normed,torch.cat([block.mlp.gate.weight,block.mlp.value.weight])).chunk(2,dim=-1)
        x=x+block.mlp.out(F.silu(gate)*value)
    return F.rms_norm(x,(x.shape[-1],),self.norm.weight,eps=1e-6)


class FastRefinedGPT(RefinedGPT):
    features=fast_features


def build_model(config):
    return FastRefinedGPT(config)
