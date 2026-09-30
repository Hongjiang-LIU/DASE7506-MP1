"""Causal learned short-prefix embeddings; no external assets or hidden state."""
import torch
from torch import nn
from torch.nn import functional as F
from efficient import EfficientGPT
from refined import RefinedGPT


class SparseGPT(EfficientGPT):
    def __init__(self,config):
        super().__init__(config)
        dim=config.get('ngram_dim',48)
        self.ngram_buckets=config.get('ngram_buckets',[16381,65521])
        self.short_embeddings=nn.ModuleList([nn.Embedding(size,dim) for size in self.ngram_buckets])
        self.short_projections=nn.ModuleList([nn.Linear(dim,config['width'],bias=False) for _ in self.ngram_buckets])
        for table,projection in zip(self.short_embeddings,self.short_projections):
            nn.init.normal_(table.weight,std=.02)
            nn.init.normal_(projection.weight,std=.05)

    def features(self,ids):
        x=self.token(ids)
        length=ids.shape[1]
        for history,(size,table,projection) in enumerate(zip(self.ngram_buckets,self.short_embeddings,self.short_projections),start=2):
            if length < history:
                continue
            code=torch.zeros_like(ids[:,history-1:])
            for offset in range(history):
                code=(code*1000003+ids[:,offset:offset+code.shape[1]]+1)%size
            addition=projection(table(code))
            x=x+F.pad(addition,(0,0,history-1,0))
        if self.pos is not None:
            x=x+self.pos(torch.arange(length,device=ids.device))
        dropout=self.config.get('dropout',.2)
        x=F.dropout(x,p=dropout,training=self.training)
        for block in self.blocks:
            batch,length,width=x.shape
            q,k,v=block.qkv(block.norm1(x)).view(batch,length,3,block.heads,width//block.heads).permute(2,0,3,1,4)
            if block.rotary is not None:
                q=block.rotary(q).to(v.dtype);k=block.rotary(k).to(v.dtype)
            attended=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            branch=block.proj(attended.transpose(1,2).reshape(batch,length,width))
            x=x+F.dropout(branch,p=dropout,training=self.training)
            x=x+F.dropout(block.mlp(block.norm2(x)),p=dropout,training=self.training)
        return self.norm(x)

    predict_log_probs=RefinedGPT.predict_log_probs


def build_model(config):
    return SparseGPT(config)
