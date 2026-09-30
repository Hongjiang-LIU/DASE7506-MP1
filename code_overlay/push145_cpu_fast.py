"""Same trained predictor, FP32 real-valued inference with fewer temporaries."""
import math
import torch
from torch.nn import functional as F
from push145_cache_hybrid import LearnedCacheHybrid


def pair_rotary(x, rotary):
    length=x.shape[-2]
    cosine=rotary.cos[:,:,:length,::2]
    sine=rotary.sin[:,:,:length,::2]
    even,odd=x[...,::2],x[...,1::2]
    result=torch.empty_like(x)
    result[...,::2]=torch.addcmul(even*cosine,odd,sine,value=-1)
    result[...,1::2]=torch.addcmul(odd*cosine,even,sine)
    return result


class FastLearnedCache(LearnedCacheHybrid):
    def _features_part(self, ids):
        x=self.token(ids)
        if self.pos is not None:
            x=x+self.pos(torch.arange(ids.shape[1],device=ids.device))
        for block in self.blocks:
            batch,length,width=x.shape
            q,k,v=block.qkv(block.norm1(x)).view(batch,length,3,block.heads,width//block.heads).permute(2,0,3,1,4)
            if block.rotary is not None:
                q=pair_rotary(q,block.rotary)
                k=pair_rotary(k,block.rotary)
            attended=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            x=x+block.proj(attended.transpose(1,2).reshape(batch,length,width))
            x=x+block.mlp(block.norm2(x))
        return self.norm(x)

    def features(self, ids):
        # This wrapper is for evaluation only; all parameters remain unchanged.
        if self.training:
            return super().features(ids)
        size=self.config.get('cpu_feature_microbatch',32)
        if ids.shape[0]<=size:
            return self._features_part(ids)
        return torch.cat([self._features_part(part) for part in ids.split(size)],0)

    def cache_attention_and_gate(self, features):
        _,length,_=features.shape
        normalized=F.normalize(features.float(),dim=-1)
        scores=normalized@normalized.transpose(-1,-2)
        scores.mul_(self.config.get('cache_theta',15.))
        query=self.pointer_query(normalized).float()
        key=self.pointer_key(normalized).float()
        scores.add_(query@key.transpose(-1,-2),alpha=1/math.sqrt(query.shape[-1]))
        allowed=torch.ones(length,length,device=features.device,dtype=torch.bool).tril(-1)
        scores.masked_fill_(~allowed,-1e4)
        attention=scores.softmax(-1)
        position=torch.arange(length,device=features.device,dtype=torch.float32)
        history=(position/max(1,self.context-1))[None,:].expand(features.shape[0],-1)
        gate_input=torch.cat([normalized,attention.amax(-1)[...,None],
            attention.square().sum(-1)[...,None],history[...,None]],-1)
        gate=self.pointer_gate(gate_input).float().sigmoid()*self.config.get('pointer_max_weight',.5)
        return attention,gate*(position>0)[None,:,None]

    @torch.no_grad()
    def neural_and_cache(self, ids):
        features=self.features(ids)
        logits=self.head(features).float()
        logits.div_(self.config.get('temperature',1.2))
        probabilities=logits.softmax(-1)
        del logits
        attention,gate=self.cache_attention_and_gate(features)
        values=torch.cat([ids[:,1:],ids.new_zeros(ids.shape[0],1)],dim=1)
        probabilities.mul_(1-gate)
        probabilities.scatter_add_(-1,values[:,None,:].expand(-1,ids.shape[1],-1),attention*gate)
        return probabilities


def build_model(config):
    return FastLearnedCache(config)
