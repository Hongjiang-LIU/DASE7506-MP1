"""Exact within-window prefix cache: observed tokens only, no retained state."""
import torch
from fast_refined import FastRefinedGPT


def exact_cache(ids,maximum=4):
    batch,length=ids.shape
    positions=torch.arange(length,device=ids.device)
    causal=positions[None,:] < positions[:,None]
    codes=ids.clone()
    attention=torch.zeros(batch,length,length,device=ids.device,dtype=torch.float32)
    evidence=torch.zeros(batch,length,device=ids.device,dtype=torch.float32)
    for history in range(1,min(maximum,length)+1):
        if history>1:
            codes[:,history-1:] += ids[:,:length-history+1] << (11*(history-1))
        allowed=causal & (positions[None,:]>=history-1) & (positions[:,None]>=history-1)
        matches=(codes[:,:,None]==codes[:,None,:]) & allowed
        count=matches.sum(-1).float()
        nonempty=count>0
        attention=torch.where(nonempty[:,:,None],matches.float()/count.clamp_min(1)[:,:,None],attention)
        evidence=torch.where(nonempty,count,evidence)
    values=torch.cat([ids[:,1:],ids.new_zeros(batch,1)],dim=1)
    result=torch.zeros(batch,length,2048,device=ids.device,dtype=torch.float32)
    result.scatter_add_(-1,values[:,None,:].expand(-1,length,-1),attention)
    return result,evidence


class ExactRefined(FastRefinedGPT):
    @torch.no_grad()
    def predict_log_probs(self,ids):
        features=self.features(ids)
        probabilities=(self.head(features).float()/self.config.get('temperature',1.)).softmax(-1)
        memory,evidence=exact_cache(ids,self.config['exact_history'])
        scale=self.config.get('exact_scale',0.)
        gate=(evidence>0).float() if scale==0 else evidence/(evidence+scale)
        mix=(gate*self.config['cache_weight'])[:,:,None]
        probabilities.mul_(1-mix).add_(memory*mix)
        return probabilities.clamp_min_(1e-30).log_()


def build_model(config):
    return ExactRefined(config)
