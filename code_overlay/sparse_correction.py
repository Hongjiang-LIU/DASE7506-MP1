"""Normalized sparse train-count corrections with the neural LM as backoff."""
import numpy as np
import torch
from fast_ngram_hybrid import FastHybrid
from refined import cache_probs
from ngram_hybrid import prefix_codes


def lookup_context(arrays,x,order):
    codes,valid=prefix_codes(x,order-1)
    contexts=arrays[f'contexts_{order}']
    pos=np.searchsorted(contexts,codes)
    safe=np.minimum(pos,len(contexts)-1)
    valid=valid & (pos<len(contexts)) & (contexts[safe]==codes)
    return codes,valid,safe


def corrected_targets(arrays,x,y,neural,minimum,maximum,strength,scale):
    result=neural.ravel().copy()
    target=y.ravel().clip(min=0)
    for order in range(minimum,maximum+1):
        codes,valid,safe=lookup_context(arrays,x,order)
        totals=arrays[f'totals_{order}'][safe]
        mix=strength*totals/(totals+scale)
        keys=arrays[f'keys_{order}'];query=(codes<<11)|target
        pos=np.searchsorted(keys,query);selected=np.minimum(pos,len(keys)-1)
        found=(pos<len(keys)) & (keys[selected]==query)
        p=np.where(found,arrays[f'counts_{order}'][selected]/totals,0.)
        result=np.where(valid,(1-mix)*result+mix*p,result).astype(np.float32)
    return result.reshape(y.shape)


class SparseCorrectionGPT(FastHybrid):
    @torch.no_grad()
    def predict_log_probs(self,ids):
        features=self.features(ids)
        probabilities=(self.head(features).float()/self.config.get('temperature',1.)).softmax(-1)
        weight=self.config.get('cache_weight',0.)
        if weight:
            if self.config.get('memory_kind')=='exact':
                from exact_refined import exact_cache
                memory,evidence=exact_cache(ids,self.config['exact_history'])
                scale=self.config.get('exact_scale',0.)
                gate=(evidence>0).float() if scale==0 else evidence/(evidence+scale)
                mix=(gate*weight)[:,:,None]
                probabilities.mul_(1-mix).add_(memory*mix)
            else:
                memory=cache_probs(features,ids,self.config.get('cache_theta',20.))
                memory[:,0]=probabilities[:,0]
                probabilities.mul_(1-weight).add_(memory,alpha=weight)
            del memory
        x=ids.detach().cpu().numpy();arrays=self.ngram.numpy_tables()
        coefficient=np.ones(x.size,dtype=np.float32);additions=[]
        strength,scale=self.config['correction_strength'],self.config['correction_scale']
        for order in range(self.config['correction_max'],self.config['correction_min']-1,-1):
            _,valid,safe=lookup_context(arrays,x,order)
            rows=np.flatnonzero(valid);selected=safe[valid]
            totals=arrays[f'totals_{order}'][selected]
            mix=np.float32(strength)*totals/(totals+np.float32(scale))
            offsets=arrays[f'offsets_{order}'];starts=offsets[selected];lengths=offsets[selected+1]-starts
            rr=np.repeat(rows,lengths)
            ii=np.repeat(starts,lengths)+np.arange(lengths.sum())-np.repeat(np.cumsum(lengths)-lengths,lengths)
            words=arrays[f'keys_{order}'][ii]&2047
            masses=arrays[f'counts_{order}'][ii]*np.repeat(mix/totals,lengths)*coefficient[rr]
            additions.append((rr,words,masses));coefficient[rows]*=1-mix
        output=probabilities.cpu().numpy().reshape(x.size,2048)
        output*=coefficient[:,None]
        for rows,words,masses in reversed(additions):
            output[rows,words]+=masses
        result=torch.from_numpy(output.reshape(*x.shape,2048)).to(ids.device)
        return result.clamp_min_(1e-30).log_()


def build_model(config):
    return SparseCorrectionGPT(config)
