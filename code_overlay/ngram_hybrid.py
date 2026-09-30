"""Compact train-only exact-key n-gram model mixed with a causal neural model.

All persistent tables are learned from train text. Temporary prefixes reset at
every call and every example. No target argument is accepted at inference.
"""
import numpy as np
import torch
from torch import nn
from refined import RefinedGPT, cache_probs

VOCAB = 2048
BITS = 11


def prefix_codes(ids, history):
    """Exact base-2048 encodings, independently for each input row."""
    ids = np.asarray(ids, dtype=np.int64)
    if history > ids.shape[1]:
        return np.zeros(ids.size,dtype=np.int64),np.zeros(ids.size,dtype=bool)
    result = ids.copy()
    if history <= 4:
        for offset in range(1, history):
            result[:, offset:] += ids[:, :-offset] << (BITS*offset)
    else:
        # Fixed 52-bit rolling hash for longer train contexts; any collisions
        # consistently merge training counts, never introduce held-out data.
        result = np.zeros_like(ids)
        for offset in range(history-1,-1,-1):
            shifted = np.zeros_like(ids)
            shifted[:,offset:] = ids[:,:ids.shape[1]-offset] if offset else ids
            result = (result * 1000003 + shifted + 1) & ((1<<52)-1)
    valid = np.broadcast_to(np.arange(ids.shape[1]) >= history-1, ids.shape)
    return result.ravel(), valid.ravel()


class NgramTables(nn.Module):
    def __init__(self, shapes):
        super().__init__()
        for name, shape in shapes.items():
            dtype = torch.int64 if name.startswith(('keys_', 'contexts_', 'offsets_')) else torch.float32
            self.register_buffer(name, torch.empty(shape, dtype=dtype))
        self._static = {}
        self.register_load_state_dict_post_hook(lambda module, incompatible: module._static.clear())

    def numpy_tables(self):
        return {name: value.detach().cpu().numpy() for name, value in self.named_buffers()}

    @torch.no_grad()
    def probabilities(self, ids, discount=.75, max_order=5):
        # The CPU count predictor uses float32, including all interpolation.
        x = ids.detach().cpu().numpy()
        arrays = self.numpy_tables()
        # Memoize ONLY training-derived probabilities, never evaluation inputs.
        if discount not in self._static:
            bigram=np.broadcast_to(arrays['unigram'],(VOCAB,VOCAB)).copy()
            ctx=arrays['contexts_2']; offsets=arrays['offsets_2']; lengths=np.diff(offsets)
            totals=arrays['totals_2']
            bigram[ctx] *= (np.float32(discount)*lengths.astype(np.float32)/totals)[:,None]
            rows=np.repeat(ctx,lengths)
            bigram[rows,arrays['keys_2'] & (VOCAB-1)] += (arrays['counts_2']-np.float32(discount))/np.repeat(totals,lengths)
            self._static[discount]=bigram
        coefficient = np.ones(x.size,dtype=np.float32)
        additions = []
        # Expand recursive backoff algebra sparsely; only one dense scaling.
        for order in range(max_order,2,-1):
            codes, valid = prefix_codes(x, order-1)
            contexts = arrays[f'contexts_{order}']
            if not len(contexts):
                continue
            positions = np.searchsorted(contexts, codes)
            safe = np.minimum(positions, len(contexts)-1)
            valid = valid & (positions < len(contexts)) & (contexts[safe] == codes)
            row_ids = np.flatnonzero(valid)
            selected = safe[valid]
            offsets = arrays[f'offsets_{order}']
            starts, ends = offsets[selected], offsets[selected+1]
            lengths = ends-starts
            totals = arrays[f'totals_{order}'][selected]
            rows = np.repeat(row_ids, lengths)
            indices = np.repeat(starts, lengths) + np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths)-lengths, lengths)
            word_ids = arrays[f'keys_{order}'][indices] & (VOCAB-1)
            masses = (arrays[f'counts_{order}'][indices]-np.float32(discount))/np.repeat(totals, lengths)
            masses *= coefficient[rows]
            additions.append((rows,word_ids,masses))
            coefficient[row_ids] *= np.float32(discount)*lengths.astype(np.float32)/totals
        result = self._static[discount][x.ravel()]
        result *= coefficient[:,None]
        for rows,word_ids,masses in reversed(additions):
            result[rows,word_ids] += masses
        return torch.from_numpy(result.reshape(*x.shape, VOCAB)).to(ids.device)


def reliability_gate(arrays,x,maximum,minimum=3,scale=5.):
    evidence=np.zeros(x.size,dtype=np.float32)
    for order in range(minimum,maximum+1):
        codes,valid=prefix_codes(x,order-1)
        contexts=arrays[f'contexts_{order}']
        if not len(contexts):
            continue
        pos=np.searchsorted(contexts,codes)
        safe=np.minimum(pos,len(contexts)-1)
        valid=valid & (pos<len(contexts)) & (contexts[safe]==codes)
        evidence[valid]=arrays[f'totals_{order}'][safe[valid]]
    if scale==0:
        return (evidence>0).astype(np.float32).reshape(x.shape)
    return (evidence/(evidence+np.float32(scale))).reshape(x.shape)


class HybridGPT(RefinedGPT):
    def __init__(self, config):
        super().__init__(config)
        self.ngram = NgramTables(config['ngram_shapes'])

    @torch.no_grad()
    def predict_log_probs(self, ids):
        features = self.features(ids)
        probabilities = (self.head(features).float()/self.config.get('temperature',1.)).softmax(-1)
        cache_weight = self.config.get('cache_weight',0.)
        if cache_weight:
            if self.config.get('memory_kind')=='exact':
                from exact_refined import exact_cache
                memory,evidence=exact_cache(ids,self.config['exact_history'])
                scale=self.config.get('exact_scale',0.)
                gate=(evidence>0).float() if scale==0 else evidence/(evidence+scale)
                mix=(gate*cache_weight)[:,:,None]
                probabilities.mul_(1-mix).add_(memory*mix)
            else:
                memory = cache_probs(features,ids,self.config.get('cache_theta',20.))
                memory[:,0] = probabilities[:,0]
                probabilities.mul_(1-cache_weight).add_(memory,alpha=cache_weight)
            del memory
        ngram_p = self.ngram.probabilities(ids, self.config['ngram_discount'], self.config['ngram_order'])
        weight = self.config['ngram_weight']
        if self.config.get('ngram_gate_min_order',0):
            gate=reliability_gate(self.ngram.numpy_tables(),ids.detach().cpu().numpy(),self.config['ngram_order'],
                                  self.config['ngram_gate_min_order'],self.config['ngram_gate_scale'])
            mix=torch.from_numpy(gate).to(ids.device).unsqueeze(-1)*weight
            probabilities.mul_(1-mix).add_(ngram_p*mix)
        else:
            probabilities.mul_(1-weight).add_(ngram_p,alpha=weight)
        return probabilities.clamp_min_(1e-30).log_()


def build_model(config):
    return HybridGPT(config)
