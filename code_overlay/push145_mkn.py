"""Train-derived, normalized count-dependent-discount n-gram probabilities.

Only fixed train-derived distributions are cached across inference calls. Input
prefixes and every temporary backoff coefficient are recreated for each batch.
The compact integer encodings are lossless; probabilities remain FP32.
"""
import numpy as np
import torch
from torch import nn
from lowrank import LowRankGPT
from refined import cache_probs
from ngram_hybrid import prefix_codes

VOCAB = 2048


class MKNTables(nn.Module):
    def __init__(self, shapes):
        super().__init__()
        for name, shape in shapes.items():
            dtype = (torch.int64 if name.startswith('contexts_') else
                     torch.int32 if name.startswith('offsets_') else
                     torch.int16 if name.startswith('words_') else torch.float32)
            self.register_buffer(name, torch.empty(shape, dtype=dtype))
        self._static = None
        self.register_load_state_dict_post_hook(lambda module, incompatible: setattr(module, '_static', None))

    def numpy_tables(self):
        return {name: value.detach().cpu().numpy() for name, value in self.named_buffers()}

    @torch.no_grad()
    def probabilities(self, ids, max_order=5):
        x = ids.detach().cpu().numpy()
        arrays = self.numpy_tables()
        if self._static is None:
            bigram = np.broadcast_to(arrays['unigram'], (VOCAB, VOCAB)).copy()
            contexts = arrays['contexts_2']
            lengths = np.diff(arrays['offsets_2'])
            bigram[contexts] *= arrays['backoffs_2'][:, None]
            rows = np.repeat(contexts, lengths)
            bigram[rows, arrays['words_2']] += arrays['masses_2']
            self._static = bigram
        coefficient = np.ones(x.size, dtype=np.float32)
        additions = []
        for order in range(max_order, 2, -1):
            contexts = arrays[f'contexts_{order}']
            if not len(contexts):
                continue
            codes, valid = prefix_codes(x, order-1)
            positions = np.searchsorted(contexts, codes)
            safe = np.minimum(positions, len(contexts)-1)
            valid = valid & (positions < len(contexts)) & (contexts[safe] == codes)
            row_ids = np.flatnonzero(valid)
            selected = safe[valid]
            offsets = arrays[f'offsets_{order}']
            starts, ends = offsets[selected].astype(np.int64), offsets[selected+1].astype(np.int64)
            lengths = ends-starts
            rows = np.repeat(row_ids, lengths)
            indices = np.repeat(starts, lengths) + np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths)-lengths, lengths)
            words = arrays[f'words_{order}'][indices]
            masses = arrays[f'masses_{order}'][indices] * coefficient[rows]
            additions.append((rows, words, masses))
            coefficient[row_ids] *= arrays[f'backoffs_{order}'][selected]
        result = self._static[x.ravel()]
        result *= coefficient[:, None]
        for rows, words, masses in reversed(additions):
            result[rows, words] += masses
        return torch.from_numpy(result.reshape(*x.shape, VOCAB)).to(ids.device)


class MKNLowRank(LowRankGPT):
    def __init__(self, config):
        super().__init__(config)
        self.ngram = MKNTables(config['mkn_shapes'])

    @torch.no_grad()
    def predict_log_probs(self, ids):
        features = self.features(ids)
        probabilities = (self.head(features).float()/self.config.get('temperature', 1.)).softmax(-1)
        cache_weight = self.config.get('cache_weight', 0.)
        if cache_weight:
            memory = cache_probs(features, ids, self.config.get('cache_theta', 20.))
            memory[:, 0] = probabilities[:, 0]
            probabilities.mul_(1-cache_weight).add_(memory, alpha=cache_weight)
            del memory
        ngram = self.ngram.probabilities(ids, self.config.get('mkn_order', 5))
        weight = self.config['mkn_weight']
        probabilities.mul_(1-weight).add_(ngram, alpha=weight)
        return probabilities.clamp_min_(1e-30).log_()


def build_model(config):
    return MKNLowRank(config)
