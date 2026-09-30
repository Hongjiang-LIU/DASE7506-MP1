"""A 57-parameter train-only statistical mixture gate; target-free inference.

The full train-derived MKN table and every neural/cache weight remain frozen.
The extra statistics features share the SAME lookup that produces the MKN
distribution. No training rows, OOF tables, or evaluated prefixes are persisted
in this inference module.
"""
import copy
import math

import numpy as np
import torch
from torch import nn

from ngram_hybrid import prefix_codes
from push145_mkn import MKNTables, VOCAB
from push150_expand import ExpandedSignedCache, PROJECTIONS


FEATURE_COUNT = 5
NORMALIZATION_EPSILON = 1e-6
CONFIDENCE_EPSILON = 1e-6
MAX_WEIGHT = .5
FEATURE_NAMES = ('neural_cache_max_logit', 'pointer_gate', 'high_order_mass',
                 'highest_order_divided_by_5', 'log1p_highest_context_branches')


class AuxiliaryMKNTables(MKNTables):
    def __init__(self, shapes):
        super().__init__(shapes)
        self._aux_static = None
        self.register_load_state_dict_post_hook(
            lambda module, incompatible: setattr(module, '_aux_static', None))

    @torch.no_grad()
    def probabilities_and_aux(self, ids, max_order=5):
        if max_order != 5:
            raise ValueError('The declared gate uses order-five MKN.')
        x = ids.detach().cpu().numpy()
        arrays = self.numpy_tables()
        if self._static is None:
            # Deliberately retain the original FP32 probability arithmetic.
            bigram = np.broadcast_to(arrays['unigram'], (VOCAB, VOCAB)).copy()
            contexts = arrays['contexts_2']
            lengths = np.diff(arrays['offsets_2'])
            bigram[contexts] *= arrays['backoffs_2'][:, None]
            rows = np.repeat(contexts, lengths)
            bigram[rows, arrays['words_2']] += arrays['masses_2']
            self._static = bigram
        if self._aux_static is None:
            aux = np.zeros((VOCAB, 2), dtype=np.float32)
            contexts = arrays['contexts_2']
            aux[contexts, 0] = 2
            aux[contexts, 1] = np.log1p(np.diff(arrays['offsets_2'])).astype(np.float32)
            self._aux_static = aux
        highest = self._aux_static[x.ravel()].copy()
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
            more_specific = order > highest[row_ids, 0]
            selected_rows = row_ids[more_specific]
            highest[selected_rows, 0] = order
            highest[selected_rows, 1] = np.log1p(lengths[more_specific]).astype(np.float32)
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
        aux = np.column_stack((np.float32(1)-coefficient, highest[:, 0]/np.float32(5),
                               highest[:, 1])).astype(np.float32)
        return (torch.from_numpy(result.reshape(*x.shape, VOCAB)).to(ids.device),
                torch.from_numpy(aux.reshape(*x.shape, 3)).to(ids.device))

    @torch.no_grad()
    def probabilities(self, ids, max_order=5):
        return self.probabilities_and_aux(ids, max_order)[0]


def gate_features(neural_probabilities, pointer_gate, statistics_aux):
    maximum = neural_probabilities.amax(-1).clamp(CONFIDENCE_EPSILON, 1-CONFIDENCE_EPSILON)
    confidence = maximum.log()-torch.log1p(-maximum)
    if pointer_gate.shape == (*maximum.shape, 1):
        pointer_gate = pointer_gate.squeeze(-1)
    return torch.cat((confidence[..., None], pointer_gate[..., None], statistics_aux), dim=-1).float()


class TinyMixtureGate(nn.Module):
    def __init__(self, initial_weight):
        super().__init__()
        if not 0 < initial_weight < MAX_WEIGHT:
            raise ValueError('Parent statistical weight must be strictly between zero and .5.')
        self.register_buffer('feature_mean', torch.zeros(FEATURE_COUNT, dtype=torch.float32))
        self.register_buffer('feature_std', torch.ones(FEATURE_COUNT, dtype=torch.float32))
        self.network = nn.Sequential(nn.Linear(5, 8), nn.SiLU(), nn.Linear(8, 1))
        nn.init.zeros_(self.network[-1].weight)
        fraction = initial_weight/MAX_WEIGHT
        nn.init.constant_(self.network[-1].bias, math.log(fraction/(1-fraction)))

    def forward(self, features, offset=0.):
        normalized = (features.float()-self.feature_mean)/self.feature_std.clamp_min(NORMALIZATION_EPSILON)
        return MAX_WEIGHT*torch.sigmoid(self.network(normalized).squeeze(-1)+offset)


class AdaptiveMKNCache(ExpandedSignedCache):
    def __init__(self, config):
        config = copy.deepcopy(config)
        # Original signed-cache weights use exactly these five factored ranks.
        if 'push150_ranks' not in config:
            config['push150_ranks'] = dict.fromkeys(PROJECTIONS, config['lowrank_rank'])
        super().__init__(config)
        if not self.config.get('mkn_shapes') or self.config.get('mkn_order') != 5:
            raise ValueError('Full order-five compact MKN tables are required.')
        self.ngram = AuxiliaryMKNTables(self.config['mkn_shapes'])
        self.mixture_gate = TinyMixtureGate(self.config['mkn_weight'])

    @torch.no_grad()
    def neural_and_pointer(self, ids):
        # Same full FP32 neural/cache calculation as ExpandedSignedCache.
        features = self.features(ids)
        logits = self.head(features).float()
        logits.div_(self.config.get('temperature', 1.15))
        probabilities = logits.softmax(-1)
        del logits
        attention, pointer_gate = self.cache_attention_and_gate(features)
        values = torch.cat([ids[:, 1:], ids.new_zeros(ids.shape[0], 1)], dim=1)
        probabilities.mul_(1-pointer_gate)
        probabilities.scatter_add_(-1, values[:, None, :].expand(-1, ids.shape[1], -1), attention*pointer_gate)
        return probabilities, pointer_gate.squeeze(-1)

    @torch.no_grad()
    def predict_log_probs(self, ids):
        probabilities, pointer_gate = self.neural_and_pointer(ids)
        counts, aux = self.ngram.probabilities_and_aux(ids, 5)
        features = gate_features(probabilities, pointer_gate, aux)
        weight = self.mixture_gate(features, self.config.get('push150_gate_offset', 0.))[..., None]
        # No extra vocabulary-sized product temporary and no second lookup.
        probabilities.mul_(1-weight).add_(counts.mul_(weight))
        return probabilities.clamp_min_(1e-30).log_()

    def training_log_probs(self, ids):
        raise RuntimeError('This declared experiment trains only the tiny head on OOF feature rows.')


def model_from_parent(parent, seed=181):
    if parent.get('implementation') not in ('push147_signed_cache', 'push150_expand'):
        raise ValueError('Unsupported gate parent implementation.')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AdaptiveMKNCache(parent['config'])
    original = parent['model']
    expected = {key for key in model.state_dict() if not key.startswith('mixture_gate.')}
    if set(original) != expected:
        raise ValueError('Parent state does not map exactly to the gate base model.')
    for key, value in original.items():
        target = model.state_dict()[key]
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f'Parent state shape/dtype mismatch: {key}')
    loaded = model.load_state_dict(original, strict=False)
    if loaded.unexpected_keys or any(not key.startswith('mixture_gate.') for key in loaded.missing_keys):
        raise AssertionError('Only new gate tensors may be absent.')
    model.eval().requires_grad_(False)
    return model


def assert_frozen_parent(parent, model):
    state = model.state_dict()
    expected = {key for key in state if not key.startswith('mixture_gate.')}
    if expected != set(parent['model']):
        raise AssertionError('Frozen base state key set changed.')
    for key, value in parent['model'].items():
        actual = state[key].detach().cpu()
        if actual.dtype != value.dtype or actual.shape != value.shape or not torch.equal(actual, value):
            raise AssertionError(f'Frozen neural/statistical tensor changed: {key}')
    for key, value in parent['config'].items():
        if model.config.get(key) != value:
            raise AssertionError(f'Frozen parent configuration changed: {key}')


def build_model(config):
    return AdaptiveMKNCache(config)
