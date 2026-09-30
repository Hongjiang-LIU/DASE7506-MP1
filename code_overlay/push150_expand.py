"""Configurable projection capacity with the existing signed causal cache.

Rank zero means a dense projection. RoPE, attention, RMSNorm, SwiGLU, the
signed cache scalars/gate, and MKN interpolation are inherited unchanged.
Only the no-grad cache scatter avoids a full-vocabulary memory temporary.
"""
import copy
import math

import torch
from torch import nn

from efficient import EfficientGPT
from lowrank import LowRankLinear
from push145_cache import LearnedCacheGPT
from push147_signed_cache import SignedDistanceCache


PROJECTIONS = ('qkv', 'proj', 'gate', 'value', 'out')
LAYOUTS = {
    'expanded': dict(qkv=160, proj=0, gate=144, value=144, out=144),
    'dense': dict.fromkeys(PROJECTIONS, 0),
    'control': dict.fromkeys(PROJECTIONS, 128),
}


def validate_layout(layout, width, hidden):
    if not isinstance(layout, dict) or set(layout) != set(PROJECTIONS):
        raise ValueError('push150_ranks must specify exactly qkv/proj/gate/value/out')
    result = {}
    for name in PROJECTIONS:
        rank = layout[name]
        limit = width if name in ('qkv', 'proj') else min(width, hidden)
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank <= limit:
            raise ValueError(f'Invalid {name} rank {rank!r}; expected 0..{limit}, with 0=dense')
        result[name] = rank
    return result


def projection_modules(model):
    for index, block in enumerate(model.blocks):
        for name in ('qkv', 'proj'):
            yield f'blocks.{index}.{name}', name, getattr(block, name)
        for name in ('gate', 'value', 'out'):
            yield f'blocks.{index}.mlp.{name}', name, getattr(block.mlp, name)


class ExpandedSignedCache(SignedDistanceCache):
    def __init__(self, config):
        config = copy.deepcopy(config)
        width = config['width']
        hidden = config.get('ffn_hidden', math.ceil((8*width/3)/16)*16)
        layout = validate_layout(config.get('push150_ranks', LAYOUTS['expanded']), width, hidden)
        config['push150_ranks'] = layout
        config.setdefault('lowrank_rank', min(128, width, hidden))
        super().__init__(config)
        for block in self.blocks:
            for container, name in ((block, 'qkv'), (block, 'proj'),
                                    (block.mlp, 'gate'), (block.mlp, 'value'), (block.mlp, 'out')):
                old = getattr(container, name)
                inputs, outputs = old.left.in_features, old.right.out_features
                rank = layout[name]
                replacement = (nn.Linear(inputs, outputs, bias=False) if rank == 0
                               else LowRankLinear(inputs, outputs, rank))
                setattr(container, name, replacement)

    # Preserve exactly the existing real-valued RoPE/attention mathematics.
    features = EfficientGPT.features

    def neural_and_cache(self, ids):
        if torch.is_grad_enabled():
            return LearnedCacheGPT.neural_and_cache(self, ids)
        features = self.features(ids)
        logits = self.head(features).float()
        logits.div_(self.config.get('temperature', 1.15))
        probabilities = logits.softmax(-1)
        del logits
        attention, gate = self.cache_attention_and_gate(features)
        values = torch.cat([ids[:, 1:], ids.new_zeros(ids.shape[0], 1)], dim=1)
        probabilities.mul_(1-gate)
        probabilities.scatter_add_(-1, values[:, None, :].expand(-1, ids.shape[1], -1), attention*gate)
        return probabilities

    # SignedDistanceCache inherits a deliberately differentiable training path
    # and a no-grad MKN prediction path. Do not replace one with the other.


def build_model(config):
    return ExpandedSignedCache(config)
