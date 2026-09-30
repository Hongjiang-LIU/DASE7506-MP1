"""Four bounded, learned exact-suffix priors on the existing causal pointer.

Only input ids are used: a match ending at source j<t copies ids[j+1], which
is already observed by position t. No input, prefix, attention or output is
retained between calls. Global train-derived MKN interpolation is unchanged.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from push150_expand import ExpandedSignedCache


SUFFIX_ORDERS = (1, 2, 3, 4)
SUFFIX_BOUNDS = (-4., 4.)
SUFFIX_PARAMETER_NAMES = tuple(f'cache_suffix_bias_{order}' for order in SUFFIX_ORDERS)


def suffix_match_masks(ids, vocab_size=2048):
    """Yield (order, causal B*T*T Boolean mask), with no hash collisions.

    The fixed course vocabulary needs 11 bits/token, hence at most 44 bits
    for four tokens. Invalid short prefixes are explicitly masked, not
    represented by a padding token that might occur in real input. Codes
    and masks are call-local; only one new order is constructed at a time.
    """
    if ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected a two-dimensional integer input tensor')
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < 2:
        raise ValueError('Expected an integer vocabulary size >= 2')
    bits = (vocab_size-1).bit_length()
    if bits*max(SUFFIX_ORDERS) > 63:
        raise ValueError('Exact four-token codes must fit signed int64')
    length = ids.shape[1]
    if length < 1:
        raise ValueError('An input window cannot be empty')
    tokens = ids.to(dtype=torch.int64)
    codes = tokens.clone()
    position = torch.arange(length, device=ids.device)
    causal = position[None, :] < position[:, None]
    for order in SUFFIX_ORDERS:
        if order > length:
            break
        if order > 1:
            codes[:, order-1:] += tokens[:, :length-order+1] << (bits*(order-1))
        valid_prefix = position >= order-1
        allowed = causal & valid_prefix[:, None] & valid_prefix[None, :]
        matches = codes[:, :, None] == codes[:, None, :]
        matches.logical_and_(allowed)
        yield order, matches


class SuffixSignedCache(ExpandedSignedCache):
    """Phase-A architecture plus four separate zero-dimensional FP32 biases."""

    def __init__(self, config):
        config = copy.deepcopy(config)
        if tuple(config.get('push150_suffix_orders', SUFFIX_ORDERS)) != SUFFIX_ORDERS:
            raise ValueError('Phase C fixes the exact suffix orders to 1..4')
        if tuple(config.get('push150_suffix_bounds', SUFFIX_BOUNDS)) != SUFFIX_BOUNDS:
            raise ValueError('Phase C fixes each suffix bias bound to [-4,4]')
        config['push150_suffix_orders'] = list(SUFFIX_ORDERS)
        config['push150_suffix_bounds'] = list(SUFFIX_BOUNDS)
        super().__init__(config)
        for name in SUFFIX_PARAMETER_NAMES:
            self.register_parameter(name, nn.Parameter(torch.zeros((), dtype=torch.float32)))

    def suffix_coefficients(self):
        return tuple(getattr(self, name).clamp(*SUFFIX_BOUNDS) for name in SUFFIX_PARAMETER_NAMES)

    @torch.no_grad()
    def project_cache_parameters(self):
        super().project_cache_parameters()
        for name in SUFFIX_PARAMETER_NAMES:
            getattr(self, name).clamp_(*SUFFIX_BOUNDS)

    def cache_attention_and_gate(self, features, ids):
        """Every caller must supply the same input ids used for these features."""
        if (ids.ndim != 2 or features.ndim != 3 or features.shape[:2] != ids.shape
                or features.device != ids.device or not 1 <= ids.shape[1] <= self.context):
            raise ValueError('Features and ids must describe the same nonempty input window')
        _, length, _ = features.shape
        normalized = F.normalize(features.float(), dim=-1)
        theta, alpha = self.cache_scales()
        similarity = normalized@normalized.transpose(-1, -2)
        scores = theta*similarity
        query = self.pointer_query(normalized).float()
        key = self.pointer_key(normalized).float()
        scores = scores+(query@key.transpose(-1, -2))/math.sqrt(query.shape[-1])
        scores = scores-alpha*self._cache_log_distance[:length, :length]
        coefficients = self.suffix_coefficients()
        for order, matches in suffix_match_masks(ids, self.config['vocab']):
            # The differentiable path cannot mutate scores needed by autograd.
            # In inference a call-local in-place add avoids another B*T*T copy.
            bonus = coefficients[order-1]*matches
            if torch.is_grad_enabled():
                scores = scores+bonus
            else:
                scores.add_(bonus)
        scores = scores.masked_fill(~self._cache_allowed[:length, :length], -1e4)
        attention = scores.softmax(-1)
        position = torch.arange(length, device=features.device, dtype=torch.float32)
        history = (position/max(1, self.context-1))[None, :].expand(features.shape[0], -1)
        gate_input = torch.cat([normalized, attention.amax(-1)[..., None],
                                attention.square().sum(-1)[..., None], history[..., None]], -1)
        gate = self.pointer_gate(gate_input).float().sigmoid()*self.config.get('pointer_max_weight', .5)
        return attention, gate*(position > 0)[None, :, None]

    def neural_and_cache(self, ids):
        features = self.features(ids)
        logits = self.head(features).float()
        if torch.is_grad_enabled():
            probabilities = (logits/self.config.get('temperature', 1.15)).softmax(-1)
        else:
            logits.div_(self.config.get('temperature', 1.15))
            probabilities = logits.softmax(-1)
        del logits
        attention, gate = self.cache_attention_and_gate(features, ids)
        values = torch.cat([ids[:, 1:], ids.new_zeros(ids.shape[0], 1)], dim=1)
        if torch.is_grad_enabled():
            memory = torch.zeros_like(probabilities)
            memory.scatter_add_(-1, values[:, None, :].expand(-1, ids.shape[1], -1), attention)
            return probabilities*(1-gate)+memory*gate
        probabilities.mul_(1-gate)
        probabilities.scatter_add_(-1, values[:, None, :].expand(-1, ids.shape[1], -1), attention*gate)
        return probabilities

    def training_log_probs(self, ids):
        # Do not inherit the old explicit LearnedCacheGPT call: it lacks ids
        # at the cache API. Training never reads global statistics.
        return self.neural_and_cache(ids).clamp_min(1e-30).log()


def training_loss(model, ids, targets, auxiliary_weight):
    """Original train-only objective, with ids passed explicitly to the cache.

    targets are used only for the final likelihood gather, never for suffix
    matching, features, attention, gate or candidate cache values.
    """
    features = model.features(ids)
    logits = model.head(features).float()/model.config.get('temperature', 1.15)
    probabilities = logits.softmax(-1)
    attention, gate = model.cache_attention_and_gate(features, ids)
    values = torch.cat([ids[:, 1:], ids.new_zeros(ids.shape[0], 1)], dim=1)
    memory = torch.zeros_like(probabilities)
    memory.scatter_add_(-1, values[:, None, :].expand(-1, ids.shape[1], -1), attention)
    mixture = probabilities*(1-gate)+memory*gate
    target = targets.unsqueeze(-1)
    mixed_nll = -mixture.gather(-1, target).clamp_min(1e-30).log().mean()
    if auxiliary_weight:
        neural_nll = -probabilities.gather(-1, target).clamp_min(1e-30).log().mean()
        return (1-auxiliary_weight)*mixed_nll+auxiliary_weight*neural_nll
    return mixed_nll


def build_model(config):
    return SuffixSignedCache(config)
