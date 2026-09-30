"""Two-scalar extension of the existing strictly within-window learned cache.

For allowed j<t, attention logits add -alpha*log(t-j), a power-law recency
prior, and learn bounded positive semantic sharpness theta. Initial alpha=0
and theta=the old cache_theta reproduce the old mechanism. No extra attention
branch, external data, cross-window state, or count-table training loss is used.

Training drivers must call project_cache_parameters() after optimizer.step().
This implements projected optimization at alpha=0 without a dead initial
gradient, while forward clamping guarantees valid bounds even before projection.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from push145_cache import LearnedCacheGPT
from push145_cache_hybrid import LearnedCacheHybrid


class RecencyLearnedCache(LearnedCacheHybrid):
    NEW_PARAMETER_NAMES = frozenset(('cache_log_theta_delta', 'cache_recency_strength'))
    THETA_MIN, THETA_MAX, RECENCY_MAX = 5., 40., 4.

    def __init__(self, config):
        super().__init__(config)
        self._theta_initial = float(config.get('cache_theta', 15.))
        if not self.THETA_MIN <= self._theta_initial <= self.THETA_MAX:
            raise ValueError('Initial cache_theta must lie in the fixed [5,40] interval.')
        self._theta_log_lower = math.log(self.THETA_MIN/self._theta_initial)
        self._theta_log_upper = math.log(self.THETA_MAX/self._theta_initial)
        self.cache_log_theta_delta = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.cache_recency_strength = nn.Parameter(torch.zeros((), dtype=torch.float32))
        # Deterministic structural buffers, never evaluation-prefix state/assets.
        position = torch.arange(self.context, dtype=torch.float32)
        distance = (position[:, None]-position[None, :]).clamp_min(1.)
        self.register_buffer('_cache_log_distance', distance.log(), persistent=False)
        self.register_buffer('_cache_allowed', torch.ones(self.context, self.context, dtype=torch.bool).tril(-1), persistent=False)

    def cache_scales(self):
        log_delta = self.cache_log_theta_delta.clamp(self._theta_log_lower, self._theta_log_upper)
        theta = (self._theta_initial*log_delta.exp()).clamp(self.THETA_MIN, self.THETA_MAX)
        alpha = self.cache_recency_strength.clamp(0., self.RECENCY_MAX)
        return theta, alpha

    @torch.no_grad()
    def project_cache_parameters(self):
        """Call after each optimizer update; parameters remain ordinary FP32."""
        self.cache_log_theta_delta.clamp_(self._theta_log_lower, self._theta_log_upper)
        self.cache_recency_strength.clamp_(0., self.RECENCY_MAX)

    def load_parent_state_dict(self, state_dict):
        """Load EVERY old weight exactly, allowing ONLY the two new scalars.

        Fresh/resumed push147 checkpoints should instead use normal strict
        load_state_dict(), preserving their already trained new parameters.
        """
        required = set(self.state_dict())-self.NEW_PARAMETER_NAMES
        supplied = set(state_dict)
        if supplied != required:
            raise RuntimeError(f'Old-parent state mismatch: missing={sorted(required-supplied)}, unexpected={sorted(supplied-required)}')
        loaded = self.load_state_dict(state_dict, strict=False)
        if set(loaded.missing_keys) != self.NEW_PARAMETER_NAMES or loaded.unexpected_keys:
            raise RuntimeError('Parent loading allowed an undeclared difference.')
        with torch.no_grad():
            self.cache_log_theta_delta.zero_()
            self.cache_recency_strength.zero_()
        return loaded

    def cache_attention_and_gate(self, features):
        _, length, _ = features.shape
        normalized = F.normalize(features.float(), dim=-1)
        theta, alpha = self.cache_scales()
        similarity = normalized@normalized.transpose(-1, -2)
        scores = theta*similarity
        query = self.pointer_query(normalized).float()
        key = self.pointer_key(normalized).float()
        scores = scores+(query@key.transpose(-1, -2))/math.sqrt(query.shape[-1])
        scores = scores-alpha*self._cache_log_distance[:length, :length]
        scores = scores.masked_fill(~self._cache_allowed[:length, :length], -1e4)
        attention = scores.softmax(-1)
        position = torch.arange(length, device=features.device, dtype=torch.float32)
        history = (position/max(1, self.context-1))[None, :].expand(features.shape[0], -1)
        gate_input = torch.cat([normalized, attention.amax(-1)[..., None],
                                attention.square().sum(-1)[..., None], history[..., None]], -1)
        gate = self.pointer_gate(gate_input).float().sigmoid()*self.config.get('pointer_max_weight', .5)
        # Position zero has no available prediction/value pair. Its memory
        # contribution and its memory-related gradient are exactly zero.
        return attention, gate*(position > 0)[None, :, None]

    def neural_and_cache(self, ids):
        if torch.is_grad_enabled():
            return LearnedCacheGPT.neural_and_cache(self, ids)
        # Preserve the previously audited FP32 memory-efficient inference path.
        return LearnedCacheHybrid.neural_and_cache(self, ids)

    def training_log_probs(self, ids):
        # Explicit differentiable path; counts never participate in training.
        return LearnedCacheGPT.neural_and_cache(self, ids).clamp_min(1e-30).log()

    @torch.no_grad()
    def predict_log_probs(self, ids):
        probabilities = self.neural_and_cache(ids)
        if hasattr(self, 'ngram'):
            if self.config.get('mkn_shapes'):
                counts = self.ngram.probabilities(ids, self.config.get('mkn_order', 5))
                weight = self.config['mkn_weight']
            else:
                counts = self.ngram.probabilities(ids, self.config['ngram_discount'], self.config['ngram_order'])
                weight = self.config['ngram_weight']
            probabilities.mul_(1-weight).add_(counts, alpha=weight)
        return probabilities.clamp_min_(1e-30).log_()


def build_model(config):
    return RecencyLearnedCache(config)
