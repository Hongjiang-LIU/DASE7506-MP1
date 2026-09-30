"""Signed-distance hypothesis; the original nonnegative run is unchanged.

This hypothesis follows train/validation observations that a nonnegative
distance coefficient remained at its zero boundary. A negative coefficient
can counter a semantic attention bias toward the immediate prior prediction,
whose cache value is the *current* token, not the unknown next token.

Only the allowable distance coefficient changes, from [0,4] to [-4,4]. The
two-parameter architecture, theta bounds, causal mask, state reset, training
loss, old-parent loading, and full FP32 statistical mixture remain inherited.
There is no additional candidate-selection or data-loading behavior here.
"""
import torch
from push147_cache import RecencyLearnedCache


class SignedDistanceCache(RecencyLearnedCache):
    RECENCY_MIN = -4.

    def cache_scales(self):
        theta, _ = super().cache_scales()
        alpha = self.cache_recency_strength.clamp(self.RECENCY_MIN, self.RECENCY_MAX)
        return theta, alpha

    @torch.no_grad()
    def project_cache_parameters(self):
        self.cache_log_theta_delta.clamp_(self._theta_log_lower, self._theta_log_upper)
        self.cache_recency_strength.clamp_(self.RECENCY_MIN, self.RECENCY_MAX)


def build_model(config):
    return SignedDistanceCache(config)
