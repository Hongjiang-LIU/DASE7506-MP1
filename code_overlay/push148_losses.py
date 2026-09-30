"""Train-only dual-dropout consistency; no inference changes or teacher assets.

Implements the symmetric-distribution consistency idea from R-Drop:
https://arxiv.org/abs/2106.14448 . Both stochastic views retain gradients.
The caller controls train/eval mode. Statistical tables must not participate in
the model's differentiable training_log_probs method.
"""
import math

import torch


def _positive_distribution(log_probabilities):
    # The existing model returns finite FP32 logs with a 1e-30 probability
    # floor. Repeat that floor here for robustness, and normalize before KL.
    probabilities = log_probabilities.float().exp().clamp_min(1e-30)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return probabilities, probabilities.log()


def dual_view_loss(model, ids, targets, coefficient):
    """Return mean two-view hard NLL + coefficient * symmetric KL and metrics.

    Even coefficient=0 performs both independent calls, for an equal-view
    control. Reduction is sum over the vocabulary, then mean over batch/time;
    neither KL argument is detached. Returned metrics alone are detached.
    No mode changes, corpus I/O, persistent state, or count-table lookup occur.
    """
    coefficient = float(coefficient)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Consistency coefficient must be finite and nonnegative.')
    if ids.ndim != 2 or targets.shape != ids.shape:
        raise ValueError('ids and targets must have the same [batch,time] shape.')

    first = model.training_log_probs(ids).float()
    second = model.training_log_probs(ids).float()
    if first.ndim != 3 or first.shape[:2] != ids.shape or second.shape != first.shape:
        raise ValueError('Both views must return [batch,time,vocabulary] log probabilities.')
    target = targets.unsqueeze(-1)
    hard = .5 * (-first.gather(-1, target).mean() - second.gather(-1, target).mean())

    p, log_p = _positive_distribution(first)
    q, log_q = _positive_distribution(second)
    # Algebraically .5*(KL(p||q)+KL(q||p)); exactly zero for identical
    # distributions, and both p/q paths remain differentiable.
    symmetric_kl = .5 * ((p-q) * (log_p-log_q)).sum(dim=-1).mean()
    loss = hard + coefficient * symmetric_kl
    return loss, {'hard': hard.detach(), 'kl': symmetric_kl.detach()}
