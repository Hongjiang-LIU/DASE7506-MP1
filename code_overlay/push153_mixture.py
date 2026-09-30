"""Stateless causal responsibility interpolation of the two existing experts.

N already includes the neural signed pointer; M is the frozen train-only MKN.
The only menu is disabled / kappa 8 / 32 / 128, with base MKN weight .15.
No optimizer, label argument, learned parameter, or cross-window accumulator
is introduced. Every responsibility uses a token already present in ``ids``.
"""
import torch

from push150_expand import ExpandedSignedCache


BASE_WEIGHT = .15
KAPPAS = (8, 32, 128)


def validate_kappa(kappa):
    if kappa is not None and (type(kappa) is not int or kappa not in KAPPAS):
        raise ValueError('push153_kappa must be None, 8, 32, or 128.')
    return kappa


def weights_from_observed_targets(neural_observed, mkn_observed, kappa):
    """Return [B,H+1] weights from [B,H] *already observed* target probabilities.

    Column j of either input is expert output j's probability for input token
    x[j+1]. Output column t uses input columns 0..t-1, never input column t.
    Thus production passes H=T-1 gathered probabilities. A scoring shortcut
    holding [B,T] probabilities for evaluator targets MUST pass [:,:-1]: its
    final target is not an input token and is never used for adaptation.

    w[0]=a; w[t]=(kappa*a + sum_{j<t} a*m[j]/((1-a)*n[j]+a*m[j]))
                    / (kappa+t), where a=.15.

    Inputs must be ordinary probabilities in FP32 or FP64 on one device.
    Frozen MKN supplies strictly positive probabilities, so no probability
    clipping, extra epsilon, or additional tunable policy is necessary.
    A nonpositive or nonfinite denominator is an error, never a fallback to
    another predictor. The helper never mutates its inputs or state.
    """
    validate_kappa(kappa)
    if not isinstance(neural_observed, torch.Tensor) or not isinstance(mkn_observed, torch.Tensor):
        raise TypeError('Observed probabilities must be tensors.')
    if neural_observed.ndim != 2 or mkn_observed.shape != neural_observed.shape:
        raise ValueError('Observed probabilities must have matching [batch,history] shapes.')
    if neural_observed.dtype not in (torch.float32, torch.float64):
        raise TypeError('Observed probabilities must use FP32 or FP64.')
    if mkn_observed.dtype != neural_observed.dtype or mkn_observed.device != neural_observed.device:
        raise ValueError('Observed probabilities must share dtype and device.')
    batch, history = neural_observed.shape
    prior = neural_observed.new_full((batch, 1), BASE_WEIGHT)
    if kappa is None:
        return neural_observed.new_full((batch, history+1), BASE_WEIGHT)
    if history == 0:
        return prior
    numerator = mkn_observed*BASE_WEIGHT
    denominator = neural_observed*(1-BASE_WEIGHT)+numerator
    if not (torch.isfinite(denominator) & (denominator > 0)).all().item():
        raise ValueError('Responsibility mixture denominators must be finite and positive.')
    responsibility = numerator/denominator
    count = torch.arange(1, history+1, dtype=neural_observed.dtype,
                         device=neural_observed.device)
    adapted = (responsibility.cumsum(dim=1)+kappa*BASE_WEIGHT)/(count+kappa)
    return torch.cat((prior, adapted), dim=1)


class CausalResponsibilityMixture(ExpandedSignedCache):
    """Same parameters/buffers as the parent; only the final mixture varies."""

    def __init__(self, config):
        validate_kappa(config.get('push153_kappa'))
        if not config.get('mkn_shapes') or config.get('mkn_weight') != BASE_WEIGHT:
            raise ValueError('push153 requires frozen MKN tables and mkn_weight=.15.')
        super().__init__(config)
        self.config.setdefault('push153_kappa', None)

    @torch.no_grad()
    def predict_log_probs(self, ids):
        kappa = validate_kappa(self.config['push153_kappa'])
        # Both branches are the actual parent method, not an algebraic rewrite.
        # Disabled is bitwise identical, and T=1 has no observed prediction.
        if kappa is None or ids.shape[1] == 1:
            return super().predict_log_probs(ids)
        probabilities = self.neural_and_cache(ids)
        counts = self.ngram.probabilities(ids, self.config.get('mkn_order', 5))
        next_inputs = ids[:, 1:, None]
        neural_observed = probabilities[:, :-1].gather(-1, next_inputs).squeeze(-1)
        mkn_observed = counts[:, :-1].gather(-1, next_inputs).squeeze(-1)
        weights = weights_from_observed_targets(neural_observed, mkn_observed, kappa)[..., None]
        # Preserve the parent's exact fixed-mixture arithmetic at position zero.
        first = probabilities[:, 0].clone()
        first.mul_(1-BASE_WEIGHT).add_(counts[:, 0], alpha=BASE_WEIGHT)
        # counts is a fresh output of MKNTables.probabilities, not its fixed
        # cached train-derived table. No full-vocabulary temporary is created.
        probabilities.mul_(1-weights)
        counts.mul_(weights)
        probabilities.add_(counts)
        probabilities[:, 0].copy_(first)
        return probabilities.clamp_min_(1e-30).log_()


def build_model(config):
    return CausalResponsibilityMixture(config)
