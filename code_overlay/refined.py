"""Validation-calibrated GPT with optional strictly within-window neural cache."""
import torch
from torch.nn import functional as F
from student import StudentGPT


def cache_probs(features, ids, theta):
    # Feature j predicts ids[j+1]. It is available to query t only when j < t.
    batch, length = ids.shape
    norm = F.normalize(features.float(), dim=-1)
    similarity = norm @ norm.transpose(-1, -2)
    causal = torch.ones(length, length, device=ids.device, dtype=torch.bool).tril(-1)
    scores = (theta * similarity).masked_fill(~causal, float('-inf'))
    # The first row has no history; it is ignored by the mixing weights below.
    scores[:, 0, 0] = 0
    attention = scores.softmax(dim=-1)
    values = torch.cat([ids[:, 1:], ids.new_zeros(batch, 1)], dim=1)
    result = features.new_zeros(batch, length, 2048, dtype=torch.float32)
    result.scatter_add_(-1, values[:, None, :].expand(-1, length, -1), attention)
    return result


class RefinedGPT(StudentGPT):
    @torch.no_grad()
    def predict_log_probs(self, ids):
        features = self.features(ids)
        weight = self.config.get('cache_weight', 0.)
        logits = self.head(features).float()
        logits.div_(self.config.get('temperature', 1.))
        if weight == 0:
            return F.log_softmax(logits, dim=-1)
        probabilities = F.softmax(logits, dim=-1)
        del logits
        memory = cache_probs(features, ids, self.config.get('cache_theta', 20.))
        # Empty-prefix positions retain their neural probability distribution.
        memory[:, 0] = probabilities[:, 0]
        probabilities.mul_(1-weight).add_(memory, alpha=weight)
        return probabilities.clamp_min_(1e-30).log_()


def build_model(config):
    return RefinedGPT(config)
