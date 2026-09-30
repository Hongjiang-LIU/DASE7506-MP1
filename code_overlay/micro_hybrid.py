"""Same independent-window predictor, smaller internal batch for CPU locality."""
import torch
from fast_ngram_hybrid import FastHybrid


class MicroHybrid(FastHybrid):
    @torch.no_grad()
    def predict_log_probs(self,ids):
        size=self.config.get('inference_microbatch',8)
        return torch.cat([FastHybrid.predict_log_probs(self,part) for part in ids.split(size)],dim=0)


def build_model(config):
    return MicroHybrid(config)
