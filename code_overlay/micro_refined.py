import torch
from fast_refined import FastRefinedGPT


class MicroRefined(FastRefinedGPT):
    @torch.no_grad()
    def predict_log_probs(self,ids):
        size=self.config.get('inference_microbatch',8)
        return torch.cat([FastRefinedGPT.predict_log_probs(self,part) for part in ids.split(size)],dim=0)


def build_model(config):
    return MicroRefined(config)
