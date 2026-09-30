"""Inference-efficient learned pointer with frozen train-derived statistics."""
import torch
from push145_cache import LearnedCacheGPT
from push145_mkn import MKNTables


class LearnedCacheHybrid(LearnedCacheGPT):
    def __init__(self, config):
        super().__init__(config)
        if config.get('mkn_shapes'):
            self.ngram = MKNTables(config['mkn_shapes'])

    @torch.no_grad()
    def neural_and_cache(self, ids):
        features = self.features(ids)
        probabilities = (self.head(features).float()/self.config.get('temperature',1.15)).softmax(-1)
        attention, gate = self.cache_attention_and_gate(features)
        values = torch.cat([ids[:,1:],ids.new_zeros(ids.shape[0],1)],dim=1)
        memory = torch.zeros_like(probabilities)
        memory.scatter_add_(-1,values[:,None,:].expand(-1,ids.shape[1],-1),attention)
        # Algebraically identical, without three additional full-vocabulary tensors.
        probabilities.mul_(1-gate).add_(memory.mul_(gate))
        return probabilities

    @torch.no_grad()
    def predict_log_probs(self, ids):
        probabilities = self.neural_and_cache(ids)
        if self.config.get('mkn_shapes'):
            counts = self.ngram.probabilities(ids,self.config.get('mkn_order',5))
            weight = self.config['mkn_weight']
        else:
            counts = self.ngram.probabilities(ids,self.config['ngram_discount'],self.config['ngram_order'])
            weight = self.config['ngram_weight']
        probabilities.mul_(1-weight).add_(counts,alpha=weight)
        return probabilities.clamp_min_(1e-30).log_()


def build_model(config):
    return LearnedCacheHybrid(config)
