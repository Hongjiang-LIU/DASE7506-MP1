"""Trainable, strictly causal per-window cache; no persistent evaluation state."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from lowrank import LowRankGPT
from ngram_hybrid import NgramTables


class LearnedCacheGPT(LowRankGPT):
    def __init__(self, config):
        super().__init__(config)
        width = config['width']
        rank = config.get('pointer_rank', 32)
        self.pointer_query = nn.Linear(width, rank, bias=False)
        self.pointer_key = nn.Linear(width, rank, bias=False)
        nn.init.normal_(self.pointer_query.weight, std=0.02)
        # Zero residual starts at exactly the original normalized-feature cache.
        nn.init.zeros_(self.pointer_key.weight)
        self.pointer_gate = nn.Sequential(nn.Linear(width + 3, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.pointer_gate[-1].weight)
        initial = config.get('cache_weight', .05) / config.get('pointer_max_weight', .5)
        nn.init.constant_(self.pointer_gate[-1].bias, math.log(initial / (1-initial)))
        if config.get('ngram_shapes'):
            self.ngram = NgramTables(config['ngram_shapes'])

    def cache_attention_and_gate(self, features):
        _, length, _ = features.shape
        normalized = F.normalize(features.float(), dim=-1)
        similarity = normalized @ normalized.transpose(-1, -2)
        query = self.pointer_query(normalized).float()
        key = self.pointer_key(normalized).float()
        scores = self.config.get('cache_theta', 15.) * similarity
        scores = scores + (query @ key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
        allowed = torch.ones(length, length, device=features.device, dtype=torch.bool).tril(-1)
        scores = scores.masked_fill(~allowed, -1e4)
        attention = scores.softmax(-1)
        position = torch.arange(length, device=features.device, dtype=torch.float32)
        history = (position / max(1, self.context-1))[None, :].expand(features.shape[0], -1)
        maximum = attention.amax(-1)
        concentration = attention.square().sum(-1)
        gate_input = torch.cat([normalized, maximum[...,None], concentration[...,None], history[...,None]], -1)
        gate = self.pointer_gate(gate_input).float().sigmoid() * self.config.get('pointer_max_weight', .5)
        # No preceding prediction/value pair exists at the first position.
        gate = gate * (position > 0)[None, :, None]
        return attention, gate

    def neural_and_cache(self, ids):
        features = self.features(ids)
        probabilities = (self.head(features).float() / self.config.get('temperature', 1.15)).softmax(-1)
        attention, gate = self.cache_attention_and_gate(features)
        values = torch.cat([ids[:,1:], ids.new_zeros(ids.shape[0], 1)], dim=1)
        memory = torch.zeros_like(probabilities)
        memory.scatter_add_(-1, values[:,None,:].expand(-1, ids.shape[1], -1), attention)
        return probabilities * (1-gate) + memory * gate

    def training_log_probs(self, ids):
        # Training uses only neural+pointer likelihood, never training-table self matches.
        return self.neural_and_cache(ids).clamp_min(1e-30).log()

    @torch.no_grad()
    def predict_log_probs(self, ids):
        probabilities = self.neural_and_cache(ids)
        if hasattr(self, 'ngram'):
            counts = self.ngram.probabilities(ids, self.config['ngram_discount'], self.config['ngram_order'])
            weight = self.config['ngram_weight']
            probabilities.mul_(1-weight).add_(counts, alpha=weight)
        return probabilities.clamp_min_(1e-30).log_()


def build_model(config):
    return LearnedCacheGPT(config)
