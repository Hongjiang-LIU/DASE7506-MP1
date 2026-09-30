"""Nonuniform projection ranks with a differentiable causal learned cache.

The inference interface is the fixed course interface. Training deliberately
excludes the train-derived count tables, as in the previous learned-cache model.
No input-dependent state is retained between calls or independent windows.
"""
import torch
from torch import nn

from lowrank import LowRankLinear
from push145_cache import LearnedCacheGPT
from push145_mkn import MKNTables


DEFAULT_RANK_LAYOUT = dict(qkv=160, proj=0, gate=112, value=112, out=112)
PROJECTION_NAMES = ("qkv", "proj", "gate", "value", "out")


def validate_layout(layout, width, hidden):
    """Rank zero denotes a dense projection; positive ranks must be feasible."""
    if not isinstance(layout, dict) or set(layout) != set(PROJECTION_NAMES):
        raise ValueError("rank_layout must contain exactly qkv/proj/gate/value/out")
    checked = {}
    for name in PROJECTION_NAMES:
        rank = layout[name]
        limit = width if name in ("qkv", "proj") else min(width, hidden)
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank <= limit:
            raise ValueError(f"Invalid rank {rank!r} for {name}; require 0..{limit}")
        checked[name] = rank
    return checked


def projection_modules(model):
    """Yield checkpoint-compatible names, semantic names, and module objects."""
    for index, block in enumerate(model.blocks):
        for name in ("qkv", "proj"):
            yield f"blocks.{index}.{name}", name, getattr(block, name)
        for name in ("gate", "value", "out"):
            yield f"blocks.{index}.mlp.{name}", name, getattr(block.mlp, name)


class RankLayoutCache(LearnedCacheGPT):
    def __init__(self, config):
        config = dict(config)
        width = config["width"]
        # The existing model computes this default in StudentGPT as well.
        import math
        hidden = config.get("ffn_hidden", math.ceil((8 * width / 3) / 16) * 16)
        layout = validate_layout(config.get("rank_layout", DEFAULT_RANK_LAYOUT), width, hidden)
        config["rank_layout"] = layout
        config.setdefault("lowrank_rank", min(128, width, hidden))
        if config.get("mkn_shapes"):
            # Avoid allocating a second obsolete count model in the superclass.
            config.pop("ngram_shapes", None)
        super().__init__(config)
        for block in self.blocks:
            for container, name in ((block, "qkv"), (block, "proj"),
                                    (block.mlp, "gate"), (block.mlp, "value"),
                                    (block.mlp, "out")):
                old = getattr(container, name)
                in_features = old.left.in_features
                out_features = old.right.out_features
                rank = layout[name]
                replacement = (nn.Linear(in_features, out_features, bias=False)
                               if rank == 0 else LowRankLinear(in_features, out_features, rank))
                setattr(container, name, replacement)
        if config.get("mkn_shapes"):
            self.ngram = MKNTables(config["mkn_shapes"])

    # Crucial: neural_and_cache/training_log_probs remain the DIFFERENTIABLE
    # LearnedCacheGPT methods, not the no_grad hybrid inference override.
    @torch.no_grad()
    def _inference_neural_and_cache(self, ids):
        # Match the existing optimized inference algebra while keeping the
        # training path separate. Only B*T*T cache weights, no B*T*V memory copy.
        features = self.features(ids)
        logits = self.head(features).float()
        logits.div_(self.config.get("temperature", 1.15))
        probabilities = logits.softmax(-1)
        del logits
        attention, gate = self.cache_attention_and_gate(features)
        values = torch.cat([ids[:, 1:], ids.new_zeros(ids.shape[0], 1)], dim=1)
        probabilities.mul_(1 - gate)
        probabilities.scatter_add_(-1, values[:, None, :].expand(-1, ids.shape[1], -1),
                                   attention * gate)
        return probabilities

    @torch.no_grad()
    def predict_log_probs(self, ids):
        probabilities = self._inference_neural_and_cache(ids)
        if hasattr(self, "ngram"):
            if self.config.get("mkn_shapes"):
                counts = self.ngram.probabilities(ids, self.config.get("mkn_order", 5))
                weight = self.config["mkn_weight"]
            else:
                counts = self.ngram.probabilities(
                    ids, self.config["ngram_discount"], self.config["ngram_order"])
                weight = self.config["ngram_weight"]
            if not 0 <= weight <= 1:
                raise ValueError("Count interpolation weight must be between zero and one")
            probabilities.mul_(1 - weight).add_(counts, alpha=weight)
        return probabilities.clamp_min_(1e-30).log_()


def build_model(config):
    return RankLayoutCache(config)
