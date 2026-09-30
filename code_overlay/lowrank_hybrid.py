from lowrank import LowRankGPT
from ngram_hybrid import NgramTables,HybridGPT


class LowRankHybrid(LowRankGPT):
    def __init__(self,config):
        super().__init__(config)
        self.ngram=NgramTables(config['ngram_shapes'])
    predict_log_probs=HybridGPT.predict_log_probs


def build_model(config):
    return LowRankHybrid(config)
