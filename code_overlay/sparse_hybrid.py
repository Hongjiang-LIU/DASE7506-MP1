from sparse_ngram import SparseGPT
from ngram_hybrid import NgramTables,HybridGPT
from fast_refined import fast_features


class SparseHybrid(SparseGPT):
    def __init__(self,config):
        super().__init__(config)
        self.ngram=NgramTables(config['ngram_shapes'])
    features=fast_features
    predict_log_probs=HybridGPT.predict_log_probs


def build_model(config):
    return SparseHybrid(config)
