from sparse_ngram import SparseGPT
from fast_refined import fast_features


class FastSparseGPT(SparseGPT):
    features=fast_features


def build_model(config):
    return FastSparseGPT(config)
