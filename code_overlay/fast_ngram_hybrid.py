from ngram_hybrid import HybridGPT
from fast_refined import fast_features


class FastHybrid(HybridGPT):
    features=fast_features


def build_model(config):
    return FastHybrid(config)
