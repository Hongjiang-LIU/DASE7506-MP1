"""Dense existing neural parent plus compact exact train-only n-gram tables.

New hypothesis prompted by reduced inference-asset size, not an expansion of
the already completed 12-choice modified-discount calibration screen.
"""
from efficient import EfficientGPT
from push145_mkn import MKNTables, MKNLowRank


class DenseCompact(EfficientGPT):
    def __init__(self, config):
        super().__init__(config)
        self.ngram = MKNTables(config['mkn_shapes'])

    predict_log_probs = MKNLowRank.predict_log_probs


def build_model(config):
    return DenseCompact(config)
