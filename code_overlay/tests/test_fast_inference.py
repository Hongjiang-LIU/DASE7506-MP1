import unittest
import torch
from refined import RefinedGPT
from fast_refined import FastRefinedGPT


class FastInferenceTests(unittest.TestCase):
    def test_matches_original_fp32(self):
        torch.set_num_threads(2)
        config=dict(vocab=2048,width=32,heads=4,depth=2,context=256,
                    temperature=1.1,cache_weight=.05,cache_theta=15.)
        original=RefinedGPT(config).eval();fast=FastRefinedGPT(config).eval()
        fast.load_state_dict(original.state_dict())
        ids=torch.randint(2048,(2,256))
        torch.testing.assert_close(original.predict_log_probs(ids),fast.predict_log_probs(ids),atol=3e-6,rtol=1e-6)
