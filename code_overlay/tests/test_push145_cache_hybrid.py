import unittest
import torch
from push145_cache import LearnedCacheGPT
from push145_cache_hybrid import LearnedCacheHybrid


class InferenceAlgebraTests(unittest.TestCase):
    def test_inplace_mixture_equals_gradient_enabled_mixture(self):
        torch.set_num_threads(2)
        torch.manual_seed(113)
        cfg=dict(vocab=2048,width=32,heads=4,depth=2,context=256,ffn_hidden=64,
                 lowrank_rank=16,dropout=0.,temperature=1.15,cache_weight=.05,cache_theta=15.,pointer_rank=8)
        source=LearnedCacheGPT(cfg).eval()
        with torch.no_grad():
            source.pointer_key.weight.normal_(0,.5)
            source.pointer_gate[-1].weight.normal_(0,.5)
        fast=LearnedCacheHybrid(cfg).eval()
        fast.load_state_dict(source.state_dict())
        ids=torch.randint(2048,(2,45))
        with torch.no_grad():
            original=source.neural_and_cache(ids)
            optimized=fast.neural_and_cache(ids)
        torch.testing.assert_close(original,optimized,atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
