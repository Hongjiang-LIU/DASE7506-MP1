import unittest
import numpy as np
import torch
from push145_cache_hybrid import LearnedCacheHybrid
from push145_cpu_fast import FastLearnedCache
from push145_mkn_build import effective_counts,make_tables


class FastCPUContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(131)
        tokens=np.tile(np.asarray([1,2,3,1,2,4,2,3,4,2,1,8],dtype=np.int64),20)
        buffers,_=make_tables(effective_counts(tokens),scale=.7)
        config=dict(vocab=2048,width=32,heads=4,depth=2,context=256,ffn_hidden=64,
            lowrank_rank=16,dropout=0.,temperature=1.2,cache_weight=.05,cache_theta=15.,
            pointer_rank=8,mkn_order=5,mkn_weight=.15,cpu_feature_microbatch=32,
            mkn_shapes={name:list(value.shape) for name,value in buffers.items()})
        self.reference=LearnedCacheHybrid(config).eval()
        self.reference.ngram.load_state_dict(buffers)
        with torch.no_grad():
            self.reference.pointer_key.weight.normal_(0,.5)
            self.reference.pointer_gate[-1].weight.normal_(0,.5)
        self.model=FastLearnedCache(config).eval()
        self.model.load_state_dict(self.reference.state_dict())

    def test_full_predictor_fp32_and_probability_equivalence(self):
        ids=torch.randint(0,10,(2,39))
        output=self.model.predict_log_probs(ids)
        self.assertEqual(output.dtype,torch.float32)
        torch.testing.assert_close(output,self.reference.predict_log_probs(ids),atol=4e-6,rtol=1e-6)
        torch.testing.assert_close(output.logsumexp(-1),torch.zeros(2,39),atol=1e-6,rtol=0)

    def test_causality_and_short_prefix(self):
        ids=torch.randint(0,10,(2,47))
        output=self.model.predict_log_probs(ids)
        changed=ids.clone();changed[:,19:]=torch.randint(0,2048,(2,28))
        torch.testing.assert_close(output[:,:19],self.model.predict_log_probs(changed)[:,:19],atol=4e-6,rtol=1e-6)
        torch.testing.assert_close(output[:,:19],self.model.predict_log_probs(ids[:,:19]),atol=4e-6,rtol=1e-6)

    def test_row_independence_and_window_reset(self):
        ids=torch.randint(0,10,(2,31))
        expected=self.model.predict_log_probs(ids)
        torch.testing.assert_close(expected[:1],self.model.predict_log_probs(ids[:1]),atol=4e-6,rtol=1e-6)
        self.model.predict_log_probs(torch.randint(2048,(2,256)))
        torch.testing.assert_close(expected,self.model.predict_log_probs(ids),atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
