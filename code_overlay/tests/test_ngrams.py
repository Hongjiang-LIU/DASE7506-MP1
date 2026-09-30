import unittest
import numpy as np
import torch
from build_train_ngrams import build_tables
from ngram_hybrid import NgramTables,HybridGPT,prefix_codes
from select_ngram_hybrid import target_probabilities
from test_contract import ContractTests


def small_tables():
    return build_tables(np.tile(np.array([7,8,9,10,7,8,11,10,8,7,10],dtype=np.int64),20),1)[0]


class NgramTests(unittest.TestCase):
    def setUp(self):
        self.buffers=small_tables()
        self.model=NgramTables({k:list(v.shape) for k,v in self.buffers.items()})
        self.model.load_state_dict(self.buffers)

    def test_normalized_and_scoring_matches_full_distribution(self):
        x=torch.tensor([[7,8,9,10,7],[10,7,8,11,0]])
        y=torch.tensor([[8,9,10,7,8],[7,8,11,1,2]])
        for order in (3,4,5):
            p=self.model.probabilities(x,.75,order)
            torch.testing.assert_close(p.sum(-1),torch.ones_like(x,dtype=torch.float32),atol=2e-6,rtol=0)
            target=target_probabilities({k:v.numpy() for k,v in self.buffers.items()},x.numpy(),y.numpy(),.75,order)
            np.testing.assert_allclose(p.gather(-1,y[...,None]).squeeze(-1).numpy(),target,atol=1e-7,rtol=1e-6)

    def test_no_future_or_cross_example_state(self):
        x=torch.tensor([[7,8,9,10,7]])
        original=self.model.probabilities(x)
        changed=x.clone();changed[0,3:]=500
        torch.testing.assert_close(original[:,:3],self.model.probabilities(changed)[:,:3])
        self.model.probabilities(torch.tensor([[99,98,97,96,95]]))
        torch.testing.assert_close(original,self.model.probabilities(x))

    def test_long_prefix_matches_training_hash_and_handles_short_rows(self):
        x=np.arange(24,dtype=np.int64).reshape(2,12)
        for history in (5,7,11):
            codes,valid=prefix_codes(x,history)
            codes=codes.reshape(x.shape)
            for row in range(2):
                for pos in range(history-1,12):
                    expected=0
                    for token in x[row,pos-history+1:pos+1]:
                        expected=(expected*1000003+int(token)+1)&((1<<52)-1)
                    self.assertEqual(codes[row,pos],expected)
        _,valid=prefix_codes(x[:,:3],11)
        self.assertFalse(valid.any())


class HybridContractTests(ContractTests):
    def setUp(self):
        torch.set_num_threads(2)
        buffers=small_tables()
        self.model=HybridGPT(dict(vocab=2048,width=32,heads=4,depth=2,context=256,
            temperature=1.1,cache_weight=.05,cache_theta=15.,ngram_weight=.2,
            ngram_discount=.75,ngram_order=5,
            ngram_shapes={k:list(v.shape) for k,v in buffers.items()})).eval()
        self.model.ngram.load_state_dict(buffers)
