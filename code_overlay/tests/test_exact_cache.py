import unittest
import torch
from exact_refined import ExactRefined,exact_cache
from test_contract import ContractTests


class ExactCacheAlignment(unittest.TestCase):
    def test_only_observed_successors_and_prefix_backoff(self):
        x=torch.tensor([[7,8,7,8,7,99]])
        p,count=exact_cache(x,4)
        self.assertEqual(count[0,0].item(),0)
        self.assertEqual(p[0,2,8].item(),1)
        self.assertEqual(p[0,4,8].item(),1)
        self.assertEqual(p[0,4,99].item(),0)
        changed=x.clone();changed[0,5]=3
        torch.testing.assert_close(p[:,:5],exact_cache(changed,4)[0][:,:5])


class ExactContractTests(ContractTests):
    def setUp(self):
        torch.set_num_threads(2)
        self.model=ExactRefined(dict(vocab=2048,width=32,heads=4,depth=2,context=256,
            temperature=1.1,cache_weight=.2,exact_history=4,exact_scale=1.)).eval()
