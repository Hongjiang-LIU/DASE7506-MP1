import unittest
import torch
from refined import build_model, cache_probs
from test_contract import ContractTests


class RefinedContractTests(ContractTests):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.model = build_model(dict(vocab=2048, width=32, heads=4, depth=2,
                                     context=256, temperature=1.1,
                                     cache_weight=0.2, cache_theta=20.)).eval()


class CacheAlignmentTests(unittest.TestCase):
    def test_only_observed_successors_receive_cache_mass(self):
        features = torch.ones(1, 4, 8)
        ids = torch.tensor([[7, 8, 9, 10]])
        p = cache_probs(features, ids, 20.)
        self.assertAlmostEqual(p[0, 1, 8].item(), 1.)
        self.assertAlmostEqual(p[0, 1, 9].item(), 0.)
        self.assertAlmostEqual(p[0, 2, 8].item(), 0.5)
        self.assertAlmostEqual(p[0, 2, 9].item(), 0.5)
        self.assertAlmostEqual(p[0, 2, 10].item(), 0.)
        torch.testing.assert_close(p[0, 3, 8:11], torch.full((3,), 1/3))


if __name__ == '__main__':
    unittest.main()
