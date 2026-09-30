import unittest
import numpy as np
import torch
from build_train_ngrams import build_tables
from push145_mkn_compact_build import convert, compare_distributions
from push145_mkn_dense import build_model


class TestCompactTables(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        ids = np.random.default_rng(14).integers(0, 8, size=4000, dtype=np.int64)
        self.original, _ = build_tables(ids, 2, trigram_minimum=1)
        self.compact = convert(self.original)

    def test_bitwise_full_probability_equivalence(self):
        x = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 1, 2, 7, 4, 6]])
        result = compare_distributions(self.original, self.compact, x)
        self.assertTrue(result['bitwise_equal'])

    def test_dense_model_contract(self):
        torch.manual_seed(16)
        model = build_model(dict(vocab=2048, width=32, heads=4, depth=2, context=256,
                                 temperature=1.15, cache_weight=.05, cache_theta=15.,
                                 mkn_weight=.15, mkn_order=5,
                                 mkn_shapes={k: list(v.shape) for k, v in self.compact.items()})).eval()
        model.ngram.load_state_dict(self.compact)
        x = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 1, 2, 7, 4, 6]])
        changed = x.clone(); changed[0, 4:] = 100
        a, b = model.predict_log_probs(x), model.predict_log_probs(changed)
        torch.testing.assert_close(a[0, :4], b[0, :4], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(a[1], b[1], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(a.logsumexp(-1), torch.zeros(x.shape), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(a[:1], model.predict_log_probs(x[:1]), atol=1e-5, rtol=1e-5)
        model.predict_log_probs(changed)
        torch.testing.assert_close(a, model.predict_log_probs(x), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
