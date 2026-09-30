import unittest
import numpy as np
import torch
from push145_mkn_build import effective_counts, make_tables, discounts
from push145_mkn import MKNTables, build_model
from push145_mkn_select import target_probabilities


class TestMKN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        rng = np.random.default_rng(917)
        cls.ids = rng.integers(0, 9, size=1000, dtype=np.int64)
        cls.effective = effective_counts(cls.ids)

    def tables(self, scale=1.):
        buffers, _ = make_tables(self.effective, scale)
        model = MKNTables({k: list(v.shape) for k, v in buffers.items()})
        model.load_state_dict(buffers)
        return model, {k: v.numpy() for k, v in buffers.items()}

    def test_normalized_positive(self):
        x = torch.tensor([[1, 2, 3, 4, 5, 6], [9, 11, 12, 15, 17, 18]])
        for scale in (.7, 1., 1.3):
            model, _ = self.tables(scale)
            p = model.probabilities(x)
            self.assertEqual(p.dtype, torch.float32)
            self.assertTrue(torch.all(p > 0))
            torch.testing.assert_close(p.sum(-1), torch.ones(x.shape), atol=1e-6, rtol=1e-6)

    def test_shortcut_matches_target_free_distribution(self):
        x = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 0, 5, 3, 8, 4]])
        y = torch.tensor([[2, 3, 4, 5, 6, 8], [0, 5, 3, 8, 4, 2]])
        model, arrays = self.tables()
        expected = model.probabilities(x).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        np.testing.assert_allclose(target_probabilities(arrays, x.numpy(), y.numpy()), expected, rtol=1e-6, atol=1e-8)

    def test_causality_and_row_independence(self):
        model, _ = self.tables()
        x = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 0, 5, 3, 8, 4]])
        before = model.probabilities(x)
        changed = x.clone(); changed[0, 4:] = 17
        after = model.probabilities(changed)
        torch.testing.assert_close(before[0, :4], after[0, :4], rtol=0, atol=0)
        torch.testing.assert_close(before[1], after[1], rtol=0, atol=0)
        model.probabilities(changed)
        torch.testing.assert_close(before[:1], model.probabilities(x[:1]), rtol=0, atol=0)

    def test_removed_probability_mass_is_backed_off(self):
        effective = {1: (np.array([0, 1, 2]), np.array([2, 2, 2])),
                     2: (np.array([0, 1, 2]), np.array([1, 2, 3])),
                     3: (np.array([0, 1, 2]), np.array([1, 2, 3]))}
        arrays, _ = make_tables(effective, 1., minimum=2, trigram_minimum=2)
        # All counts total 6 even though the count-one entry was dropped.
        d, _ = discounts(effective[3][1])
        expected = np.array([(2-d[1])/6, (3-d[2])/6])
        np.testing.assert_allclose(arrays['masses_3'].numpy(), expected, rtol=1e-6)
        np.testing.assert_allclose(arrays['backoffs_3'].numpy(), 1-expected.sum(), rtol=1e-6)

    def test_exact_compact_dtypes_and_no_target_inference_arg(self):
        import inspect
        model, arrays = self.tables()
        self.assertEqual(list(inspect.signature(model.probabilities).parameters), ['ids', 'max_order'])
        self.assertEqual(arrays['words_3'].dtype, np.int16)
        self.assertEqual(arrays['contexts_3'].dtype, np.int64)
        self.assertEqual(arrays['offsets_3'].dtype, np.int32)

    def test_complete_neural_hybrid_contract(self):
        buffers, _ = make_tables(self.effective, .7)
        torch.manual_seed(902)
        model = build_model(dict(vocab=2048, width=32, heads=4, depth=2, context=256,
                                 lowrank_rank=8, temperature=1.15, cache_weight=.05,
                                 cache_theta=15., mkn_weight=.15, mkn_order=5,
                                 mkn_shapes={k: list(v.shape) for k, v in buffers.items()})).eval()
        model.ngram.load_state_dict(buffers)
        x = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 0, 5, 3, 8, 4]])
        changed = x.clone(); changed[0, 4:] = 17
        a, b = model.predict_log_probs(x), model.predict_log_probs(changed)
        torch.testing.assert_close(a[0, :4], b[0, :4], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(a[1], b[1], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(a.logsumexp(-1), torch.zeros(x.shape), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(a[:1], model.predict_log_probs(x[:1]), rtol=1e-5, atol=1e-5)
        model.predict_log_probs(changed)
        torch.testing.assert_close(a, model.predict_log_probs(x), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
