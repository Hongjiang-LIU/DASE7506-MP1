import unittest
from unittest.mock import patch
import numpy as np
import torch
from build_train_ngrams import build_tables
from push145_mkn_compact_build import convert
from push145_cache_hybrid import LearnedCacheHybrid
from push145_cache import LearnedCacheGPT
from push147_cache import RecencyLearnedCache


class RecencyCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        ids = np.random.default_rng(147).integers(0, 10, size=4000, dtype=np.int64)
        original, _ = build_tables(ids, minimum=2, trigram_minimum=1)
        cls.tables = convert(original)

    def setUp(self):
        torch.manual_seed(147)
        self.config = dict(vocab=2048, width=32, heads=4, depth=2, ffn_hidden=64,
            context=256, lowrank_rank=16, dropout=0., temperature=1.25,
            cache_weight=.05, cache_theta=15., pointer_rank=8,
            mkn_shapes={k: list(v.shape) for k, v in self.tables.items()},
            mkn_order=5, mkn_weight=.15)
        self.parent = LearnedCacheHybrid(dict(self.config)).eval()
        self.parent.ngram.load_state_dict(self.tables)
        torch.nn.init.normal_(self.parent.pointer_key.weight, std=.07)
        torch.nn.init.normal_(self.parent.pointer_gate[-1].weight, std=.04)
        self.model = RecencyLearnedCache(dict(self.config)).eval()
        self.model.load_parent_state_dict(self.parent.state_dict())
        self.x = torch.tensor([[1, 2, 3, 1, 2, 3, 4, 1, 2, 3], [5, 6, 5, 6, 5, 6, 7, 8, 5, 6]])

    def test_initialization_preserves_every_old_weight_and_prediction(self):
        theta, alpha = self.model.cache_scales()
        self.assertEqual(theta.item(), 15.)
        self.assertEqual(alpha.item(), 0.)
        for name, value in self.parent.state_dict().items():
            torch.testing.assert_close(value, self.model.state_dict()[name], rtol=0, atol=0)
        before = self.parent.predict_log_probs(self.x)
        after = self.model.predict_log_probs(self.x)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertEqual(set(self.model.state_dict())-set(self.parent.state_dict()), self.model.NEW_PARAMETER_NAMES)

    def test_loading_rejects_any_other_missing_weight(self):
        bad = dict(self.parent.state_dict())
        bad.pop('pointer_query.weight')
        with self.assertRaises(RuntimeError):
            self.model.load_parent_state_dict(bad)

    def test_training_loss_initially_matches_parent_base(self):
        self.parent.train()
        self.model.train()
        y = (self.x+1) % 10
        torch.manual_seed(177)
        expected = LearnedCacheGPT.neural_and_cache(self.parent, self.x).clamp_min(1e-30).log()
        torch.manual_seed(177)
        actual = self.model.training_log_probs(self.x)
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        a = -expected.gather(-1, y[..., None]).mean()
        b = -actual.gather(-1, y[..., None]).mean()
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_nonzero_recency_causality_independence_reset_normalization(self):
        with torch.no_grad():
            self.model.cache_recency_strength.fill_(.7)
            self.model.cache_log_theta_delta.fill_(.15)
        original = self.model.predict_log_probs(self.x)
        changed = self.x.clone(); changed[:, 6:] = (changed[:, 6:]+31) % 2048
        torch.testing.assert_close(original[:, :6], self.model.predict_log_probs(changed)[:, :6], atol=2e-6, rtol=1e-6)
        changed = self.x.clone(); changed[1] = 100
        torch.testing.assert_close(original[0], self.model.predict_log_probs(changed)[0], atol=0, rtol=0)
        self.model.predict_log_probs(changed)
        torch.testing.assert_close(original, self.model.predict_log_probs(self.x), atol=0, rtol=0)
        torch.testing.assert_close(original[:1], self.model.predict_log_probs(self.x[:1]), atol=2e-6, rtol=1e-6)
        torch.testing.assert_close(original.logsumexp(-1), torch.zeros(self.x.shape), atol=1e-6, rtol=0)
        self.assertEqual(original.dtype, torch.float32)

    def test_first_position_has_no_memory_and_short_prefix_matches(self):
        with torch.no_grad():
            self.model.cache_recency_strength.fill_(1.2)
            features = self.model.features(self.x)
            _, gate = self.model.cache_attention_and_gate(features)
            self.assertTrue(torch.equal(gate[:, 0], torch.zeros_like(gate[:, 0])))
            neural = (self.model.head(features).float()/self.model.config['temperature']).softmax(-1)
            actual = self.model.neural_and_cache(self.x)
            torch.testing.assert_close(actual[:, 0], neural[:, 0], atol=0, rtol=0)
        complete = self.model.predict_log_probs(self.x)
        for length in (1, 2, 5):
            prefix = self.model.predict_log_probs(self.x[:, :length])
            torch.testing.assert_close(prefix, complete[:, :length], atol=3e-6, rtol=1e-6)

    def test_new_parameters_have_finite_nonzero_gradients_and_counts_unused(self):
        self.model.train()
        y = torch.tensor([[2, 3, 1, 2, 3, 4, 1, 2, 3, 1], [6, 5, 6, 5, 6, 7, 8, 5, 6, 5]])
        with patch.object(self.model.ngram, 'probabilities', side_effect=AssertionError('Training read count tables')):
            loss = -self.model.training_log_probs(self.x).gather(-1, y[..., None]).mean()
            loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name in self.model.NEW_PARAMETER_NAMES:
            parameter = dict(self.model.named_parameters())[name]
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad))
            self.assertGreater(parameter.grad.abs().item(), 0.)
        self.assertGreater(self.model.token.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(self.model.pointer_gate[-1].weight.grad.abs().sum().item(), 0.)

    def test_bounds_projection_and_recency_prior(self):
        with torch.no_grad():
            self.model.cache_log_theta_delta.fill_(1e6)
            self.model.cache_recency_strength.fill_(-1.)
        theta, alpha = self.model.cache_scales()
        self.assertLessEqual(theta.item(), 40.)
        self.assertGreaterEqual(theta.item(), 39.99999)
        self.assertEqual(alpha.item(), 0.)
        self.model.project_cache_parameters()
        self.assertEqual(self.model.cache_recency_strength.item(), 0.)
        with torch.no_grad():
            self.model.cache_log_theta_delta.fill_(-1e6)
            self.model.cache_recency_strength.fill_(1.)
            self.model.pointer_key.weight.zero_()
        theta, alpha = self.model.cache_scales()
        self.assertEqual(theta.item(), 5.)
        # Equal features isolate the power-law distance prior. For t=4,
        # j=3 has distance1 and j=0 distance4: their attention ratio is 4.
        features = torch.ones(1, 5, self.config['width'])
        attention, _ = self.model.cache_attention_and_gate(features)
        self.assertAlmostEqual((attention[0, 4, 3]/attention[0, 4, 0]).item(), 4., places=5)

    def test_strict_roundtrip_retains_learned_new_parameters(self):
        with torch.no_grad():
            self.model.cache_log_theta_delta.fill_(.2)
            self.model.cache_recency_strength.fill_(.8)
        restored = RecencyLearnedCache(dict(self.config)).eval()
        restored.load_state_dict(self.model.state_dict(), strict=True)
        torch.testing.assert_close(self.model.predict_log_probs(self.x), restored.predict_log_probs(self.x), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
