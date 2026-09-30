from unittest.mock import patch
import torch
import test_push147_cache as original_tests
from push147_cache import RecencyLearnedCache
from push147_signed_cache import SignedDistanceCache


class SignedDistanceTests(original_tests.RecencyCacheTests):
    """Inherit the complete unsigned contract plus negative-distance checks."""

    def setUp(self):
        super().setUp()
        self.model = SignedDistanceCache(dict(self.config)).eval()
        self.model.load_parent_state_dict(self.parent.state_dict())

    def test_zero_is_strictly_equivalent_to_unsigned_model(self):
        unsigned = RecencyLearnedCache(dict(self.config)).eval()
        unsigned.load_state_dict(self.model.state_dict(), strict=True)
        torch.testing.assert_close(unsigned.predict_log_probs(self.x), self.model.predict_log_probs(self.x), atol=0, rtol=0)
        self.assertEqual(set(unsigned.state_dict()), set(self.model.state_dict()))

    def test_bounds_projection_and_recency_prior(self):
        # Override only the original test's nonnegative-bound expectation.
        with torch.no_grad():
            self.model.cache_log_theta_delta.fill_(1e6)
            self.model.cache_recency_strength.fill_(-1.)
        theta, alpha = self.model.cache_scales()
        self.assertLessEqual(theta.item(), 40.)
        self.assertGreaterEqual(theta.item(), 39.99999)
        self.assertEqual(alpha.item(), -1.)
        self.model.project_cache_parameters()
        self.assertEqual(self.model.cache_recency_strength.item(), -1.)
        with torch.no_grad():
            self.model.cache_log_theta_delta.fill_(-1e6)
            self.model.pointer_key.weight.zero_()
        theta, _ = self.model.cache_scales()
        self.assertEqual(theta.item(), 5.)
        # With alpha=-1 and identical features, distance4 receives exactly
        # four times distance1's weight, still exclusively on allowed j<t.
        attention, gate = self.model.cache_attention_and_gate(torch.ones(1, 5, self.config['width']))
        self.assertAlmostEqual((attention[0, 4, 0]/attention[0, 4, 3]).item(), 4., places=5)
        self.assertEqual(attention[0, 4, 4].item(), 0.)
        self.assertEqual(gate[0, 0, 0].item(), 0.)
        with torch.no_grad():
            self.model.cache_recency_strength.fill_(-1e6)
        self.assertEqual(self.model.cache_scales()[1].item(), -4.)
        self.model.project_cache_parameters()
        self.assertEqual(self.model.cache_recency_strength.item(), -4.)
        with torch.no_grad():
            self.model.cache_recency_strength.fill_(1e6)
        self.assertEqual(self.model.cache_scales()[1].item(), 4.)
        self.model.project_cache_parameters()
        self.assertEqual(self.model.cache_recency_strength.item(), 4.)

    def test_negative_distance_causal_normalized_short_prefix_and_reset(self):
        with torch.no_grad():
            self.model.cache_recency_strength.fill_(-1.1)
            self.model.cache_log_theta_delta.fill_(.12)
        expected = self.model.predict_log_probs(self.x)
        changed = self.x.clone(); changed[:, 6:] = (changed[:, 6:]+23) % 2048
        torch.testing.assert_close(expected[:, :6], self.model.predict_log_probs(changed)[:, :6], atol=2e-6, rtol=1e-6)
        changed = self.x.clone(); changed[1] = 19
        torch.testing.assert_close(expected[0], self.model.predict_log_probs(changed)[0], atol=0, rtol=0)
        for length in (1, 2, 5):
            torch.testing.assert_close(expected[:, :length], self.model.predict_log_probs(self.x[:, :length]), atol=3e-6, rtol=1e-6)
        self.model.predict_log_probs(changed)
        torch.testing.assert_close(expected, self.model.predict_log_probs(self.x), atol=0, rtol=0)
        torch.testing.assert_close(expected.logsumexp(-1), torch.zeros(self.x.shape), atol=1e-6, rtol=0)
        self.assertEqual(expected.dtype, torch.float32)

    def test_negative_distance_gradients_and_no_count_training(self):
        with torch.no_grad():
            self.model.cache_recency_strength.fill_(-.6)
        self.model.train()
        y = torch.tensor([[2, 3, 1, 2, 3, 4, 1, 2, 3, 1], [6, 5, 6, 5, 6, 7, 8, 5, 6, 5]])
        with patch.object(self.model.ngram, 'probabilities', side_effect=AssertionError('Training read count tables')):
            loss = -self.model.training_log_probs(self.x).gather(-1, y[..., None]).mean()
            loss.backward()
        for name in self.model.NEW_PARAMETER_NAMES:
            gradient = dict(self.model.named_parameters())[name].grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient))
            self.assertGreater(gradient.abs().item(), 0.)

    def test_negative_coefficient_strict_checkpoint_roundtrip(self):
        with torch.no_grad():
            self.model.cache_recency_strength.fill_(-.8)
        restored = SignedDistanceCache(dict(self.config)).eval()
        restored.load_state_dict(self.model.state_dict(), strict=True)
        self.assertEqual(restored.cache_recency_strength.item(), self.model.cache_recency_strength.item())
        torch.testing.assert_close(self.model.predict_log_probs(self.x), restored.predict_log_probs(self.x), atol=0, rtol=0)
