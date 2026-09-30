import unittest
import torch
from push145_cache import LearnedCacheGPT
from refined import cache_probs


class LearnedCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(103)
        self.model = LearnedCacheGPT(dict(vocab=2048, width=32, heads=4, depth=2,
            ffn_hidden=64, context=256, lowrank_rank=16, dropout=0.,
            temperature=1.15, cache_weight=.05, cache_theta=15., pointer_rank=8)).eval()

    def test_initialization_matches_original_cache(self):
        ids = torch.randint(2048, (2,31))
        with torch.no_grad():
            f = self.model.features(ids)
            p = (self.model.head(f).float()/1.15).softmax(-1)
            memory = cache_probs(f, ids, 15.)
            memory[:,0] = p[:,0]
            expected = p*.95 + memory*.05
            actual = self.model.predict_log_probs(ids).exp()
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)

    def test_causal_independent_reset_and_normalized(self):
        ids = torch.randint(2048,(2,37))
        original = self.model.predict_log_probs(ids)
        changed = ids.clone(); changed[:,18:] = torch.randint(2048,(2,19))
        torch.testing.assert_close(original[:,:18], self.model.predict_log_probs(changed)[:,:18], atol=2e-6, rtol=1e-6)
        changed = ids.clone(); changed[1] = 9
        torch.testing.assert_close(original[0], self.model.predict_log_probs(changed)[0], atol=0, rtol=0)
        self.model.predict_log_probs(torch.randint(2048,(3,37)))
        torch.testing.assert_close(original, self.model.predict_log_probs(ids), atol=0, rtol=0)
        torch.testing.assert_close(original.logsumexp(-1), torch.zeros(2,37), atol=1e-6, rtol=0)
        first = self.model.predict_log_probs(ids[:,:1])
        torch.testing.assert_close(first, original[:,:1], atol=3e-6, rtol=1e-6)

    def test_learned_head_has_finite_gradients(self):
        self.model.train()
        ids = torch.tensor([[1,2,3,1,2,3,4,1,2]])
        y = torch.tensor([[2,3,1,2,3,4,1,2,3]])
        loss = -self.model.training_log_probs(ids).gather(-1,y[...,None]).mean()
        loss.backward()
        for name,p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertGreater(self.model.pointer_gate[-1].weight.grad.abs().sum().item(), 0.)
        self.assertGreater(self.model.pointer_key.weight.grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
