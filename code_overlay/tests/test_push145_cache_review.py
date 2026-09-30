"""Independent CPU-only checks of trained-cache prefix and padding behavior."""
import unittest
import torch
from push145_cache import LearnedCacheGPT


class TrainedCacheReviewTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(711)
        self.model = LearnedCacheGPT(dict(vocab=2048, width=32, heads=4, depth=2,
            ffn_hidden=64, context=256, lowrank_rank=16, dropout=0.,
            temperature=1.15, cache_weight=.05, cache_theta=15., pointer_rank=8)).eval()
        # Exercise a genuinely learned, nonconstant head rather than only its
        # initialization, where the key and final gate weights are zero.
        with torch.no_grad():
            self.model.pointer_key.weight.normal_(0., .5)
            self.model.pointer_gate[-1].weight.normal_(0., .5)

    def test_nonzero_head_prefix_and_padding_equivalence(self):
        ids = torch.randint(2048, (2, 19))
        full = torch.randint(2048, (2, 256))
        full[:, :19] = ids
        short = self.model.predict_log_probs(ids)
        padded = self.model.predict_log_probs(full)
        torch.testing.assert_close(short, padded[:, :19], atol=4e-6, rtol=1e-6)
        full[:, 19:] = 0
        torch.testing.assert_close(short, self.model.predict_log_probs(full)[:, :19],
                                   atol=4e-6, rtol=1e-6)
        torch.testing.assert_close(short.logsumexp(-1), torch.zeros(2,19), atol=1e-6, rtol=0.)

    def test_first_position_is_exactly_neural(self):
        ids = torch.randint(2048, (2, 13))
        with torch.no_grad():
            neural = (self.model.head(self.model.features(ids)).float()/1.15).softmax(-1)
            actual = self.model.predict_log_probs(ids).exp()
        torch.testing.assert_close(actual[:,0], neural[:,0], atol=1e-8, rtol=1e-6)

    def test_training_and_inference_distribution_agree_without_tables(self):
        ids = torch.randint(2048, (2, 23))
        with torch.no_grad():
            training = self.model.training_log_probs(ids)
            inference = self.model.predict_log_probs(ids)
        torch.testing.assert_close(training, inference, atol=0., rtol=0.)


if __name__ == '__main__':
    unittest.main()
