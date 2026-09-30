import math
import unittest
import numpy as np
import torch
from build_train_ngrams import build_tables
from push145_mkn_compact_build import convert
from push145_cache_hybrid import LearnedCacheHybrid
from push145_mkn_select import target_probabilities
from push146_calibrate import TEMPERATURES, MIXTURE_WEIGHTS, neural_cache_targets, score_grid


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(146)
        ids = np.random.default_rng(146).integers(0, 9, size=3000, dtype=np.int64)
        old, _ = build_tables(ids, minimum=2, trigram_minimum=1)
        self.buffers = convert(old)
        self.model = LearnedCacheHybrid(dict(vocab=2048, width=32, heads=4, depth=2,
            ffn_hidden=64, context=256, lowrank_rank=16, dropout=0.,
            temperature=1.2, cache_weight=.05, cache_theta=15., pointer_rank=8,
            mkn_shapes={k: list(v.shape) for k, v in self.buffers.items()},
            mkn_order=5, mkn_weight=.15)).eval()
        self.model.ngram.load_state_dict(self.buffers)
        # Nontrivial learned pointer weights exercise the trained-head pathway.
        torch.nn.init.normal_(self.model.pointer_key.weight, std=.03)
        torch.nn.init.normal_(self.model.pointer_gate[-1].weight, std=.02)

    def test_exactly_declared_grid(self):
        self.assertEqual(TEMPERATURES, (1.20, 1.25, 1.30, 1.35, 1.40))
        self.assertEqual(MIXTURE_WEIGHTS, (.10, .15, .20))
        self.assertEqual(len(TEMPERATURES)*len(MIXTURE_WEIGHTS), 15)

    def test_all_shortcut_settings_match_actual_predictor(self):
        x = torch.tensor([[1, 2, 3, 1, 2, 3, 4], [5, 6, 5, 6, 5, 7, 0]])
        y = torch.tensor([[2, 3, 1, 2, 3, 4, 5], [6, 5, 6, 5, 7, 0, -100]])
        state_before = {k: v.clone() for k, v in self.model.state_dict().items()}
        gathered = neural_cache_targets(self.model, x, y)
        arrays = {k: v.numpy() for k, v in self.buffers.items()}
        counts = target_probabilities(arrays, x.numpy(), y.numpy())
        neural = {k: v.numpy() for k, v in gathered.items()}
        trials, best = score_grid(neural, counts, y.numpy(), 40)
        self.assertEqual(len(trials), 15)
        self.assertEqual(best['validation_bpb'], min(row['validation_bpb'] for row in trials))
        valid = y != -100
        for row in trials:
            self.model.config.update(temperature=row['temperature'], mkn_weight=row['mkn_weight'])
            logp = self.model.predict_log_probs(x)
            direct = -logp.gather(-1, y.clamp_min(0)[..., None]).squeeze(-1)[valid].double().sum().item()/math.log(2)/40
            self.assertLess(abs(direct-row['validation_bpb']), 1e-6)
        for name, tensor in self.model.state_dict().items():
            torch.testing.assert_close(state_before[name], tensor, atol=0, rtol=0)

    def test_invalid_measurements_rejected(self):
        values = {temperature: np.ones((1, 2), np.float32)*.5 for temperature in TEMPERATURES}
        with self.assertRaises(ValueError):
            score_grid(values, np.ones((1, 2), np.float32), np.zeros((1, 2), np.int64), 0)
        with self.assertRaises(ValueError):
            score_grid(values, np.ones((1, 2), np.float32), np.full((1, 2), -100), 3)


if __name__ == '__main__':
    unittest.main()
