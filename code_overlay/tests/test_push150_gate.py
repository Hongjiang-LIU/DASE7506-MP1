"""Synthetic gate/OOF checks only: no real corpus, checkpoint or GPU use."""
import ast
import copy
import inspect
import math
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from common import PROTOCOL
from push145_mkn import MKNTables
from push145_mkn_build import effective_counts, make_tables
from push147_signed_cache import SignedDistanceCache
from push150_expand import ExpandedSignedCache
from push150_crossfit import (contiguous_folds, effective_counts_segments,
                              complete_window_starts, fold_windows)
import push150_gate as gate
import push150_gate_train as trainer


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.ids = np.random.default_rng(181).integers(0, 6, 1800, dtype=np.int64)
        cls.buffers, _ = make_tables(effective_counts(cls.ids), .55)

    def parent(self, weight=.15, implementation='push147_signed_cache'):
        torch.manual_seed(181)
        config = dict(vocab=2048, width=16, heads=4, depth=1, ffn_hidden=32,
                      context=256, lowrank_rank=4, dropout=0., pointer_rank=4,
                      cache_theta=15., cache_weight=.05, temperature=1.25,
                      mkn_weight=weight, mkn_order=5, discount_scale=.55,
                      mkn_shapes={k: list(v.shape) for k, v in self.buffers.items()})
        if implementation == 'push150_expand':
            config['push150_ranks'] = dict(qkv=6, proj=0, gate=6, value=6, out=6)
            model = ExpandedSignedCache(config)
        else:
            model = SignedDistanceCache(config)
        model.ngram.load_state_dict(self.buffers)
        with torch.no_grad():
            model.cache_recency_strength.fill_(-.4)
            model.cache_log_theta_delta.fill_(.03)
            model.pointer_key.weight.normal_(0, .03)
        model.eval()
        parent = dict(protocol=PROTOCOL, implementation=implementation, config=copy.deepcopy(model.config),
                      model=model.state_dict(), train_tokens=376832000)
        return parent, model

    def input(self):
        return torch.tensor([[1, 2, 1, 2, 3, 4, 5, 1, 2], [5, 4, 3, 2, 1, 0, 5, 3, 1]])

    def test_auxiliary_lookup_full_distribution_is_bitwise_original(self):
        shapes = {k: list(v.shape) for k, v in self.buffers.items()}
        reference = MKNTables(shapes); reference.load_state_dict(self.buffers)
        auxiliary = gate.AuxiliaryMKNTables(shapes); auxiliary.load_state_dict(self.buffers)
        x = self.input()
        for length in (1, 2, 4, 9):
            expected = reference.probabilities(x[:, :length])
            actual, features = auxiliary.probabilities_and_aux(x[:, :length])
            torch.testing.assert_close(expected, actual, atol=0, rtol=0)
            self.assertEqual(features.shape, (*x[:, :length].shape, 3))
            self.assertTrue(torch.isfinite(features).all())
            self.assertTrue(torch.all((features[..., :2] >= 0) & (features[..., :2] <= 1)))
            self.assertTrue(torch.all(features[..., 2] >= 0))
        # Short prefixes cannot contain a high-order context at position zero.
        _, features = auxiliary.probabilities_and_aux(x)
        torch.testing.assert_close(features[:, 0, 0], torch.zeros(2), atol=0, rtol=0)
        self.assertTrue(torch.all(features[:, 0, 1] <= .4))

    def test_prediction_performs_only_one_three_order_lookup(self):
        parent, _ = self.parent()
        model = gate.model_from_parent(parent)
        with mock.patch.object(gate, 'prefix_codes', wraps=gate.prefix_codes) as lookup:
            model.predict_log_probs(self.input())
        self.assertEqual(lookup.call_count, 3)

    def test_all_three_parent_weights_initialize_correctly(self):
        x = self.input()
        for weight in (.1, .15, .2):
            parent, source = self.parent(weight)
            model = gate.model_from_parent(parent)
            self.assertEqual(sum(p.numel() for p in model.mixture_gate.parameters()), 57)
            torch.testing.assert_close(model.predict_log_probs(x), source.predict_log_probs(x), atol=3e-6, rtol=2e-6)
            features = torch.randn(2, 9, 5)
            torch.testing.assert_close(model.mixture_gate(features), torch.full((2, 9), weight), atol=2e-8, rtol=1e-7)
            gate.assert_frozen_parent(parent, model)

    def test_expanded_dense_projection_parent_maps_without_weight_change(self):
        parent, source = self.parent(.2, implementation='push150_expand')
        model = gate.model_from_parent(parent)
        gate.assert_frozen_parent(parent, model)
        torch.testing.assert_close(source.predict_log_probs(self.input()), model.predict_log_probs(self.input()), atol=3e-6, rtol=2e-6)
        self.assertIsInstance(model.blocks[0].proj, torch.nn.Linear)

    def test_causality_row_independence_short_prefix_and_normalization(self):
        parent, _ = self.parent()
        model = gate.model_from_parent(parent)
        with torch.no_grad():
            model.mixture_gate.network[-1].weight.normal_(0, .1)
        x = self.input(); before = model.predict_log_probs(x)
        changed = x.clone(); changed[0, 5:] = 7
        after = model.predict_log_probs(changed)
        torch.testing.assert_close(before[0, :5], after[0, :5], atol=0, rtol=0)
        torch.testing.assert_close(before[1], after[1], atol=0, rtol=0)
        for length in (1, 3, 6):
            torch.testing.assert_close(before[:, :length], model.predict_log_probs(x[:, :length]), atol=3e-6, rtol=2e-6)
        torch.testing.assert_close(before.logsumexp(-1), torch.zeros(x.shape), atol=1e-6, rtol=0)
        model.predict_log_probs(changed)
        torch.testing.assert_close(before, model.predict_log_probs(x), atol=0, rtol=0)
        with torch.no_grad():
            neural, pointer = model.neural_and_pointer(x)
            pure = (model.head(model.features(x)).float()/model.config['temperature']).softmax(-1)
        torch.testing.assert_close(pointer[:, 0], torch.zeros(2), atol=0, rtol=0)
        torch.testing.assert_close(neural[:, 0], pure[:, 0], atol=0, rtol=0)

    def test_all_four_offset_shortcuts_match_full_target_free_prediction(self):
        parent, _ = self.parent()
        model = gate.model_from_parent(parent)
        with torch.no_grad():
            model.mixture_gate.network[-1].weight.normal_(0, .03)
            x = self.input(); y = x.roll(-1, 1)
            neural, pointer = model.neural_and_pointer(x)
            counts, aux = model.ngram.probabilities_and_aux(x)
            features = gate.gate_features(neural, pointer, aux)
            neural_target = neural.gather(-1, y[..., None]).squeeze(-1)
            count_target = counts.gather(-1, y[..., None]).squeeze(-1)
            for offset in trainer.OFFSETS:
                weight = model.mixture_gate(features, offset)
                mixed = (1-weight)*neural_target+weight*count_target
                model.config['push150_gate_offset'] = offset
                actual = model.predict_log_probs(x).gather(-1, y[..., None]).squeeze(-1)
                torch.testing.assert_close(mixed.log(), actual, atol=2e-6, rtol=2e-6)

    def test_only_small_head_learns_from_seven_column_train_rows(self):
        parent, _ = self.parent()
        model = gate.model_from_parent(parent)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        model.mixture_gate.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.mixture_gate.parameters(), lr=.001, weight_decay=.0001)
        rows = torch.randn(64, 7); rows[:, 5] = .05; rows[:, 6] = .4
        loss = trainer.train_target_loss(model.mixture_gate, rows)
        loss.backward(); optimizer.step()
        self.assertTrue(math.isfinite(loss.item()))
        self.assertFalse(torch.equal(before['mixture_gate.network.2.bias'], model.state_dict()['mixture_gate.network.2.bias']))
        gate.assert_frozen_parent(parent, model)
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if not name.startswith('mixture_gate.')))
        with self.assertRaises(RuntimeError):
            model.training_log_probs(self.input())

    def test_raw_segment_union_matches_original_for_one_segment(self):
        expected = effective_counts(self.ids)
        actual = effective_counts_segments([self.ids])
        for order in range(1, 6):
            np.testing.assert_array_equal(expected[order][0], actual[order][0])
            np.testing.assert_array_equal(expected[order][1], actual[order][1])

    def test_no_artificial_join_ngrams_or_double_counted_predecessor_types(self):
        segments = [np.array([1, 2]), np.array([1, 2]), np.array([3, 4])]
        effective = effective_counts_segments(segments, maximum=2)
        keys, counts = effective[2]
        self.assertNotIn((2 << 11)|1, keys.tolist())
        self.assertNotIn((2 << 11)|3, keys.tolist())
        self.assertEqual(int(counts[np.where(keys == ((1 << 11)|2))[0][0]]), 2)
        lower_keys, lower_counts = effective[1]
        self.assertEqual(int(lower_counts[np.where(lower_keys == 2)[0][0]]), 1)

    def test_fold_tables_exclude_held_region_and_windows_include_no_boundary_token(self):
        ids = np.arange(10023, dtype=np.int64)
        regions = contiguous_folds(len(ids))
        self.assertEqual(len(regions), 10)
        self.assertEqual(regions[0][0], 0); self.assertEqual(regions[-1][1], len(ids))
        self.assertTrue(all(start % 256 == 0 for start, _ in regions))
        for start, end in regions:
            count = 0
            for x, y in fold_windows(ids, (start, end), batch_size=3):
                self.assertEqual(x.shape[1], 256)
                self.assertGreaterEqual(int(x.min()), start)
                self.assertLess(int(y.max()), end)
                torch.testing.assert_close(y, x+1, atol=0, rtol=0)
                count += x.numel()
            self.assertEqual(count, len(complete_window_starts(start, end))*256)
        segments = [np.array([1, 1, 2]), np.array([777, 777, 777]), np.array([3, 4, 3])]
        keys, _ = effective_counts_segments([segments[0], segments[2]], 2)[2]
        self.assertFalse(np.any((keys & 2047) == 777))

    def test_normalization_uses_only_train_feature_columns(self):
        rows = np.arange(70, dtype=np.float32).reshape(10, 7)
        mean, std = trainer.train_normalization(rows)
        changed = rows.copy(); changed[:, 5:] = 1e8
        changed_mean, changed_std = trainer.train_normalization(changed)
        np.testing.assert_array_equal(mean, changed_mean)
        np.testing.assert_array_equal(std, changed_std)
        np.testing.assert_allclose(mean, rows[:, :5].mean(axis=0))
        self.assertEqual(mean.dtype, np.float32); self.assertEqual(std.dtype, np.float32)
        constant = np.ones((5, 7), np.float32)
        _, std = trainer.train_normalization(constant)
        np.testing.assert_array_equal(std, np.full(5, 1e-6, np.float32))

    def test_plan_grid_recipe_and_no_test_loader(self):
        self.assertEqual(trainer.OFFSETS, (-.5, 0., .5, 1.))
        self.assertEqual((trainer.SEED, trainer.EPOCHS, trainer.BATCH_SIZE), (181, 6, 8192))
        rows = trainer.choice_rows([5., 4., 6., 7., 8.], 10)
        self.assertEqual(len(rows), 5); self.assertIsNone(rows[0]['offset'])
        with mock.patch.object(Path, 'read_bytes', side_effect=AssertionError('No I/O permitted')):
            with self.assertRaises(ValueError):
                trainer.load_one_split('test', Path('unused'))
        for module in (gate, trainer):
            tree = ast.parse(inspect.getsource(module))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotEqual(node.func.id, 'load_data')
        self.assertEqual(list(inspect.signature(gate.AdaptiveMKNCache.predict_log_probs).parameters), ['self', 'ids'])

    def test_aux_cache_invalidates_after_loading_other_fold(self):
        shapes = {k: list(v.shape) for k, v in self.buffers.items()}
        model = gate.AuxiliaryMKNTables(shapes); model.load_state_dict(self.buffers)
        model.probabilities_and_aux(self.input())
        self.assertIsNotNone(model._static); self.assertIsNotNone(model._aux_static)
        model.load_state_dict(self.buffers)
        self.assertIsNone(model._static); self.assertIsNone(model._aux_static)

    def test_parent_state_and_config_are_not_modified(self):
        parent, _ = self.parent()
        before = copy.deepcopy(parent)
        gate.model_from_parent(parent)
        self.assertEqual(parent['config'], before['config'])
        for key, value in parent['model'].items():
            torch.testing.assert_close(value, before['model'][key], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
