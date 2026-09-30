"""Synthetic-only checks; no corpus, CUDA, or full evaluation is invoked."""
import ast
import copy
import inspect
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import push150_mkn_sweep as sweep
from push145_mkn_build import effective_counts, make_tables
from push147_signed_cache import SignedDistanceCache


class MKNSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.ids = np.random.default_rng(150).integers(0, 8, size=2400, dtype=np.int64)
        cls.original, _ = make_tables(effective_counts(cls.ids), .7, minimum=2, trigram_minimum=1)
        cls.state = {'ngram.'+name: value for name, value in cls.original.items()}

    def model(self):
        torch.manual_seed(150)
        config = dict(vocab=2048, width=32, heads=4, depth=1, ffn_hidden=64,
                      context=256, lowrank_rank=8, dropout=0., pointer_rank=8,
                      temperature=1.25, cache_theta=15., cache_weight=.05,
                      discount_scale=.7, mkn_order=5, mkn_weight=.15,
                      mkn_shapes={k: list(v.shape) for k, v in self.original.items()})
        model = SignedDistanceCache(config).eval()
        model.ngram.load_state_dict(self.original)
        with torch.no_grad():
            model.cache_recency_strength.fill_(-.4)
            model.cache_log_theta_delta.fill_(.03)
            model.pointer_key.weight.normal_(0, .03)
        return model

    def test_fixed_nine_choices(self):
        self.assertEqual(sweep.SCALES, (.40, .55, .70))
        self.assertEqual(sweep.WEIGHTS, (.10, .15, .20))
        self.assertEqual(sweep.TEMPERATURE, 1.25)
        self.assertEqual(len(sweep.SCALES)*len(sweep.WEIGHTS), 9)

    def test_rebuild_control_is_bitwise_equal_and_pruning_fixed(self):
        variants, parity = sweep.rebuild_variants(self.ids, self.state)
        self.assertEqual(parity['buffers'], 21)
        for scale, (tables, stats) in variants.items():
            self.assertEqual([item['minimum'] for item in stats], [1, 1, 2, 2])
            for name, tensor in self.original.items():
                self.assertEqual(tensor.dtype, tables[name].dtype)
                self.assertEqual(tensor.shape, tables[name].shape)
                if scale == .7 or name.startswith(('words_', 'contexts_', 'offsets_')) or name == 'unigram':
                    self.assertTrue(torch.equal(tensor, tables[name]))

    def test_parity_rejects_changed_missing_and_wrong_dtype(self):
        changed = {k: v.clone() for k, v in self.original.items()}
        changed['masses_2'][0] += .01
        with self.assertRaises(AssertionError):
            sweep.assert_exact_statistics(self.state, changed)
        changed = dict(self.original); changed.pop('unigram')
        with self.assertRaises(AssertionError):
            sweep.assert_exact_statistics(self.state, changed)
        changed = dict(self.original); changed['unigram'] = changed['unigram'].double()
        with self.assertRaises(AssertionError):
            sweep.assert_exact_statistics(self.state, changed)

    def test_all_nine_shortcuts_match_full_signed_predictor(self):
        model = self.model()
        x = torch.tensor([[1, 2, 1, 2, 3, 4], [5, 5, 6, 2, 1, 3]])
        y = torch.tensor([[2, 1, 2, 3, 4, 1], [5, 6, 2, 1, 3, -100]])
        with torch.no_grad():
            neural = model.neural_and_cache(x).gather(-1, y.clamp_min(0)[..., None]).squeeze(-1).numpy()
        variants, _ = sweep.rebuild_variants(self.ids, self.state)
        count_targets = {scale: sweep.target_probabilities({k: v.numpy() for k, v in tables.items()},
                         x.numpy(), y.numpy()) for scale, (tables, _) in variants.items()}
        trials, best = sweep.score_grid(neural, count_targets, y.numpy(), 40)
        self.assertEqual(len(trials), 9)
        self.assertEqual(best['validation_bpb'], min(row['validation_bpb'] for row in trials))
        for row in trials:
            model.ngram.load_state_dict(variants[row['discount_scale']][0])
            model.config.update(mkn_weight=row['mkn_weight'])
            logp = model.predict_log_probs(x)
            torch.testing.assert_close(logp.logsumexp(-1), torch.zeros(x.shape), rtol=1e-6, atol=1e-6)
            direct = -logp.gather(-1, y.clamp_min(0)[..., None]).squeeze(-1)[y != -100].double().sum().item()/math.log(2)/40
            self.assertLess(abs(direct-row['validation_bpb']), 1e-6)

    def test_candidate_keeps_every_neural_weight_and_parent_unchanged(self):
        model = self.model()
        parent = dict(implementation=sweep.IMPLEMENTATION, config=copy.deepcopy(model.config),
                      model=model.state_dict(), train_targets=376832000)
        old = copy.deepcopy(parent)
        tables, _ = make_tables(effective_counts(self.ids), .4)
        best = dict(discount_scale=.4, mkn_weight=.2, temperature=1.25, mkn_order=5)
        candidate = sweep.candidate_from_parent(parent, tables, best)
        sweep.assert_neural_and_config_unchanged(parent, candidate)
        self.assertEqual(candidate['train_targets'], 376832000)
        self.assertEqual(parent['config'], old['config'])
        for name, tensor in parent['model'].items():
            self.assertTrue(torch.equal(tensor, old['model'][name]))
        candidate['model']['pointer_query.weight'][0, 0] += 1
        with self.assertRaises(AssertionError):
            sweep.assert_neural_and_config_unchanged(parent, candidate)

    def test_control_candidate_reproduces_parent_short_prefix_and_causality(self):
        model = self.model()
        parent = dict(implementation=sweep.IMPLEMENTATION, config=copy.deepcopy(model.config), model=model.state_dict())
        choice = dict(discount_scale=.7, mkn_weight=.15, temperature=1.25, mkn_order=5)
        candidate = sweep.candidate_from_parent(parent, self.original, choice)
        other = SignedDistanceCache(candidate['config']).eval()
        other.load_state_dict(candidate['model'])
        x = torch.tensor([[1, 2, 1, 2, 3, 4], [4, 3, 2, 1, 2, 1]])
        before = model.predict_log_probs(x)
        torch.testing.assert_close(before, other.predict_log_probs(x), atol=0, rtol=0)
        for length in (1, 2, 3, 5):
            torch.testing.assert_close(before[:, :length], other.predict_log_probs(x[:, :length]), atol=2e-6, rtol=2e-6)
        changed = x.clone(); changed[0, 4:] = 7
        after = other.predict_log_probs(changed)
        torch.testing.assert_close(before[0, :4], after[0, :4], atol=0, rtol=0)
        torch.testing.assert_close(before[1], after[1], atol=0, rtol=0)

    def test_invalid_grid_measurements_rejected(self):
        neural = np.full((1, 2), .5, np.float32)
        counts = {scale: neural.copy() for scale in sweep.SCALES}
        for targets, size in ((np.zeros((1, 2)), 0), (np.full((1, 2), -100), 5)):
            with self.assertRaises(ValueError):
                sweep.score_grid(neural, counts, targets, size)
        counts[.4][0, 0] = np.nan
        with self.assertRaises(ValueError):
            sweep.score_grid(neural, counts, np.zeros((1, 2)), 5)

    def test_no_test_loader_or_common_load_data(self):
        with mock.patch.object(Path, 'read_bytes', side_effect=AssertionError('I/O attempted')):
            with self.assertRaises(ValueError):
                sweep.load_one_split('test', Path('not-used'))
        tree = ast.parse(inspect.getsource(sweep))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'common':
                self.assertNotIn('load_data', [item.name for item in node.names])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotEqual(node.func.id, 'load_data')

    def test_json_and_checkpoint_writes_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            record, checkpoint = Path(directory)/'record.json', Path(directory)/'small.pt'
            sweep.write_json_new(record, {'synthetic': True})
            sweep.save_checkpoint_new(checkpoint, {'x': torch.tensor([1.])})
            with self.assertRaises(FileExistsError):
                sweep.write_json_new(record, {'synthetic': False})
            with self.assertRaises(FileExistsError):
                sweep.save_checkpoint_new(checkpoint, {'x': torch.tensor([2.])})


if __name__ == '__main__':
    unittest.main()
