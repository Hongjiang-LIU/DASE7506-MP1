"""Independent small CPU checks; no dataset files, CUDA, or scorer calls."""
import ast
import copy
import inspect
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

import push146_train as trainer
from push145_cache import LearnedCacheGPT
from push145_cache_hybrid import LearnedCacheHybrid
from push145_mkn_build import effective_counts, make_tables


class Push146ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        ids = np.random.default_rng(614).integers(0, 7, size=500, dtype=np.int64)
        cls.tables, _ = make_tables(effective_counts(ids), scale=.7)

    def setUp(self):
        torch.manual_seed(131)
        self.config = dict(vocab=2048, width=32, heads=4, depth=2,
            ffn_hidden=64, context=256, lowrank_rank=16, dropout=.2,
            temperature=1.20, cache_weight=.05, cache_theta=15., pointer_rank=8,
            pointer_max_weight=.5, mkn_weight=.15, mkn_order=5,
            mkn_shapes={k: list(v.shape) for k, v in self.tables.items()})
        self.original = LearnedCacheHybrid(self.config).eval()
        self.original.ngram.load_state_dict(self.tables)
        self.parent = dict(config=copy.deepcopy(self.config),
            model={k: v.detach().clone() for k, v in self.original.state_dict().items()},
            train_tokens=180224000)
        self.train = LearnedCacheGPT(trainer.training_config(self.config, 0.))
        self.train.load_state_dict(trainer.neural_state(self.parent['model']), strict=True)
        self.ids = torch.tensor([[1, 2, 3, 1, 2, 3, 4, 1], [3, 2, 1, 3, 2, 1, 4, 3]])
        self.targets = torch.tensor([[2, 3, 1, 2, 3, 4, 1, 2], [2, 1, 3, 2, 1, 4, 3, 2]])

    def checkpoint(self, model=None, step=0, weights='raw'):
        return trainer.make_checkpoint(model or self.train, self.parent, self.config,
                                       'synthetic-parent', 131, step, weights)

    def test_training_config_is_deep_copied_and_has_no_statistical_keys(self):
        source = copy.deepcopy(self.config)
        source.update(ngram_shapes={'unigram': [2048]}, ngram_weight=.9,
                      discount_scale=.7, metadata={'nested': [1]})
        before = copy.deepcopy(source)
        config = trainer.training_config(source, .1)
        self.assertFalse(any(k.startswith(('ngram_', 'mkn_')) for k in config))
        self.assertNotIn('discount_scale', config)
        self.assertEqual(config['temperature'], 1.15)
        self.assertEqual(config['dropout'], .1)
        config['metadata']['nested'].append(2)
        self.assertEqual(source, before)

    def test_train_loss_has_gradients_and_never_depends_on_statistics(self):
        self.assertFalse(hasattr(self.train, 'ngram'))
        logp = self.train.training_log_probs(self.ids)
        loss = -logp.gather(-1, self.targets[..., None]).mean()
        self.assertTrue(loss.requires_grad)
        loss.backward()
        for name, parameter in self.train.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for parameter in (self.train.token.weight, self.train.pointer_key.weight,
                          self.train.pointer_gate[-1].weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)
        poisoned = {k: torch.full_like(v, 0) if k.startswith('ngram.') else v
                    for k, v in self.parent['model'].items()}
        poisoned['ngram.unused_poison'] = torch.tensor(float('nan'))
        self.train.load_state_dict(trainer.neural_state(poisoned), strict=True)
        torch.testing.assert_close(logp.detach(), self.train.training_log_probs(self.ids),
                                   atol=0., rtol=0.)

    def test_step_zero_full_hybrid_reproduces_parent_despite_training_temperature(self):
        checkpoint = self.checkpoint()
        evaluator = LearnedCacheHybrid(checkpoint['config']).eval()
        evaluator.load_state_dict(checkpoint['model'], strict=True)
        torch.testing.assert_close(evaluator.predict_log_probs(self.ids),
                                   self.original.predict_log_probs(self.ids), atol=0., rtol=0.)
        self.assertEqual(checkpoint['training_config']['temperature'], 1.15)
        self.assertEqual(checkpoint['config']['temperature'], 1.20)
        self.assertEqual(checkpoint['config']['mkn_weight'], .15)
        self.assertEqual(checkpoint['train_tokens'], 180224000)

    def test_checkpoint_roundtrip_tied_embedding_snapshot_and_cost(self):
        checkpoint = self.checkpoint(step=4000)
        self.assertEqual(checkpoint['train_tokens'], 212992000)
        self.assertEqual(checkpoint['selected_step'], 4000)
        self.assertEqual(checkpoint['model']['head.weight'].data_ptr(),
                         checkpoint['model']['token.weight'].data_ptr())
        snapshot = checkpoint['model']['token.weight'].clone()
        with torch.no_grad():
            self.train.token.weight.add_(.01)
        torch.testing.assert_close(checkpoint['model']['token.weight'], snapshot, atol=0., rtol=0.)
        stream = io.BytesIO()
        torch.save(checkpoint, stream)
        stream.seek(0)
        restored = torch.load(stream, map_location='cpu', weights_only=True)
        evaluator = LearnedCacheHybrid(restored['config']).eval()
        evaluator.load_state_dict(restored['model'], strict=True)
        self.assertIs(evaluator.head.weight, evaluator.token.weight)
        torch.testing.assert_close(evaluator.predict_log_probs(self.ids),
                                   self.original.predict_log_probs(self.ids), atol=0., rtol=0.)

    def test_raw_and_ema_use_own_neural_parameters_and_identical_statistics(self):
        averaged = copy.deepcopy(self.train).eval()
        self.assertIs(averaged.head.weight, averaged.token.weight)
        with torch.no_grad():
            self.train.token.weight.add_(.125)
            for target, current in zip(averaged.parameters(), self.train.parameters()):
                target.lerp_(current, .01)
        raw, ema = self.checkpoint(weights='raw'), self.checkpoint(averaged, weights='ema')
        self.assertFalse(torch.equal(raw['model']['token.weight'], ema['model']['token.weight']))
        torch.testing.assert_close(ema['model']['token.weight'],
                                   self.parent['model']['token.weight'] + .00125,
                                   atol=1e-8, rtol=1e-6)
        for key, value in self.parent['model'].items():
            if key.startswith('ngram.'):
                torch.testing.assert_close(raw['model'][key], value, atol=0., rtol=0.)
                torch.testing.assert_close(ema['model'][key], value, atol=0., rtol=0.)

    def test_actual_sampling_assignments_include_last_legal_start(self):
        # Run only the actual two sampling assignments extracted from main().
        # Force randint's maximum valid output to exercise the endpoint exactly.
        tree = ast.parse(inspect.getsource(trainer.main))
        assignments = [node for node in ast.walk(tree)
                       if isinstance(node, ast.Assign) and len(node.targets) == 1
                       and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id in ('starts', 'batch')]
        self.assertEqual(len(assignments), 2)
        module = ast.fix_missing_locations(ast.Module(body=assignments, type_ignores=[]))
        code = compile(module, '<actual-trainer-sampling>', 'exec')
        for length in (257, 513):
            tokens = torch.arange(length)
            scope = dict(torch=torch, tokens=tokens, device=torch.device('cpu'),
                         rng=torch.Generator().manual_seed(131))
            def last_start(high, size, generator=None):
                self.assertEqual(high, length - 256)
                return torch.full(size, high-1, dtype=torch.long)
            with patch.object(torch, 'randint', side_effect=last_start):
                exec(code, scope)
            self.assertEqual(tuple(scope['batch'].shape), (32, 257))
            self.assertTrue(torch.all(scope['batch'][:, -1] == length-1))
            self.assertTrue(torch.all(scope['starts'] == length-257))


if __name__ == '__main__':
    unittest.main()
