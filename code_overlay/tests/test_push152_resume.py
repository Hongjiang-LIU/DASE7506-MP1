"""Tiny synthetic CPU state/metadata checks, never real checkpoints or data."""
import ast
import copy
import inspect
import io
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common import PROTOCOL, make_model
import push150_oof_train as previous
import push152_resume as trainer


class ResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        random.seed(211); np.random.seed(211); torch.manual_seed(211)

    def models(self):
        config = dict(vocab=2048, width=16, heads=2, depth=1, context=256,
            ffn_hidden=32, rope=True, lowrank_rank=4, pointer_rank=4,
            dropout=.2, temperature=1.15, cache_weight=.05, cache_theta=15.,
            push150_ranks=dict(qkv=6, proj=0, gate=4, value=4, out=4))
        model, _ = make_model('push150_expand', config, torch.device('cpu'))
        averages = {label: copy.deepcopy(model).eval().requires_grad_(False) for label in trainer.EMA_DECAYS}
        opts, _ = trainer.build_optimizers(model, 'muon', 7.5e-6, 7.5e-5, .00025, .0005)
        sampler = torch.Generator().manual_seed(211)
        return model, averages, opts, sampler

    def advance(self, model, averages, opts, sampler):
        ranges = torch.tensor([[0,22],[31,72]], dtype=torch.long)
        tokens = torch.arange(80)%13+1
        starts, keep = trainer.sample_starts_and_coin(sampler, ranges, 2)
        batch = tokens[starts[:,None]+torch.arange(9)]
        extras = (random.random(), float(np.random.random()))
        model.train()
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        loss = trainer.control_loss(model, batch[:,:-1], batch[:,1:])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        for opt in opts:
            opt.step()
        model.project_cache_parameters()
        trainer.update_averages(model, averages)
        return starts, keep, loss.detach(), extras

    def old_state(self):
        model, averages, opts, sampler = self.models()
        self.advance(model, averages, opts, sampler)
        identity = dict(implementation='push150_expand', recipe=dict(steps=12000,
            objective='control', training_config=copy.deepcopy(model.config)))
        log = dict(completed_steps=12000, status='completed', parent_train_targets=540672000)
        state = previous.pack_train_state(model, averages['ema99'], opts, sampler, identity, 12000, log)
        return state

    def equal(self, left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(type(left), type(right)); self.assertEqual(len(left), len(right))
            for x, y in zip(left, right):
                self.equal(x, y)
        else:
            self.assertEqual(left, right)

    def test_omitted_count_lookup_preserves_exact_control_loss_and_gradients(self):
        model, _, _, _ = self.models(); other = copy.deepcopy(model)
        ids = torch.tensor([[1,2,1,2],[3,4,3,4]])
        rng = torch.get_rng_state().clone()
        expected = previous.training_loss(model, ids, ids.roll(-1, 1), torch.full((2,4), .9), 'control', True)
        torch.set_rng_state(rng)
        actual = trainer.control_loss(other, ids, ids.roll(-1, 1))
        self.equal(actual, expected)
        expected.backward(); actual.backward()
        for x, y in zip(model.parameters(), other.parameters()):
            self.equal(x.grad, y.grad)

    def test_keep_forks_raw_and_matching_moments_not_ema(self):
        old = self.old_state(); original = copy.deepcopy(old)
        model, averages, opts, sampler = self.models()
        trainer.fork_from_old_state(old, model, averages, opts, sampler, 'optimizer-keep')
        self.equal(trainer.model_state(model), old['raw'])
        self.equal(trainer.model_state(averages['ema99']), old['ema'])
        self.equal(trainer.model_state(averages['ema999']), old['ema'])
        for opt, before in zip(opts, old['optimizers']):
            self.equal(opt.state_dict()['state'], before['state'])
        self.assertTrue(any(not torch.equal(old['raw'][key], old['ema'][key]) for key in old['raw']))
        self.equal(old, original)
        self.assertIs(model.token.weight, model.head.weight)

    def test_reset_changes_only_optimizer_states_not_initial_rng_or_parameters(self):
        old = self.old_state()
        model, avg, opts, sampler = self.models()
        trainer.fork_from_old_state(old, model, avg, opts, sampler, 'optimizer-keep')
        keep_rng = trainer.capture_rng(sampler)
        other, other_avg, fresh, other_sampler = self.models()
        trainer.fork_from_old_state(old, other, other_avg, fresh, other_sampler, 'optimizer-reset')
        self.equal(trainer.capture_rng(other_sampler), keep_rng)
        self.equal(trainer.model_state(model), trainer.model_state(other))
        for label in trainer.EMA_DECAYS:
            self.equal(trainer.model_state(avg[label]), trainer.model_state(other_avg[label]))
        self.assertTrue(all(opt.state for opt in opts))
        self.assertTrue(all(not opt.state for opt in fresh))

    def test_declared_constant_rates_override_old_lr_and_base_lr(self):
        old = self.old_state(); model, avg, opts, sampler = self.models()
        layout = trainer.fork_from_old_state(old, model, avg, opts, sampler, 'optimizer-keep')
        for opt, details in zip(opts, layout):
            for group, item in zip(opt.param_groups, details['groups']):
                self.assertEqual(group['lr'], trainer.RATES[item['kind']])
                self.assertEqual(group['base_lr'], group['lr'])

    def test_changed_old_recipe_or_step_rejected_before_model_mutation(self):
        old = self.old_state()
        for change in ('step', 'objective', 'architecture'):
            model, avg, opts, sampler = self.models(); before = trainer.model_state(model)
            wrong = copy.deepcopy(old)
            if change == 'step':
                wrong['step'] = 11999
            elif change == 'objective':
                wrong['identity']['recipe']['objective'] = 'joint'
            else:
                wrong['identity']['recipe']['training_config']['width'] = 32
            with self.subTest(change=change), self.assertRaises(ValueError):
                trainer.fork_from_old_state(wrong, model, avg, opts, sampler, 'optimizer-keep')
            self.equal(before, trainer.model_state(model))

    def test_parameter_order_mismatch_is_rejected(self):
        old = self.old_state(); model, avg, opts, sampler = self.models()
        old['optimizers'][0]['param_groups'][0]['params'].reverse()
        with self.assertRaises(ValueError):
            trainer.fork_from_old_state(old, model, avg, opts, sampler, 'optimizer-keep')

    def test_two_ema_updates_have_declared_decay_and_no_grad(self):
        model, avg, _, _ = self.models()
        before = {label: trainer.model_state(value) for label, value in avg.items()}
        with torch.no_grad():
            next(model.parameters()).add_(.1)
        trainer.update_averages(model, avg)
        for label, value in avg.items():
            for name, parameter in value.named_parameters():
                expected = before[label][name].lerp(dict(model.named_parameters())[name].detach(), 1.-trainer.EMA_DECAYS[label])
                self.equal(parameter, expected)
            self.assertFalse(any(parameter.requires_grad for parameter in value.parameters()))

    def test_roundtrip_exact_next_stochastic_update_with_two_emas_and_optimizers(self):
        old = self.old_state(); model, avg, opts, sampler = self.models()
        trainer.fork_from_old_state(old, model, avg, opts, sampler, 'optimizer-keep')
        identity = dict(recipe={'steps':8000}, plan_sha256='synthetic', phase_name='optimizer-keep')
        self.advance(model, avg, opts, sampler)
        state = trainer.pack_state(model, avg, opts, sampler, identity, 1, dict(completed_steps=1))
        stream = io.BytesIO(); torch.save(state, stream); stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        expected_result = self.advance(model, avg, opts, sampler)
        expected_state = trainer.pack_state(model, avg, opts, sampler, identity, 2, dict(completed_steps=2))
        other, other_avg, other_opts, other_sampler = self.models()
        trainer.restore_state(saved, other, other_avg, other_opts, other_sampler, identity)
        self.equal(self.advance(other, other_avg, other_opts, other_sampler), expected_result)
        self.equal(trainer.pack_state(other, other_avg, other_opts, other_sampler, identity, 2,
            dict(completed_steps=2)), expected_state)

    def test_resume_identity_and_missing_ema_rejected_before_mutation(self):
        model, avg, opts, sampler = self.models()
        identity = dict(recipe={'steps':8000}, plan_sha256='synthetic')
        state = trainer.pack_state(model, avg, opts, sampler, identity, 0, dict(completed_steps=0))
        before = trainer.model_state(model)
        with self.assertRaises(ValueError):
            trainer.restore_state(state, model, avg, opts, sampler, dict(identity, plan_sha256='changed'))
        self.equal(before, trainer.model_state(model))
        del state['averages']['ema999']
        with self.assertRaises(ValueError):
            trainer.restore_state(state, model, avg, opts, sampler, identity)

    def test_fourteen_validation_choices_no_step_zero_duplicate(self):
        model, avg, _, _ = self.models()
        self.assertEqual([name for name, _ in trainer.validation_choices(0, model, avg)], ['raw','ema99'])
        menus = [trainer.validation_choices(step, model, avg) for step in (0,2000,4000,6000,8000)]
        self.assertEqual(sum(map(len, menus)), 14)

    def test_snapshot_uses_current_tables_and_union_lineage_once(self):
        model, _, _, _ = self.models()
        parent = dict(config=dict(model.config, **trainer.FIXED_VALIDATION),
            model={'ngram.unigram': torch.arange(8, dtype=torch.float32)}, ancestry={'shared':'same-trajectory'})
        identity = dict(plan_sha256='synthetic', source_sha256={})
        saved = trainer.make_snapshot(model, parent, identity, {}, 8000, 'ema999')
        self.assertEqual(saved['train_tokens'], 704512000)
        self.assertIs(saved['model']['head.weight'], saved['model']['token.weight'])
        self.equal(saved['model']['ngram.unigram'], parent['model']['ngram.unigram'])
        self.assertEqual(saved['config']['temperature'], 1.3)
        self.assertFalse(any('optimizer' in key or 'oof' in key for key in saved['model']))

    def test_fold_geometry_and_sampler_keep_entire_windows_inside_folds(self):
        count = 7680
        folds = [dict(fold=i,start=start,end=end,valid_random_starts=end-start-256)
            for i, (start, end) in enumerate(trainer.contiguous_folds(count))]
        ranges = trainer.validate_fold_geometry(folds, count)
        sampler = torch.Generator().manual_seed(211)
        before = torch.get_rng_state().clone()
        for _ in range(3):
            starts, keep = trainer.sample_starts_and_coin(sampler, ranges)
            self.assertIsInstance(keep, bool)
            for start in starts.tolist():
                self.assertEqual(sum(fold['start'] <= start and start+256 < fold['end'] for fold in folds), 1)
        self.equal(torch.get_rng_state(), before)
        folds[0]['end'] += 1
        with self.assertRaises(ValueError):
            trainer.validate_fold_geometry(folds, count)

    def test_sources_include_new_driver_and_no_test_or_oof_array_loader(self):
        sources = trainer.source_digests()
        self.assertIn('push152_resume.py', sources)
        self.assertIn('push150_oof_train.py', sources)
        self.assertIn('push148_muon.py', sources)
        tree = ast.parse(inspect.getsource(trainer.load_allowed_data))
        split_loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For) and
            isinstance(node.target, ast.Name) and node.target.id == 'split')
        self.assertEqual(ast.literal_eval(split_loop.iter), ('train','validation'))
        self.assertNotIn('gather_window_probabilities', inspect.getsource(trainer.run))
        self.assertNotIn('load_prepared_cache', inspect.getsource(trainer.run))

    def test_exclusive_snapshots_and_preflight_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trainer.save_new({'synthetic': torch.tensor(1)}, root/'snapshot.pt')
            with self.assertRaises(FileExistsError):
                trainer.save_new({}, root/'snapshot.pt')
            args = SimpleNamespace(run_dir=root, phase_name='optimizer-keep', _run_directory_confirmed=True)
            for _ in range(2):
                with patch.object(trainer, 'parse_args', return_value=args), patch.object(trainer, 'run',
                    side_effect=ValueError('synthetic preflight')):
                    with self.assertRaises(ValueError):
                        trainer.main()
            self.assertEqual(len(list(root.glob('preflight-failure-*.json'))), 2)

    def test_fixed_plan_menu_and_lrs_reject_changes(self):
        shared = dict(training_split='train', selection_split='validation', context=256,
            batch_size=32, targets_per_step=8192, training_precision='bf16', validation_precision='fp32',
            validation_device='cuda', training_temperature=1.15, validation_temperature=1.3,
            dropout=.2, mkn_weight=.15, mkn_order=5, discount_scale=.7, gradient_clip=1.,
            optimizer='muon', adamw_betas=[.9,.95], backbone_weight_decay=.1,
            pointer_weight_decay=.01, scalar_weight_decay=0., muon_momentum=.95, muon_ns_steps=5,
            cpu_threads=4, no_calibration_grid=True)
        phase = dict(state_sha256=trainer.OLD_STATE_SHA, source_step=12000, source_lineage_targets=638976000,
            steps=8000, eval_every=2000, learning_rates=dict(trainer.RATES), schedule='constant',
            ema=dict(existing_decay=.99, new_decay=.999), variants={
                name: dict(keep_optimizer_state=(name == 'optimizer-keep')) for name in trainer.RUN_NAMES})
        plan = dict(round='push152', parent=dict(sha256=trainer.STATISTICS_PARENT_SHA,
            implementation='push150_expand', selected_lineage_targets=638976000),
            shared=shared, phase_a=phase, external_training_data_or_weights=False, evaluation_network_access=False)
        for name in trainer.RUN_NAMES:
            self.assertEqual(trainer.verify_plan(plan, name), phase['variants'][name])
        for scope, key, value in [('shared','training_split','validation'), ('shared','dropout',0.),
            ('phase_a','steps',10000), ('phase_a','schedule','cosine')]:
            wrong = copy.deepcopy(plan); wrong[scope][key] = value
            with self.subTest(scope=scope, key=key), self.assertRaises(ValueError):
                trainer.verify_plan(wrong, 'optimizer-keep')
        wrong = copy.deepcopy(plan); wrong['phase_a']['learning_rates']['muon'] = .0005
        with self.assertRaises(ValueError):
            trainer.verify_plan(wrong, 'optimizer-keep')


if __name__ == '__main__':
    unittest.main()
