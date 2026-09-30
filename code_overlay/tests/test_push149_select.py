"""Synthetic metadata-only fixtures; no real model, GPU or corpus is loaded."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from common import sha
import push149_select as selector


class Push149SelectorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        sources = {}
        for name in selector.SOURCE_SHA256:
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('synthetic source '+name).encode())
            sources[name] = sha(path)
        self.addCleanup(patch.stopall)
        patch.dict(selector.SOURCE_SHA256, sources, clear=True).start()
        parent = self.root/'runs'/selector.PARENT_RUN
        parent.mkdir(parents=True)
        (parent/'checkpoint.pt').write_bytes(b'synthetic parent, not a model')
        self.parent_sha = sha(parent/'checkpoint.pt')
        patch.object(selector, 'PARENT_SHA256', self.parent_sha).start()
        parent_training = self.root/'runs/push148-muon'
        parent_training.mkdir(parents=True)
        (parent_training/'checkpoint.pt').write_bytes(b'synthetic previous EMA')
        previous_sha = sha(parent_training/'checkpoint.pt')
        self.put(parent_training/'metrics.json', dict(parent_train_targets=278528000,
            best=dict(step=8000, selected_lineage_targets=selector.PARENT_VIEW_TARGETS, checkpoint_sha256=previous_sha)))
        self.put(parent/'metrics.json', dict(checkpoint_sha256=self.parent_sha, parent_sha256=previous_sha,
            new_training_targets=0, test_used=False, full_validation=dict(bpb=1.4494),
            best=dict(temperature=1.25, mkn_weight=.15)))
        self.plan = dict(runs=[dict(name=name, optimizer=kind, steps=4000, views=1, batch_size=32,
            rdrop_coefficient=0, muon_lr=.0005) for name,kind in zip(selector.RUNS, ('adamw','muon'))],
            common=dict(selector.COMMON_RECIPE), max_training_runs=2,
            calibration=dict(max_settings=30, max_parents=2, temperatures=[1.2,1.25,1.3,1.35,1.4],
                             statistical_weights=[.1,.15,.2]), minimum_validation_improvement=.0005,
            parent_sha256=self.parent_sha, parent_train_targets=selector.PARENT_VIEW_TARGETS)
        self.put(self.root/'runs/push149_plan.json', self.plan)
        self.make_run(0, 1.4491)
        self.make_run(1, 1.4487)

    @staticmethod
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def training_path(self, index=1):
        return self.root/'runs'/('push149-'+selector.RUNS[index])/'metrics.json'

    def calibration_path(self, index=1):
        return self.root/'runs'/('push149-'+selector.RUNS[index]+'-calibrated')/'metrics.json'

    def make_run(self, index, score, completed=2000, selected=1000, status='partial'):
        recipe = self.plan['runs'][index]
        name = recipe['name']
        train = self.training_path(index).parent
        calibration = self.calibration_path(index).parent
        train.mkdir(parents=True, exist_ok=True)
        calibration.mkdir(parents=True, exist_ok=True)
        (train/'checkpoint.pt').write_bytes(('synthetic trained '+name).encode())
        (calibration/'checkpoint.pt').write_bytes(('synthetic calibrated '+name).encode())
        choices = [dict(step=0, weights='raw')]+[dict(step=step, weights=kind)
            for step in range(1000, completed+1, 1000) for kind in ('raw','ema')]
        sources = ['push148_train.py','push148_losses.py','push147_train.py']
        if index == 1:
            sources.append('push148_muon.py')
        training = dict(parent_sha256=self.parent_sha, parent_train_targets=selector.PARENT_VIEW_TARGETS,
            test_used=False, steps=4000, completed_steps=completed, status=status,
            batch_size=32, dropout_views_per_sample=1, incremental_search_targets=completed*8192,
            incremental_sampled_label_targets=completed*8192, seconds=20.+index,
            seed=163, eval_every=1000, optimizer=recipe['optimizer'], backbone_lr=.0000075,
            head_lr=.000075, scalar_lr=.00025, muon_lr=.0005, rdrop_coefficient=0,
            ema_decay=.99, training_precision='bf16', muon_momentum=.95, muon_ns_steps=5,
            training_implementation='push147_signed_cache', inference_implementation='push147_signed_cache',
            inference_architecture_unchanged=True, no_statistical_tables_in_training=True,
            source_splits=['train','validation'], source_sha256={name:selector.SOURCE_SHA256[name] for name in sources},
            implementation_sha256=selector.SOURCE_SHA256['push147_signed_cache.py'],
            training_config=dict(context=256, temperature=1.15, dropout=.2),
            validation_config=dict(context=256, temperature=1.25, dropout=.2, mkn_weight=.15),
            history=[dict(step=completed)], validation_history=choices,
            best=dict(step=selected, weights='ema', checkpoint_sha256=sha(train/'checkpoint.pt'),
                selected_lineage_targets=selector.PARENT_VIEW_TARGETS+selected*8192,
                new_selected_sampled_label_targets=selected*8192, scalars={'cache_recency_strength':-.4}))
        self.put(train/'metrics.json', training)
        if status == 'partial':
            self.put(train/'termination.json', dict(status='stopped', completed_steps=completed,
                metrics_sha256=sha(train/'metrics.json'), reason='Synthetic explicit termination'))
        trials = [dict(temperature=t,mkn_weight=w,validation_bpb=score if (t,w)==(1.25,.15) else score+.01)
                  for t,w in sorted(selector.GRID)]
        self.put(calibration/'metrics.json', dict(test_used=False, trials=trials, seconds_total=2.,
            parent_sha256=sha(train/'checkpoint.pt'), checkpoint_sha256=sha(calibration/'checkpoint.pt'),
            best=dict(temperature=1.25,mkn_weight=.15,validation_bpb=score),
            full_validation=dict(bpb=score+1e-8), precision='fp32', device='cuda:0',
            implementation='push147_signed_cache', implementation_sha256=selector.SOURCE_SHA256['push147_signed_cache.py'],
            script_sha256=selector.SOURCE_SHA256['push146_calibrate.py'],
            evaluator_sha256=selector.SOURCE_SHA256['evaluate.py'], tokenizer_sha256=selector.SOURCE_SHA256['data/tokenizer.json'],
            new_training_targets=0))

    def mutate(self, path, **changes):
        value = json.loads(path.read_text()); value.update(changes); self.put(path, value)

    def test_actual_counts_selected_lineage_and_schema(self):
        result = selector.collect_selection(self.root, 201)
        self.assertEqual(result['selection_status'], 'new_candidate')
        self.assertEqual(result['all_new_search_processed_view_targets'], 2*2000*8192)
        self.assertEqual(result['all_new_search_sampled_label_occurrences'], 2*2000*8192)
        self.assertEqual(result['winner']['selected_lineage_targets'], selector.PARENT_VIEW_TARGETS+1000*8192)
        self.assertEqual(result['winner']['selected_run'], 'push149-muon-low-calibrated')
        self.assertEqual(result['total_raw_ema_validation_choices'], 10)
        self.assertEqual(result['total_calibration_choices'], 30)
        self.assertEqual(result['unit_tests']['executions'], 201)
        self.assertFalse(result['test_used'])

    def test_completed_counts_and_nine_choices(self):
        for i in (0,1):
            self.make_run(i, 1.4487+i*.0001, completed=4000, selected=4000, status='completed')
        result = selector.collect_selection(self.root, 202)
        self.assertEqual(result['all_new_search_targets'], 65536000)
        self.assertEqual(result['total_raw_ema_validation_choices'], 18)

    def test_recipe_changes_refuse_selection(self):
        original = json.loads(self.training_path().read_text())
        for key,value in dict(optimizer='adamw', seed=164, eval_every=2000, steps=8000,
            backbone_lr=.00003, head_lr=.0003, scalar_lr=.001, muon_lr=.002,
            dropout_views_per_sample=2, batch_size=16, rdrop_coefficient=.1).items():
            with self.subTest(key=key):
                self.put(self.training_path(), dict(original, **{key:value}))
                result = selector.collect_selection(self.root, 203)
                self.assertEqual(result['selection_status'], 'refused')
        self.put(self.training_path(), original)

    def test_changed_source_file_fails_closed(self):
        (self.root/'push148_train.py').write_bytes(b'altered synthetic code')
        with self.assertRaisesRegex(ValueError, 'source hash'):
            selector.collect_selection(self.root, 204)

    def test_recorded_source_hash_mismatch_refuses(self):
        record = json.loads(self.training_path().read_text())
        record['source_sha256']['push148_muon.py'] = '0'*64
        self.put(self.training_path(), record)
        self.assertEqual(selector.collect_selection(self.root, 205)['selection_status'], 'refused')

    def test_corrupted_parent_checkpoint_fails_closed(self):
        (self.root/'runs'/selector.PARENT_RUN/'checkpoint.pt').write_bytes(b'changed parent')
        with self.assertRaisesRegex(ValueError, 'Parent checkpoint'):
            selector.collect_selection(self.root, 206)

    def test_parent_lineage_mismatch_fails_closed(self):
        path = self.root/'runs/push148-muon/metrics.json'
        self.mutate(path, parent_train_targets=0)
        with self.assertRaisesRegex(ValueError, 'Parent training lineage'):
            selector.collect_selection(self.root, 207)

    def test_trained_checkpoint_corruption_refuses(self):
        (self.training_path().parent/'checkpoint.pt').write_bytes(b'changed candidate')
        self.assertEqual(selector.collect_selection(self.root, 208)['selection_status'], 'refused')

    def test_failed_run_keeps_actual_cost_and_best_checkpoint(self):
        self.mutate(self.training_path(), status='failed', failure_type='AssertionError', failure_message='')
        result = selector.collect_selection(self.root, 209)
        self.assertTrue(result['candidates'][1]['selection_eligible'])
        self.assertEqual(result['candidates'][1]['recorded_status'], 'failed')
        self.assertEqual(result['candidates'][1]['new_search_targets'], 2000*8192)

    def test_partial_without_calibration_retains_parent(self):
        self.calibration_path().unlink()
        result = selector.collect_selection(self.root, 210)
        self.assertEqual(result['selection_status'], 'retain_parent')
        self.assertEqual(result['winner']['selected_run'], selector.PARENT_RUN)
        self.assertIsNone(result['winner']['selected_step'])
        self.assertEqual(result['total_calibration_choices'], 15)

    def test_completed_without_calibration_refuses(self):
        self.make_run(1, 1.448, completed=4000, status='completed')
        self.calibration_path().unlink()
        self.assertEqual(selector.collect_selection(self.root, 211)['selection_status'], 'refused')

    def test_running_without_explicit_stop_refuses(self):
        self.mutate(self.training_path(), status='running')
        (self.training_path().parent/'termination.json').unlink()
        self.assertEqual(selector.collect_selection(self.root, 212)['selection_status'], 'refused')

    def test_missing_run_refuses_incomplete_accounting(self):
        self.training_path().unlink()
        result = selector.collect_selection(self.root, 213)
        self.assertEqual(result['selection_status'], 'refused')
        self.assertFalse(result['all_training_accounting_present'])

    def test_plan_caps_or_learning_rate_cannot_change(self):
        self.plan['common']['backbone_adamw_lr'] = .00003
        self.put(self.root/'runs/push149_plan.json', self.plan)
        with self.assertRaisesRegex(ValueError, 'Plan common recipe'):
            selector.collect_selection(self.root, 214)

    def test_duplicate_validation_choice_refuses(self):
        value = json.loads(self.training_path().read_text())
        value['validation_history'].append(value['validation_history'][-1])
        self.put(self.training_path(), value)
        self.assertEqual(selector.collect_selection(self.root, 215)['selection_status'], 'refused')

    def test_exclusive_output_and_required_measured_test_count(self):
        result = selector.collect_selection(self.root, 216)
        output = self.root/'runs/push149_selection.json'
        selector.write_selection(output, result)
        with self.assertRaises(FileExistsError):
            selector.write_selection(output, result)
        with self.assertRaises(ValueError):
            selector.collect_selection(self.root, 0)


if __name__ == '__main__':
    unittest.main()
