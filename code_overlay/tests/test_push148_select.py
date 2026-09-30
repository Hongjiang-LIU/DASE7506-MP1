import json
from pathlib import Path
import tempfile
import unittest
from common import sha
from push148_select import RUNS, GRID, PARENT_RUN, PARENT_VIEW_TARGETS, collect_selection, write_selection


class SelectorFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        parent_dir = self.root/'runs'/PARENT_RUN
        parent_dir.mkdir(parents=True)
        (parent_dir/'checkpoint.pt').write_bytes(b'synthetic parent; not a torch model')
        self.parent_sha = sha(parent_dir/'checkpoint.pt')
        self.put(parent_dir/'metrics.json', dict(checkpoint_sha256=self.parent_sha, test_used=False,
            full_validation=dict(bpb=1.46), best=dict(temperature=1.25, mkn_weight=.15)))
        runs = [dict(name=name, steps=8000, views=2 if name in ('control-dual', 'rdrop010', 'rdrop050') else 1,
                     batch_size=16 if name in ('control-dual', 'rdrop010', 'rdrop050') else 32) for name in RUNS]
        self.put(self.root/'runs/push148_plan.json', dict(runs=runs, max_training_runs=5,
            calibration=dict(max_settings=75), parent_sha256=self.parent_sha, parent_train_targets=PARENT_VIEW_TARGETS))
        for index, declared in enumerate(runs):
            self.make_run(declared, 1.459-index*.0001)

    @staticmethod
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def make_run(self, declared, score, actual=4000, selected=2000, status='partial'):
        name = declared['name']
        train_dir = self.root/'runs'/('push148-'+name)
        calibrated_dir = self.root/'runs'/('push148-'+name+'-calibrated')
        train_dir.mkdir(parents=True, exist_ok=True)
        calibrated_dir.mkdir(parents=True, exist_ok=True)
        (train_dir/'checkpoint.pt').write_bytes(('synthetic training '+name).encode())
        (calibrated_dir/'checkpoint.pt').write_bytes(('synthetic calibration '+name).encode())
        labels = declared['batch_size']*256
        history = [dict(step=0, weights='raw'), dict(step=selected, weights='raw'), dict(step=selected, weights='ema')]
        training = dict(parent_sha256=self.parent_sha, parent_train_targets=PARENT_VIEW_TARGETS,
            test_used=False, steps=8000, batch_size=declared['batch_size'],
            incremental_search_targets=actual*8192, seconds=40.+RUNS.index(name),
            history=[dict(step=actual)], validation_history=history,
            best=dict(step=selected, weights='ema', checkpoint_sha256=sha(train_dir/'checkpoint.pt'),
                      selected_lineage_targets=PARENT_VIEW_TARGETS+selected*8192,
                      scalars={'cache_recency_strength': -.4}))
        if name != 'control-single':
            training.update(completed_steps=actual, status=status, dropout_views_per_sample=declared['views'],
                            incremental_sampled_label_targets=actual*labels)
            training['best']['new_selected_sampled_label_targets'] = selected*labels
        self.put(train_dir/'metrics.json', training)
        if actual < 8000 and status != 'failed':
            self.put(train_dir/'termination.json', dict(status='stopped', completed_steps=actual,
                metrics_sha256=sha(train_dir/'metrics.json'), reason='Explicit synthetic fixture termination'))
        trials = [dict(temperature=t, mkn_weight=w, validation_bpb=score if (t, w) == (1.25, .15) else score+.01) for t, w in sorted(GRID)]
        calibration = dict(test_used=False, trials=trials, seconds_total=3.,
            parent_sha256=sha(train_dir/'checkpoint.pt'), checkpoint_sha256=sha(calibrated_dir/'checkpoint.pt'),
            best=dict(temperature=1.25, mkn_weight=.15, validation_bpb=score),
            full_validation=dict(bpb=score+1e-8), precision='fp32', device='cuda:0')
        self.put(calibrated_dir/'metrics.json', calibration)

    def test_actual_counts_and_selected_lineage_not_plan_caps(self):
        result = collect_selection(self.root, 171)
        self.assertEqual(result['all_new_search_processed_view_targets'], 5*4000*8192)
        self.assertEqual(result['all_new_search_sampled_label_occurrences'], 4000*(2*8192+3*4096))
        self.assertEqual(result['winner']['selected_lineage_targets'], PARENT_VIEW_TARGETS+2000*8192)
        self.assertEqual(result['winner']['selected_run'], 'push148-muon-calibrated')
        self.assertEqual(result['total_raw_ema_validation_choices'], 15)
        self.assertEqual(result['total_calibration_choices'], 75)
        self.assertEqual(result['unit_tests']['executions'], 171)
        self.assertAlmostEqual(result['winner']['validation_bpb_gpu_fp32'], 1.4586+1e-8, places=14)

    def test_failed_partial_checkpoint_can_be_compared(self):
        path = self.root/'runs/push148-rdrop010/metrics.json'
        data = json.loads(path.read_text()); data.update(status='failed', failure_type='SyntheticFailure', failure_message='Explicit fixture failure')
        self.put(path, data)
        result = collect_selection(self.root, 172)
        row = next(row for row in result['candidates'] if row['name'] == 'rdrop010')
        self.assertTrue(row['selection_eligible'])
        self.assertEqual(row['recorded_status'], 'failed')

    def test_hash_corruption_refuses_entire_decision(self):
        (self.root/'runs/push148-muon/checkpoint.pt').write_bytes(b'changed synthetic data')
        result = collect_selection(self.root, 173)
        self.assertEqual(result['selection_status'], 'refused')
        self.assertIsNone(result['winner'])

    def test_missing_calibration_is_explicitly_ineligible(self):
        (self.root/'runs/push148-muon-calibrated/metrics.json').unlink()
        result = collect_selection(self.root, 174)
        row = result['candidates'][-1]
        self.assertFalse(row['selection_eligible'])
        self.assertIn('No completed calibration', row['selection_rejection'])
        self.assertEqual(result['total_calibration_choices'], 60)

    def test_refuses_overwrite_and_requires_measured_test_count(self):
        result = collect_selection(self.root, 175)
        destination = self.root/'runs/push148_selection.json'
        write_selection(destination, result)
        with self.assertRaises(FileExistsError):
            write_selection(destination, result)
        with self.assertRaises(ValueError):
            collect_selection(self.root, 0)

    def test_running_without_termination_refuses_selection(self):
        path = self.root/'runs/push148-muon/metrics.json'
        data = json.loads(path.read_text()); data['status'] = 'running'
        self.put(path, data)
        (path.parent/'termination.json').unlink()
        result = collect_selection(self.root, 176)
        self.assertEqual(result['selection_status'], 'refused')
        self.assertIsNone(result['winner'])

    def test_unstarted_planned_run_refuses_selection(self):
        (self.root/'runs/push148-muon/metrics.json').unlink()
        result = collect_selection(self.root, 177)
        self.assertEqual(result['selection_status'], 'refused')
        self.assertFalse(result['all_training_accounting_present'])

    def test_completed_run_without_calibration_refuses_selection(self):
        plan = json.loads((self.root/'runs/push148_plan.json').read_text())
        self.make_run(plan['runs'][-1], 1.458, actual=8000, status='completed')
        (self.root/'runs/push148-muon-calibrated/metrics.json').unlink()
        self.assertEqual(collect_selection(self.root, 178)['selection_status'], 'refused')


if __name__ == '__main__':
    unittest.main()
