"""Dataset-free evidence fixtures: no predictor, checkpoint corpus or GPU."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import push150_phase_bc_report as report


def measurement(bpb=1.4, seconds=1.):
    return dict(bpb=bpb, nll_nats=bpb*math.log(2)*1148007,
                targets=376599, utf8_bytes=1148007, seconds=seconds)


def history(parent_targets=500):
    rows = [dict(step=0, weights='raw', session=0, **measurement(1.5))]
    rows += [dict(step=step, weights=kind, session=0,
                  **measurement(1.5-step/100000-(.001 if kind == 'ema' else 0)))
             for step in (1000,2000,3000,4000) for kind in ('raw','ema')]
    best = dict(rows[-1], selected_lineage_targets=parent_targets+4000*8192)
    return dict(status='completed', completed_steps=4000, resume_history=[], discarded_completed_steps=0,
                validation_history=rows, best=best)


def gate_records():
    folds = report.fold_specifications()
    positions = sum(row['training_positions'] for row in folds)
    common = dict(new_neural_training_targets=0, retained_training_positions=positions)
    build = dict(common, folds=folds, parity_count_scan_token_occurrences=report.TRAIN_TOKENS,
                 ten_crossfit_count_scan_token_occurrences=9*report.TRAIN_TOKENS,
                 excluded_cross_fold_target_positions=9)
    extracted = dict(common, neural_extraction_target_occurrences=positions,
                     initialization_parity_neural_forward_target_occurrences=2048)
    trained = dict(common, parameters=57, seed=181, gate_target_occurrences=6*positions,
                   optimizer_updates=6*((positions+8191)//8192),
                   epochs=[dict(epoch=i, target_occurrences=positions, cumulative_gate_target_occurrences=i*positions,
                                mean_train_nll=1.) for i in range(1,7)])
    return build, extracted, trained


def repair_fixture():
    rows = []
    for repeat, batches, names in ((0,(8,16,32),('baseline','candidate')),
            (1,(32,16,8),('candidate','baseline')), (2,(16,32,8),('baseline','candidate'))):
        for mb in batches:
            for name in names:
                seconds = 7.+repeat*.1 if name == 'baseline' else 42.+repeat*.2+mb/100
                rows.append(dict(repeat=repeat, microbatch=mb, name=name, median_scoring_seconds=seconds,
                    repeats=[measurement(2. if name == 'baseline' else 1.4472849220037163, seconds)],
                    peak_process_working_set_bytes=2**30, checkpoint_bytes=50*2**20))
    candidates = []
    for mb in (8,16,32):
        times = {name:[r['median_scoring_seconds'] for r in rows if r['microbatch']==mb and r['name']==name]
                 for name in ('baseline','candidate')}
        medians = {name:sorted(values)[1] for name,values in times.items()}
        candidates.append(dict(microbatch=mb,times=times,medians=medians,
            cpu_ratio=medians['candidate']/medians['baseline'],bpb_max_difference=0.,peak_ram_gib=1.,
            eligible_for_training=False, pair_ratios=[c/b for b,c in zip(times['baseline'],times['candidate'])],
            max_candidate_min_baseline_ratio=max(times['candidate'])/min(times['baseline'])))
    return dict(measurements=rows,candidates=candidates,winner=None)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_c_recipe_matches_control_except_suffix_objective(self):
        config = dict(dropout=.1,temperature=1.4,mkn_weight=.2,mkn_order=5,mkn_shapes={'unigram':[2048]},
                      discount_scale=.55,width=320,push150_ranks={'qkv':160})
        control = report.expected_suffix_recipe(config,False)
        joint = report.expected_suffix_recipe(config,True)
        self.assertEqual({k:v for k,v in control.items() if k!='objective'},
                         {k:v for k,v in joint.items() if k!='objective'})
        self.assertEqual((control['steps'],control['seed'],control['warmup'],control['eval_every']), (4000,191,100,1000))
        self.assertEqual(control['tail_snapshot_start_step'],3000)
        self.assertFalse(any(key.startswith('mkn_') or key=='discount_scale' for key in control['training_config']))
        self.assertEqual(control['validation_config']['discount_scale'],.55)
        self.assertEqual(control['validation_config']['temperature'],1.25)

    def test_nine_choices_and_selected_lineage(self):
        metrics = history()
        self.assertEqual(report.verify_suffix_history(metrics,500)['step'],4000)
        metrics['best']['selected_lineage_targets'] += 8192
        with self.assertRaisesRegex(ValueError,'lineage'):
            report.verify_suffix_history(metrics,500)

    def test_missing_reordered_extra_validation_rejected(self):
        for mutate in (lambda rows: rows.pop(), lambda rows: rows.reverse(), lambda rows: rows.append(copy.deepcopy(rows[-1]))):
            metrics = history(); mutate(metrics['validation_history'])
            with self.assertRaisesRegex(ValueError,'nine'):
                report.verify_suffix_history(metrics,500)

    def test_resumed_or_failed_training_never_silently_selected(self):
        metrics = history(); metrics['resume_history'] = [{'session':1}]
        with self.assertRaisesRegex(ValueError,'manual'):
            report.verify_suffix_history(metrics,500)
        metrics = history(); metrics['status'] = 'failed'
        with self.assertRaisesRegex(ValueError,'incomplete'):
            report.verify_suffix_history(metrics,500)

    def test_nonminimum_snapshot_rejected(self):
        metrics = history(); metrics['best']['bpb'] += .01
        with self.assertRaisesRegex(ValueError,'minimum'):
            report.verify_suffix_history(metrics,500)

    def test_fold_positions_and_ten_count_scans(self):
        records = gate_records(); positions = report.verify_gate_training_counts(*records)
        self.assertGreater(positions,3_000_000)
        self.assertLess(positions,report.TRAIN_TOKENS)
        for fold in records[0]['folds']:
            start,end = fold['excluded_token_interval']
            self.assertEqual(start%256,0)
            for first in range(start,end-256,256):
                self.assertLess(first+256,end)
            self.assertNotIn([start,end],fold['included_token_intervals'])

    def test_gate_six_epochs_are_nonzero_label_cost_not_neural_updates(self):
        build,extract,train = gate_records()
        report.verify_gate_training_counts(build,extract,train)
        self.assertGreater(train['gate_target_occurrences'],0)
        self.assertEqual(train['new_neural_training_targets'],0)
        train['gate_target_occurrences'] = 0
        with self.assertRaisesRegex(ValueError,'cost'):
            report.verify_gate_training_counts(build,extract,train)

    def test_self_fold_and_cross_boundary_accounting_rejected(self):
        build,extract,train = gate_records()
        build['folds'][0]['included_token_intervals'][0] = build['folds'][0]['excluded_token_interval']
        with self.assertRaisesRegex(ValueError,'OOF'):
            report.verify_gate_training_counts(build,extract,train)
        build,extract,train = gate_records()
        build['folds'][0]['complete_training_windows'] += 1
        with self.assertRaisesRegex(ValueError,'OOF'):
            report.verify_gate_training_counts(build,extract,train)

    def test_partial_epochs_and_extra_neural_updates_rejected(self):
        build,extract,train = gate_records(); train['epochs'].pop()
        with self.assertRaisesRegex(ValueError,'epoch'):
            report.verify_gate_training_counts(build,extract,train)
        build,extract,train = gate_records(); extract['new_neural_training_targets'] = 1
        with self.assertRaisesRegex(ValueError,'neural'):
            report.verify_gate_training_counts(build,extract,train)

    def test_all_repair_passes_required_and_original_winner_not_used(self):
        repair = repair_fixture(); report.verify_repaired_timings(repair)
        repair['winner'] = {'microbatch':32}
        with self.assertRaisesRegex(ValueError,'cannot authorize'):
            report.verify_repaired_timings(repair)
        repair = repair_fixture(); repair['measurements'].pop()
        with self.assertRaisesRegex(ValueError,'18'):
            report.verify_repaired_timings(repair)

    def test_repair_ratio_and_eligibility_recomputed(self):
        repair = repair_fixture(); repair['candidates'][0]['eligible_for_training'] = True
        with self.assertRaisesRegex(ValueError,'eligibility'):
            report.verify_repaired_timings(repair)
        repair = repair_fixture(); repair['candidates'][0]['cpu_ratio'] = 4.
        with self.assertRaisesRegex(ValueError,'ratio'):
            report.verify_repaired_timings(repair)

    def test_gate_threshold_does_not_remove_small_gain_winner(self):
        rows = [dict(name='parent',selection_eligible=True,validation_bpb_gpu_fp32=1.44),
                dict(name='gate',selection_eligible=True,validation_bpb_gpu_fp32=1.4398)]
        selected = report.select_candidates(rows,1.44,[])
        self.assertEqual(selected['winner']['name'],'gate')
        self.assertFalse(selected['passes_predeclared_improvement_gate'])
        rows[1]['validation_bpb_gpu_fp32'] = 1.439
        self.assertTrue(report.select_candidates(rows,1.44,[])['passes_predeclared_improvement_gate'])

    def test_any_branch_integrity_error_refuses_global_choice(self):
        rows = [dict(name='good',selection_eligible=True,validation_bpb_gpu_fp32=1.4)]
        selected = report.select_candidates(rows,1.5,[{'name':'failed-gate'}])
        self.assertEqual(selected['status'],'refused'); self.assertIsNone(selected['winner'])

    def test_corpus_text_and_unrelated_array_hash_refused_before_open(self):
        with patch.object(Path,'open',side_effect=AssertionError('No file I/O')):
            for path in ('data/wikitext_test.txt','data/wikitext_train.txt','output/test.npy'):
                with self.assertRaises(ValueError):
                    report.sha(Path(path))

    def test_safe_feature_hash_and_changed_derived_asset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory = root/'train_features'; directory.mkdir()
            path = directory/'fold-00.npy'; path.write_bytes(b'synthetic header only')
            record = dict(path='train_features/fold-00.npy',bytes=path.stat().st_size,sha256=report.sha(path))
            self.assertEqual(report.artifact(root,record),path)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                report.artifact(root,record)

    def test_unit_evidence_source_change_and_execution_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'tests').mkdir()
            source = root/'synthetic.py'; source.write_text('pass\n')
            test = root/'tests/test_synthetic.py'; test.write_text('pass\n')
            record = dict(successful=True,failures=[],errors=[],skipped=[],test_text_loaded=False,tests_run=3,
                          source_sha256={'synthetic.py':report.sha(source)},
                          unit_source_sha256={'test_synthetic.py':report.sha(test)})
            path = root/'units.json'; path.write_text(json.dumps(record))
            result = report.verify_units(root,[path])
            self.assertEqual(result['execution_count'],3); self.assertFalse(result['this_report_covered'])
            source.write_text('changed\n')
            with self.assertRaisesRegex(ValueError,'source changed'):
                report.verify_units(root,[path])

    def test_source_graph_requires_exact_frozen_local_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'entry.py').write_text('from dependency import x\n')
            (root/'dependency.py').write_text('x=1\n')
            hashes = {name:report.sha(root/name) for name in ('entry.py','dependency.py')}
            with patch.object(report,'FROZEN_SOURCES',hashes):
                self.assertEqual(report.source_graph(root,['entry.py']),hashes)
                (root/'dependency.py').write_text('x=2\n')
                with self.assertRaisesRegex(ValueError,'changed'):
                    report.source_graph(root,['entry.py'])

    def test_real_frozen_graph_is_source_only_and_complete_for_both_drivers(self):
        # Only read Python ASTs/hashes, never execute the drivers or data loaders.
        for implementation,suffix in (('push150_expand',False),('push147_signed_cache',False),('push150_suffix',True)):
            sources = report.training_sources(report.ROOT,implementation,suffix)
            self.assertIn(implementation+'.py',sources)
            self.assertIn('push148_muon.py',sources)
            if suffix:
                self.assertTrue({'push150_suffix_init.py','push150_suffix_train.py','push150_suffix_calibrate.py'} <= set(sources))

    def test_old_control_calibrator_without_new_lineage_field_is_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory = root/'control-calibrated'; directory.mkdir()
            parent_sha = 'a'*64
            parent = dict(protocol=report.a.PROTOCOL, implementation='push150_expand',train_tokens=500,
                          config={'temperature':1.25,'mkn_weight':.15}, model={'synthetic':torch.zeros(1)})
            trials = [dict(temperature=t,mkn_weight=w,validation_bpb=1.4+index*.001,
                           validation_nll_nats=(1.4+index*.001)*math.log(2)*1148007)
                      for index,(t,w) in enumerate((t,w) for t in (1.2,1.25,1.3,1.35,1.4) for w in (.1,.15,.2))]
            calibrated = copy.deepcopy(parent)
            calibrated['config'].update(temperature=1.2,mkn_weight=.1)
            calibrated['calibration_parent_sha256'] = parent_sha
            path = directory/'checkpoint.pt'; torch.save(calibrated,path)
            metrics = dict(parent_sha256=parent_sha,parent_config=parent['config'],test_used=False,new_training_targets=0,
                implementation='push150_expand',precision='fp32',device='cuda:0',threads=2,
                script_sha256=report.a.CALIBRATE_SHA,implementation_sha256=report.FROZEN_SOURCES['push150_expand.py'],
                evaluator_sha256=report.a.TRAIN_SOURCES['evaluate.py'],tokenizer_sha256=report.a.DATA_SHA['data/tokenizer.json'],
                validation_sha256=report.a.DATA_SHA['data/wikitext_validation.txt'],validation_target_occurrences=376599,
                validation_utf8_bytes=1148007,seconds_total=1.,trials=trials,best=trials[0],
                full_validation=measurement(),shortcut_full_bpb_absolute_difference=0.,shortcut_matches_full_predictor=True,
                checkpoint_sha256=report.sha(path),checkpoint_bytes=path.stat().st_size)
            plan = dict(parent_sha256=parent_sha,trials=15,test_used=False,temperatures=[1.2,1.25,1.3,1.35,1.4],
                        mkn_weights=[.1,.15,.2],split='validation',script_sha256=report.a.CALIBRATE_SHA)
            for filename,payload in (('metrics.json',metrics),('trials.json',trials),('experiment_plan.json',plan)):
                (directory/filename).write_text(json.dumps(payload))
            row = {}
            report.verify_calibration(root,directory,parent,parent_sha,False,row)
            self.assertTrue(row['selection_eligible'])
            self.assertEqual(row['selected_lineage_targets'],500)
            self.assertEqual(row['calibration_choices'],15)
            metrics['parent_sha256'] = 'b'*64
            (directory/'metrics.json').write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError,'parent differs'):
                report.verify_calibration(root,directory,parent,parent_sha,False,{})

    def test_missing_a_keeps_observed_failed_training_cost(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory = root/'runs/push150-suffix-control'; directory.mkdir(parents=True)
            metrics = dict(completed_steps=1200,discarded_completed_steps=0,incremental_search_targets=1200*8192,
                incremental_sampled_label_targets=1200*8192,lineage_new_targets=1200*8192,seconds=12.,status='failed',
                failure_type='Synthetic',failure_message='fixture',failed_attempt_cost_note='partial unknown',validation_history=[])
            (directory/'metrics.json').write_text(json.dumps(metrics))
            result = report.collect_report(root, [root/'missing-units.json'])
            self.assertEqual(result['status'],'refused'); self.assertIsNone(result['winner'])
            observed = result['observed_evidence']['suffix_runs'][0]
            self.assertEqual(observed['new_search_processed_view_targets'],1200*8192)
            self.assertEqual(observed['failure_type'],'Synthetic')
            self.assertEqual(result['cost_summary']['phase_c_new_label_occurrences'],1200*8192)
            self.assertFalse(result['cost_summary']['accounting_verified'])

    def test_report_writer_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'report.json'
            report.a.write_report(path, {'status':'refused'})
            with self.assertRaises(FileExistsError):
                report.a.write_report(path, {'status':'changed'})
            self.assertEqual(json.loads(path.read_text()),{'status':'refused'})


if __name__ == '__main__':
    unittest.main()
