"""Synthetic metadata only; no corpus/checkpoints/scorer/GPU/results execution."""
import ast
import copy
import inspect
import math
from pathlib import Path
import tempfile
import unittest

import push150_results as report


class ResultTests(unittest.TestCase):
    def ledger(self, implementation='push150_suffix'):
        winner=dict(checkpoint_sha256='a'*64,implementation=implementation)
        freeze=dict(frozen_at_utc='2026-09-30T00:00:00+00:00',implementation_sha256='b'*64,inference_assets_mib=43.)
        launch=dict(created_at_utc='2026-09-30T00:00:01+00:00',candidate_pass_cap=1,
            device='cpu',precision='fp32',threads=4,affinity_mask=15,candidate_sha256='a'*64,
            baseline_sha256=report.BASELINE_SHA,evaluator_sha256=report.SCORER_SHA,tokenizer_sha256=report.TOKENIZER_SHA,
            planned_passes=[dict(name=name,sha256='a'*64 if name=='candidate' else report.BASELINE_SHA) for name in report.PASS_NAMES])
        records={}; rows=[]
        for index,name in enumerate(report.PASS_NAMES):
            digest='a'*64 if name=='candidate' else report.BASELINE_SHA
            seconds=35. if name=='candidate' else 8.+index
            bpb=1.44 if name=='candidate' else 2.1
            measured=dict(protocol='7506-mp1-wt2-v2',split='test',precision='fp32',device='cpu',device_name='CPU',
                checkpoint_sha256=digest,implementation_sha256='b'*64 if name=='candidate' else report.BASELINE_MODEL_SHA,
                evaluator_sha256=report.SCORER_SHA,tokenizer_sha256=report.TOKENIZER_SHA,
                targets=428405,utf8_bytes=1292013,bpb=bpb,nll_nats=bpb*math.log(2)*1292013,seconds=seconds)
            record=dict(started=dict(index=index,name=name,checkpoint_sha256=digest,
                started_at_utc=f'2026-09-30T00:00:{2+index*2:02d}+00:00'),
                process=dict(returncode=0,completed_at_utc=f'2026-09-30T00:00:{3+index*2:02d}+00:00'),
                measured=measured,resource=dict(cpu_affinity_mask=15,peak_process_working_set_bytes=2**30),
                output_sha256='c'*64,resources_sha256='d'*64)
            records[name]=record
            rows.append(dict(name=name,output=name+'.json',output_sha256='c'*64,resources_sha256='d'*64,
                             **measured,peak_ram_gib=1.))
        summary=dict(status='completed',limits_pass=True,candidate_scoring_process_launches=1,
            baseline_scoring_process_launches=2,no_post_test_selection=True,measurements=rows,
            completed_at_utc='2026-09-30T00:00:08+00:00',test_bpb=1.44,
            cpu_ratio_to_baseline_mean=35/9,cpu_ratio_to_faster_baseline=35/8,peak_ram_gib=1.,inference_assets_mib=43.)
        return launch,records,summary,freeze,winner

    def costs(self):
        training={}; calibration={}
        for index,name in enumerate(report.TRAIN_RUNS):
            completed=20 if index<3 else 4 if index<5 else 12
            parent=report.ORIGINAL_PARENT_TARGETS if index<3 else report.ORIGINAL_PARENT_TARGETS+20*8192
            training[name]=dict(status='completed',test_used=False,completed_steps=completed,steps=completed,
                discarded_completed_steps=0,batch_size=32,dropout_views_per_sample=1,
                incremental_search_targets=completed*8192,incremental_sampled_label_targets=completed*8192,
                parent_train_targets=parent,best=dict(step=completed,weights='ema',selected_lineage_targets=parent+completed*8192),
                validation_history=[{}]*(21 if index<3 else 9 if index<5 else 13),seconds=10.)
            calibration[name]=dict(test_used=False,new_training_targets=0,trials=[{}]*15,
                best=dict(validation_bpb=1.44+index*.00001),full_validation=dict(bpb=1.44+index*.00001),seconds_total=2.)
        gate={phase:dict(status='completed',test_used=False,seconds_total=1.) for phase in ('build','extract','train','score')}
        gate['train'].update(gate_target_occurrences=600,new_neural_training_targets=0)
        gate['build'].update(parity_count_scan_token_occurrences=100,ten_crossfit_count_scan_token_occurrences=900)
        gate['extract'].update(neural_extraction_target_occurrences=100,initialization_parity_neural_forward_target_occurrences=2048)
        gate['score']['choices']=[{}]*5
        a,c,d=60*8192,8*8192,24*8192
        bc=dict(accounting_verified=True,phase_a_new_neural_view_targets=a,phase_c_new_label_occurrences=c,
                gate_label_occurrences=600,all_recorded_new_gradient_label_occurrences=a+c+600)
        overall=dict(accounting_verified=True,phase_abc_recorded_new_gradient_label_occurrences=a+c+600,
            phase_d_new_label_occurrences=d,all_recorded_new_gradient_label_occurrences=a+c+d+600,
            shared_B_count_scan_token_occurrences_added_again_in_D=0,
            shared_original149_ancestry_targets=report.ORIGINAL_PARENT_TARGETS,missing_or_partial_work_remains_unquantified=False)
        selection=dict(phase_bc={'record':{'cost_summary':bc}},cost_summary=overall,
            winner={'selected_lineage_targets':123},followon_launches={'seconds_total_inclusive_not_added_to_child_costs':150.})
        return (selection,training,calibration,gate,dict(seconds_total=1.,count_scan_token_occurrences=100),
                dict(seconds_total=2.,trials=[{}]*9),dict(costs={'new_count_scan_token_occurrences':0,'seconds_total':3.}))

    def test_valid_ledger_supports_suffix_and_unchanged_d_implementation(self):
        for implementation in ('push150_suffix','push150_expand'):
            value=report.verify_ledger_records(*self.ledger(implementation))
            self.assertTrue(value['limits_pass']);self.assertEqual(value['test_bpb'],1.44)
            self.assertEqual(value['candidate_scoring_passes'],1)

    def test_failed_partial_or_resource_rejected_summary_refuses(self):
        for key,value in (('status','failed'),('status','running'),('limits_pass',False),('candidate_scoring_process_launches',2),
                          ('baseline_scoring_process_launches',1),('no_post_test_selection',False)):
            data=self.ledger();data[2][key]=value
            with self.assertRaises(ValueError):report.verify_ledger_records(*data)

    def test_changed_actual_result_or_resource_rejected_even_with_pass_flag(self):
        for field,value in (('targets',428404),('utf8_bytes',1292012),('bpb',1.43),('checkpoint_sha256','e'*64),('seconds',float('nan'))):
            data=self.ledger();data[1]['candidate']['measured'][field]=value
            with self.assertRaises(ValueError):report.verify_ledger_records(*data)
        data=self.ledger();data[1]['candidate']['resource']['peak_process_working_set_bytes']=5*2**30
        with self.assertRaises(ValueError):report.verify_ledger_records(*data)

    def test_recomputed_speed_limit_not_just_boolean(self):
        data=self.ledger();data[1]['candidate']['measured']['seconds']=50.
        data[2]['measurements'][1]['seconds']=50.;data[2]['cpu_ratio_to_baseline_mean']=50/9
        data[2]['cpu_ratio_to_faster_baseline']=50/8
        with self.assertRaises(ValueError):report.verify_ledger_records(*data)

    def test_faster_baseline_not_an_undeclared_extra_limit(self):
        data=self.ledger();data[1]['candidate']['measured']['seconds']=44.
        data[2]['measurements'][1]['seconds']=44.;data[2]['cpu_ratio_to_baseline_mean']=44/9
        data[2]['cpu_ratio_to_faster_baseline']=44/8
        result=report.verify_ledger_records(*data)
        self.assertTrue(result['limits_pass']);self.assertGreater(result['ratio_to_faster_baseline'],5)

    def test_pre_freeze_or_overlapping_passes_refused(self):
        data=self.ledger();data[0]['created_at_utc']='2026-09-29T23:59:59+00:00'
        with self.assertRaises(ValueError):report.verify_ledger_records(*data)
        data=self.ledger();data[1]['candidate']['started']['started_at_utc']='2026-09-30T00:00:02+00:00'
        with self.assertRaises(ValueError):report.verify_ledger_records(*data)

    def test_failed_exit_and_missing_or_reordered_launches_refused(self):
        data=self.ledger();data[1]['candidate']['process']['returncode']=1
        with self.assertRaises(ValueError):report.verify_ledger_records(*data)
        data=self.ledger();data[2]['measurements'].reverse()
        with self.assertRaises(ValueError):report.verify_ledger_records(*data)
        data=self.ledger();data[0]['planned_passes'].pop()
        with self.assertRaises(ValueError):report.verify_ledger_records(*data)

    def test_costs_count_shared_tables_parent_and_gate_once(self):
        result=report.summarize_costs(*self.costs())
        self.assertEqual(result['all_new_gradient_label_occurrences'],92*8192+600)
        self.assertEqual(result['new_neural_sampled_label_occurrences'],92*8192)
        self.assertEqual(result['shared_original149_ancestry_targets_counted_once'],376832000)
        self.assertEqual(result['count_scan_token_occurrences'],dict(mkn=100,gate_full_parity=100,gate_tenfold=900,phase_d_additional=0))
        self.assertEqual(result['training_loop_seconds'],70.)
        self.assertEqual(result['calibration_seconds'],14.)
        self.assertEqual(result['raw_ema_validation_choices'],107)
        self.assertEqual(result['neural_calibration_choices'],105)
        self.assertEqual(result['followon_launch_seconds_inclusive_not_added_again'],150.)

    def test_double_counted_gate_or_shared_scans_refuses(self):
        values=self.costs();values[0]['cost_summary']['all_recorded_new_gradient_label_occurrences']+=600
        with self.assertRaises(ValueError):report.summarize_costs(*values)
        values=self.costs();values[-1]['costs']['new_count_scan_token_occurrences']=100
        with self.assertRaises(ValueError):report.summarize_costs(*values)

    def test_actual_completed_and_best_selected_steps_not_plan_cap(self):
        values=self.costs();training=values[1]['oof-joint'];training['best'].update(step=0,selected_lineage_targets=training['parent_train_targets'])
        result=report.summarize_costs(*values)
        self.assertEqual(result['training_runs'][-1]['selected_step'],0)
        self.assertEqual(result['training_runs'][-1]['sampled_label_occurrences'],12*8192)
        values=self.costs();values[1]['oof-joint']['completed_steps']=11
        with self.assertRaises(ValueError):report.summarize_costs(*values)

    def test_partial_training_unconfirmed_calibration_refuse(self):
        values=self.costs();values[1]['oof-joint']['status']='failed'
        with self.assertRaises(ValueError):report.summarize_costs(*values)
        values=self.costs();values[2]['oof-joint']['trials'].pop()
        with self.assertRaises(ValueError):report.summarize_costs(*values)
        values=self.costs();values[2]['oof-joint']['best']['validation_bpb']-=.001
        with self.assertRaises(ValueError):report.summarize_costs(*values)

    def test_comparison_uses_dynamic_actual_scores(self):
        costs=report.summarize_costs(*self.costs())
        result=report.comparison(costs['training_runs'],'oof-joint','oof-control','single seed')
        self.assertLess(result['validation_improvement'],0)
        self.assertTrue(result['processed_targets_match'])

    def test_metadata_only_no_model_or_evaluation_calls(self):
        source=inspect.getsource(report);tree=ast.parse(source)
        imported={node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
        self.assertFalse({'torch','evaluate','common','push150_select'} & imported)
        calls={ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)}
        self.assertFalse({'torch.load','score','make_model','load_data','np.load'} & calls)
        self.assertNotIn('read_bytes',calls)
        self.assertEqual(len(report.ZIP_NAMES),8)

    def test_exclusive_output_and_corpus_read_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            destination=Path(directory)/'RESULTS.json'
            report.write_new(destination,{'verified':True})
            with self.assertRaises(FileExistsError):report.write_new(destination,{'verified':False})
        with self.assertRaises(ValueError):report.sha(Path('wikitext_test.txt'))
        with self.assertRaises(ValueError):report.read(Path('wikitext_train.txt'))


if __name__=='__main__':unittest.main()
