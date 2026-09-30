"""Tiny synthetic evidence fixtures: no corpus, real checkpoints, or scoring."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import push150_phase_a_report as report


def measurement(bpb=1.44,seconds=4.):
    return dict(bpb=bpb,nll_nats=bpb*math.log(2)*1148007,
                targets=376599,utf8_bytes=1148007,seconds=seconds)


def training_fixture():
    choices=[(0,'raw')]+[(step,kind) for step in range(2000,20001,2000) for kind in ('raw','ema')]
    history=[dict(step=step,weights=kind,session=0,**measurement(1.45-index*.0001))
             for index,(step,kind) in enumerate(choices)]
    best=dict(history[-1],selected_lineage_targets=540672000)
    return dict(completed_steps=20000,discarded_completed_steps=0,incremental_search_targets=163840000,
                incremental_sampled_label_targets=163840000,lineage_new_targets=163840000,
                seconds=800.,status='completed',resume_history=[],validation_history=history,best=best)


def calibration_fixture():
    trials=[dict(temperature=t,mkn_weight=w,validation_bpb=1.44+index*.0001)
            for index,(t,w) in enumerate(sorted(report.GRID))]
    return dict(parent_sha256='training-sha',test_used=False,new_training_targets=0,
        implementation='push150_expand',implementation_sha256=report.TRAIN_SOURCES['push150_expand.py'],
        script_sha256=report.CALIBRATE_SHA,evaluator_sha256=report.TRAIN_SOURCES['evaluate.py'],
        tokenizer_sha256=report.DATA_SHA['data/tokenizer.json'],
        validation_sha256=report.DATA_SHA['data/wikitext_validation.txt'],precision='fp32',device='cuda:0',
        trials=trials,best=trials[0],full_validation=measurement(1.440000004),
        shortcut_full_bpb_absolute_difference=abs(1.440000004-1.44),shortcut_matches_full_predictor=True,
        validation_target_occurrences=376599,validation_utf8_bytes=1148007)


class PhaseAReportTests(unittest.TestCase):
    def test_validation_measurement_rejects_wrong_targets_bytes_nonfinite_or_nll(self):
        self.assertEqual(report.verify_measurement(measurement()),1.44)
        for key,value in (('targets',428405),('utf8_bytes',1292013),('bpb',float('nan')),
                          ('seconds',0),('nll_nats',42)):
            row=measurement(); row[key]=value
            with self.assertRaises(ValueError):
                report.verify_measurement(row)

    def test_all_raw_ema_choices_and_minimum_are_required(self):
        training=training_fixture()
        self.assertEqual(report.verify_history(training)['step'],20000)
        for change in ('missing','duplicate','best','lineage','future'):
            bad=copy.deepcopy(training)
            if change=='missing': bad['validation_history'].pop()
            if change=='duplicate': bad['validation_history'][2]=bad['validation_history'][1]
            if change=='best': bad['best']['bpb']=0.1
            if change=='lineage': bad['best']['selected_lineage_targets']=376832000
            if change=='future': bad['completed_steps']=18000
            with self.subTest(change=change),self.assertRaises(ValueError):
                report.verify_history(bad)

    def test_failed_partial_accounting_retains_all_completed_work(self):
        training=training_fixture()
        training.update(completed_steps=1234,discarded_completed_steps=123,
            incremental_search_targets=1357*8192,incremental_sampled_label_targets=1357*8192,
            lineage_new_targets=1234*8192,status='failed',failure_type='KeyboardInterrupt',
            failure_message='',failed_attempt_cost_note='Unknown partial attempt work')
        actual=report.training_accounting(training)
        self.assertEqual(actual['new_search_processed_view_targets'],1357*8192)
        self.assertEqual(actual['new_search_sampled_label_occurrences'],1357*8192)
        self.assertEqual(actual['failure_type'],'KeyboardInterrupt')
        training['incremental_search_targets']=1234*8192
        with self.assertRaises(ValueError):
            report.training_accounting(training)

    def test_resumed_history_is_explicitly_refused_until_manual_merged_accounting(self):
        training=training_fixture(); training['resume_history']=[{'resumed_from_step':2000}]
        with self.assertRaisesRegex(ValueError,'manual merged-history'):
            report.verify_history(training)

    def test_calibration_grid_parent_and_full_confirmation_are_checked(self):
        fixture=calibration_fixture()
        self.assertEqual(report.verify_calibration(fixture,'training-sha'),1.440000004)
        for change in ('parent','duplicate','best','full','device','source'):
            bad=copy.deepcopy(fixture)
            if change=='parent': bad['parent_sha256']='archive-sha-not-scoring-sha'
            if change=='duplicate': bad['trials'][1]=bad['trials'][0]
            if change=='best': bad['best']=bad['trials'][2]
            if change=='full': bad['full_validation']=measurement(1.45)
            if change=='device': bad['device']='cpu'
            if change=='source': bad['script_sha256']='wrong'
            with self.subTest(change=change),self.assertRaises(ValueError):
                report.verify_calibration(bad,'training-sha')

    def test_recipe_has_fixed_cost_and_keeps_statistics_out_of_training(self):
        parent=dict(temperature=1.25,dropout=.2,mkn_weight=.15,mkn_order=5,
                    mkn_shapes={'unigram':[2048]},discount_scale=.7,push150_ranks={'qkv':160})
        old=copy.deepcopy(parent)
        recipe=report.expected_recipe({'schedule':'wsd'},parent)
        self.assertEqual(recipe['steps'],20000)
        self.assertEqual(recipe['batch_size']*recipe['context'],8192)
        self.assertEqual(recipe['warmup'],200)
        self.assertEqual(recipe['decay_steps'],4000)
        self.assertEqual(recipe['training_config']['temperature'],1.15)
        self.assertFalse(any(key.startswith('mkn_') for key in recipe['training_config']))
        self.assertEqual(recipe['validation_config']['mkn_shapes'],parent['mkn_shapes'])
        recipe['training_config']['push150_ranks']['qkv']=99
        self.assertEqual(parent,old)

    def test_file_boundary_refuses_all_corpus_text_before_open(self):
        with patch.object(Path,'open',side_effect=AssertionError('Opened forbidden text')):
            for path in ('data/wikitext_test.txt','data/wikitext_train.txt','data/wikitext_validation.txt'):
                with self.assertRaises(ValueError): report.sha(path)
                with self.assertRaises(ValueError): report.read(path)
        with self.assertRaises(ValueError):
            report.child(Path('fixture-root'),'../outside.json')

    def test_exclusive_result_creation_preserves_refusal_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'refusal.json'
            result={'status':'refused','evidence_errors':[{'name':'example','error':'missing evidence'}]}
            report.write_report(path,result)
            with self.assertRaises(FileExistsError): report.write_report(path,{'status':'wrong'})
            self.assertEqual(json.loads(path.read_text()),result)

    def test_feasibility_uses_time_ram_checkpoint_size_and_parent_score(self):
        names=['baseline-before','control','expanded','dense','baseline-after']
        initializers={name:{'report':{'checkpoint_sha256':name+'-sha'}} for name in names[1:4]}
        measurements=[]
        for name in names:
            baseline=name.startswith('baseline')
            seconds=8. if baseline else 32.
            measurements.append(dict(name=name,split='validation',repeats=[measurement(1.4472849220037163,seconds)],
                threads=4,cpu_affinity_mask=15,median_scoring_seconds=seconds,
                evaluator_sha256=report.TRAIN_SOURCES['evaluate.py'],
                checkpoint_sha256=report.BASELINE_SHA if baseline else name+'-sha',
                implementation_sha256=report.BASELINE_IMPL_SHA if baseline else report.TRAIN_SOURCES['push150_expand.py'],
                peak_process_working_set_bytes=2**30,checkpoint_bytes=40*2**20))
        record=dict(status='completed',phase_plan_sha256=report.PLAN_SHA,test_used=False,
                    device='cpu',precision='fp32',threads=4,affinity_mask=15,split='validation',
                    measurements=measurements,baseline_mean_seconds=8.,
                    candidates={name:dict(cpu_ratio=4.,parent_bpb_absolute_difference=0.,
                        preliminary_feasible=True,training_authorized_by_phase_a=name!='dense') for name in names[1:4]})
        def fake_sha(path):
            return report.BASELINE_IMPL_SHA if Path(path).name=='model.py' else report.BASELINE_SHA
        with patch.object(report,'sha',side_effect=fake_sha):
            report.verify_feasibility(Path('fixture'),record,initializers)
            for key,value in (('peak_process_working_set_bytes',5*2**30),('checkpoint_bytes',64*2**20)):
                bad=copy.deepcopy(record); bad['measurements'][2][key]=value
                with self.assertRaisesRegex(ValueError,'feasibility result'):
                    report.verify_feasibility(Path('fixture'),bad,initializers)
                bad['candidates']['expanded'].update(preliminary_feasible=False,training_authorized_by_phase_a=False)
                report.verify_feasibility(Path('fixture'),bad,initializers)

    def synthetic_collect(self,training_scores,mkn_score,invalid_run=None):
        root=Path('fixture-root')
        parent_score=1.447284978111573
        config=dict(temperature=1.25,mkn_weight=.15,discount_scale=.7)
        parent=dict(train_tokens=376832000,config=config,implementation='push147_signed_cache')
        plan=dict(parent_sha256=report.PARENT_SHA,parent_train_targets=376832000,
            parent_validation_bpb_gpu_fp32=parent_score,runs=[
                dict(name=name,initialization='expanded' if name.startswith('expanded') else 'control',
                     schedule='wsd' if name.endswith('wsd') else 'cosine') for name in report.RUNS])
        parent_metrics=dict(checkpoint_sha256=report.PARENT_SHA,test_used=False,full_validation=measurement(parent_score))
        def fake_read(path):
            path=Path(path)
            if path.name=='push150_phase_a_plan.json': return plan
            if path.name=='metrics.json': return parent_metrics
            if path.name=='summary.json': return {'status':'synthetic-feasibility'}
            raise AssertionError('Unexpected read '+str(path))
        def fake_sha(path):
            path=Path(path)
            if path.name=='push150_phase_a_plan.json': return report.PLAN_SHA
            if path.name=='checkpoint.pt': return report.PARENT_SHA
            if path.name=='push150_init.py': return report.INIT_SHA
            if path.name=='push146_calibrate.py': return report.CALIBRATE_SHA
            if path.name=='tokenizer.json': return report.DATA_SHA['data/tokenizer.json']
            if path.name=='manifest.json': return report.DATA_SHA['data/manifest.json']
            return 'synthetic-sha'
        def fake_training(root,declared,init,feasibility,row):
            row.update(new_search_processed_view_targets=163840000,new_search_sampled_label_occurrences=163840000,
                       training_loop_seconds=800.,raw_ema_validation_choices=21,calibration_choices=15,calibration_seconds=6.)
            if declared['name']==invalid_run: raise ValueError('Synthetic incomplete/changed evidence')
            row.update(selection_eligible=True,validation_bpb_gpu_fp32=training_scores[declared['name']],
                       selected_run='push150-'+declared['name']+'-calibrated',checkpoint_sha256='selected-'+declared['name'])
        def fake_mkn(root,parent,row):
            row.update(selection_eligible=True,validation_bpb_gpu_fp32=mkn_score,selected_run='push150-mkn',
                       calibration_choices=9,calibration_seconds=6.,count_scan_token_occurrences=3613343)
        initializers={name:{'report':{},'report_sha256':'sha'} for name in ('control','expanded','dense')}
        with patch.object(report,'read',side_effect=fake_read),patch.object(report,'sha',side_effect=fake_sha), \
             patch.object(report,'checkpoint',return_value=parent),patch.object(report,'verify_sources'), \
             patch.object(report,'collect_initializers',return_value=initializers),patch.object(report,'verify_feasibility'), \
             patch.object(report,'collect_training',side_effect=fake_training),patch.object(report,'collect_mkn',side_effect=fake_mkn):
            return report.collect_report(root,222)

    def test_selection_minimum_gate_and_total_work_are_separate(self):
        scores={'control-cosine':1.4472,'control-wsd':1.4473,'expanded-cosine':1.4474}
        result=self.synthetic_collect(scores,1.44699)
        self.assertEqual(result['status'],'phase_a_choice_complete')
        self.assertEqual(result['winner']['name'],'mkn-discount')
        self.assertFalse(result['passes_predeclared_improvement_gate'])
        self.assertEqual(result['all_new_search_processed_view_targets'],491520000)
        self.assertEqual(result['total_recorded_raw_ema_validation_choices'],63)
        self.assertEqual(result['total_recorded_calibration_choices'],54)
        self.assertFalse(result['final_cpu_audit_completed'])
        scores['expanded-cosine']=1.44
        improved=self.synthetic_collect(scores,1.44699)
        self.assertEqual(improved['winner']['name'],'expanded-cosine')
        self.assertTrue(improved['passes_predeclared_improvement_gate'])

    def test_incomplete_or_changed_one_run_refuses_winner_without_erasing_work(self):
        scores={name:1.44 for name in report.RUNS}
        result=self.synthetic_collect(scores,1.43,invalid_run='control-wsd')
        self.assertEqual(result['status'],'refused')
        self.assertIsNone(result['winner'])
        self.assertEqual(result['all_new_search_processed_view_targets'],491520000)
        self.assertEqual(result['evidence_errors'][0]['name'],'control-wsd')
        self.assertEqual(len(result['candidates']),5)


if __name__=='__main__':
    unittest.main()
