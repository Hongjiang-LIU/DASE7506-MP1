"""Synthetic metadata only: no corpus, model imports, GPU, or real scorer."""
import copy
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import push151_finalize as f


def measurement(bpb=1.43, seconds=32., checkpoint='a'*64, split='validation', official=False):
    targets, size = (376599,1148007) if split == 'validation' else (428405,1292013)
    result = dict(bpb=bpb, nll_nats=bpb*math.log(2)*size, seconds=seconds, targets=targets, utf8_bytes=size,
        split=split, precision='fp32', checkpoint_sha256=checkpoint, implementation_sha256=f.IMPLEMENTATION_SHA,
        evaluator_sha256=f.SCORER_SHA, device='cuda:0', test_used=False)
    if official:
        result.update(protocol='7506-mp1-wt2-v2', tokenizer_sha256=f.TOKENIZER_SHA, device='cpu',device_name='CPU')
    return result


def selection_fixture():
    rows = []
    menu = [('A','control',1.44),('A','tail-ema-average',1.4397),('A','last-raw-ema-average',1.4401),
            ('B','information-equal',1.4405),('B','information-plus8mib',1.4394),('C','combined',1.439)]
    for index,(phase,name,bpb) in enumerate(menu):
        digest = f.PARENT_SHA if index == 0 else str(index)*64
        rows.append(dict(phase=phase,name=name,checkpoint='runs/push151-fixture/'+name+'/checkpoint.pt',
            checkpoint_sha256=digest,validation_bpb_gpu_fp32=bpb,measured=measurement(bpb,checkpoint=digest),
            improvement_vs_parent=1.44-bpb, passes_validation_improvement_gate=(1.44-bpb >= .0005)))
    ranked = sorted((row for row in rows if row['passes_validation_improvement_gate']),
                    key=lambda row:row['validation_bpb_gpu_fp32'])
    return dict(status='validation_qualified_candidates_need_resource_audit',plan_sha256=f.PLAN_SHA,
        selection_split='validation',test_used=False,all_declared_A_B_C_accounted=True,minimum_improvement=.0005,
        selected_shared_ancestor_targets=f.LINEAGE,new_neural_gradient_targets=0,phase_c_status='completed',
        full_validation_passes=7,candidates=rows,validation_qualified_in_rank_order=ranked)


def timing_fixture():
    order = [(0,'baseline'),(0,'candidate'),(1,'candidate'),(1,'baseline'),(2,'baseline'),(2,'candidate')]
    rows = []
    for repeat,name in order:
        score = measurement(1.43 if name == 'candidate' else 2., 31.5 if name == 'candidate' else 7.)
        for key in ('split','precision','checkpoint_sha256','implementation_sha256','evaluator_sha256','device','test_used'):
            del score[key]
        rows.append(dict(repeat=repeat,name=name,peak_process_working_set_bytes=2**30,**score))
    return dict(test_used=False,split='validation',device='cpu',precision='fp32',threads=4,affinity_mask=15,
        candidate_sha256='a'*64,baseline_sha256=f.BASELINE_SHA,evaluator_sha256=f.SCORER_SHA,
        script_sha256=f.TIMING_SCRIPT_SHA,measurements=rows,cpu_time_ratio=4.5)


def comparison(atol=1e-5, rtol=1e-5, exact=False):
    return dict(elements=2048,maximum_absolute_difference=0.,tolerance_violation_elements=0,
        within_declared_tolerance=True,atol=atol,rtol=rtol,exact_equal=exact,
        zero_tolerance_denominator_violation=False,maximum_tolerance_ratio=0.)


def diagnostic_batches_fixture():
    batches=[]
    for name in ('random_seed127','supplied_train_first512'):
        rows=[]
        for length in (1,2,3,19,63,128,255):
            rows.append(dict(length=length,masked_attention_maximum_t_greater_than_zero=0.,
                effective_forbidden_attention_maximum=0.,first_position_gate_maximum=0.,
                same_shape_future_change=comparison(4e-6,1e-6),mkn_prefix=comparison(0.,0.,True),
                supplemental_fp32_log_prefix=comparison(),
                fp64_reference_log_prefix=comparison(1e-12,0.),
                fp64_reference_neural_pointer_prefix=comparison(1e-12,0.),
                supplemental_fp64_reference_log_prefix=comparison(1e-10,0.),
                supplemental_fp64_reference_neural_pointer_prefix=comparison(1e-10,0.),
                fp32_short_versus_fp64_reference=comparison(),
                fp32_short_versus_same_full_fp64_reference=comparison(1e-4,0.),
                production_fp32_log_prefix=comparison(5e-6,1e-6)))
        batches.append(dict(name=name,full_length=256,full_masked_attention_maximum_t_greater_than_zero=0.,
            full_effective_forbidden_attention_maximum=0.,full_first_position_gate_maximum=0.,
            old_audit_same_shape_future_128=comparison(4e-6,1e-6),
            old_audit_example_independence=comparison(4e-6,1e-6),
            old_audit_reset_exact=comparison(0.,0.,True),fp32_full_versus_fp64_reference=comparison(),prefixes=rows))
    return batches


class PrecisionTests(unittest.TestCase):
    def test_all_fourteen_prefixes_and_two_fixtures_required(self):
        batches=diagnostic_batches_fixture(); value=f.check_diagnostic_batches(batches)
        self.assertEqual(len(value['actual_shape_change_checks']),14)
        self.assertFalse(value['legacy_short_prefix_check_failed'])
        batches[1]['prefixes'].pop()
        with self.assertRaises(ValueError): f.check_diagnostic_batches(batches)

    def test_legacy_failure_retained_not_reclassified(self):
        batches=diagnostic_batches_fixture(); old=batches[1]['prefixes'][3]['production_fp32_log_prefix']
        old.update(maximum_absolute_difference=1.6e-5,tolerance_violation_elements=6,
                   within_declared_tolerance=False,maximum_tolerance_ratio=1.5)
        result=f.check_diagnostic_batches(batches)
        self.assertTrue(result['legacy_short_prefix_check_failed'])
        self.assertEqual(result['legacy_total_mismatched_elements'],6)

    def test_full_future_attention_and_first_position_gate_exactly_zero(self):
        for key in ('full_masked_attention_maximum_t_greater_than_zero',
                    'full_effective_forbidden_attention_maximum','full_first_position_gate_maximum'):
            batches=diagnostic_batches_fixture(); batches[0][key]=1e-20
            with self.assertRaises(ValueError): f.check_diagnostic_batches(batches)

    def test_prefix_attention_and_mkn_must_be_exact(self):
        batches=diagnostic_batches_fixture(); batches[0]['prefixes'][2]['effective_forbidden_attention_maximum']=1e-20
        with self.assertRaises(ValueError): f.check_diagnostic_batches(batches)
        batches=diagnostic_batches_fixture(); batches[0]['prefixes'][2]['mkn_prefix']['exact_equal']=False
        with self.assertRaises(ValueError): f.check_diagnostic_batches(batches)

    def test_fp64_error_fixed_bound_no_relaxation(self):
        for name in ('fp64_reference_log_prefix','fp64_reference_neural_pointer_prefix'):
            batches=diagnostic_batches_fixture(); value=batches[0]['prefixes'][0][name]
            value.update(maximum_absolute_difference=1.1e-10,tolerance_violation_elements=1,within_declared_tolerance=False)
            with self.assertRaises(ValueError): f.check_diagnostic_batches(batches)

    def test_production_must_match_same_full_fp64_reference(self):
        batches=diagnostic_batches_fixture(); value=batches[1]['prefixes'][4]['fp32_short_versus_same_full_fp64_reference']
        value.update(tolerance_violation_elements=1,within_declared_tolerance=False,maximum_tolerance_ratio=1.01)
        with self.assertRaises(ValueError): f.check_diagnostic_batches(batches)

    def test_same_shape_remains_stricter_and_reset_exact(self):
        batches=diagnostic_batches_fixture(); batches[0]['old_audit_same_shape_future_128']['atol']=1e-5
        with self.assertRaises(ValueError): f.check_diagnostic_batches(batches)
        batches=diagnostic_batches_fixture(); batches[0]['old_audit_reset_exact']['maximum_absolute_difference']=1e-20
        with self.assertRaises(ValueError): f.check_diagnostic_batches(batches)

    def test_nonfinite_or_inconsistent_diagnostic_refused(self):
        for key,value in [('maximum_absolute_difference',float('nan')),('within_declared_tolerance',False),
                          ('maximum_tolerance_ratio',float('inf')),('elements',0)]:
            batches=diagnostic_batches_fixture(); batches[0]['prefixes'][0]['supplemental_fp32_log_prefix'][key]=value
            with self.assertRaises(ValueError): f.check_diagnostic_batches(batches)


class SelectionTests(unittest.TestCase):
    def test_best_qualified_candidate_only(self):
        value = selection_fixture()
        self.assertEqual(f.ranked_candidate(value)['name'],'combined')

    def test_rank_override_or_omission_refused(self):
        for operation in ('reverse','omit','duplicate'):
            value = selection_fixture()
            if operation == 'reverse': value['validation_qualified_in_rank_order'].reverse()
            elif operation == 'omit': value['validation_qualified_in_rank_order'].pop(0)
            else: value['validation_qualified_in_rank_order'].append(value['validation_qualified_in_rank_order'][0])
            with self.subTest(operation=operation), self.assertRaises(ValueError): f.ranked_candidate(value)

    def test_missing_or_added_predictor_refused(self):
        for add in (False,True):
            value = selection_fixture()
            if add: value['candidates'].append(copy.deepcopy(value['candidates'][-1]))
            else: value['candidates'].pop()
            with self.assertRaises(ValueError): f.ranked_candidate(value)

    def test_test_use_wrong_plan_or_new_training_refused(self):
        for key,value in [('test_used',True),('plan_sha256','x'*64),('new_neural_gradient_targets',8192),
                          ('selected_shared_ancestor_targets',2*f.LINEAGE),('all_declared_A_B_C_accounted',False),
                          ('full_validation_passes',6),('minimum_improvement',0)]:
            record = selection_fixture(); record[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): f.ranked_candidate(record)

    def test_misrepresented_improvement_refused(self):
        record = selection_fixture(); record['candidates'][1]['passes_validation_improvement_gate'] = True
        with self.assertRaises(ValueError): f.ranked_candidate(record)

    def test_short_score_or_modified_scorer_refused(self):
        for key,value in [('targets',1),('utf8_bytes',1),('precision','bf16'),('device','cpu'),
                          ('evaluator_sha256','x'*64),('nll_nats',1.)]:
            record = selection_fixture(); record['candidates'][-1]['measured'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): f.ranked_candidate(record)


class TimingTests(unittest.TestCase):
    def test_complete_three_pairs(self):
        record = f.check_timing(timing_fixture(),{'checkpoint_sha256':'a'*64})
        self.assertEqual(record['ratio_of_medians'],4.5)

    def test_exact_limit_allowed(self):
        value = timing_fixture()
        for row in value['measurements']:
            if row['name'] == 'candidate': row['seconds'] = 35.
        value['cpu_time_ratio'] = 5.
        self.assertEqual(f.check_timing(value,{'checkpoint_sha256':'a'*64})['ratio_of_medians'],5.)

    def test_worst_single_ratio_not_extra_gate(self):
        value = timing_fixture(); value['measurements'][1]['seconds'] = 40.
        result = f.check_timing(value,{'checkpoint_sha256':'a'*64})
        self.assertGreater(result['max_candidate_divided_by_min_baseline'],5.)
        self.assertFalse(result['extra_conservative_gate_required'])

    def test_missing_reordered_or_failed_pair_refused(self):
        variants = []
        value=timing_fixture(); value['measurements'].pop(); variants.append(value)
        value=timing_fixture(); value['measurements'].reverse(); variants.append(value)
        value=timing_fixture(); value['cpu_time_ratio']=1.; variants.append(value)
        value=timing_fixture(); value['measurements'][1]['peak_process_working_set_bytes']=4*2**30+1; variants.append(value)
        value=timing_fixture(); value['candidate_sha256']='x'*64; variants.append(value)
        value=timing_fixture(); value['script_sha256']='x'*64; variants.append(value)
        for value in variants:
            with self.assertRaises(ValueError): f.check_timing(value,{'checkpoint_sha256':'a'*64})

    def test_nonfinite_durations_and_counts_refused(self):
        for key,wrong in [('seconds',float('nan')),('seconds',0.),('targets',1),('utf8_bytes',1)]:
            value = timing_fixture(); value['measurements'][1][key] = wrong
            with self.assertRaises(ValueError): f.check_timing(value,{'checkpoint_sha256':'a'*64})

    def test_raw_six_files_must_match_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); value=timing_fixture(); summary=root/'summary.json'; f.write_new(summary,value)
            for row in value['measurements']:
                name=row['name']; score={key:item for key,item in row.items() if key not in ('name','repeat','peak_process_working_set_bytes')}
                f.write_new(root/(name+'-'+str(row['repeat'])+'.json'),dict(split='validation',threads=4,
                    cpu_affinity_mask=15,checkpoint_sha256='a'*64 if name=='candidate' else f.BASELINE_SHA,
                    implementation_sha256=f.IMPLEMENTATION_SHA if name=='candidate' else f.BASELINE_MODEL_SHA,
                    evaluator_sha256=f.SCORER_SHA,repeats=[score],peak_process_working_set_bytes=2**30))
            result,manifest=f.verify_timing_files(root,summary,value,{'checkpoint_sha256':'a'*64},{'measurement':{'bpb':1.43}})
            self.assertEqual(len(manifest),7)
            altered=copy.deepcopy(value); altered['measurements'][1]['seconds']=32.
            with self.assertRaises(ValueError):
                f.verify_timing_files(root,summary,altered,{'checkpoint_sha256':'a'*64},{'measurement':{'bpb':1.43}})


class MetadataTests(unittest.TestCase):
    def test_no_corpus_read_even_when_missing(self):
        for function in (f.sha,f.read):
            with self.assertRaises(ValueError): function(Path('missing/WikiText_test.txt'))

    def test_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(f.child(directory,'runs/fixture.json'),(Path(directory)/'runs/fixture.json').resolve())
            for name in ('../test','/test','C:/test','runs/./test','runs//test',''):
                with self.assertRaises(ValueError): f.child(directory,name)

    def test_exclusive_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'record.json'; f.write_new(path,{'one':1}); old=f.sha(path)
            with self.assertRaises(FileExistsError): f.write_new(path,{'one':2})
            self.assertEqual(f.sha(path),old)

    def test_manifest_changed_source_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.json'; f.write_new(path,{'one':1})
            f.verify_manifest(directory,{'fixture.json':f.sha(path)})
            with self.assertRaises(ValueError): f.verify_manifest(directory,{'fixture.json':'x'*64})

    def test_unit_evidence_requires_current_complete_source_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'tests').mkdir(); (root/'runs').mkdir()
            source=root/'push151_finalize.py'; source.write_text('# synthetic finalizer',encoding='utf-8')
            test=root/'tests/test_push151_finalize.py'; test.write_text('# synthetic unit',encoding='utf-8')
            unit=dict(successful=True,tests_run=3,failures=[],errors=[],skipped=[],test_text_loaded=False,
                source_sha256={source.name:f.sha(source)},unit_source_sha256={test.name:f.sha(test)})
            path=root/f.UNIT; f.write_new(path,unit)
            self.assertEqual(f.verify_units(root,path)['executions'],3)
            (root/'push151_added.py').write_text('# untested',encoding='utf-8')
            with self.assertRaises(ValueError): f.verify_units(root,path)

    def test_freeze_verification_checks_full_record_and_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); selection=selection_fixture(); sp=root/'selection.json'; f.write_new(sp,selection)
            winner=f.ranked_candidate(selection); run=(root/winner['checkpoint']).parent; run.mkdir(parents=True)
            audit=run/'audit.json'; f.write_new(audit,{'assets':{}})
            timing=root/'timing.json'; f.write_new(timing,{'fixture':True})
            current=dict(checkpoint_sha256=winner['checkpoint_sha256'],test_bpb_at_freeze=None,
                         timing_record='timing.json',audit_record=f.relative(root,audit),evidence_sha256={'timing.json':f.sha(timing)},
                         work_evidence_sha256={'fixture.py':'a'*64})
            frozen=dict(current,frozen_at_utc='2026-01-01T00:00:00Z'); fp=run/'candidate_frozen.json'; f.write_new(fp,frozen)
            original_verify=f.verify_manifest
            def metadata_manifest(path, manifest):
                if manifest == {'fixture.py':'a'*64}: return
                return original_verify(path,manifest)
            with patch.object(f,'preflight',return_value=(fp,current)), patch.object(f,'verify_manifest',side_effect=metadata_manifest):
                self.assertEqual(f.verify_freeze(root,sp)['winner'],winner)
                current['test_bpb_at_freeze']=1.3
                with self.assertRaises(ValueError): f.verify_freeze(root,sp)


class FinalProtocolTests(unittest.TestCase):
    def test_official_complete_measurement(self):
        value=measurement(split='test',official=True)
        f.check_measurement(value,{'cpu_affinity_mask':15,'peak_process_working_set_bytes':2**30},'a'*64,f.IMPLEMENTATION_SHA)

    def test_changed_official_fields_refused(self):
        for key,value in [('split','validation'),('targets',428404),('utf8_bytes',1292012),('device','cuda:0'),
                          ('tokenizer_sha256','x'*64),('precision','bf16'),('seconds',float('inf'))]:
            measured=measurement(split='test',official=True); measured[key]=value
            with self.assertRaises(ValueError):
                f.check_measurement(measured,{'cpu_affinity_mask':15,'peak_process_working_set_bytes':2**30},'a'*64,f.IMPLEMENTATION_SHA)

    def test_ram_and_affinity_required(self):
        for resource in ({},{'cpu_affinity_mask':15},{'cpu_affinity_mask':3,'peak_process_working_set_bytes':2**30}):
            with self.assertRaises(ValueError): f.check_measurement(measurement(split='test',official=True),resource,'a'*64,f.IMPLEMENTATION_SHA)

    def test_summary_preserves_fluctuation_and_five_limit(self):
        rows=[dict(name='baseline-before',seconds=7.),dict(name='candidate',seconds=40.,bpb=1.43,peak_ram_gib=2.),
              dict(name='baseline-after',seconds=9.)]
        result=f.test_summary(rows,{'inference_assets_mib':60.})
        self.assertTrue(result['limits_pass']); self.assertEqual(result['cpu_ratio_to_baseline_mean'],5.)
        self.assertGreater(result['cpu_ratio_to_faster_baseline'],5.)
        self.assertEqual(result['measurements'],rows)

    def test_summary_failure_never_selects_another_candidate(self):
        rows=[dict(name='baseline-before',seconds=7.),dict(name='candidate',seconds=40.,bpb=1.6,peak_ram_gib=5.),
              dict(name='baseline-after',seconds=7.)]
        result=f.test_summary(rows,{'inference_assets_mib':60.})
        self.assertFalse(result['limits_pass']); self.assertEqual(result['test_bpb'],1.6)
        self.assertTrue(result['no_post_test_selection']); self.assertTrue(result['no_automatic_retry'])

    def test_existing_or_wrong_ledger_never_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); ledger=root/f.FINAL_TEST; ledger.mkdir(parents=True)
            with patch.object(f.subprocess,'run') as launch:
                with self.assertRaises(ValueError): f.run_test(root,root/'selection.json',ledger)
                with self.assertRaises(ValueError): f.run_test(root,root/'selection.json',root/'other')
                launch.assert_not_called()

    def test_process_failure_preserves_launch_and_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'runs').mkdir(); checkpoint=root/'candidate.pt'; checkpoint.write_bytes(b'fake')
            fp=root/'freeze.json'; f.write_new(fp,{'fixture':1}); sp=root/'selection.json'; f.write_new(sp,{'fixture':2})
            (root/'push151_finalize.py').write_text('# metadata fixture',encoding='utf-8')
            evidence=dict(winner={'checkpoint_sha256':'a'*64},checkpoint=checkpoint,freeze_path=fp,audit={})
            with patch.object(f,'verify_freeze',return_value=evidence), patch.object(f.subprocess,'run',
                    return_value=subprocess.CompletedProcess([],3,'partial stdout','fixture failure')) as launch:
                with self.assertRaises(ValueError): f.run_test(root,sp,root/f.FINAL_TEST)
                self.assertEqual(launch.call_count,1)
                failure=f.read(root/f.FINAL_TEST/'failure.json')
                self.assertEqual(failure['launches_started'],['baseline-before'])
                self.assertTrue(failure['partial_test_exposure_must_be_disclosed'])
                self.assertEqual(f.read(root/f.FINAL_TEST/'baseline-before.process.json')['stdout'],'partial stdout')
                with self.assertRaises(ValueError): f.run_test(root,sp,root/f.FINAL_TEST)
                self.assertEqual(launch.call_count,1)

    def test_keyboard_interrupt_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'runs').mkdir(); fp=root/'freeze.json'; sp=root/'selection.json'
            f.write_new(fp,{'fixture':1}); f.write_new(sp,{'fixture':2})
            (root/'push151_finalize.py').write_text('# metadata fixture',encoding='utf-8')
            evidence=dict(winner={'checkpoint_sha256':'a'*64},checkpoint=root/'fake.pt',freeze_path=fp,audit={})
            with patch.object(f,'verify_freeze',return_value=evidence), patch.object(f.subprocess,'run',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt): f.run_test(root,sp,root/f.FINAL_TEST)
            failure=f.read(root/f.FINAL_TEST/'failure.json')
            self.assertEqual(failure['failure_type'],'KeyboardInterrupt')
            self.assertTrue((root/f.FINAL_TEST/'baseline-before.started.json').exists())

    def test_success_executes_exactly_three_fixed_passes_and_binds_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'runs').mkdir(); checkpoint=root/'candidate.pt'; checkpoint.write_bytes(b'fake')
            fp=root/'freeze.json'; f.write_new(fp,{'fixture':1}); sp=root/'selection.json'; f.write_new(sp,{'fixture':2})
            (root/'push151_finalize.py').write_text('# metadata fixture',encoding='utf-8')
            evidence=dict(winner={'checkpoint_sha256':'a'*64},checkpoint=checkpoint,freeze_path=fp,audit={'inference_assets_mib':50.})
            names=[]
            def synthetic_process(command,**kwargs):
                output=Path(command[command.index('--output')+1]); name=output.stem; names.append(name)
                self.assertEqual(command[1],'-B'); self.assertEqual(command[2],str(root/'official_resource_run.py'))
                self.assertEqual(command[command.index('--split')+1],'test')
                candidate=name=='candidate'
                measured=measurement(1.43 if candidate else 2.,32. if candidate else 8.,
                    'a'*64 if candidate else f.BASELINE_SHA,split='test',official=True)
                measured['implementation_sha256']=f.IMPLEMENTATION_SHA if candidate else f.BASELINE_MODEL_SHA
                f.write_new(output,measured)
                f.write_new(output.with_suffix('.resources.json'),dict(cpu_affinity_mask=15,
                    peak_process_working_set_bytes=2**30,evaluator_output=str(output)))
                return subprocess.CompletedProcess(command,0,'synthetic only','')
            with patch.object(f,'verify_freeze',return_value=evidence), patch.object(f.subprocess,'run',side_effect=synthetic_process), \
                    patch('builtins.print'):
                report=f.run_test(root,sp,root/f.FINAL_TEST)
            self.assertEqual(names,['baseline-before','candidate','baseline-after'])
            self.assertEqual(report['candidate_scoring_process_launches'],1)
            self.assertEqual(report['cpu_ratio_to_baseline_mean'],4.)
            for row in report['measurements']:
                self.assertEqual(row['output_sha256'],f.sha(root/f.FINAL_TEST/row['output']))
            self.assertEqual(f.read(root/f.FINAL_TEST/'summary.json'),report)


if __name__ == '__main__':
    unittest.main()
