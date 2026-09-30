"""Dataset-free finalization fixtures: no models, corpus, GPU or real scorer."""
import copy
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import push152_finalize as f


def measurement(bpb=1.43, seconds=31.5, digest='a'*64, split='validation'):
    targets,size = (376599,1148007) if split == 'validation' else (428405,1292013)
    return dict(bpb=bpb,nll_nats=bpb*math.log(2)*size,token_ppl=20.,seconds=seconds,targets=targets,utf8_bytes=size,
        protocol='7506-mp1-wt2-v2',split=split,precision='fp32',checkpoint_sha256=digest,
        implementation_sha256=f.IMPLEMENTATION_SHA,evaluator_sha256=f.SCORER_SHA,
        tokenizer_sha256=f.TOKENIZER_SHA,device='cpu',device_name='CPU',test_used=False)


def timing_fixture():
    order = [(0,'baseline'),(0,'candidate'),(1,'candidate'),(1,'baseline'),(2,'baseline'),(2,'candidate')]
    rows = []
    for repeat,name in order:
        score = measurement(1.43 if name == 'candidate' else 2.,31.5 if name == 'candidate' else 7.)
        score = {key:score[key] for key in ('bpb','nll_nats','token_ppl','seconds','targets','utf8_bytes')}
        rows.append(dict(name=name,repeat=repeat,peak_process_working_set_bytes=2**30,**score))
    return dict(test_used=False,split='validation',device='cpu',precision='fp32',threads=4,affinity_mask=15,
        candidate_sha256='a'*64,baseline_sha256=f.BASELINE_SHA,evaluator_sha256=f.SCORER_SHA,
        script_sha256=f.TIMING_SCRIPT_SHA,measurements=rows,cpu_time_ratio=4.5,
        peak_process_working_set_bytes=2**30)


def write_timing(root, rank=0, record=None):
    record = timing_fixture() if record is None else record
    directory = f.resource_dir(root,rank)/'timing'; directory.mkdir(parents=True)
    f.write_new(directory/'summary.json',record)
    for row in record['measurements']:
        candidate = row['name'] == 'candidate'
        score = {key:value for key,value in row.items() if key not in ('name','repeat','peak_process_working_set_bytes')}
        f.write_new(directory/(row['name']+'-'+str(row['repeat'])+'.json'),dict(split='validation',threads=4,
            cpu_affinity_mask=15,checkpoint_sha256='a'*64 if candidate else f.BASELINE_SHA,
            implementation_sha256=f.IMPLEMENTATION_SHA if candidate else f.BASELINE_MODEL_SHA,
            evaluator_sha256=f.SCORER_SHA,repeats=[score],peak_process_working_set_bytes=row['peak_process_working_set_bytes']))
    return record


def selection_fixture():
    winner = dict(name='control',phase='B',step=20000,weights='ema',parameters=5597331,
        checkpoint='runs/push152-control/fixture.pt',checkpoint_sha256='a'*64,
        selected_lineage_targets=f.LINEAGE+20000*8192,validation_bpb_gpu_fp32=1.43)
    return dict(round='push152',status='validation_qualified_candidates_need_resource_audit',plan_sha256=f.PLAN_SHA,
        selection_split='validation',test_used=False,all_declared_runs_accounted=True,
        all_completed_runs_have_exact_predeclared_validation_menu=True,minimum_improvement=.001,
        maximum_resource_candidates=3,no_new_calibration_or_averaging=True,script_sha256='s'*64,
        created_utc='2026-01-01T00:00:00+00:00',resource_audit_candidates=[winner],
        validation_qualified_in_rank_order=[winner],source_sha256={'push152_select.py':'s'*64},
        evidence_sha256={f.PLAN:f.PLAN_SHA},parent=dict(sha256=f.PARENT_SHA,validation_bpb_gpu_fp32=1.439826631072287))


def comparison(atol=1e-5,rtol=1e-5,exact=False):
    return dict(elements=2048,maximum_absolute_difference=0.,tolerance_violation_elements=0,
        within_declared_tolerance=True,atol=atol,rtol=rtol,exact_equal=exact,
        zero_tolerance_denominator_violation=False,maximum_tolerance_ratio=0.)


def diagnostic_fixture():
    batches=[]
    for name in ('random_seed127','supplied_train_first512'):
        prefixes=[]
        for length in (1,2,3,19,63,128,255):
            prefixes.append(dict(length=length,masked_attention_maximum_t_greater_than_zero=0.,
                effective_forbidden_attention_maximum=0.,first_position_gate_maximum=0.,
                same_shape_future_change=comparison(4e-6,1e-6),mkn_prefix=comparison(0.,0.,True),
                supplemental_fp32_log_prefix=comparison(),fp64_reference_log_prefix=comparison(1e-12,0.),
                fp64_reference_neural_pointer_prefix=comparison(1e-12,0.),
                supplemental_fp64_reference_log_prefix=comparison(1e-10,0.),
                supplemental_fp64_reference_neural_pointer_prefix=comparison(1e-10,0.),
                fp32_short_versus_fp64_reference=comparison(),
                fp32_short_versus_same_full_fp64_reference=comparison(1e-4,0.),
                production_fp32_log_prefix=comparison(5e-6,1e-6)))
        batches.append(dict(name=name,full_length=256,full_masked_attention_maximum_t_greater_than_zero=0.,
            full_effective_forbidden_attention_maximum=0.,full_first_position_gate_maximum=0.,
            old_audit_same_shape_future_128=comparison(4e-6,1e-6),old_audit_example_independence=comparison(4e-6,1e-6),
            old_audit_reset_exact=comparison(0.,0.,True),fp32_full_versus_fp64_reference=comparison(),prefixes=prefixes))
    return dict(status='completed',checkpoint_sha256='a'*64,implementation='push150_expand',
        implementation_sha256=f.IMPLEMENTATION_SHA,plan_sha256=f.PLAN_SHA,numerical_audit_plan_sha256=f.PLAN_SHA,
        selection_sha256='s'*64,diagnostic_script_sha256=f.DIAGNOSTIC_SCRIPT_SHA,supplemental_audit_script_sha256='b'*64,
        test_text_loaded_or_hashed=False,training_performed=False,official_scoring_performed=False,
        checkpoint_or_inference_sources_modified=False,validation_text_loaded=True,validation_text_used_by_diagnostic=False,
        requested_lengths=[1,2,3,19,63,128,255],criteria={'shape_tolerance_disclosure':'supplemental not teacher approval'},
        runtime=dict(device='cpu',precision='fp32',threads=4,affinity_mask=15),
        fp64_reference={key:True for key in ('neural_pointer_float64','dropout_disabled',
            'original_fp32_rope_distance_buffers_cast_to_float64','count_table_values_original_fp32_cast_to_float64',
            'inference_implementation_unchanged','no_model_double_shortcut')},
        train_prefix=dict(split='train',used_tokens=512),batches=batches,source_sha256={'fixture.py':'f'*64},
        summary={'predeclared_supplemental_diagnostic_pass':True})


class PrecisionIntegrationTests(unittest.TestCase):
    def call(self, diagnostic, audit_change=None):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'runs').mkdir()
            f.write_new(root/f.PLAN,{'audit':diagnostic['criteria']})
            path=f.resource_dir(root,0); path.mkdir()
            checks=f.check_diagnostic_batches(diagnostic['batches'])
            diagnostic=dict(diagnostic,checks=checks)
            f.write_new(path/'audit.diagnostic.json',diagnostic)
            precision=dict(**checks,plan_sha256=f.PLAN_SHA,plan_file=f.PLAN,diagnostic_file='audit.diagnostic.json',
                diagnostic_sha256=f.sha(path/'audit.diagnostic.json'),diagnostic_script='push150_numerical_diagnostic.py',
                diagnostic_script_sha256=f.DIAGNOSTIC_SCRIPT_SHA,supplemental_audit_script='push152_audit.py',
                supplemental_audit_script_sha256='b'*64,legacy_failure_reclassified_as_pass=False,
                teacher_short_prefix_tolerance_not_specified=True,shape_change_tolerance=dict(atol=1e-5,rtol=1e-5),
                same_shape_tolerance_unchanged=dict(atol=4e-6,rtol=1e-6),reset_atol=0.,reset_rtol=0.,
                neither_predictor_nor_official_scorer_modified=True,criterion_note='supplemental not teacher approval')
            audit=dict(checkpoint_sha256='a'*64,selection_sha256='s'*64,audit_script_sha256='b'*64,
                precision_diagnostic=precision,source_sha256={'fixture.py':'f'*64})
            if audit_change: audit_change(audit)
            return f.verify_precision(root,0,audit,{'push152_audit.py':'b'*64})

    def test_all14prefixes_and_fixed_identity_bind(self):
        self.assertEqual(len(self.call(diagnostic_fixture())),1)

    def test_legacy_failures_remain_disclosed_without_reclassification(self):
        diagnostic=diagnostic_fixture()
        diagnostic['batches'][0]['prefixes'][3]['production_fp32_log_prefix'].update(
            maximum_absolute_difference=1.8e-5,tolerance_violation_elements=7,within_declared_tolerance=False,
            maximum_tolerance_ratio=1.5)
        self.assertEqual(len(self.call(diagnostic)),1)
        with self.assertRaises(ValueError): self.call(diagnostic,lambda audit:audit['precision_diagnostic'].update(
            legacy_failure_reclassified_as_pass=True))

    def test_changed_plan_source_or_test_access_refused(self):
        for key,wrong in [('plan_sha256','x'*64),('test_text_loaded_or_hashed',True),
                          ('supplemental_audit_script_sha256','x'*64),('validation_text_used_by_diagnostic',True)]:
            diagnostic=diagnostic_fixture(); diagnostic[key]=wrong
            with self.subTest(key=key),self.assertRaises(ValueError): self.call(diagnostic)

    def test_false_full_double_reference_or_missing_fixture_refused(self):
        diagnostic=diagnostic_fixture(); diagnostic['fp64_reference']['no_model_double_shortcut']=False
        with self.assertRaises(ValueError): self.call(diagnostic)
        diagnostic=diagnostic_fixture(); diagnostic['batches'].pop()
        with self.assertRaises(ValueError): self.call(diagnostic)

    def test_future_mass_or_reference_error_cannot_be_hidden_in_summary(self):
        diagnostic=diagnostic_fixture(); diagnostic['batches'][0]['full_effective_forbidden_attention_maximum']=1e-20
        with self.assertRaises(ValueError): self.call(diagnostic)
        diagnostic=diagnostic_fixture()
        diagnostic['batches'][1]['prefixes'][0]['fp64_reference_log_prefix']['maximum_absolute_difference']=1.1e-10
        with self.assertRaises(ValueError): self.call(diagnostic)


class SelectionTests(unittest.TestCase):
    def call(self, record, fresh=None, rank=0):
        root = Path('synthetic-root').resolve()
        module = SimpleNamespace(__file__=root/'push152_select.py',collect=lambda _:copy.deepcopy(record if fresh is None else fresh))
        digests = {'push151_finalize.py':f.LEGACY_FINALIZER_SHA,'push152_plan.json':f.PLAN_SHA,
                   'push152_select.py':'s'*64,'fixture.pt':'a'*64,'push152_selection.json':'z'*64}
        with patch.object(f,'read',return_value=record),patch.object(f,'sha',side_effect=lambda path:digests[Path(path).name]), \
             patch.object(f,'verify_manifest'),patch.object(f.importlib,'import_module',return_value=module):
            return f.verify_selection(root,root/f.SELECTION,rank)

    def test_selection_recomputed_allowing_only_new_timestamp(self):
        value = selection_fixture(); fresh = copy.deepcopy(value); fresh['created_utc']='2026-01-02T00:00:00+00:00'
        self.assertEqual(self.call(value,fresh)[1]['selected_lineage_targets'],802816000)

    def test_recomputed_cost_or_menu_change_rejected(self):
        value = selection_fixture(); fresh = copy.deepcopy(value); fresh['new_cost']=8192
        with self.assertRaises(ValueError): self.call(value,fresh)

    def test_all_five_runs_and_no_test_or_extra_grid_required(self):
        for key,wrong in [('all_declared_runs_accounted',False),('all_completed_runs_have_exact_predeclared_validation_menu',False),
                          ('test_used',True),('minimum_improvement',.0005),('maximum_resource_candidates',4),
                          ('no_new_calibration_or_averaging',False)]:
            value=selection_fixture(); value[key]=wrong
            with self.subTest(key=key),self.assertRaises(ValueError): self.call(value)

    def test_lineage_and_parent_must_match_real_selected_step(self):
        for delta in (-8192,8192):
            value=selection_fixture(); value['resource_audit_candidates'][0]['selected_lineage_targets']+=delta
            with self.assertRaises(ValueError): self.call(value)
        value=selection_fixture(); value['parent']['sha256']='x'*64
        with self.assertRaises(ValueError): self.call(value)

    def test_at_least_point001_improvement(self):
        value=selection_fixture(); value['resource_audit_candidates'][0]['validation_bpb_gpu_fp32']=1.439
        with self.assertRaises(ValueError): self.call(value)

    def test_only_first_three_ranks_and_no_arbitrary_rank(self):
        for rank in (-1,3,True,1):
            with self.subTest(rank=rank),self.assertRaises(ValueError): self.call(selection_fixture(),rank=rank)


class TimingTests(unittest.TestCase):
    def test_all_six_fresh_pairs_and_exact_five_allowed(self):
        value=timing_fixture()
        for row in value['measurements']:
            if row['name']=='candidate': row['seconds']=35.
        value['cpu_time_ratio']=5.
        result=f.timing_details(value,{'checkpoint_sha256':'a'*64})
        self.assertTrue(result['cpu_limit_pass']); self.assertEqual(result['ratio_of_medians'],5.)

    def test_single_worst_ratio_is_reported_not_extra_gate(self):
        value=timing_fixture(); value['measurements'][1]['seconds']=40.
        result=f.timing_details(value,{'checkpoint_sha256':'a'*64})
        self.assertTrue(result['cpu_limit_pass']); self.assertGreater(result['max_candidate_divided_by_min_baseline'],5.)
        self.assertFalse(result['extra_conservative_gate_required'])

    def test_failed_ratio_is_retained_for_explicit_resource_disposition(self):
        value=timing_fixture()
        for row in value['measurements']:
            if row['name']=='candidate': row['seconds']=36.
        value['cpu_time_ratio']=36/7
        self.assertFalse(f.timing_details(value,{'checkpoint_sha256':'a'*64})['cpu_limit_pass'])

    def test_complete_order_identity_positive_finite_counts_required(self):
        variations=[]
        value=timing_fixture(); value['measurements'].pop(); variations.append(value)
        value=timing_fixture(); value['measurements'].reverse(); variations.append(value)
        for key,wrong in [('seconds',float('nan')),('targets',1),('utf8_bytes',1),('peak_process_working_set_bytes',0)]:
            value=timing_fixture(); value['measurements'][1][key]=wrong; variations.append(value)
        value=timing_fixture(); value['cpu_time_ratio']=1.; variations.append(value)
        value=timing_fixture(); value['candidate_sha256']='x'*64; variations.append(value)
        for value in variations:
            with self.assertRaises(ValueError): f.timing_details(value,{'checkpoint_sha256':'a'*64})

    def test_candidate_ram_does_not_use_baseline_peak_as_extra_gate(self):
        value=timing_fixture(); value['measurements'][0]['peak_process_working_set_bytes']=5*2**30
        value['peak_process_working_set_bytes']=5*2**30
        self.assertTrue(f.timing_details(value,{'checkpoint_sha256':'a'*64})['ram_limit_pass'])
        value['measurements'][1]['peak_process_working_set_bytes']=4*2**30+1
        self.assertFalse(f.timing_details(value,{'checkpoint_sha256':'a'*64})['ram_limit_pass'])

    def test_raw_six_files_and_audit_score_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); write_timing(root)
            result,manifest=f.verify_timing_files(root,0,{'checkpoint_sha256':'a'*64},{'measurement':{'bpb':1.43}})
            self.assertTrue(result['cpu_limit_pass']); self.assertEqual(len(manifest),7)
            with self.assertRaises(ValueError):
                f.verify_timing_files(root,0,{'checkpoint_sha256':'a'*64},{'measurement':{'bpb':1.4301}})

    def test_raw_implementation_or_resource_overage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); value=timing_fixture()
            for row in value['measurements']:
                if row['name']=='candidate': row['seconds']=36.
            value['cpu_time_ratio']=36/7; write_timing(root,record=value)
            with self.assertRaises(ValueError):
                f.verify_timing_files(root,0,{'checkpoint_sha256':'a'*64},{'measurement':{'bpb':1.43}})
            result,_=f.verify_timing_files(root,0,{'checkpoint_sha256':'a'*64},{'measurement':{'bpb':1.43}},False)
            self.assertFalse(result['cpu_limit_pass'])


class PriorRankTests(unittest.TestCase):
    def test_absent_or_running_evidence_cannot_be_interpreted_as_failure(self):
        fake=SimpleNamespace(check_prior_dispositions=lambda *_:(_ for _ in ()).throw(ValueError('Missing earlier audit')))
        with patch.object(f,'load_auditor',return_value=fake),self.assertRaises(ValueError):
            f.verify_prior_ranks(Path('synthetic'),{'resource_audit_candidates':[{}]},1,{})

    def test_completed_prior_candidate_cannot_be_skipped_if_passed(self):
        outcome=dict(rank=0,stage='timing_resources',failed=True,evidence='runs/r0/summary.json')
        fake=SimpleNamespace(check_prior_dispositions=lambda *_:[outcome])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'runs').mkdir(); f.write_new(root/f.SELECTION,{})
            with patch.object(f,'load_auditor',return_value=fake), \
                 patch.object(f,'verify_audit',return_value=({'asset_limit_pass':True,'ram_limit_pass':True},{})), \
                 patch.object(f,'verify_timing_files',return_value=({'cpu_limit_pass':True,'ram_limit_pass':True},{})), \
                 self.assertRaises(ValueError):
                f.verify_prior_ranks(root,{'resource_audit_candidates':[{}]},1,{})

    def test_completed_measured_failure_can_advance_and_is_preserved(self):
        outcome=dict(rank=0,stage='timing_resources',failed=True,evidence='runs/r0/summary.json')
        fake=SimpleNamespace(check_prior_dispositions=lambda *_:[outcome])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'runs').mkdir(); f.write_new(root/f.SELECTION,{})
            with patch.object(f,'load_auditor',return_value=fake), \
                 patch.object(f,'verify_audit',return_value=({'asset_limit_pass':True,'ram_limit_pass':True},{})), \
                 patch.object(f,'verify_timing_files',return_value=({'cpu_limit_pass':False,'ram_limit_pass':True},{})):
                self.assertEqual(f.verify_prior_ranks(root,{'resource_audit_candidates':[{}]},1,{})[0],[outcome])


class UnitAndMetadataTests(unittest.TestCase):
    def unit_receipts(self, work):
        root=work/'MP1_student_starter/code'; (root/'tests').mkdir(parents=True); (root/'runs').mkdir()
        for name in ('push152_train.py','push152_select.py','push152_finalize.py'):
            (root/name).write_text('# '+name,encoding='utf-8')
        for name in ('test_push152_train.py','test_push152_select.py','test_push152_finalize.py'):
            (root/'tests'/name).write_text('# '+name,encoding='utf-8')
        for name in ('push152_audit.py','push151_audit.py','push150_numerical_diagnostic.py',
                     'push152_audit_unit_check.py','test_push152_audit.py'):
            (work/name).write_text('# '+name,encoding='utf-8')
        base=dict(successful=True,tests_run=5,failures=[],errors=[],skipped=[],test_text_loaded=False)
        prior={'push152_train.py':f.sha(root/'push152_train.py')}
        prior_tests={'test_push152_train.py':f.sha(root/'tests/test_push152_train.py')}
        f.write_new(root/f.PRETRAINING_UNIT,dict(base,source_sha256=prior,unit_source_sha256=prior_tests))
        final=dict(base,source_sha256={path.name:f.sha(path) for path in root.glob('push152*.py')},
                   unit_source_sha256={path.name:f.sha(path) for path in (root/'tests').glob('test_push152*.py')})
        f.write_new(root/f.UNIT,final)
        audit=dict(base,source_sha256={name:f.sha(work/name) for name in
            ('push152_audit.py','push151_audit.py','push150_numerical_diagnostic.py','push152_audit_unit_check.py')},
            unit_source_sha256={'test_push152_audit.py':f.sha(work/'test_push152_audit.py')})
        f.write_new(root/f.AUDIT_UNIT,audit)
        return root,prior,prior_tests

    def test_pretraining_final_and_separate_work_audit_receipts_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory); root,prior,tests=self.unit_receipts(work)
            with patch.object(f,'PRETRAINING_SOURCES',prior),patch.object(f,'PRETRAINING_TESTS',tests), \
                 patch.object(f,'LEGACY_AUDIT_SHA',f.sha(work/'push151_audit.py')), \
                 patch.object(f,'DIAGNOSTIC_SCRIPT_SHA',f.sha(work/'push150_numerical_diagnostic.py')):
                units,sources=f.verify_units(root,root/f.UNIT,root/f.AUDIT_UNIT)
                self.assertEqual(units['final_executions'],10)
                self.assertEqual(len(sources),5)
                (root/'push152_new_untested.py').write_text('# not in final suite',encoding='utf-8')
                with self.assertRaises(ValueError): f.verify_units(root,root/f.UNIT,root/f.AUDIT_UNIT)

    def test_work_auditor_source_mutation_invalidates_its_own_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory); root,prior,tests=self.unit_receipts(work)
            with patch.object(f,'PRETRAINING_SOURCES',prior),patch.object(f,'PRETRAINING_TESTS',tests), \
                 patch.object(f,'LEGACY_AUDIT_SHA',f.sha(work/'push151_audit.py')), \
                 patch.object(f,'DIAGNOSTIC_SCRIPT_SHA',f.sha(work/'push150_numerical_diagnostic.py')):
                (work/'push152_audit.py').write_text('# changed after tests',encoding='utf-8')
                with self.assertRaises(ValueError): f.verify_units(root,root/f.UNIT,root/f.AUDIT_UNIT)

    def test_dataset_free_units_need_positive_count_and_no_failures(self):
        value=dict(successful=True,tests_run=1,failures=[],errors=[],skipped=[],test_text_loaded=False)
        f.check_unit_result(value)
        for key,wrong in [('tests_run',0),('tests_run',True),('errors',['fixture']),('skipped',['fixture']),('test_text_loaded',True)]:
            changed=dict(value); changed[key]=wrong
            with self.assertRaises(ValueError): f.check_unit_result(changed)

    def test_no_corpus_even_hash(self):
        for function in (f.read,f.sha):
            with self.assertRaises(ValueError): function('missing/wikitext_test.txt')

    def test_paths_are_scoped_and_records_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in ('../outside','C:/outside','/outside','runs/./thing',''):
                with self.assertRaises(ValueError): f.child(directory,value)
            path=Path(directory)/'fixture.json'; f.write_new(path,{'one':1}); old=f.sha(path)
            with self.assertRaises(FileExistsError): f.write_new(path,{'two':2})
            self.assertEqual(old,f.sha(path))

    def test_global_freeze_excludes_other_ranks_and_test_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'runs').mkdir(); f.write_new(root/f.FREEZE,{'rank':0})
            with patch.object(f,'preflight') as preflight,self.assertRaises(ValueError):
                f.create_freeze(root,root/f.SELECTION,1,root/f.UNIT,root/f.AUDIT_UNIT)
            preflight.assert_not_called()

    def test_freeze_verify_rechecks_every_field_and_accepts_extra_preserved_failure_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'runs').mkdir(); cp=root/'candidate.pt'; cp.write_bytes(b'synthetic')
            audit=root/'audit.json'; f.write_new(audit,{'fixture':True})
            current=dict(validation_rank=0,checkpoint='candidate.pt',selected_validation_record={'checkpoint_sha256':'a'*64},
                audit_record='audit.json',test_bpb_at_freeze=None,evidence_sha256={'audit.json':f.sha(audit)},work_evidence_sha256={})
            frozen=dict(current,frozen_at_utc='2026-01-01T00:00:00Z')
            f.write_new(root/f.FREEZE,frozen)
            real_verify=f.verify_manifest
            def manifest(path,values):
                if values: real_verify(path,values)
            with patch.object(f,'preflight',return_value=(root/f.FREEZE,current)),patch.object(f,'verify_manifest',side_effect=manifest):
                self.assertEqual(f.verify_freeze(root,root/f.SELECTION)['checkpoint'],cp)
                current['test_bpb_at_freeze']=1.4
                with self.assertRaises(ValueError): f.verify_freeze(root,root/f.SELECTION)


class FinalTestProtocolTests(unittest.TestCase):
    def fixtures(self, root):
        (root/'runs').mkdir(); checkpoint=root/'candidate.pt'; checkpoint.write_bytes(b'synthetic')
        freeze=root/f.FREEZE; f.write_new(freeze,{'fixture':1})
        selection=root/f.SELECTION; f.write_new(selection,{'fixture':2})
        (root/'push152_finalize.py').write_text('# synthetic metadata',encoding='utf-8')
        return selection,dict(winner={'checkpoint_sha256':'a'*64},checkpoint=checkpoint,freeze_path=freeze,
                              audit={'inference_assets_mib':55.})

    def test_exactly_three_commands_candidate_only_once_and_hashes_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); selection,evidence=self.fixtures(root); names=[]
            def process(command,**kwargs):
                self.assertEqual(command[1],'-B'); self.assertEqual(Path(command[2]),root/'official_resource_run.py')
                self.assertEqual(command[command.index('--split')+1],'test')
                output=Path(command[command.index('--output')+1]); names.append(output.stem)
                candidate=output.stem=='candidate'
                measured=measurement(1.43 if candidate else 2.,32. if candidate else 8.,
                                     'a'*64 if candidate else f.BASELINE_SHA,'test')
                measured['implementation_sha256']=f.IMPLEMENTATION_SHA if candidate else f.BASELINE_MODEL_SHA
                f.write_new(output,measured)
                f.write_new(output.with_suffix('.resources.json'),dict(cpu_affinity_mask=15,
                    peak_process_working_set_bytes=2**30,evaluator_output=str(output)))
                return subprocess.CompletedProcess(command,0,'synthetic','')
            with patch.object(f,'verify_freeze',return_value=evidence),patch.object(f.subprocess,'run',side_effect=process),patch('builtins.print'):
                result=f.run_test(root,selection,root/f.FINAL_TEST)
            self.assertEqual(names,['baseline-before','candidate','baseline-after'])
            self.assertEqual(result['candidate_scoring_process_launches'],1)
            self.assertEqual(result['cpu_ratio_to_baseline_mean'],4.)
            for row in result['measurements']:
                self.assertEqual(row['output_sha256'],f.sha(root/f.FINAL_TEST/row['output']))

    def test_process_failure_or_interrupt_is_terminal_and_preserves_launch(self):
        for interrupted in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory); selection,evidence=self.fixtures(root)
                with patch.object(f,'verify_freeze',return_value=evidence),patch.object(f.subprocess,'run',
                        side_effect=KeyboardInterrupt if interrupted else None,
                        return_value=subprocess.CompletedProcess([],9,'partial stdout','failed')) as launch:
                    with self.assertRaises(KeyboardInterrupt if interrupted else ValueError):
                        f.run_test(root,selection,root/f.FINAL_TEST)
                    self.assertEqual(launch.call_count,1)
                    failure=f.read(root/f.FINAL_TEST/'failure.json')
                    self.assertEqual(failure['launches_started'],['baseline-before'])
                    self.assertTrue(failure['partial_test_exposure_must_be_disclosed'])
                    with self.assertRaises(ValueError): f.run_test(root,selection,root/f.FINAL_TEST)
                    self.assertEqual(launch.call_count,1)

    def test_wrong_ledger_or_existing_attempt_never_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/f.FINAL_TEST).mkdir(parents=True)
            with patch.object(f.subprocess,'run') as launch:
                for ledger in (root/f.FINAL_TEST,root/'alternative'):
                    with self.assertRaises(ValueError): f.run_test(root,root/f.SELECTION,ledger)
                launch.assert_not_called()

    def test_full_official_counts_protocol_and_affinity_required(self):
        for key,wrong in [('targets',1),('utf8_bytes',1),('split','validation'),('device','cuda:0'),
                          ('precision','bf16'),('tokenizer_sha256','x'*64),('seconds',float('inf'))]:
            row=measurement(split='test'); row[key]=wrong
            with self.assertRaises(ValueError):
                f.check_measurement(row,{'cpu_affinity_mask':15,'peak_process_working_set_bytes':2**30},'a'*64,f.IMPLEMENTATION_SHA)

    def test_failed_test_resources_never_trigger_reselection(self):
        rows=[dict(name='baseline-before',seconds=7.),dict(name='candidate',seconds=50.,bpb=1.6,peak_ram_gib=5.),
              dict(name='baseline-after',seconds=9.)]
        result=f.test_summary(rows,{'inference_assets_mib':55.})
        self.assertFalse(result['limits_pass']); self.assertEqual(result['test_bpb'],1.6)
        self.assertTrue(result['no_post_test_selection']); self.assertTrue(result['no_automatic_retry'])


if __name__ == '__main__':
    unittest.main()
