"""Metadata-only resource-gated push152 freeze and one-shot official test.

Only ``test`` executes the unchanged official scoring wrapper. No model is
imported and no corpus is read or hashed by this module. Resource candidates
are visited in the predeclared order; missing evidence is never a failure.
"""
import argparse
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from push151_finalize import (require, number, sha, read, write_new, child, relative,
    verify_manifest, check_score, check_measurement, test_summary,
    check_diagnostic_batches, CONTRACT_KEYS, FIXED_KEYS, BASELINE_SHA,
    BASELINE_MODEL_SHA, SCORER_SHA, TOKENIZER_SHA, IMPLEMENTATION_SHA,
    OFFICIAL_WRAPPER_SHA, TIMING_SCRIPT_SHA, MEASURE_SCRIPT_SHA,
    DIAGNOSTIC_SCRIPT_SHA)

ROOT = Path(__file__).resolve().parent
PLAN = 'runs/push152_plan.json'
PLAN_SHA = 'd15cd6276ace1894a37db7c005453c61acd2e5acb7e934cb6790f3860aa06b51'
PARENT_SHA = 'c72cffff6b4152492518473f46a274b53272f59dacb80072cd9e1497d2e0bdf5'
PARENT_AUDIT = 'runs/push151-combination/audit.json'
PARENT_AUDIT_SHA = '5a4731f6b6174901aa27b4f0e342564ad2b2a22dc5e7a6efad6362c891cb41e1'
LEGACY_FINALIZER_SHA = '4536fc7d83cbcb2dd060748b3ff8a7bce3e57548ca80ccde20f5350b4df87e4b'
LEGACY_AUDIT_SHA = '5c348e35298b1cc1b4ebb43a97747c8709cd274048d2ae4368326ffbf2a1eb8a'
LINEAGE = 638976000
SELECTION = 'runs/push152_selection.json'
PRETRAINING_UNIT = 'runs/push152_unit_tests_pretraining.json'
UNIT = 'runs/push152_unit_tests_final.json'
AUDIT_UNIT = 'runs/push152_audit_unit_tests_final.json'
FREEZE = 'runs/push152_candidate_frozen.json'
FINAL_TEST = 'runs/push152-final-test'
PRETRAINING_SOURCES = {
    'push152_init.py': '0ef6a17e093fcf19b43b212d22fe0ba41f60e4550f8ccc3cf48b47754b86c98b',
    'push152_launch.py': 'fb8b0789f793e668c463db348c4b01e78923c8e35454dffc8f07af359f3e3f51',
    'push152_preflight.py': '72e55de2b8041f1974264ea95ce49ca89689998d729d0c73ae88ccffa1365a36',
    'push152_resume.py': 'acf14022eeb1e28fd003c0315db078e1d3196fc542df89175fd9aaeab4cb7ad3',
    'push152_train.py': '7044375f667bffc320e928d93ba38ee369b2172b7bb8b1476cbac9ef913d2b4c',
    'push152_unit_check.py': '7eb7504cadadae8653d42cf33e75f312745b450563c3291357799eccf7ec064e'}
PRETRAINING_TESTS = {
    'test_push152_init.py': '3529f7c7d76c3c49e23da1da1d8424b5110302b0574746991df5d6caeafa13d7',
    'test_push152_resume.py': '495e50abc44303555a427a6546e96441c9d11212cf0f4a7148e61ac062aabbc0',
    'test_push152_train.py': '06b916b8be3ddc9e3ee7586fc3f0a3e5415699022c42d0fd2dbb7a4fc586fa4b'}


def resource_dir(root, rank):
    require(type(rank) is int and 0 <= rank < 3, 'At most three zero-based resource ranks')
    return child(root, 'runs/push152-resource-rank'+str(rank))


def check_unit_result(record):
    require(record.get('successful') is True and type(record.get('tests_run')) is int and
            record['tests_run'] > 0 and record.get('failures') == [] and record.get('errors') == [] and
            record.get('skipped') == [] and record.get('test_text_loaded') is False,
            'Dataset-free unit suite did not pass completely')


def verify_units(root, unit_path, audit_unit_path):
    root, work = Path(root), Path(root).parents[1]
    require(Path(unit_path).resolve() == child(root, UNIT) and
            Path(audit_unit_path).resolve() == child(root, AUDIT_UNIT), 'Fixed final unit receipts required')
    initial = read(child(root, PRETRAINING_UNIT))
    check_unit_result(initial)
    require(initial.get('source_sha256') == PRETRAINING_SOURCES and
            initial.get('unit_source_sha256') == PRETRAINING_TESTS,
            'Pretraining unit coverage differs from frozen executable graph')
    verify_manifest(root, PRETRAINING_SOURCES)
    verify_manifest(root/'tests', PRETRAINING_TESTS)
    final = read(unit_path)
    check_unit_result(final)
    sources = {path.name: sha(path) for path in sorted(root.glob('push152*.py'))}
    tests = {path.name: sha(path) for path in sorted((root/'tests').glob('test_push152*.py'))}
    require(final.get('source_sha256') == sources and final.get('unit_source_sha256') == tests and
            {'push152_finalize.py','push152_select.py'} <= set(sources) and
            {'test_push152_finalize.py','test_push152_select.py'} <= set(tests),
            'Final unit receipt does not cover current complete push152 sources/tests')
    audit_unit = read(audit_unit_path)
    check_unit_result(audit_unit)
    audit_sources = {name: sha(work/name) for name in
        ('push152_audit.py','push151_audit.py','push150_numerical_diagnostic.py','push152_audit_unit_check.py')}
    audit_tests = {'test_push152_audit.py': sha(work/'test_push152_audit.py')}
    require(audit_sources['push151_audit.py'] == LEGACY_AUDIT_SHA and
            audit_sources['push150_numerical_diagnostic.py'] == DIAGNOSTIC_SCRIPT_SHA and
            audit_unit.get('source_sha256') == audit_sources and audit_unit.get('unit_source_sha256') == audit_tests,
            'Auditor receipt/helper coverage differs')
    receipts = {name: dict(path=name,sha256=sha(child(root,name)),executions=value['tests_run'])
                for name,value in ((PRETRAINING_UNIT,initial),(UNIT,final),(AUDIT_UNIT,audit_unit))}
    return dict(pretraining=receipts[PRETRAINING_UNIT],final=receipts[UNIT],audit=receipts[AUDIT_UNIT],
                final_executions=final['tests_run']+audit_unit['tests_run']), dict(audit_sources,**audit_tests)


def timing_details(timing, winner):
    expected = dict(test_used=False,split='validation',device='cpu',precision='fp32',threads=4,
        affinity_mask=15,candidate_sha256=winner['checkpoint_sha256'],baseline_sha256=BASELINE_SHA,
        evaluator_sha256=SCORER_SHA,script_sha256=TIMING_SCRIPT_SHA)
    require(all(timing.get(key) == value for key,value in expected.items()), 'Timing identity differs')
    rows = timing['measurements']
    require([(row['repeat'],row['name']) for row in rows] ==
        [(0,'baseline'),(0,'candidate'),(1,'candidate'),(1,'baseline'),(2,'baseline'),(2,'candidate')],
        'Exactly three fresh alternating validation timing pairs required')
    for row in rows:
        check_score(row,'validation')
        require(number(row.get('peak_process_working_set_bytes')), 'Missing timing peak RAM')
    times = {name:[row['seconds'] for row in rows if row['name'] == name]
             for name in ('baseline','candidate')}
    ratio = statistics.median(times['candidate'])/statistics.median(times['baseline'])
    require(number(timing.get('cpu_time_ratio')) and abs(ratio-timing['cpu_time_ratio']) < 1e-12,
            'Timing summary ratio differs from actual complete passes')
    require(timing.get('peak_process_working_set_bytes') == max(row['peak_process_working_set_bytes'] for row in rows),
            'Timing summary all-process peak differs from original measurements')
    peak = max(row['peak_process_working_set_bytes'] for row in rows if row['name'] == 'candidate')
    return dict(baseline_seconds=times['baseline'],candidate_seconds=times['candidate'],
        ratio_of_medians=ratio,peak_process_working_set_bytes=peak,
        cpu_limit_pass=ratio <= 5,ram_limit_pass=peak <= 4*2**30,
        individual_pair_ratios=[c/b for b,c in zip(times['baseline'],times['candidate'])],
        max_candidate_divided_by_min_baseline=max(times['candidate'])/min(times['baseline']),
        extra_conservative_gate_required=False,all_records_retained=True)


def verify_timing_files(root, rank, winner, audit, require_pass=True):
    directory = resource_dir(root,rank)/'timing'
    path = directory/'summary.json'
    timing = read(path)
    summary = timing_details(timing,winner)
    manifest = {relative(root,path):sha(path)}
    for row in timing['measurements']:
        raw_path = directory/(row['name']+'-'+str(row['repeat'])+'.json')
        raw = read(raw_path)
        candidate = row['name'] == 'candidate'
        expected = dict(split='validation',threads=4,cpu_affinity_mask=15,
            checkpoint_sha256=winner['checkpoint_sha256'] if candidate else BASELINE_SHA,
            implementation_sha256=IMPLEMENTATION_SHA if candidate else BASELINE_MODEL_SHA,evaluator_sha256=SCORER_SHA)
        require(all(raw.get(key) == value for key,value in expected.items()) and len(raw.get('repeats',[])) == 1,
                'Fresh timing file identity differs')
        require(row == dict(name=row['name'],repeat=row['repeat'],**raw['repeats'][0],
                           peak_process_working_set_bytes=raw['peak_process_working_set_bytes']),
                'Timing summary did not preserve raw measurement')
        if candidate:
            require(abs(row['bpb']-audit['measurement']['bpb']) < 1e-10,
                    'Timed predictor score differs from independently audited predictor')
        manifest[relative(root,raw_path)] = sha(raw_path)
    if require_pass:
        require(summary['cpu_limit_pass'] and summary['ram_limit_pass'], 'Candidate timing/RAM exceeds teacher limit')
    return summary,manifest


def verify_precision(root, rank, audit, work_source_sha):
    directory = resource_dir(root,rank)
    precision = audit['precision_diagnostic']
    diagnostic_path = directory/'audit.diagnostic.json'
    require(sha(diagnostic_path) == precision['diagnostic_sha256'], 'Diagnostic evidence changed')
    diagnostic = read(diagnostic_path)
    require(audit['audit_script_sha256'] == work_source_sha['push152_audit.py'], 'Auditor source changed')
    criteria = read(child(root,PLAN))['audit']
    expected = dict(status='completed',checkpoint_sha256=audit['checkpoint_sha256'],
        implementation='push150_expand',implementation_sha256=IMPLEMENTATION_SHA,
        plan_sha256=PLAN_SHA,numerical_audit_plan_sha256=PLAN_SHA,selection_sha256=audit['selection_sha256'],
        diagnostic_script_sha256=DIAGNOSTIC_SCRIPT_SHA,
        supplemental_audit_script_sha256=work_source_sha['push152_audit.py'],
        test_text_loaded_or_hashed=False,training_performed=False,official_scoring_performed=False,
        checkpoint_or_inference_sources_modified=False,validation_text_loaded=True,
        validation_text_used_by_diagnostic=False,requested_lengths=[1,2,3,19,63,128,255])
    require(all(diagnostic.get(key) == value for key,value in expected.items()) and diagnostic.get('criteria') == criteria,
            'Diagnostic identity/source isolation/predeclared criterion differs')
    declaration = dict(fixed_diagnostic_passes=True,plan_sha256=PLAN_SHA,plan_file=PLAN,
        diagnostic_file='audit.diagnostic.json',diagnostic_script='push150_numerical_diagnostic.py',
        diagnostic_script_sha256=DIAGNOSTIC_SCRIPT_SHA,supplemental_audit_script='push152_audit.py',
        supplemental_audit_script_sha256=work_source_sha['push152_audit.py'],legacy_failure_reclassified_as_pass=False,
        teacher_short_prefix_tolerance_not_specified=True,shape_change_tolerance=dict(atol=1e-5,rtol=1e-5),
        same_shape_tolerance_unchanged=dict(atol=4e-6,rtol=1e-6),reset_atol=0.,reset_rtol=0.,
        neither_predictor_nor_official_scorer_modified=True,criterion_note=criteria['shape_tolerance_disclosure'])
    require(all(precision.get(key) == value for key,value in declaration.items()), 'Precision declaration differs')
    require(all(diagnostic['runtime'].get(key) == value for key,value in
        dict(device='cpu',precision='fp32',threads=4,affinity_mask=15).items()), 'Diagnostic CPU runtime differs')
    require(all(diagnostic['fp64_reference'].get(key) is True for key in
        ('neural_pointer_float64','dropout_disabled','original_fp32_rope_distance_buffers_cast_to_float64',
         'count_table_values_original_fp32_cast_to_float64','inference_implementation_unchanged','no_model_double_shortcut')),
        'FP64 reference must use the independent formula and original buffer values')
    require(diagnostic['train_prefix']['split'] == 'train' and diagnostic['train_prefix']['used_tokens'] == 512,
            'Diagnostic fixture source differs')
    checks = check_diagnostic_batches(diagnostic['batches'])
    require(diagnostic.get('checks') == checks and all(precision.get(key) == value for key,value in checks.items()) and
            diagnostic['summary'].get('predeclared_supplemental_diagnostic_pass') is True,
            'Detailed diagnostic results do not support audit verdict')
    require(diagnostic['source_sha256'] == audit['source_sha256'], 'Diagnostic/runtime source graph differs')
    return {relative(root,diagnostic_path):sha(diagnostic_path)}


def verify_audit(root, rank, winner, selection_sha, work_source_sha, require_limits=True):
    root = Path(root)
    directory = resource_dir(root,rank)
    path, checkpoint = directory/'audit.json', child(root,winner['checkpoint'])
    audit = read(path)
    expected = dict(status='completed',plan_sha256=PLAN_SHA,selection_sha256=selection_sha,
        checkpoint_sha256=winner['checkpoint_sha256'],implementation='push150_expand',
        implementation_sha256=IMPLEMENTATION_SHA,test_used=False,evaluation_network_access=False,
        full_validation_passes=1,fp64_reference_deleted_before_official_score=True,
        cpu_timing_ratio_not_measured_by_this_audit=True,all_limits_pass_not_claimed_without_separate_paired_timing=True)
    require(all(audit.get(key) == value for key,value in expected.items()) and
            audit.get('selected_candidate') == winner, 'Audit identity/candidate differs')
    require(audit.get('resource_rank') == rank and
            audit.get('train_targets_on_checkpoint') == winner['selected_lineage_targets'] and
            audit.get('parameters') == winner['parameters'] and
            audit.get('unchanged_statistical_buffers') == 21 and audit.get('shared_ancestor_targets') == LINEAGE and
            audit.get('selected_new_training_targets') == winner['step']*8192 and
            audit.get('new_training_targets') == audit.get('audit_new_training_targets') == 0,
            'Audit rank/lineage/statistics/parameters differ')
    for group,keys in [('contract_checks',CONTRACT_KEYS),('fixed_file_checks',FIXED_KEYS)]:
        values = audit.get(group,{})
        require(keys <= set(values) and all(value is True for value in values.values()), 'Audit check failed: '+group)
    require(number(audit.get('peak_process_working_set_bytes')) and
            audit['peak_ram_gib'] == audit['peak_process_working_set_bytes']/2**30,
            'Actual audit peak RAM evidence differs')
    measured = audit['measurement']
    check_score(measured,'validation')
    require(all(measured.get(key) == value for key,value in dict(device='cpu',precision='fp32',split='validation',
        test_used=False,checkpoint_sha256=winner['checkpoint_sha256'],implementation_sha256=IMPLEMENTATION_SHA,
        evaluator_sha256=SCORER_SHA,tokenizer_sha256=TOKENIZER_SHA).items()), 'Actual CPU audit score identity differs')
    require(abs(measured['bpb']-winner['validation_bpb_gpu_fp32']) <= 1e-5,
            'CPU predictor differs materially from validation-selected GPU predictor')
    require(sha(root/PARENT_AUDIT) == PARENT_AUDIT_SHA, 'Fixed parent asset manifest changed')
    parent = read(root/PARENT_AUDIT)
    assets = {name.replace('\\','/'):value for name,value in audit['assets'].items()}
    old = {name.replace('\\','/'):value for name,value in parent['assets'].items()}
    require(len(assets) == len(audit['assets']) and set(assets) == set(old) and
            all(assets[name] == value for name,value in old.items() if name != 'checkpoint.pt'),
            'Inherited predictor dependency graph changed or incomplete')
    require(audit.get('asset_manifest_basis') == dict(original_audit=PARENT_AUDIT,
        original_audit_sha256=PARENT_AUDIT_SHA,identical_runtime_files_only_checkpoint_replaced=True) and
        audit.get('audit_support_source_sha256') == {'push151_audit.py':LEGACY_AUDIT_SHA},
        'Audit runtime/helper manifest provenance differs')
    manifest = {relative(root,path):sha(path),PARENT_AUDIT:PARENT_AUDIT_SHA}
    total = 0
    for name,asset in assets.items():
        actual = checkpoint if name == 'checkpoint.pt' else child(root,name)
        require(sha(actual) == asset['sha256'] and actual.stat().st_size == asset['bytes'], 'Audit asset changed: '+name)
        manifest[relative(root,actual)] = asset['sha256']
        total += asset['bytes']
    require(assets['checkpoint.pt']['sha256'] == winner['checkpoint_sha256'] and
            total == audit['inference_assets_bytes'] and total/2**20 == audit['inference_assets_mib'] and total > 0,
            'Inference asset identity/byte sum differs')
    require(audit.get('asset_limit_pass') is (total <= 64*2**20) and
            audit.get('ram_limit_pass') is (audit['peak_ram_gib'] <= 4), 'Audit resource booleans differ')
    if require_limits:
        require(audit['asset_limit_pass'] and audit['ram_limit_pass'], 'Audit exceeds teacher resource limits')
    verify_manifest(root,audit['source_sha256'])
    verify_manifest(root,audit['development_source_sha256'])
    require(all(assets.get(name,{}).get('sha256') == digest for name,digest in audit['source_sha256'].items()),
            'Audited source absent from counted inference assets')
    manifest.update(audit['source_sha256'])
    manifest.update(audit['development_source_sha256'])
    manifest.update(verify_precision(root,rank,audit,work_source_sha))
    return audit,manifest


def load_auditor(root, work_source_sha):
    """Import only the pure audit metadata API, never its run() function."""
    work = Path(root).parents[1]
    for name in ('push152_audit.py','push151_audit.py','push150_numerical_diagnostic.py'):
        require(sha(work/name) == work_source_sha[name], 'Auditor dependency changed: '+name)
    sys.path.insert(0,str(work))
    try:
        spec = importlib.util.spec_from_file_location('_push152_finalizer_audit_metadata',work/'push152_audit.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def verify_prior_ranks(root, selection, rank, work_source_sha):
    auditor = load_auditor(root,work_source_sha)
    pins = {}
    outcomes = auditor.check_prior_dispositions(Path(root),selection,rank,pins)
    manifest = {relative(root,path):digest for path,digest in pins.items()}
    # Independently verify every completed prior audit and six-pass timing; a
    # terminal process failure is allowed only through the auditor's explicit
    # command/started/process/failure identity checks, never missing output.
    for outcome in outcomes:
        prior_rank = outcome['rank']
        prior = selection['resource_audit_candidates'][prior_rank]
        directory = resource_dir(root,prior_rank)
        if outcome['stage'] != 'audit':
            audit,evidence = verify_audit(root,prior_rank,prior,sha(Path(root)/SELECTION),work_source_sha,False)
            manifest.update(evidence)
            if outcome['stage'] == 'timing_resources':
                timing,evidence = verify_timing_files(root,prior_rank,prior,audit,False)
                require(not (timing['cpu_limit_pass'] and timing['ram_limit_pass'] and
                    audit['asset_limit_pass'] and audit['ram_limit_pass']), 'Earlier candidate passed; cannot skip it')
                manifest.update(evidence)
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.log'):
                manifest[relative(root,path)] = sha(path)
    return outcomes,manifest


def verify_selection(root, selection_path, rank):
    root = Path(root)
    resource_dir(root,rank)
    require(Path(selection_path).resolve() == child(root,SELECTION), 'Fixed push152 selection path required')
    require(sha(child(root,PLAN)) == PLAN_SHA and sha(root/'push151_finalize.py') == LEGACY_FINALIZER_SHA,
            'Predeclared plan or frozen pure helper changed')
    selection = read(selection_path)
    require(selection.get('status') == 'validation_qualified_candidates_need_resource_audit' and
            selection.get('round') == 'push152' and selection.get('plan_sha256') == PLAN_SHA and
            selection.get('selection_split') == 'validation' and selection.get('test_used') is False and
            selection.get('all_declared_runs_accounted') is True and
            selection.get('all_completed_runs_have_exact_predeclared_validation_menu') is True and
            selection.get('minimum_improvement') == .001 and selection.get('maximum_resource_candidates') == 3 and
            selection.get('no_new_calibration_or_averaging') is True, 'Incomplete/test-based/changed selection')
    require(selection['script_sha256'] == sha(root/'push152_select.py'), 'Selector source changed')
    selector = importlib.import_module('push152_select')
    require(Path(selector.__file__).resolve() == (root/'push152_select.py').resolve(), 'Wrong selector module root')
    actual = selector.collect(root)
    require({key:value for key,value in actual.items() if key != 'created_utc'} ==
            {key:value for key,value in selection.items() if key != 'created_utc'},
            'Selection no longer matches complete source/trial/cost evidence')
    ranked = selection['resource_audit_candidates']
    require(0 <= rank < len(ranked) <= 3 and ranked == selection['validation_qualified_in_rank_order'][:3],
            'Rank is not in the fixed first-three qualified menu')
    winner = ranked[rank]
    require(winner['selected_lineage_targets'] == LINEAGE+winner['step']*8192 and
            selection['parent']['sha256'] == PARENT_SHA and
            selection['parent']['validation_bpb_gpu_fp32']-winner['validation_bpb_gpu_fp32'] >= .001,
            'Selected lineage or improvement gate differs')
    verify_manifest(root,selection['source_sha256'])
    verify_manifest(root,selection['evidence_sha256'])
    manifest = dict(selection['evidence_sha256'],**selection['source_sha256'])
    manifest[SELECTION] = sha(selection_path)
    require(sha(child(root,winner['checkpoint'])) == winner['checkpoint_sha256'], 'Selected checkpoint changed')
    return selection,winner,manifest


def preflight(root, selection_path, rank, unit_path, audit_unit_path):
    root = Path(root)
    selection,winner,manifest = verify_selection(root,selection_path,rank)
    units,work_files = verify_units(root,unit_path,audit_unit_path)
    auditor = load_auditor(root,work_files)
    plan = read(child(root,PLAN))
    auditor.verify_plan(plan)
    verified_rank,verified_winner = auditor.verify_selection(selection,winner['checkpoint'],winner['checkpoint_sha256'],plan)
    require(verified_rank == rank and verified_winner == winner, 'Independent auditor ranking differs')
    previous,prior_manifest = verify_prior_ranks(root,selection,rank,work_files)
    # A lower-ranked resource trial is never justified after this rank passes.
    for later in range(rank+1,3):
        directory = resource_dir(root,later)
        require(not any((directory/name).exists() for name in
            ('audit.started.json','audit.json','audit.failure.json','timing/summary.json','timing.failure.json')),
            'A lower-ranked resource attempt exists after the first passing candidate')
    audit,audit_files = verify_audit(root,rank,winner,sha(selection_path),work_files)
    require(audit.get('earlier_rank_failure_evidence') == previous, 'Audited prior-rank dispositions changed')
    require(not (resource_dir(root,rank)/'audit.failure.json').exists() and
            not (resource_dir(root,rank)/'timing.failure.json').exists(), 'Selected rank has conflicting failure evidence')
    timing,timing_files = verify_timing_files(root,rank,winner,audit)
    fixed = {'evaluate.py':SCORER_SHA,'data/tokenizer.json':TOKENIZER_SHA,
        'model.py':BASELINE_MODEL_SHA,'runs/baseline-s17/checkpoint.pt':BASELINE_SHA,
        'push150_expand.py':IMPLEMENTATION_SHA,'official_resource_run.py':OFFICIAL_WRAPPER_SHA,
        'push145_fresh_benchmark.py':TIMING_SCRIPT_SHA,'measure_validation.py':MEASURE_SCRIPT_SHA,
        'push151_finalize.py':LEGACY_FINALIZER_SHA}
    verify_manifest(root,fixed)
    for values in (fixed,prior_manifest,audit_files,timing_files):
        manifest.update(values)
    for name in (PRETRAINING_UNIT,UNIT,AUDIT_UNIT):
        manifest[name] = sha(child(root,name))
    for path in [*root.glob('push152*.py'),*(root/'tests').glob('test_push152*.py')]:
        manifest[relative(root,path)] = sha(path)
    checkpoint = child(root,winner['checkpoint'])
    record = dict(round='push152',status='frozen_after_validation_resource_gates',plan_sha256=PLAN_SHA,
        selection_record=SELECTION,selection_record_sha256=sha(selection_path),selected_validation_record=winner,
        validation_rank=rank,earlier_rank_failure_evidence=previous,maximum_resource_candidates=3,
        first_passing_candidate_locked=True,post_test_fallback_allowed=False,
        checkpoint=winner['checkpoint'],checkpoint_sha256=winner['checkpoint_sha256'],checkpoint_bytes=checkpoint.stat().st_size,
        implementation='push150_expand',implementation_sha256=IMPLEMENTATION_SHA,
        all_declared_runs_accounted=True,selection_split='validation',minimum_validation_improvement=.001,
        test_bpb_at_freeze=None,this_predictor_test_scored_before_freeze=False,test_used_for_tuning=False,
        prior_other_predictors_test_exposure_disclosed=True,
        known_push151_test_bpb_disclosure_only=plan['parent']['previous_test_bpb_disclosure_only'],
        prior_test_history_is_exhaustive=False,shared_ancestor_targets=LINEAGE,
        selected_lineage_targets=winner['selected_lineage_targets'],train_processed_targets=winner['selected_lineage_targets'],
        selected_new_training_targets=winner['step']*8192,development_costs=selection['costs'],
        same_budget_endpoint_comparisons=selection['same_budget_endpoint_comparisons'],parameters=audit['parameters'],
        unit_tests=units,audit_record=relative(root,resource_dir(root,rank)/'audit.json'),
        audit_sha256=sha(resource_dir(root,rank)/'audit.json'),
        timing_record=relative(root,resource_dir(root,rank)/'timing/summary.json'),
        timing_record_sha256=sha(resource_dir(root,rank)/'timing/summary.json'),validation_timing=timing,
        validation_bpb_cpu_fp32=audit['measurement']['bpb'],inference_assets_bytes=audit['inference_assets_bytes'],
        inference_assets_mib=audit['inference_assets_mib'],validation_audit_peak_ram_gib=audit['peak_ram_gib'],
        contract_checks_all_pass=True,numerical_tolerance_is_supplemental_not_teacher_authorization=True,
        legacy_short_prefix_check_failed=audit['precision_diagnostic']['legacy_short_prefix_check_failed'],
        legacy_total_mismatched_elements=audit['precision_diagnostic']['legacy_total_mismatched_elements'],
        official_scorer_unchanged=True,official_test_wrapper_sha256=OFFICIAL_WRAPPER_SHA,
        baseline_checkpoint_sha256=BASELINE_SHA,baseline_implementation_sha256=BASELINE_MODEL_SHA,
        finalizer_source_sha256=sha(root/'push152_finalize.py'),evaluation_network_access=False,
        full_test_protocol='One official CPU FP32 candidate pass bracketed by fixed baselines; threads4 affinity15; no retry or post-test selection.',
        evidence_sha256=manifest,work_evidence_sha256=work_files)
    return child(root,FREEZE),record


def create_freeze(root, selection_path, rank, unit_path, audit_unit_path):
    root = Path(root)
    require(not child(root,FINAL_TEST).exists() and not child(root,FREEZE).exists(),
            'Global freeze/test ledger already exists; cannot choose another candidate')
    path,record = preflight(root,selection_path,rank,unit_path,audit_unit_path)
    # Retain launch, partial-work and resource records without touching corpus
    # text or final-test logs. Selected source/evidence fields remain exact.
    directories = ['runs/push152-launches','runs/push152-expanded-init','runs/push152-expanded-preflight']
    directories.extend('runs/push152-resource-rank'+str(index) for index in range(rank+1))
    for directory in directories:
        for evidence in child(root,directory).rglob('*'):
            if evidence.is_file() and evidence.suffix in ('.json','.log'):
                record['evidence_sha256'][relative(root,evidence)] = sha(evidence)
    record['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_new(path,record)
    return path,record


def verify_freeze(root, selection_path):
    """Read-only and usable after official test for the packaging check."""
    root = Path(root)
    path = child(root,FREEZE)
    frozen = read(path)
    verify_manifest(root,frozen['evidence_sha256'])
    verify_manifest(root.parents[1],frozen['work_evidence_sha256'])
    current_path,current = preflight(root,selection_path,frozen['validation_rank'],child(root,UNIT),child(root,AUDIT_UNIT))
    require(current_path == path and frozen.get('frozen_at_utc'), 'Missing global frozen identity/time')
    for key,value in current.items():
        if key in ('evidence_sha256','work_evidence_sha256'):
            require(all(frozen[key].get(name) == digest for name,digest in value.items()), 'Frozen evidence changed')
        else:
            require(frozen.get(key) == value, 'Frozen identity changed: '+key)
    return dict(freeze=frozen,freeze_path=path,winner=frozen['selected_validation_record'],
        checkpoint=child(root,frozen['checkpoint']),audit=read(child(root,frozen['audit_record'])))


def run_test(root, selection_path, run_dir):
    root,run_dir = Path(root),Path(run_dir)
    require(run_dir.resolve() == child(root,FINAL_TEST), 'Exactly one fixed final-test ledger directory')
    require(not run_dir.exists(), 'Do not retry or overwrite a final-test attempt')
    evidence = verify_freeze(root,selection_path)
    candidate_sha = evidence['winner']['checkpoint_sha256']
    specs = [('baseline-before',root/'runs/baseline-s17/checkpoint.pt',BASELINE_SHA,BASELINE_MODEL_SHA),
             ('candidate',evidence['checkpoint'],candidate_sha,IMPLEMENTATION_SHA),
             ('baseline-after',root/'runs/baseline-s17/checkpoint.pt',BASELINE_SHA,BASELINE_MODEL_SHA)]
    run_dir.mkdir(parents=True,exist_ok=False)
    started,rows,launches = time.perf_counter(),[],[]
    identity = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=PLAN_SHA,
        selection_sha256=sha(selection_path),freeze_sha256=sha(evidence['freeze_path']),
        finalizer_source_sha256=sha(root/'push152_finalize.py'),official_wrapper_sha256=OFFICIAL_WRAPPER_SHA,
        evaluator_sha256=SCORER_SHA,tokenizer_sha256=TOKENIZER_SHA,candidate_sha256=candidate_sha,
        baseline_sha256=BASELINE_SHA,planned_passes=[dict(name=name,checkpoint=str(path),sha256=digest)
            for name,path,digest,_ in specs],candidate_pass_cap=1,device='cpu',precision='fp32',threads=4,affinity_mask=15,
        failure_policy='Preserve every launch/output/failure; no retry, fallback, or predictor changes.')
    try:
        write_new(run_dir/'launch_plan.json',identity)
        for index,(name,checkpoint,digest,implementation_sha) in enumerate(specs):
            verify_freeze(root,selection_path)
            require(sha(selection_path) == identity['selection_sha256'] and
                    sha(evidence['freeze_path']) == identity['freeze_sha256'] and
                    sha(root/'push152_finalize.py') == identity['finalizer_source_sha256'],
                    'Frozen orchestration evidence changed during scoring')
            output = run_dir/(name+'.json')
            command = [sys.executable,'-B',str(root/'official_resource_run.py'),'--checkpoint',str(checkpoint),
                       '--split','test','--output',str(output)]
            write_new(run_dir/(name+'.started.json'),dict(index=index,name=name,
                started_at_utc=datetime.now(timezone.utc).isoformat(),command=command,
                launch_plan_sha256=sha(run_dir/'launch_plan.json'),checkpoint_sha256=digest))
            launches.append(name)
            process = subprocess.run(command,cwd=root,capture_output=True,text=True,encoding='utf-8',errors='replace')
            write_new(run_dir/(name+'.process.json'),dict(returncode=process.returncode,stdout=process.stdout,
                stderr=process.stderr,completed_at_utc=datetime.now(timezone.utc).isoformat()))
            require(process.returncode == 0,'Official scoring process failed: '+name)
            resources_path = output.with_suffix('.resources.json')
            measured,resources = read(output),read(resources_path)
            check_measurement(measured,resources,digest,implementation_sha)
            require(Path(resources['evaluator_output']).resolve() == output.resolve(), 'Resource/output path differs')
            rows.append(dict(**measured,name=name,output=output.name,output_sha256=sha(output),
                resources_sha256=sha(resources_path),peak_ram_gib=resources['peak_process_working_set_bytes']/2**30))
            print(json.dumps(rows[-1],allow_nan=False),flush=True)
        verify_freeze(root,selection_path)
        require(sha(selection_path) == identity['selection_sha256'] and sha(evidence['freeze_path']) == identity['freeze_sha256'],
                'Freeze changed during official scoring')
        report = test_summary(rows,evidence['audit'])
        report.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),
            launch_plan_sha256=sha(run_dir/'launch_plan.json'),freeze_sha256=identity['freeze_sha256'],
            checkpoint_sha256=candidate_sha,seconds_total=time.perf_counter()-started)
        write_new(run_dir/'summary.json',report)
        return report
    except BaseException as error:
        write_new(run_dir/'failure.json',dict(status='failed',failure_type=type(error).__name__,failure_message=str(error),
            launches_started=launches,measurements_completed=rows,seconds_total=time.perf_counter()-started,
            no_automatic_retry=True,partial_test_exposure_must_be_disclosed=bool(launches)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    freeze = sub.add_parser('freeze')
    freeze.add_argument('--rank',type=int,choices=range(3),default=0)
    freeze.add_argument('--selection',type=Path,default=ROOT/SELECTION)
    freeze.add_argument('--unit-evidence',type=Path,default=ROOT/UNIT)
    freeze.add_argument('--audit-unit-evidence',type=Path,default=ROOT/AUDIT_UNIT)
    test = sub.add_parser('test')
    test.add_argument('--selection',type=Path,default=ROOT/SELECTION)
    test.add_argument('--run-dir',type=Path,default=ROOT/FINAL_TEST)
    args = parser.parse_args()
    if args.command == 'freeze':
        path,record = create_freeze(ROOT,args.selection,args.rank,args.unit_evidence,args.audit_unit_evidence)
        print(json.dumps(dict(output=str(path),**record),indent=2,allow_nan=False),flush=True)
    else:
        print(json.dumps(run_test(ROOT,args.selection,args.run_dir),indent=2,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
