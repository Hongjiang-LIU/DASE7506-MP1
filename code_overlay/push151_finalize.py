"""Metadata-only push151 freeze and exclusive one-candidate official test ledger.

Only the explicitly invoked ``test`` subcommand launches the unchanged official
scorer. Import, freeze, and metadata tests never import a model or read a corpus.
There is no rank override, resource-failure fallback, retry, or test selection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
PLAN = 'runs/push151_plan.json'
PLAN_SHA = '3ba6539a20ba74c7989b9733e755d48bd7529ae9446785e918ceb65f049063aa'
PARENT_SHA = 'afaf5f98c4c2148942454bb9295a41f7d41209b804c28db96676a5d9886eb9c7'
BASELINE_SHA = 'b7fdb67c8ef14b5166c0de1f17cf306f34800b41dfe97da856c9812cdb1f9701'
BASELINE_MODEL_SHA = '6cce142a31d396f017319543f50d70f30fa6da740729af01c8baed13c6f671f9'
SCORER_SHA = '128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d'
TOKENIZER_SHA = '020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e'
IMPLEMENTATION_SHA = 'd0340092e2a2882113533f05f4668c48b721ee4416842910c2ba2ba1758e1d45'
OFFICIAL_WRAPPER_SHA = '9a6240cde92c082ae83121a0afc8ad7ce4729e491c52d12fe6aeb90738838a1f'
TIMING_SCRIPT_SHA = '5e0a5173dd00a1b3519877172d1dcb3d195a2a106a17cd6256d7b55cb78bb99e'
MEASURE_SCRIPT_SHA = '2b625ce21edeb1a6094b3efde455cab4bc27b5292dd0ace03ee9c21861b85448'
DIAGNOSTIC_SCRIPT_SHA = '92f033e1cc7b3532924d2d6bbde6eee1d0a8fa73c1c076f32e340f63201ebe25'
PARENT_AUDIT_SHA = '7a570b0ebc7c934c803244fafbb81124e14abcac16026a6d4c0805c006db3aab'
AUDIT_SCRIPT_SHA = '5c348e35298b1cc1b4ebb43a97747c8709cd274048d2ae4368326ffbf2a1eb8a'
LINEAGE = 638976000
SELECTION = 'runs/push151_validation_selection.json'
UNIT = 'runs/push151_unit_tests_final.json'
TIMING = 'runs/push151-fresh-validation-timing/summary.json'
FINAL_TEST = 'runs/push151-final-test'
PHASE_DIRS = ('runs/push151-average', 'runs/push151-statistics', 'runs/push151-combination')
CONTRACT_KEYS = {'future_invariance', 'example_independence', 'window_reset',
                 'short_prefix_equivalence', 'normalized', 'finite'}
FIXED_KEYS = {'common.py', 'evaluate.py', 'model.py', 'tests/test_contract.py',
              'data/manifest.json', 'data/tokenizer.json', 'data/wikitext_train.txt',
              'data/wikitext_validation.txt'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, positive=True):
    return (type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0))


def sha(path):
    path = Path(path)
    require(not path.name.lower().startswith('wikitext_'), 'No corpus read/hash in finalizer')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def read(path):
    require(not Path(path).name.lower().startswith('wikitext_'), 'Metadata only')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, record):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')


def child(root, relative):
    name = str(relative).replace('\\', '/')
    require(name and not name.startswith('/') and ':' not in name and
            all(part not in ('', '.', '..') for part in name.split('/')), 'Unsafe relative path')
    resolved = (Path(root)/name).resolve()
    require(resolved.is_relative_to(Path(root).resolve()), 'Path escaped root')
    return resolved


def relative(root, path):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def verify_manifest(root, manifest):
    require(isinstance(manifest, dict) and manifest, 'Missing evidence/source manifest')
    for name, digest in manifest.items():
        require(sha(child(root, name)) == digest, 'Evidence/source changed: '+name)


def check_score(row, split, checkpoint_sha=None, implementation_sha=None, official=False):
    targets, utf8_bytes = (376599, 1148007) if split == 'validation' else (428405, 1292013)
    require(row.get('targets') == targets and row.get('utf8_bytes') == utf8_bytes,
            'Incomplete '+split+' measurement')
    for key in ('bpb', 'nll_nats', 'seconds'):
        require(number(row.get(key)), 'Invalid scoring quantity: '+key)
    require(abs(row['bpb']-row['nll_nats']/math.log(2)/utf8_bytes) < 1e-10,
            'BPB denominator differs')
    if checkpoint_sha is not None:
        expected = dict(split=split, precision='fp32', checkpoint_sha256=checkpoint_sha,
                        implementation_sha256=implementation_sha, evaluator_sha256=SCORER_SHA)
        require(all(row.get(key) == value for key, value in expected.items()), 'Scoring identity differs')
        if split == 'validation':
            require(row.get('test_used') is False and str(row.get('device', '')).startswith('cuda'),
                    'Development selection must use GPU FP32 validation only')
    if official:
        require(row.get('protocol') == '7506-mp1-wt2-v2' and
                row.get('tokenizer_sha256') == TOKENIZER_SHA and
                row.get('device') == 'cpu' and row.get('device_name') == 'CPU', 'Official protocol differs')


def ranked_candidate(selection):
    require(selection.get('status') == 'validation_qualified_candidates_need_resource_audit' and
            selection.get('plan_sha256') == PLAN_SHA and selection.get('selection_split') == 'validation' and
            selection.get('test_used') is False and selection.get('all_declared_A_B_C_accounted') is True and
            selection.get('minimum_improvement') == .0005 and
            selection.get('selected_shared_ancestor_targets') == LINEAGE and
            selection.get('new_neural_gradient_targets') == 0, 'Incomplete/invalid push151 selection')
    rows = selection['candidates']
    expected = [('A','control'), ('A','tail-ema-average'), ('A','last-raw-ema-average'),
                ('B','information-equal'), ('B','information-plus8mib')]
    require(selection.get('phase_c_status') in ('completed','skipped'), 'Unaccounted combination')
    if selection['phase_c_status'] == 'completed':
        expected.append(('C','combined'))
    require([(row['phase'], row['name']) for row in rows] == expected, 'Changed candidate menu/order')
    require(rows[0]['checkpoint_sha256'] == PARENT_SHA, 'Missing unchanged parent')
    require(len({row['checkpoint'] for row in rows}) == len(rows), 'Duplicate candidate paths')
    parent = rows[0]['validation_bpb_gpu_fp32']
    for row in rows:
        measured = row['measured']
        check_score(measured, 'validation', row['checkpoint_sha256'], IMPLEMENTATION_SHA)
        require(row['validation_bpb_gpu_fp32'] == measured['bpb'], 'Candidate score differs')
        improvement = parent-row['validation_bpb_gpu_fp32']
        require(abs(improvement-row['improvement_vs_parent']) < 1e-14 and
                row['passes_validation_improvement_gate'] is (improvement >= .0005), 'Improvement gate differs')
    ranked = sorted((row for row in rows if row['passes_validation_improvement_gate']),
                    key=lambda row: row['validation_bpb_gpu_fp32'])
    require(ranked and selection.get('validation_qualified_in_rank_order') == ranked,
            'Qualified ranking differs; no fallback or rank override')
    require(selection.get('full_validation_passes') == 6+int(selection['phase_c_status'] == 'completed'),
            'Development scoring cost differs')
    return ranked[0]


def verify_selection(root, selection_path):
    root = Path(root)
    require(Path(selection_path).resolve() == child(root, SELECTION), 'Fixed push151 selection path required')
    require(sha(child(root, PLAN)) == PLAN_SHA, 'Predeclared plan changed')
    plan, selection = read(child(root, PLAN)), read(selection_path)
    winner = ranked_candidate(selection)
    require(selection['script_sha256'] == sha(root/'push151_select.py'), 'Selector changed')
    require(abs(selection['candidates'][0]['validation_bpb_gpu_fp32']-
                plan['parent']['validation_bpb_gpu_fp32']) < 1e-6, 'Parent validation did not reproduce')
    required = {PLAN, *(directory+'/'+name for directory in PHASE_DIRS[:2] for name in ('build.json','score.json')),
                PHASE_DIRS[2]+'/result.json'}
    require(required <= set(selection['evidence_sha256']), 'A/B/C evidence missing')
    verify_manifest(root, selection['evidence_sha256'])
    manifest = dict(selection['evidence_sha256'])
    manifest[SELECTION] = sha(selection_path)
    for source in [plan['parent'], *plan['phase_a']['snapshots'].values()]:
        require(sha(child(root, source['path'])) == source['sha256'], 'Ancestral checkpoint changed')
        manifest[source['path']] = source['sha256']
    phase_rows = {}
    for phase, directory, names in [('A', PHASE_DIRS[0], ['control','tail-ema-average','last-raw-ema-average']),
                                    ('B', PHASE_DIRS[1], ['control','information-equal','information-plus8mib'])]:
        build, score = read(child(root, directory+'/build.json')), read(child(root, directory+'/score.json'))
        require(build.get('status') == score.get('status') == 'completed' and
                build.get('test_used') is False and score.get('test_used') is False and
                build['plan_sha256'] == score['plan_sha256'] == PLAN_SHA and
                score['build_sha256'] == sha(child(root, directory+'/build.json')),
                'Phase build/score identity differs: '+phase)
        require(build['source_sha256'] == score['source_sha256'], 'Build/score source differs')
        require(selection['construction_seconds'][phase] == build['seconds_total'] and
                selection['development_score_seconds'][phase] == score['seconds_total'],
                'Phase construction/scoring costs differ')
        if phase == 'A':
            require(build.get('new_training_targets') == score.get('new_training_targets') == 0 and
                    build.get('shared_trajectory_union_targets') == score.get('shared_trajectory_union_targets') == LINEAGE,
                    'Average phase ancestry/gradient accounting differs')
        else:
            require(build.get('new_gradient_targets') == score.get('new_gradient_targets') == 0 and
                    build.get('inherited_shared_lineage_targets') == LINEAGE and
                    selection['statistical_train_tokens_scanned'] == build['count_source_token_occurrences'],
                    'Statistical phase ancestry/scanning accounting differs')
        verify_manifest(root, build['source_sha256'])
        manifest.update(build['source_sha256'])
        require([row['name'] for row in build['candidates']] == names and
                [row['name'] for row in score['candidates']] == names, 'Incomplete phase menu')
        current = []
        for built, measured in zip(build['candidates'], score['candidates']):
            require(sha(child(root, built['checkpoint'])) == built['checkpoint_sha256'] ==
                    measured['checkpoint_sha256'], 'Phase checkpoint changed')
            check_score(measured, 'validation', built['checkpoint_sha256'], IMPLEMENTATION_SHA)
            manifest[built['checkpoint']] = built['checkpoint_sha256']
            current.append(dict(phase=phase, name=built['name'], checkpoint=built['checkpoint'],
                checkpoint_sha256=built['checkpoint_sha256'], validation_bpb_gpu_fp32=measured['bpb'], measured=measured))
        phase_rows[phase] = current
        require(current[0]['checkpoint_sha256'] == PARENT_SHA, 'Phase parent differs')
        selected_rows = [row for row in selection['candidates'] if row['phase'] == phase]
        expected_rows = current if phase == 'A' else current[1:]
        require(len(selected_rows) == len(expected_rows) and all(
            all(actual.get(key) == value for key, value in expected.items())
            for actual, expected in zip(selected_rows, expected_rows)), 'Selection and phase scores differ')
    require(abs(phase_rows['A'][0]['measured']['bpb']-phase_rows['B'][0]['measured']['bpb']) < 1e-8,
            'A/B control scores disagree')
    combo = read(child(root, PHASE_DIRS[2]+'/result.json'))
    best_a = min(phase_rows['A'], key=lambda row: row['validation_bpb_gpu_fp32'])
    best_b = min(phase_rows['B'], key=lambda row: row['validation_bpb_gpu_fp32'])
    active = best_a['name'] != 'control' and best_b['name'] != 'control'
    require(combo['plan_sha256'] == PLAN_SHA and combo.get('test_used') is False and
            combo['script_sha256'] == sha(root/'push151_select.py') and
            combo['status'] == selection['phase_c_status'] == ('completed' if active else 'skipped') and
            combo['phase_a_score_sha256'] == manifest[PHASE_DIRS[0]+'/score.json'] and
            combo['phase_b_score_sha256'] == manifest[PHASE_DIRS[1]+'/score.json'] and
            combo.get('new_gradient_targets') == 0, 'Combination conditions/evidence differ')
    require(selection['combination_seconds_including_build_and_score'] == combo['seconds_total'],
            'Combination work accounting differs')
    if active:
        require(combo['source_a'] == best_a and combo['source_b'] == best_b and
                combo.get('shared_ancestor_targets') == LINEAGE, 'Combination ancestors differ')
        validation_path = child(root, PHASE_DIRS[2]+'/validation.json')
        require(combo['measurement'] == read(validation_path) == selection['candidates'][-1]['measured'],
                'Combination validation differs')
        manifest[relative(root, validation_path)] = sha(validation_path)
    for row in selection['candidates']:
        require(sha(child(root, row['checkpoint'])) == row['checkpoint_sha256'], 'Candidate changed')
        manifest[row['checkpoint']] = row['checkpoint_sha256']
    return selection, winner, manifest


def verify_units(root, unit_path):
    root = Path(root)
    require(Path(unit_path).resolve() == child(root, UNIT), 'Fixed final unit evidence required')
    unit = read(unit_path)
    require(unit.get('successful') is True and type(unit.get('tests_run')) is int and unit['tests_run'] > 0 and
            unit.get('failures') == [] and unit.get('errors') == [] and unit.get('skipped') == [] and
            unit.get('test_text_loaded') is False, 'Final dataset-free unit suite did not pass')
    expected_source = {path.name: sha(path) for path in sorted(root.glob('push151*.py'))}
    expected_tests = {path.name: sha(path) for path in sorted((root/'tests').glob('test_push151*.py'))}
    require(expected_source and expected_tests and unit.get('source_sha256') == expected_source and
            unit.get('unit_source_sha256') == expected_tests, 'Final unit source coverage/hash differs')
    require('push151_finalize.py' in expected_source and 'test_push151_finalize.py' in expected_tests,
            'Finalizer tests absent from unit suite')
    return dict(path=UNIT, sha256=sha(unit_path), executions=unit['tests_run'])


def check_timing(timing, winner):
    expected = dict(test_used=False, split='validation', device='cpu', precision='fp32',
        threads=4, affinity_mask=15, candidate_sha256=winner['checkpoint_sha256'],
        baseline_sha256=BASELINE_SHA, evaluator_sha256=SCORER_SHA, script_sha256=TIMING_SCRIPT_SHA)
    require(all(timing.get(key) == value for key, value in expected.items()), 'Timing identity differs')
    rows = timing['measurements']
    order = [(0,'baseline'),(0,'candidate'),(1,'candidate'),(1,'baseline'),(2,'baseline'),(2,'candidate')]
    require([(row['repeat'], row['name']) for row in rows] == order, 'Need three fresh alternating timing pairs')
    times = {name: [row['seconds'] for row in rows if row['name'] == name] for name in ('baseline','candidate')}
    for row in rows:
        check_score(row, 'validation')
        require(number(row.get('peak_process_working_set_bytes')), 'Missing timing RAM evidence')
    ratio = statistics.median(times['candidate'])/statistics.median(times['baseline'])
    require(abs(ratio-timing['cpu_time_ratio']) < 1e-12 and ratio <= 5, 'Median CPU ratio exceeds five/differs')
    require(max(row['peak_process_working_set_bytes'] for row in rows if row['name'] == 'candidate') <= 4*2**30,
            'Timed candidate exceeded RAM limit')
    return dict(baseline_seconds=times['baseline'], candidate_seconds=times['candidate'], ratio_of_medians=ratio,
        individual_pair_ratios=[c/b for b,c in zip(times['baseline'],times['candidate'])],
        max_candidate_divided_by_min_baseline=max(times['candidate'])/min(times['baseline']),
        extra_conservative_gate_required=False, all_records_retained=True)


def verify_timing_files(root, timing_path, timing, winner, audit):
    manifest = {relative(root, timing_path): sha(timing_path)}
    summary = check_timing(timing, winner)
    for row in timing['measurements']:
        path = Path(timing_path).parent/(row['name']+'-'+str(row['repeat'])+'.json')
        raw = read(path)
        cp_sha = winner['checkpoint_sha256'] if row['name'] == 'candidate' else BASELINE_SHA
        impl_sha = IMPLEMENTATION_SHA if row['name'] == 'candidate' else BASELINE_MODEL_SHA
        require(raw.get('split') == 'validation' and raw.get('threads') == 4 and raw.get('cpu_affinity_mask') == 15 and
                raw.get('checkpoint_sha256') == cp_sha and raw.get('implementation_sha256') == impl_sha and
                raw.get('evaluator_sha256') == SCORER_SHA and len(raw.get('repeats', [])) == 1,
                'Fresh timing raw identity differs')
        expected = dict(name=row['name'], repeat=row['repeat'], **raw['repeats'][0],
                        peak_process_working_set_bytes=raw['peak_process_working_set_bytes'])
        require(expected == row, 'Timing summary is not the original measurement')
        if row['name'] == 'candidate':
            require(abs(row['bpb']-audit['measurement']['bpb']) < 1e-10, 'Fresh candidate score differs from audit')
        manifest[relative(root, path)] = sha(path)
    return summary, manifest


def checked_comparison(value, atol=None, rtol=None, bound=None, exact=False, required_pass=True):
    require(type(value.get('elements')) is int and value['elements'] > 0 and
            number(value.get('maximum_absolute_difference'), positive=False), 'Invalid diagnostic comparison')
    violations = value.get('tolerance_violation_elements')
    require(type(violations) is int and 0 <= violations <= value['elements'] and
            value.get('within_declared_tolerance') is (violations == 0), 'Inconsistent diagnostic verdict')
    if atol is not None:
        require(value.get('atol') == atol and value.get('rtol') == rtol, 'Diagnostic tolerance changed')
    if required_pass:
        require(violations == 0 and value.get('zero_tolerance_denominator_violation') is False and
                number(value.get('maximum_tolerance_ratio'), positive=False) and
                value['maximum_tolerance_ratio'] <= 1, 'Required diagnostic comparison failed')
    if bound is not None:
        require(value['maximum_absolute_difference'] <= bound, 'Diagnostic absolute error exceeds fixed bound')
    if exact:
        require(value.get('exact_equal') is True and value['maximum_absolute_difference'] == 0 and violations == 0,
                'Diagnostic exact equality failed')


def check_diagnostic_batches(batches):
    require([batch['name'] for batch in batches] == ['random_seed127','supplied_train_first512'], 'Diagnostic fixtures differ')
    legacy_count, rows = 0, []
    for batch in batches:
        require(batch.get('full_length') == 256 and
                all(batch.get(name) == 0 for name in ('full_masked_attention_maximum_t_greater_than_zero',
                    'full_effective_forbidden_attention_maximum','full_first_position_gate_maximum')),
                'Full-window forbidden pointer mass')
        for name in ('old_audit_same_shape_future_128','old_audit_example_independence'):
            checked_comparison(batch[name],4e-6,1e-6)
        checked_comparison(batch['old_audit_reset_exact'],0.,0.,exact=True)
        checked_comparison(batch['fp32_full_versus_fp64_reference'],bound=1e-4,required_pass=False)
        require([row['length'] for row in batch['prefixes']] == [1,2,3,19,63,128,255], 'Missing/reordered diagnostic prefixes')
        for row in batch['prefixes']:
            require(all(row.get(name) == 0 for name in ('masked_attention_maximum_t_greater_than_zero',
                    'effective_forbidden_attention_maximum','first_position_gate_maximum')), 'Forbidden prefix pointer mass')
            checked_comparison(row['same_shape_future_change'],4e-6,1e-6)
            checked_comparison(row['mkn_prefix'],0.,0.,exact=True)
            checked_comparison(row['supplemental_fp32_log_prefix'],1e-5,1e-5)
            for name in ('fp64_reference_log_prefix','fp64_reference_neural_pointer_prefix'):
                checked_comparison(row[name],1e-12,0.,bound=1e-10,required_pass=False)
                checked_comparison(row['supplemental_'+name],1e-10,0.)
            checked_comparison(row['fp32_short_versus_fp64_reference'],bound=1e-4,required_pass=False)
            checked_comparison(row['fp32_short_versus_same_full_fp64_reference'],1e-4,0.)
            old = row['production_fp32_log_prefix']
            checked_comparison(old,5e-6,1e-6,required_pass=False)
            legacy_count += old['tolerance_violation_elements']
            rows.append(dict(fixture=batch['name'],length=row['length'],
                maximum_absolute_log_error=old['maximum_absolute_difference'],
                maximum_tolerance_ratio=row['supplemental_fp32_log_prefix']['maximum_tolerance_ratio'],
                legacy_tolerance_violation_elements=old['tolerance_violation_elements'],supplemental_atol=1e-5,supplemental_rtol=1e-5))
    return dict(fixed_diagnostic_passes=True,legacy_short_prefix_check_failed=legacy_count > 0,
                legacy_total_mismatched_elements=legacy_count,actual_shape_change_checks=rows)


def verify_precision(root, checkpoint, audit):
    root, checkpoint = Path(root), Path(checkpoint)
    work = root.parents[1]
    audit_script = work/'push151_audit.py'
    helper_script = work/'push150_numerical_diagnostic.py'
    audit_sha = sha(audit_script)
    require(audit['audit_script_sha256'] == audit_sha == AUDIT_SCRIPT_SHA and sha(helper_script) == DIAGNOSTIC_SCRIPT_SHA,
            'Actual numerical audit/helper source changed')
    precision = audit['precision_diagnostic']
    expected = dict(fixed_diagnostic_passes=True,plan_sha256=PLAN_SHA,plan_file=PLAN,
        diagnostic_file='audit.diagnostic.json',diagnostic_script=helper_script.name,
        diagnostic_script_sha256=DIAGNOSTIC_SCRIPT_SHA,supplemental_audit_script=audit_script.name,
        supplemental_audit_script_sha256=audit_sha,legacy_failure_reclassified_as_pass=False,
        teacher_short_prefix_tolerance_not_specified=True,shape_change_tolerance=dict(atol=1e-5,rtol=1e-5),
        same_shape_tolerance_unchanged=dict(atol=4e-6,rtol=1e-6),reset_atol=0.,reset_rtol=0.,
        neither_predictor_nor_official_scorer_modified=True)
    require(all(precision.get(key) == value for key,value in expected.items()), 'Precision audit declaration differs')
    diagnostic_path = checkpoint.parent/'audit.diagnostic.json'
    require(sha(diagnostic_path) == precision['diagnostic_sha256'], 'Numerical diagnostic changed')
    diagnostic = read(diagnostic_path)
    expected = dict(status='completed',checkpoint_sha256=audit['checkpoint_sha256'],implementation='push150_expand',
        implementation_sha256=IMPLEMENTATION_SHA,plan_sha256=PLAN_SHA,numerical_audit_plan_sha256=PLAN_SHA,
        selection_sha256=audit['selection_sha256'],diagnostic_script_sha256=DIAGNOSTIC_SCRIPT_SHA,
        supplemental_audit_script_sha256=audit_sha,test_text_loaded_or_hashed=False,training_performed=False,
        official_scoring_performed=False,checkpoint_or_inference_sources_modified=False,
        validation_text_loaded=True,validation_text_used_by_diagnostic=False,requested_lengths=[1,2,3,19,63,128,255])
    require(all(diagnostic.get(key) == value for key,value in expected.items()), 'Diagnostic identity/source isolation differs')
    criteria = read(child(root,PLAN))['audit']
    require(diagnostic.get('criteria') == criteria and precision.get('criterion_note') == criteria['shape_tolerance_disclosure'],
            'Diagnostic did not use predeclared criteria')
    runtime = diagnostic['runtime']
    require(all(runtime.get(key) == value for key,value in dict(device='cpu',precision='fp32',threads=4,affinity_mask=15).items()),
            'Numerical diagnostic runtime differs')
    reference = diagnostic['fp64_reference']
    require(all(reference.get(key) is True for key in ('neural_pointer_float64','dropout_disabled',
        'original_fp32_rope_distance_buffers_cast_to_float64','count_table_values_original_fp32_cast_to_float64',
        'inference_implementation_unchanged','no_model_double_shortcut')), 'FP64 reference method differs')
    require(diagnostic['train_prefix']['split'] == 'train' and diagnostic['train_prefix']['used_tokens'] == 512,
            'Diagnostic fixture source differs')
    verify_manifest(root,diagnostic['source_sha256'])
    require(diagnostic['source_sha256'] == audit['source_sha256'] and
            all(audit['assets'].get(name,{}).get('sha256') == digest for name,digest in diagnostic['source_sha256'].items()),
            'Diagnostic runtime sources do not match audited assets')
    checks = check_diagnostic_batches(diagnostic['batches'])
    require(diagnostic.get('checks') == checks and all(precision.get(key) == value for key,value in checks.items()) and
            diagnostic['summary'].get('predeclared_supplemental_diagnostic_pass') is True,
            'Detailed diagnostic checks/summary differ')
    return ({relative(root,diagnostic_path):sha(diagnostic_path)},
            {audit_script.name:audit_sha,helper_script.name:DIAGNOSTIC_SCRIPT_SHA})


def verify_audit(root, checkpoint, winner):
    root, checkpoint = Path(root), Path(checkpoint)
    audit_path = checkpoint.parent/'audit.json'
    audit = read(audit_path)
    expected = dict(status='completed',checkpoint_sha256=winner['checkpoint_sha256'],implementation='push150_expand',
        implementation_sha256=IMPLEMENTATION_SHA,plan_sha256=PLAN_SHA,selection_sha256=sha(root/SELECTION),
        new_training_targets=0,full_validation_passes=1,fp64_reference_deleted_before_official_score=True,
        cpu_timing_ratio_not_measured_by_this_audit=True,all_limits_pass_not_claimed_without_separate_paired_timing=True)
    require(all(audit.get(key) == value for key,value in expected.items()) and audit.get('selected_candidate') == winner,
            'Actual audit identity/selection/ancestry differs')
    require(audit.get('test_used') is False and audit.get('evaluation_network_access') is False,
            'Audit used forbidden sources')
    for group, keys in [('contract_checks', CONTRACT_KEYS), ('fixed_file_checks', FIXED_KEYS)]:
        checks = audit.get(group, {})
        require(keys <= set(checks) and all(value is True for value in checks.values()), 'Audit contract failed: '+group)
    require(audit.get('asset_limit_pass') is True and audit.get('ram_limit_pass') is True and
            number(audit.get('peak_ram_gib')) and audit['peak_ram_gib'] <= 4 and
            audit['peak_process_working_set_bytes']/2**30 == audit['peak_ram_gib'], 'Audit resource limit failed')
    require(audit['train_targets_on_checkpoint'] == LINEAGE and number(audit.get('parameters')),
            'Audited selected lineage/parameters differ')
    check_score(audit['measurement'], 'validation')
    require(abs(audit['measurement']['bpb']-winner['validation_bpb_gpu_fp32']) <= 1e-5,
            'CPU prediction differs materially from selected GPU FP32 validation')
    assets = audit['assets']
    required_assets = {'checkpoint.pt', 'push150_expand.py', 'evaluate.py', 'common.py', 'data/tokenizer.json'}
    require(required_assets <= {name.replace('\\','/') for name in assets}, 'Inference asset graph incomplete')
    manifest = {relative(root, audit_path): sha(audit_path)}
    old_path = root/'runs/push150-oof-control-calibrated/audit.json'
    require(sha(old_path) == PARENT_AUDIT_SHA, 'Fixed parent inference asset manifest changed')
    old_assets = {name.replace('\\','/'):entry for name,entry in read(old_path)['assets'].items()}
    normalized_assets = {name.replace('\\','/'):entry for name,entry in assets.items()}
    require(len(normalized_assets) == len(assets) and set(normalized_assets) == set(old_assets) and
            all(normalized_assets[name] == entry for name,entry in old_assets.items() if name != 'checkpoint.pt'),
            'Audit omitted/modified a inherited runtime dependency')
    require(audit.get('asset_manifest_basis') == dict(original_audit=relative(root,old_path),
        original_audit_sha256=PARENT_AUDIT_SHA,identical_runtime_files_only_checkpoint_replaced=True),
        'Inference asset manifest provenance differs')
    manifest[relative(root,old_path)] = PARENT_AUDIT_SHA
    total = 0
    for name, asset in assets.items():
        path = checkpoint if name == 'checkpoint.pt' else child(root, name)
        require(sha(path) == asset['sha256'] and path.stat().st_size == asset['bytes'], 'Inference asset changed: '+name)
        manifest[relative(root, path)] = asset['sha256']
        total += asset['bytes']
    require(assets['checkpoint.pt']['sha256'] == winner['checkpoint_sha256'] and
            0 < total == audit['inference_assets_bytes'] <= 64*2**20 and
            total/2**20 == audit['inference_assets_mib'], 'Asset identity/sum/limit differs')
    require(audit['measurement'].get('device') == 'cpu' and audit['measurement'].get('precision') == 'fp32' and
            audit['measurement'].get('split') == 'validation' and audit['measurement'].get('test_used') is False and
            audit['measurement'].get('checkpoint_sha256') == winner['checkpoint_sha256'] and
            audit['measurement'].get('implementation_sha256') == IMPLEMENTATION_SHA and
            audit['measurement'].get('evaluator_sha256') == SCORER_SHA and
            audit['measurement'].get('tokenizer_sha256') == TOKENIZER_SHA, 'CPU audit scorer identity differs')
    verify_manifest(root,audit['source_sha256'])
    verify_manifest(root,audit['development_source_sha256'])
    manifest.update(audit['source_sha256'])
    manifest.update(audit['development_source_sha256'])
    diagnostic_files, work_files = verify_precision(root,checkpoint,audit)
    manifest.update(diagnostic_files)
    return audit, manifest, work_files


def preflight(root, selection_path, timing_path, unit_path):
    root = Path(root)
    selection, winner, manifest = verify_selection(root, selection_path)
    unit = verify_units(root, unit_path)
    checkpoint = child(root, winner['checkpoint'])
    audit, assets, work_files = verify_audit(root, checkpoint, winner)
    timing = read(timing_path)
    timing_summary, timing_files = verify_timing_files(root, timing_path, timing, winner, audit)
    fixed = {'evaluate.py': SCORER_SHA, 'data/tokenizer.json': TOKENIZER_SHA,
             'model.py': BASELINE_MODEL_SHA, 'runs/baseline-s17/checkpoint.pt': BASELINE_SHA,
             'push150_expand.py': IMPLEMENTATION_SHA, 'official_resource_run.py': OFFICIAL_WRAPPER_SHA,
             'push145_fresh_benchmark.py': TIMING_SCRIPT_SHA, 'measure_validation.py': MEASURE_SCRIPT_SHA}
    verify_manifest(root, fixed)
    manifest.update(fixed)
    manifest.update(assets)
    manifest.update(timing_files)
    manifest[UNIT] = unit['sha256']
    for path in [*root.glob('push151*.py'), *(root/'tests').glob('test_push151*.py')]:
        manifest[relative(root,path)] = sha(path)
    record = dict(round='push151', status='frozen_after_validation_resource_gates', plan_sha256=PLAN_SHA,
        selection_record=SELECTION, selection_record_sha256=sha(selection_path),
        selected_validation_record=winner, validation_rank=0, automatic_fallback_allowed=False,
        checkpoint=winner['checkpoint'], checkpoint_sha256=winner['checkpoint_sha256'],
        checkpoint_bytes=checkpoint.stat().st_size, implementation='push150_expand', implementation_sha256=IMPLEMENTATION_SHA,
        all_declared_A_B_C_accounted=True, selection_split='validation', test_bpb_at_freeze=None,
        this_predictor_test_scored_before_freeze=False, test_used_for_tuning=False,
        prior_other_predictors_test_exposure_disclosed=True, known_push150_test_bpb_disclosure_only=1.4580381311397583,
        prior_test_history_is_exhaustive=False, selected_lineage_targets=LINEAGE, train_processed_targets=LINEAGE,
        new_neural_gradient_targets=0, parameters=audit['parameters'],
        unit_tests=unit, audit_record=relative(root, checkpoint.parent/'audit.json'),
        audit_sha256=sha(checkpoint.parent/'audit.json'), timing_record=relative(root,timing_path),
        timing_record_sha256=sha(timing_path), validation_timing=timing_summary,
        validation_bpb_cpu_fp32=audit['measurement']['bpb'],
        inference_assets_bytes=audit['inference_assets_bytes'], inference_assets_mib=audit['inference_assets_mib'],
        validation_audit_peak_ram_gib=audit['peak_ram_gib'], contract_checks_all_pass=True,
        numerical_tolerance_is_supplemental_not_teacher_authorization=True,
        official_scorer_unchanged=True, official_test_wrapper_sha256=OFFICIAL_WRAPPER_SHA,
        baseline_checkpoint_sha256=BASELINE_SHA, baseline_implementation_sha256=BASELINE_MODEL_SHA,
        finalizer_source_sha256=sha(root/'push151_finalize.py'), evaluation_network_access=False,
        full_test_protocol='One official CPU FP32 candidate pass bracketed by fixed baseline passes; threads4 affinity15; no retry/fallback/post-test selection.',
        evidence_sha256=manifest, work_evidence_sha256=work_files)
    return checkpoint.parent/'candidate_frozen.json', record


def create_freeze(root, selection_path, timing_path, unit_path):
    root = Path(root)
    require(not child(root, FINAL_TEST).exists(), 'Final-test ledger already exists')
    path, record = preflight(root, selection_path, timing_path, unit_path)
    require(not path.exists(), 'Freeze already exists; never overwrite it')
    # Retain all small trial and launch evidence, including failures/partial work,
    # without ever opening data text or loading checkpoint contents.
    for directory in (*PHASE_DIRS, 'runs/push151-launches'):
        for evidence in child(root,directory).rglob('*'):
            if evidence.is_file() and evidence.suffix in ('.json','.npy') and evidence.name != 'candidate_frozen.json':
                record['evidence_sha256'][relative(root,evidence)] = sha(evidence)
    repair_path = path.parent/'audit_metadata_repair_plan.json'
    if repair_path.exists():
        repair = read(repair_path)
        require(repair.get('plan_sha256') == PLAN_SHA and
                repair.get('checkpoint_sha256') == record['checkpoint_sha256'] and
                repair.get('repaired_audit_source_sha256') == AUDIT_SCRIPT_SHA and
                all(repair.get(key) is False for key in ('failed_attempt_full_validation_scored',
                    'model_or_checkpoint_changed','audit_thresholds_changed','official_evaluator_changed','test_used')),
                'Metadata-only audit repair provenance differs')
        for key, digest_key in [('failed_started_record_preserved_as','failed_started_record_sha256'),
                                ('failed_record_preserved_as','failed_record_sha256')]:
            require(sha(child(path.parent,repair[key])) == repair[digest_key], 'Original audit failure changed')
        source = 'push151_audit.before_virtual_module_fix.py'
        original_sha = 'e4f3db2148fda1ba4134197052a5c30370a7aa7fd538520219e941f33e34c6d9'
        require(repair.get('original_audit_source') == 'work/'+source and
                repair.get('original_audit_source_sha256') == original_sha and
                sha(root.parents[1]/source) == original_sha, 'Original audit source was not preserved')
        record['work_evidence_sha256'][source] = original_sha
    record['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_new(path, record)
    return path, record


def verify_freeze(root, selection_path):
    root = Path(root)
    selection = read(selection_path)
    winner = ranked_candidate(selection)
    freeze_path = child(root,winner['checkpoint']).parent/'candidate_frozen.json'
    freeze = read(freeze_path)
    verify_manifest(root, freeze['evidence_sha256'])
    verify_manifest(root.parents[1],freeze['work_evidence_sha256'])
    path, current = preflight(root, selection_path, child(root,freeze['timing_record']), child(root,UNIT))
    require(path == freeze_path, 'Frozen candidate path differs')
    for key, value in current.items():
        if key in ('evidence_sha256','work_evidence_sha256'):
            require(all(freeze[key].get(name) == digest for name,digest in value.items()), 'Frozen evidence changed')
        else:
            require(freeze.get(key) == value, 'Frozen identity changed: '+key)
    require(freeze.get('frozen_at_utc'), 'Missing freeze time')
    return dict(freeze=freeze, freeze_path=freeze_path, winner=winner,
                audit=read(child(root,freeze['audit_record'])), checkpoint=child(root,winner['checkpoint']))


def check_measurement(measured, resource, checkpoint_sha, implementation_sha):
    check_score(measured, 'test', checkpoint_sha, implementation_sha, official=True)
    require(resource.get('cpu_affinity_mask') == 15 and number(resource.get('peak_process_working_set_bytes')),
            'Missing actual affinity/peak RAM evidence')


def test_summary(rows, audit):
    require([row['name'] for row in rows] == ['baseline-before','candidate','baseline-after'],
            'Incomplete ordered official test passes')
    candidate = rows[1]
    baselines = [rows[0]['seconds'],rows[2]['seconds']]
    mean_ratio = candidate['seconds']/statistics.mean(baselines)
    faster_ratio = candidate['seconds']/min(baselines)
    return dict(status='completed', measurements=rows, candidate_scoring_process_launches=1,
        baseline_scoring_process_launches=2, test_bpb=candidate['bpb'], candidate_seconds=candidate['seconds'],
        baseline_seconds=baselines, cpu_ratio_to_baseline_mean=mean_ratio,
        cpu_ratio_to_faster_baseline=faster_ratio, baseline_slow_to_fast_ratio=max(baselines)/min(baselines),
        peak_ram_gib=candidate['peak_ram_gib'], inference_assets_mib=audit['inference_assets_mib'],
        limits_pass=mean_ratio <= 5 and candidate['peak_ram_gib'] <= 4 and audit['inference_assets_mib'] <= 64,
        cpu_ratio_mean_limit_pass=mean_ratio <= 5, faster_baseline_ratio_reported_not_additional_gate=True,
        no_post_test_selection=True, no_automatic_retry=True)


def run_test(root, selection_path, run_dir):
    root, run_dir = Path(root), Path(run_dir)
    require(run_dir.resolve() == child(root,FINAL_TEST), 'Exactly one fixed final-test ledger directory')
    require(not run_dir.exists(), 'Do not retry/overwrite a final-test run')
    evidence = verify_freeze(root, selection_path)
    candidate_sha = evidence['winner']['checkpoint_sha256']
    specs = [('baseline-before',root/'runs/baseline-s17/checkpoint.pt',BASELINE_SHA,BASELINE_MODEL_SHA),
             ('candidate',evidence['checkpoint'],candidate_sha,IMPLEMENTATION_SHA),
             ('baseline-after',root/'runs/baseline-s17/checkpoint.pt',BASELINE_SHA,BASELINE_MODEL_SHA)]
    run_dir.mkdir(parents=True, exist_ok=False)
    started, rows, launches = time.perf_counter(), [], []
    identity = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=PLAN_SHA,
        selection_sha256=sha(selection_path), freeze_sha256=sha(evidence['freeze_path']),
        finalizer_source_sha256=sha(root/'push151_finalize.py'), official_wrapper_sha256=OFFICIAL_WRAPPER_SHA,
        evaluator_sha256=SCORER_SHA, tokenizer_sha256=TOKENIZER_SHA,
        candidate_sha256=candidate_sha, baseline_sha256=BASELINE_SHA,
        planned_passes=[dict(name=name,checkpoint=str(path),sha256=digest) for name,path,digest,_ in specs],
        candidate_pass_cap=1, device='cpu', precision='fp32', threads=4, affinity_mask=15,
        failure_policy='Preserve every launch/output/failure; no automatic retry or predictor changes.')
    try:
        write_new(run_dir/'launch_plan.json',identity)
        for index,(name,checkpoint,digest,implementation_sha) in enumerate(specs):
            verify_freeze(root, selection_path)
            require(sha(selection_path) == identity['selection_sha256'] and
                    sha(evidence['freeze_path']) == identity['freeze_sha256'] and
                    sha(root/'push151_finalize.py') == identity['finalizer_source_sha256'],
                    'Frozen orchestration evidence changed during final test')
            output = run_dir/(name+'.json')
            command = [sys.executable,'-B',str(root/'official_resource_run.py'),'--checkpoint',str(checkpoint),
                       '--split','test','--output',str(output)]
            write_new(run_dir/(name+'.started.json'),dict(index=index,name=name,
                started_at_utc=datetime.now(timezone.utc).isoformat(), command=command,
                launch_plan_sha256=sha(run_dir/'launch_plan.json'),checkpoint_sha256=digest))
            launches.append(name)
            process = subprocess.run(command,cwd=root,capture_output=True,text=True,encoding='utf-8',errors='replace')
            write_new(run_dir/(name+'.process.json'),dict(returncode=process.returncode,stdout=process.stdout,
                stderr=process.stderr,completed_at_utc=datetime.now(timezone.utc).isoformat()))
            require(process.returncode == 0, 'Official scoring process failed: '+name)
            resources_path = output.with_suffix('.resources.json')
            measured, resource = read(output),read(resources_path)
            check_measurement(measured,resource,digest,implementation_sha)
            require(Path(resource['evaluator_output']).resolve() == output.resolve(), 'Resource/output path differs')
            row = dict(**measured,name=name,output=output.name,output_sha256=sha(output),
                resources_sha256=sha(resources_path),peak_ram_gib=resource['peak_process_working_set_bytes']/2**30)
            rows.append(row)
            print(json.dumps(row,allow_nan=False),flush=True)
        verify_freeze(root,selection_path)
        require(sha(selection_path) == identity['selection_sha256'] and
                sha(evidence['freeze_path']) == identity['freeze_sha256'], 'Freeze changed during scoring')
        report = test_summary(rows,evidence['audit'])
        report.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),
            launch_plan_sha256=sha(run_dir/'launch_plan.json'),freeze_sha256=identity['freeze_sha256'],
            checkpoint_sha256=candidate_sha,seconds_total=time.perf_counter()-started)
        write_new(run_dir/'summary.json',report)
        return report
    except BaseException as error:
        write_new(run_dir/'failure.json',dict(status='failed',failure_type=type(error).__name__,
            failure_message=str(error),launches_started=launches,measurements_completed=rows,
            seconds_total=time.perf_counter()-started,no_automatic_retry=True,
            partial_test_exposure_must_be_disclosed=bool(launches)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    freeze = sub.add_parser('freeze')
    freeze.add_argument('--selection',type=Path,default=ROOT/SELECTION)
    freeze.add_argument('--timing',type=Path,default=ROOT/TIMING)
    freeze.add_argument('--unit-evidence',type=Path,default=ROOT/UNIT)
    test = sub.add_parser('test')
    test.add_argument('--selection',type=Path,default=ROOT/SELECTION)
    test.add_argument('--run-dir',type=Path,default=ROOT/FINAL_TEST)
    args = parser.parse_args()
    if args.command == 'freeze':
        path,record = create_freeze(ROOT,args.selection,args.timing,args.unit_evidence)
        print(json.dumps(dict(output=str(path),**record),indent=2,allow_nan=False),flush=True)
    else:
        print(json.dumps(run_test(ROOT,args.selection,args.run_dir),indent=2,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
