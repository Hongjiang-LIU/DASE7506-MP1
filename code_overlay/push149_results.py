"""Summarize existing frozen push149 evidence, without importing a model/scorer.

Only metadata, source files, checkpoints and prior archives are hashed. Corpus
text is never opened, and this program does not evaluate or train anything.
RESULTS.json is created exclusively, never overwritten.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
IMPLEMENTATION = 'push147_signed_cache'
PROTOCOL = '7506-mp1-wt2-v2'
PARENT_RUN = 'push148-muon-calibrated'
PARENT_SHA = '16c4638fe42a80e4b9465a31e1f3ecdb192c2c8426bcb4384992f4b968566f21'
PARENT_VIEWS = 344064000
PARENT_CPU_VALIDATION = 1.4493996958427136
PARENT_TEST = 1.4658432094844394
RUN_NAMES = ('control-adamw', 'muon-low')
FIXED_SCORER_SHA = '128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d'
FIXED_TOKENIZER_SHA = '020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e'
FIXED_IMPLEMENTATION_SHA = 'c20fc5411f63010cb9b3f3a9952f0c8aac851d55a8ee1c0ccc2ab5680ed901a3'
FIXED_BASELINE_MODEL_SHA = '6cce142a31d396f017319543f50d70f30fa6da740729af01c8baed13c6f671f9'
FIXED_BASELINE_CHECKPOINT_SHA = 'b7fdb67c8ef14b5166c0de1f17cf306f34800b41dfe97da856c9812cdb1f9701'
# Independent, previously recorded expectations, identical to package_push149.ps1.
PRESERVED_PACKAGES = {
    'outputs/MP1_FINAL_1.55249.zip': '7650fed64075b09a860c4637e61018e53a6890635a75c44f559c7dd6aa68b87d',
    'outputs/MP1_push145_tested_candidate.zip': 'c168aded897a7e93b4cfef190035bcd6a2d75f54722d4259c226006d8cac7d6c',
    'outputs/MP1_push146_tested_candidate.zip': '63e2d825c4c75c9e5bb0e6c45e779c0793d05af961ac93eb8691b4395db68d5f',
    'outputs/MP1_push147_tested_candidate.zip': '4be269238d2967a371c056c21fe773ebd945265e01ce8f55fb521bec470d18fb',
    'outputs/MP1_push148_tested_candidate.zip': '1e7ad29fa587f0b9ab616dcf33340fccf53813d5487a6b9698e262a851f242d2',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    path = Path(path)
    require(not path.name.startswith('wikitext_'), 'Corpus reads/hashes are prohibited here')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verified_hash(path, expected):
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), f'Invalid hash: {path}')
    require(sha(path) == expected, f'Hash mismatch: {path}')
    return expected


def child(root, relative):
    text = str(relative).replace('\\', '/')
    require(text and not text.startswith('/') and ':' not in text and
            all(part not in ('', '.', '..') for part in text.split('/')), f'Unsafe path: {relative}')
    result = (Path(root) / text).resolve()
    require(result.is_relative_to(Path(root).resolve()), f'Escaping path: {relative}')
    return result


def finite(value, label, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value) and (value > 0 if positive else value >= 0), f'Invalid {label}')
    return float(value)


def natural(value, label):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f'Invalid {label}')
    return value


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), f'Mismatch: {label}')


def verify_training_evidence(selection, plan, implementation_sha):
    rows = selection['candidates']
    require([row['name'] for row in rows] == list(RUN_NAMES), 'Incomplete candidate accounting')
    require([row['name'] for row in plan['runs']] == list(RUN_NAMES), 'Unexpected plan runs')
    require(selection['all_training_accounting_present'] is True, 'Missing actual training accounting')
    for row, recipe in zip(rows, plan['runs']):
        name = row['name']
        run_dir = ROOT/'runs'/('push149-' + name)
        path = run_dir/'metrics.json'
        verified_hash(path, row['training_metrics_sha256'])
        training = read(path)
        require(training['parent_sha256'] == PARENT_SHA and training['parent_train_targets'] == PARENT_VIEWS,
                f'Unexpected parent lineage: {name}')
        require(training['test_used'] is False and training['training_precision'] == 'bf16', f'Training protocol: {name}')
        require(training['training_implementation'] == IMPLEMENTATION and
                training['inference_implementation'] == IMPLEMENTATION and
                training['implementation_sha256'] == implementation_sha, f'Changed inference implementation: {name}')
        sources = training.get('source_sha256')
        require(sources is not None, 'Both runs must use the hashed existing push148 driver')
        require(bool(sources), f'Missing training source hashes: {name}')
        for source_name, source_sha in sources.items():
            require(re.fullmatch(r'[a-zA-Z0-9_]+\.py', source_name), 'Unexpected training source path')
            verified_hash(ROOT/source_name, source_sha)
        completed = natural(row['completed_steps'], 'completed steps')
        actual_completed = training.get('completed_steps', training['history'][-1]['step'] if training['history'] else 0)
        require(completed == actual_completed and completed <= training['steps'] <= recipe['steps'], f'Step accounting: {name}')
        views = natural(row['dropout_views_per_sample'], 'views')
        batch = natural(row['batch_size'], 'batch size')
        require(views == recipe['views'] and batch == recipe['batch_size'] == training['batch_size'], f'View/batch recipe: {name}')
        require(training.get('dropout_views_per_sample', 1) == views, f'Actual views: {name}')
        require(row['new_search_processed_view_targets'] == training['incremental_search_targets'] == completed*batch*views*256,
                f'Actual processed-view cost: {name}')
        require(row['new_search_sampled_label_occurrences'] == completed*batch*256, f'Actual sampled-label cost: {name}')
        if 'incremental_sampled_label_targets' in training:
            require(training['incremental_sampled_label_targets'] == completed*batch*256, f'Training sampled-label cost: {name}')
        close(row['training_loop_seconds'], finite(training['seconds'], 'training seconds'), f'Training time: {name}')
        choices = training['validation_history']
        require(row['raw_ema_validation_choices'] == len(choices) and
                row['raw_validation_choices'] == sum(item['weights'] == 'raw' for item in choices) and
                row['ema_validation_choices'] == sum(item['weights'] == 'ema' for item in choices), f'Validation choices: {name}')
        require(not row['evidence_errors'], f'Invalid selection evidence: {name}')
        terminal = row['terminal_evidence']
        if terminal['kind'] == 'explicit_termination_record':
            verified_hash(run_dir/'termination.json', terminal['sha256'])
        if not row['selection_eligible']:
            continue  # Keep failed/partial costs; never manufacture a calibrated score.
        calibration_dir = child(ROOT/'runs', row['selected_run'])
        verified_hash(calibration_dir/'metrics.json', row['calibration_metrics_sha256'])
        calibrated = read(calibration_dir/'metrics.json')
        verified_hash(run_dir/'checkpoint.pt', row['training_checkpoint_sha256'])
        verified_hash(calibration_dir/'checkpoint.pt', row['checkpoint_sha256'])
        require(calibrated['parent_sha256'] == row['training_checkpoint_sha256'] and
                calibrated['checkpoint_sha256'] == row['checkpoint_sha256'], f'Calibration ancestry: {name}')
        require(calibrated['test_used'] is False and calibrated['implementation'] == IMPLEMENTATION and
                calibrated['implementation_sha256'] == implementation_sha, f'Calibration implementation: {name}')
        verified_hash(ROOT/'push146_calibrate.py', calibrated['script_sha256'])
        require(calibrated['evaluator_sha256'] == FIXED_SCORER_SHA and calibrated['tokenizer_sha256'] == FIXED_TOKENIZER_SHA,
                f'Calibration scorer/tokenizer: {name}')
        close(row['validation_bpb_gpu_fp32'], calibrated['full_validation']['bpb'], f'Calibration score: {name}')
        require(row['calibration_choices'] == len(calibrated['trials']), f'Calibration choices: {name}')
        close(row['calibration_seconds'], finite(calibrated['seconds_total'], 'calibration seconds'), f'Calibration time: {name}')
        require(row['selected_lineage_targets'] == PARENT_VIEWS + row['selected_step']*batch*views*256 and
                row['selected_new_sampled_label_occurrences'] == row['selected_step']*batch*256, f'Selected lineage: {name}')
    for total_key, row_key in (
        ('all_new_search_processed_view_targets', 'new_search_processed_view_targets'),
        ('all_new_search_targets', 'new_search_processed_view_targets'),
        ('all_new_search_sampled_label_occurrences', 'new_search_sampled_label_occurrences'),
        ('total_raw_ema_validation_choices', 'raw_ema_validation_choices'),
        ('total_raw_validation_choices', 'raw_validation_choices'),
        ('total_ema_validation_choices', 'ema_validation_choices'),
        ('total_calibration_choices', 'calibration_choices'),
    ):
        require(selection[total_key] == sum(row[row_key] for row in rows), f'Actual accounting total: {total_key}')
    close(selection['total_training_loop_seconds'], sum(row['training_loop_seconds'] for row in rows), 'Total training time')
    recorded_calibration = sum(row['calibration_seconds'] for row in rows if row['calibration_seconds'] is not None)
    close(selection['total_recorded_calibration_seconds'], recorded_calibration, 'Recorded calibration time')
    close(selection['total_calibration_seconds'], recorded_calibration, 'Calibration time alias')
    require(selection['calibration_time_fully_recorded'] == all(row['calibration_seconds'] is not None for row in rows),
            'Calibration-time completeness flag')
    require(selection['total_calibration_choices'] <= plan['calibration']['max_settings'], 'Calibration budget exceeded')
    return rows


def comparison(rows, method, control):
    experiment, reference = rows[method], rows[control]
    available = experiment['selection_eligible'] and reference['selection_eligible']
    return dict(method=method, control=control, validation_comparison_available=available,
                validation_bpb_improvement=(reference['validation_bpb_gpu_fp32']-experiment['validation_bpb_gpu_fp32']) if available else None,
                completed_steps_method=experiment['completed_steps'], completed_steps_control=reference['completed_steps'],
                actual_processed_view_targets_match=experiment['new_search_processed_view_targets'] == reference['new_search_processed_view_targets'],
                actual_sampled_label_occurrences_match=experiment['new_search_sampled_label_occurrences'] == reference['new_search_sampled_label_occurrences'],
                caveat='Positive is improvement. Single seed; each run independently selected its raw/EMA step and calibration. Unequal completed work is not a matched final-step comparison.')


def main():
    output_root = PROJECT/'outputs/MP1_push149_tested_candidate'
    destination = output_root/'RESULTS.json'
    if destination.exists():
        raise FileExistsError('Never overwrite a completed result record')
    selection_path = ROOT/'runs/push149_selection.json'
    selection = read(selection_path)
    plan_path = ROOT/'runs/push149_plan.json'
    plan = read(plan_path)
    verified_hash(plan_path, selection['plan_sha256'])
    verified_hash(ROOT/'push149_select.py', selection['selection_script_sha256'])
    verified_hash(ROOT/'evaluate.py', FIXED_SCORER_SHA)
    verified_hash(ROOT/'data/tokenizer.json', FIXED_TOKENIZER_SHA)
    require(selection['selection_split'] == 'validation' and selection['test_used'] is False and
            selection['selection_status'] == 'new_candidate' and selection['passes_predeclared_improvement_gate'] is True and
            not selection['selection_integrity_errors'], 'Only a complete, validation-selected new winner may be reported here')
    require(selection['parent_checkpoint_sha256'] == plan['parent_sha256'] == PARENT_SHA and
            selection['parent_lineage_processed_view_targets'] == plan['parent_train_targets'] == PARENT_VIEWS, 'Parent lineage mismatch')
    verified_hash(ROOT/'runs'/PARENT_RUN/'checkpoint.pt', PARENT_SHA)
    close(plan['parent_validation_bpb_cpu_fp32'], PARENT_CPU_VALIDATION, 'Parent CPU validation')
    close(plan['known_parent_test_bpb_disclosure_only'], PARENT_TEST, 'Disclosed previous test score')
    winner = selection['winner']
    require(re.fullmatch(r'push149-[a-z0-9-]+', winner['selected_run']), 'Unexpected selected run')
    run_dir = child(ROOT/'runs', winner['selected_run'])
    audit_path, freeze_path = run_dir/'audit.json', run_dir/'candidate_frozen.json'
    audit, freeze = read(audit_path), read(freeze_path)
    verified_hash(selection_path, freeze['selection_record_sha256'])
    verified_hash(audit_path, freeze['audit_sha256'])
    verified_hash(ROOT/'push149_freeze.py', freeze['freeze_script_sha256'])
    require(freeze['selection_split'] == 'validation' and freeze['selected_validation_record'] == winner and
            freeze['test_used_for_tuning'] is False and freeze['this_predictor_test_scored_before_freeze'] is False and
            freeze['test_bpb_at_freeze'] is None, 'Freeze protocol mismatch')
    require(freeze['implementation'] == IMPLEMENTATION and freeze['inference_architecture_unchanged'] is True,
            'Inference architecture must remain push147_signed_cache')
    implementation_sha = verified_hash(ROOT/(IMPLEMENTATION+'.py'), freeze['implementation_sha256'])
    require(implementation_sha == FIXED_IMPLEMENTATION_SHA, 'The previous inference implementation changed')
    checkpoint_sha = verified_hash(run_dir/'checkpoint.pt', freeze['checkpoint_sha256'])
    require(checkpoint_sha == winner['checkpoint_sha256'] and
            (run_dir/'checkpoint.pt').stat().st_size == freeze['checkpoint_bytes'], 'Frozen checkpoint mismatch')
    require(audit['test_used'] is False and audit['evaluation_network_access'] is False, 'Audit protocol mismatch')
    for group in ('contract_checks', 'fixed_file_checks'):
        require(audit[group] and all(value is True for value in audit[group].values()), f'Audit failed: {group}')
    asset_bytes = 0
    for name, metadata in audit['assets'].items():
        path = run_dir/'checkpoint.pt' if name == 'checkpoint.pt' else child(ROOT, name)
        verified_hash(path, metadata['sha256'])
        require(path.stat().st_size == metadata['bytes'], f'Audited bytes changed: {name}')
        asset_bytes += metadata['bytes']
    require(audit['assets']['checkpoint.pt']['sha256'] == checkpoint_sha and asset_bytes == audit['inference_assets_bytes'],
            'Audit checkpoint/asset total mismatch')
    close(audit['inference_assets_mib'], asset_bytes/2**20, 'Audit MiB')
    require(audit['asset_limit_pass'] == (asset_bytes <= 64*2**20), 'Asset pass flag mismatch')
    close(audit['peak_ram_gib'], audit['peak_process_working_set_bytes']/2**30, 'Audit RAM')
    require(audit['ram_limit_pass'] == (audit['peak_ram_gib'] <= 4), 'Audit RAM flag mismatch')
    require(audit['asset_limit_pass'] is True and audit['ram_limit_pass'] is True, 'Frozen validation resource audit did not pass')
    require(audit['parameters'] == freeze['parameters'] == 4887699, 'Unexpected inference parameter count')
    require(audit['train_targets_on_checkpoint'] == freeze['train_processed_targets'] == winner['selected_lineage_targets'],
            'Checkpoint training-cost mismatch')
    close(audit['measurement']['bpb'], freeze['validation_bpb_cpu_fp32'], 'Frozen CPU validation')
    unit_tests = selection['unit_tests']
    require(natural(unit_tests['executions'], 'unit-test executions') > 0 and
            unit_tests['executions'] == freeze['unit_test_executions_passed'], 'Unit-test execution count mismatch')
    rows = verify_training_evidence(selection, plan, implementation_sha)
    eligible = [row for row in rows if row['selection_eligible']]
    require(winner == min(eligible, key=lambda row: row['validation_bpb_gpu_fp32']), 'Winner is not the validation minimum')
    require(selection['parent_validation_bpb_gpu_fp32']-winner['validation_bpb_gpu_fp32'] >= selection['minimum_validation_improvement'],
            'Winner did not pass predeclared validation gain')

    test_dir = ROOT/'runs/push149-final-test'
    scores, resources, evidence_hashes = {}, {}, {}
    baseline_sha = verified_hash(ROOT/'runs/baseline-s17/checkpoint.pt', FIXED_BASELINE_CHECKPOINT_SHA)
    model_sha = verified_hash(ROOT/'model.py', FIXED_BASELINE_MODEL_SHA)
    timing_path = ROOT/'runs/push149-fresh-validation-timing/summary.json'
    timing = read(timing_path)
    require(timing['test_used'] is False and timing['split'] == 'validation' and
            timing['device'] == 'cpu' and timing['precision'] == 'fp32' and
            timing['threads'] == 4 and timing['affinity_mask'] == 15, 'Validation timing protocol mismatch')
    require(timing['candidate_sha256'] == checkpoint_sha and timing['baseline_sha256'] == baseline_sha and
            timing['evaluator_sha256'] == FIXED_SCORER_SHA, 'Validation timing source hashes mismatch')
    verified_hash(ROOT/'push145_fresh_benchmark.py', timing['script_sha256'])
    validation_baseline = [finite(row['seconds'], 'validation baseline time', positive=True)
                           for row in timing['measurements'] if row['name'] == 'baseline']
    validation_candidate = [finite(row['seconds'], 'validation candidate time', positive=True)
                            for row in timing['measurements'] if row['name'] == 'candidate']
    require(len(validation_baseline) == len(validation_candidate) == 3 and len(timing['measurements']) == 6,
            'Need all three recorded validation timing pairs')
    require(freeze['validation_timing']['baseline_seconds'] == validation_baseline and
            freeze['validation_timing']['candidate_seconds'] == validation_candidate, 'Frozen validation times mismatch')
    validation_ratio = median(validation_candidate)/median(validation_baseline)
    close(timing['cpu_time_ratio'], validation_ratio, 'Validation ratio of medians')
    close(freeze['validation_timing']['ratio_of_medians'], validation_ratio, 'Frozen validation ratio')
    close(timing['conservative_time_ratio'], max(validation_candidate)/min(validation_baseline), 'Conservative validation ratio')
    require(validation_ratio <= 5, 'Validation timing gate did not pass')
    frozen_time = datetime.fromisoformat(freeze['frozen_at_utc']).timestamp()
    for label in ('baseline-before', 'candidate', 'baseline-after'):
        score_path = test_dir/(label+'.json')
        resource_path = test_dir/(label+'.resources.json')
        score, memory = read(score_path), read(resource_path)
        require(score['protocol'] == PROTOCOL and score['split'] == 'test' and
                score['precision'] == 'fp32' and score['device'] == 'cpu', f'Official scoring protocol: {label}')
        require(score['targets'] == 428405 and score['utf8_bytes'] == 1292013, f'Full-test coverage: {label}')
        require(score['evaluator_sha256'] == FIXED_SCORER_SHA and score['tokenizer_sha256'] == FIXED_TOKENIZER_SHA,
                f'Fixed scorer/tokenizer changed: {label}')
        require(score['checkpoint_sha256'] == (checkpoint_sha if label == 'candidate' else baseline_sha) and
                score['implementation_sha256'] == (implementation_sha if label == 'candidate' else model_sha), f'Scored predictor mismatch: {label}')
        finite(score['seconds'], 'scoring seconds', positive=True)
        close(finite(score['bpb'], 'BPB', positive=True), finite(score['nll_nats'], 'NLL', positive=True)/math.log(2)/1292013, f'BPB formula: {label}')
        require(memory['cpu_affinity_mask'] == 15 and
                memory['includes'] == 'entire evaluator process including all fixed dataset loading', f'Resource scope: {label}')
        output = Path(memory['evaluator_output'].replace('\\', '/'))
        output = output.resolve() if output.is_absolute() else (ROOT/output).resolve()
        require(output == score_path.resolve(), f'Resource record output mismatch: {label}')
        finite(memory['peak_process_working_set_bytes'], 'peak process bytes', positive=True)
        require(score_path.stat().st_mtime >= frozen_time, f'Score file predates freeze: {label}')
        scores[label], resources[label] = score, memory
        evidence_hashes[label] = {'score_sha256': sha(score_path), 'resources_sha256': sha(resource_path)}
    require((test_dir/'baseline-before.json').stat().st_mtime <= (test_dir/'candidate.json').stat().st_mtime <=
            (test_dir/'baseline-after.json').stat().st_mtime, 'Official bracket order is not reflected in source records')
    candidate = scores['candidate']
    before, after = scores['baseline-before']['seconds'], scores['baseline-after']['seconds']
    mean_baseline = (before+after)/2
    ratio, faster_ratio = candidate['seconds']/mean_baseline, candidate['seconds']/min(before, after)
    candidate_ram = resources['candidate']['peak_process_working_set_bytes']/2**30
    preserved = [dict(path=path, expected_sha256=verified_hash(PROJECT/path, expected), unchanged=True)
                 for path, expected in PRESERVED_PACKAGES.items()]
    by_name = {row['name']: row for row in rows}
    limits_pass = ratio <= 5 and candidate_ram <= 4 and asset_bytes <= 64*2**20
    result = dict(
        status='Frozen full-test scoring complete; recorded resource pass is not a universal timing guarantee',
        result_recorded_at_utc=datetime.now(timezone.utc).isoformat(), selected_run=winner['selected_run'],
        checkpoint_sha256=checkpoint_sha, implementation=IMPLEMENTATION, inference_architecture_unchanged=True,
        validation_bpb_cpu_fp32=audit['measurement']['bpb'], validation_bpb_gpu_fp32=winner['validation_bpb_gpu_fp32'],
        previous_candidate_validation_bpb_cpu_fp32=PARENT_CPU_VALIDATION, previous_candidate_test_bpb_cpu_fp32=PARENT_TEST,
        test_bpb_cpu_fp32=candidate['bpb'], test_candidate_scoring_passes=1, test_targets=428405, test_utf8_bytes=1292013,
        test_bpb_improvement=PARENT_TEST-candidate['bpb'], test_relative_improvement_percent=(PARENT_TEST-candidate['bpb'])/PARENT_TEST*100,
        reached_1_45_test_bpb=candidate['bpb'] <= 1.45, temperature=winner['temperature'], statistical_weight=winner['mkn_weight'],
        best_training_snapshot_scalars=winner['best_training_snapshot_scalars'], parameters=audit['parameters'], comparisons=rows,
        interpretation=dict(parent_to_adamw_control_validation_improvement=(selection['parent_validation_bpb_gpu_fp32']-by_name['control-adamw']['validation_bpb_gpu_fp32']) if by_name['control-adamw']['selection_eligible'] else None,
            muon_low_vs_adamw_control=comparison(by_name, 'muon-low', 'control-adamw'),
            caveat='Both routes restart optimization from the same parent EMA weights; momentum and EMA reset. This is a configured optimizer/LR comparison, not a pure algorithm ablation. Use validation only; do not extrapolate gains or guarantee a test target.'),
        training=dict(inherited_processed_view_targets=PARENT_VIEWS,
            new_processed_view_targets_on_selected_lineage=winner['selected_new_processed_view_targets'],
            new_sampled_label_occurrences_on_selected_lineage=winner['selected_new_sampled_label_occurrences'],
            selected_total_processed_view_targets=winner['selected_lineage_targets'],
            all_new_search_processed_view_targets=selection['all_new_search_processed_view_targets'],
            all_new_search_sampled_label_occurrences=selection['all_new_search_sampled_label_occurrences'],
            recorded_runs=len(rows), completed_runs=sum(row['recorded_status'] in ('completed', 'completed_inferred_from_history') for row in rows),
            failed_or_partial_runs=[row['name'] for row in rows if row['recorded_status'] not in ('completed', 'completed_inferred_from_history')],
            actual_per_run_accounting=rows, counts_are_actual_recorded_work_not_planned_caps=True,
            counts_are_unique_corpus_tokens=False, accounting_note=selection['accounting_note'],
            inherited_sampled_label_occurrences_not_rederived=True,
            training_precision='bf16', seed=plan['common']['seed'], context=plan['common']['context'],
            total_loop_seconds=selection['total_training_loop_seconds'],
            loop_time_scope='Recorded training plus checkpoint validation; excludes interpreter/device/data-loading setup, unrecorded failed work and human/agent effort.',
            recorded_calibration_seconds=selection['total_recorded_calibration_seconds'], calibration_time_fully_recorded=selection['calibration_time_fully_recorded'],
            calibration_trials=selection['total_calibration_choices'], raw_ema_validation_choices=selection['total_raw_ema_validation_choices'],
            raw_validation_choices=selection['total_raw_validation_choices'], ema_validation_choices=selection['total_ema_validation_choices'],
            statistical_tables='Inherited frozen train-derived MKN tables; no new table fitting this round', external_text_or_pretrained_weights=False,
            muon_attribution='Training-only adaptation of Keller Jordan Muon (MIT); full THIRD_PARTY_MUON_LICENSE.txt is required in the scoring overlay with training sources.'),
        assets=dict(bytes=asset_bytes, mib=asset_bytes/2**20, limit_mib=64, **{'pass':asset_bytes <= 64*2**20}),
        validation_audit_ram=dict(bytes=audit['peak_process_working_set_bytes'], gib=audit['peak_ram_gib'], scope=audit['memory_scope']),
        full_test_ram=dict(bytes=resources['candidate']['peak_process_working_set_bytes'], gib=candidate_ram,
            scope=resources['candidate']['includes'], limit_gib=4, **{'pass':candidate_ram <= 4}),
        full_test_all_resource_records=resources,
        full_test_timing=dict(protocol='Independent baseline-before, candidate, baseline-after official evaluator processes',
            device='cpu', precision='fp32', threads=4, affinity_mask=15, candidate_seconds=candidate['seconds'],
            baseline_before_seconds=before, baseline_after_seconds=after, mean_baseline_seconds=mean_baseline,
            ratio_to_mean_baseline=ratio, ratio_to_faster_baseline=faster_ratio,
            ratio_to_baseline_before=candidate['seconds']/before, ratio_to_baseline_after=candidate['seconds']/after,
            limit_ratio=5, pass_in_recorded_full_test_run=ratio <= 5, faster_baseline_ratio_within_limit=faster_ratio <= 5),
        validation_timing=freeze['validation_timing'], all_three_resource_limits_pass_in_recorded_full_test_run=limits_pass,
        contract_checks_all_pass=True, fixed_file_checks_all_pass=True, unittest_discovery=unit_tests,
        test_used_for_tuning=False, test_used_for_final_scoring_after_freeze=True, evaluation_network_access=False,
        prior_test_exposure_disclosed=True, prior_test_history_is_exhaustive=False,
        prior_test_exposure_scope='Recent freeze disclosures are not an exhaustive project test history. Earlier variants were tested. This round was developed and selected with train/validation only; no claim of never having seen test outcomes is made.',
        preserved_previous_packages=preserved, old_final_and_previous_candidates_overwritten=False, website_updated=False,
        complete_from_scratch_training_reproduction_bundle=False, scoring_reproduction_with_original_course_data_supported=True,
        missing_for_course_submission='Full <=10-page final report, finalized student identity, immutable code/checkpoint publication links and website submission; raw course texts and historical parent checkpoints are not included in this scoring overlay.',
        ai_assistance_disclosure='AI assisted experiment design, implementation, validation checks and documentation; student verification and final course disclosure remain required.',
        evidence_hashes=dict(selection=sha(selection_path), plan=sha(plan_path), audit=sha(audit_path), freeze=sha(freeze_path), validation_timing=sha(timing_path),
            official_score_records=evidence_hashes, official_resource_runner_current_source=sha(ROOT/'official_resource_run.py')),
        result_script_sha256=sha(__file__))
    # Run no evaluator and perform no retry. Exclusivity also covers a race with
    # another writer after the initial existence check.
    output_root.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
