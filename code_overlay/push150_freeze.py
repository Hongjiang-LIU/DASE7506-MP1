"""Freeze the complete validation-selected push150 method after CPU audit.

No dataset text is opened or hashed. All declared phase accounting must already
have passed the overall selector; existing freeze records are never overwritten.
"""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import statistics

from push150_final_test import ROOT, BASELINE_SHA, BASELINE_MODEL_SHA, SCORER_SHA, TOKENIZER_SHA
from push150_final_test import child, read, require, sha, write_new


def check_timing(timing, winner):
    expected = dict(test_used=False, split='validation', device='cpu', precision='fp32',
        threads=4, affinity_mask=15, candidate_sha256=winner['checkpoint_sha256'],
        baseline_sha256=BASELINE_SHA, evaluator_sha256=SCORER_SHA)
    require(all(timing.get(key) == value for key, value in expected.items()), 'Timing identity differs')
    rows = timing['measurements']
    require([(row['repeat'], row['name']) for row in rows] ==
            [(0, 'baseline'), (0, 'candidate'), (1, 'candidate'), (1, 'baseline'),
             (2, 'baseline'), (2, 'candidate')], 'Need all three prescribed fresh alternating pairs')
    times = {}
    for name in ('baseline', 'candidate'):
        values = [row['seconds'] for row in rows if row['name'] == name]
        require(all(isinstance(value, (float, int)) and math.isfinite(value) and value > 0
                    for value in values), 'Invalid scoring duration')
        times[name] = values
    ratio = statistics.median(times['candidate'])/statistics.median(times['baseline'])
    require(abs(ratio-timing['cpu_time_ratio']) < 1e-12 and ratio <= 5, 'Median CPU ratio exceeds five')
    require(all(row['targets'] == 376599 and row['utf8_bytes'] == 1148007 for row in rows), 'Incomplete validation timing')
    require(max(row['peak_process_working_set_bytes'] for row in rows if row['name'] == 'candidate') <= 4*2**30,
            'Timed candidate exceeded RAM budget')
    return dict(baseline_seconds=times['baseline'], candidate_seconds=times['candidate'],
        ratio_of_medians=ratio,
        individual_pair_ratios=[c/b for b, c in zip(times['baseline'], times['candidate'])],
        max_candidate_divided_by_min_baseline=max(times['candidate'])/min(times['baseline']),
        all_records_retained=True, extra_conservative_gate_required=False)


def create_record(root, selection_path, timing_path):
    root = Path(root)
    selection = read(selection_path)
    require(selection.get('selection_split') == 'validation' and selection.get('test_used') is False and
            selection.get('all_declared_phases_accounted') is True and
            selection.get('passes_predeclared_improvement_gate') is True and
            not selection.get('selection_integrity_errors'), 'Overall validation selection is incomplete or invalid')
    require(selection.get('evidence_sha256'), 'Missing complete selection evidence manifest')
    for name, digest in selection['evidence_sha256'].items():
        require(sha(child(root, name)) == digest, 'Selection evidence changed: '+name)
    unit = selection['unit_tests']
    unit_path = child(root, unit['path'])
    require(sha(unit_path) == unit['sha256'], 'Unit evidence changed')
    unit_record = read(unit_path)
    require(unit_record.get('successful') is True and unit_record['tests_run'] == unit['executions'] > 0 and
            not unit_record['failures'] and not unit_record['errors'], 'Unit tests did not pass')
    for group in ('source_sha256', 'unit_source_sha256'):
        for name, digest in unit_record[group].items():
            source = root/name if group == 'source_sha256' else root/'tests'/name
            require(sha(source) == digest, 'A tested push150 source changed: '+name)
    winner = selection['winner']
    require(re.fullmatch(r'push150-[a-z0-9-]+', winner['selected_run']), 'Expected an improved new push150 candidate')
    run_dir = child(root/'runs', winner['selected_run'])
    checkpoint = run_dir/'checkpoint.pt'
    require(sha(checkpoint) == winner['checkpoint_sha256'], 'Selected checkpoint changed')
    audit_path = run_dir/'audit.json'
    audit = read(audit_path)
    require(audit.get('test_used') is False and audit.get('evaluation_network_access') is False,
            'Audit used a forbidden source')
    for group in ('contract_checks', 'fixed_file_checks'):
        require(audit.get(group) and all(value is True for value in audit[group].values()), 'Audit check failed: '+group)
    require(audit.get('asset_limit_pass') is True and audit.get('ram_limit_pass') is True and
            0 < audit['peak_ram_gib'] <= 4, 'Actual audit resource limits failed')
    total = 0
    for name, asset in audit['assets'].items():
        path = checkpoint if name == 'checkpoint.pt' else child(root, name)
        require(sha(path) == asset['sha256'] and path.stat().st_size == asset['bytes'], 'Inference asset changed: '+name)
        total += asset['bytes']
    require(0 < total == audit['inference_assets_bytes'] <= 64*2**20 and
            abs(total/2**20-audit['inference_assets_mib']) < 1e-12, 'Inference asset sum differs/exceeds limit')
    require(audit['assets']['checkpoint.pt']['sha256'] == winner['checkpoint_sha256'], 'Audit checkpoint mismatch')
    require(audit['train_targets_on_checkpoint'] == winner['selected_lineage_targets'], 'Audit selected lineage differs')
    require(abs(audit['measurement']['bpb']-winner['validation_bpb_gpu_fp32']) < 1e-5,
            'Actual CPU prediction differs materially from selected FP32 validation result')
    timing = read(timing_path)
    timing_summary = check_timing(timing, winner)
    require(all(abs(row['bpb']-audit['measurement']['bpb']) < 1e-10
                for row in timing['measurements'] if row['name'] == 'candidate'), 'Fresh timing candidate scores differ')
    implementation = winner['implementation']
    require(re.fullmatch(r'[a-z][a-z0-9_]*', implementation), 'Unsafe implementation module')
    for name, expected in [('evaluate.py', SCORER_SHA), ('data/tokenizer.json', TOKENIZER_SHA),
                           ('model.py', BASELINE_MODEL_SHA),
                           ('runs/baseline-s17/checkpoint.pt', BASELINE_SHA)]:
        require(sha(root/name) == expected, 'Fixed protocol asset changed: '+name)
    require(not (root/'runs/push150-final-test').exists(), 'Final test ledger already exists')
    record = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        status='Complete A/B/C/D validation winner frozen after actual CPU audit and three fresh timing pairs',
        all_declared_phases_accounted=True, selection_split='validation',
        selection_record=Path(selection_path).resolve().relative_to(root.resolve()).as_posix(),
        selection_record_sha256=sha(selection_path), selected_validation_record=winner,
        checkpoint_sha256=winner['checkpoint_sha256'], checkpoint_bytes=checkpoint.stat().st_size,
        implementation=implementation, implementation_sha256=sha(root/(implementation+'.py')),
        inference_asset_hashes='audit.json:assets', audit_sha256=sha(audit_path),
        timing_record=Path(timing_path).resolve().relative_to(root.resolve()).as_posix(),
        timing_record_sha256=sha(timing_path), validation_timing=timing_summary,
        validation_bpb_cpu_fp32=audit['measurement']['bpb'],
        original_push149_validation_bpb_cpu_fp32=1.4472849220037163,
        test_bpb_at_freeze=None, this_predictor_test_scored_before_freeze=False,
        prior_other_predictors_test_exposure_disclosed=True, known_push149_test_bpb_disclosure_only=1.4635140025181628,
        prior_test_history_is_exhaustive=False, test_used_for_tuning=False,
        parameters=audit['parameters'], train_processed_targets=audit['train_targets_on_checkpoint'],
        inference_assets_mib=audit['inference_assets_mib'], validation_audit_peak_ram_gib=audit['peak_ram_gib'],
        contract_checks_all_pass=True, unit_test_executions_passed=unit['executions'],
        all_search_costs_recorded_in_selection=True, evaluation_network_access=False,
        full_test_protocol='One unchanged official CPU FP32 candidate pass bracketed by fixed baseline passes, four threads/affinity15. Every launch is recorded. No post-test tuning, selection or automatic retry.',
        final_test_runner_sha256=sha(root/'push150_final_test.py'),
        official_test_wrapper_sha256=sha(root/'official_resource_run.py'),
        baseline_implementation_sha256=BASELINE_MODEL_SHA, freeze_script_sha256=sha(__file__))
    return run_dir/'candidate_frozen.json', record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=ROOT/'runs/push150_selection.json')
    parser.add_argument('--timing', type=Path, default=ROOT/'runs/push150-fresh-validation-timing/summary.json')
    args = parser.parse_args()
    destination, record = create_record(ROOT, args.selection, args.timing)
    write_new(destination, record)
    import json
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
