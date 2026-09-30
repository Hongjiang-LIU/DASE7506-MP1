"""Freeze an audited push148 validation winner before a single candidate test."""
import argparse
import json
from datetime import datetime, timezone
from common import ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit-test-executions',required=True,type=int)
    args = parser.parse_args()
    selection_path = ROOT/'runs/push148_selection.json'
    selection = json.loads(selection_path.read_text())
    winner = selection['winner']
    run_dir = ROOT/'runs'/winner['selected_run']
    destination = run_dir/'candidate_frozen.json'
    if destination.exists():
        raise FileExistsError('Never overwrite a freeze record')
    audit = json.loads((run_dir/'audit.json').read_text())
    timing = json.loads((ROOT/'runs/push148-fresh-validation-timing/summary.json').read_text())
    assert selection['selection_split'] == 'validation' and not selection['test_used']
    assert selection['passes_predeclared_improvement_gate'] and args.unit_test_executions>0
    assert audit['asset_limit_pass'] and audit['ram_limit_pass']
    assert all(audit['contract_checks'].values()) and all(audit['fixed_file_checks'].values())
    assert timing['test_used'] is False and timing['cpu_time_ratio'] <= 5
    assert sha(run_dir/'checkpoint.pt') == winner['checkpoint_sha256'] == timing['candidate_sha256']
    for name,metadata in audit['assets'].items():
        path = run_dir/'checkpoint.pt' if name == 'checkpoint.pt' else ROOT/name
        assert sha(path) == metadata['sha256'] and path.stat().st_size == metadata['bytes']
    baseline = [row['seconds'] for row in timing['measurements'] if row['name']=='baseline']
    candidate = [row['seconds'] for row in timing['measurements'] if row['name']=='candidate']
    record = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        status='Frozen validation winner after resource audit, before one new candidate full-test pass',
        selection_split='validation',selection_record='../push148_selection.json',
        selection_record_sha256=sha(selection_path),checkpoint_sha256=winner['checkpoint_sha256'],
        checkpoint_bytes=(run_dir/'checkpoint.pt').stat().st_size,
        implementation='push147_signed_cache',implementation_sha256=sha(ROOT/'push147_signed_cache.py'),
        inference_architecture_unchanged=True,inference_asset_hashes='audit.json: assets',
        audit_sha256=sha(run_dir/'audit.json'),validation_bpb_cpu_fp32=audit['measurement']['bpb'],
        previous_candidate_validation_bpb_cpu_fp32=1.459526725695439,
        test_bpb_at_freeze=None,this_predictor_test_scored_before_freeze=False,
        prior_other_predictors_test_exposure_disclosed=True,
        recent_prior_test_bpb_for_disclosure_only=[1.4815738339536177,1.476132901503194],
        prior_test_history_is_exhaustive=False,
        selected_validation_record=winner,parameters=audit['parameters'],
        train_processed_targets=audit['train_targets_on_checkpoint'],
        contract_checks_all_pass=True,unit_test_executions_passed=args.unit_test_executions,
        inference_assets_mib=audit['inference_assets_mib'],validation_audit_peak_ram_gib=audit['peak_ram_gib'],
        validation_timing=dict(baseline_seconds=baseline,candidate_seconds=candidate,
            ratio_of_medians=timing['cpu_time_ratio'],
            max_candidate_divided_by_min_baseline=timing['conservative_time_ratio'],
            individual_pair_ratios=[c/b for b,c in zip(baseline,candidate)],
            all_records_retained=True,extra_conservative_gate_required=False),
        full_test_protocol_predeclared='One candidate full-test CPU FP32 pass through unchanged evaluate.py via official_resource_run.py, bracketed by independent fixed baseline passes; four threads and affinity15. Report all times and mean/faster-baseline ratios. No predictor changes or candidate retry after test.',
        test_used_for_tuning=False,evaluation_network_access=False,
        old_final_and_previous_candidates_preserved=True,freeze_script_sha256=sha(__file__))
    destination.write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    main()
