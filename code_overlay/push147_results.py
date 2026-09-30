"""Record results from completed frozen scoring; never load corpus or predictor."""
import json
import math
from datetime import datetime, timezone
from common import ROOT, sha


def read(path):
    return json.loads(path.read_text())


def main():
    selection = read(ROOT/'runs/push147_selection.json')
    winner = selection['winner']
    run_dir = ROOT/'runs'/winner['selected_run']
    audit = read(run_dir/'audit.json')
    freeze = read(run_dir/'candidate_frozen.json')
    assert sha(ROOT/'runs/push147_selection.json') == freeze['selection_record_sha256']
    assert sha(run_dir/'audit.json') == freeze['audit_sha256']
    test_dir = ROOT/'runs/push147-final-test'
    candidate = read(test_dir/'candidate.json')
    baseline_before = read(test_dir/'baseline-before.json')
    baseline_after = read(test_dir/'baseline-after.json')
    resources = read(test_dir/'candidate.resources.json')
    baseline_sha = sha(ROOT/'runs/baseline-s17/checkpoint.pt')
    for baseline in (baseline_before, baseline_after):
        assert baseline['checkpoint_sha256'] == baseline_sha
        assert baseline['precision'] == 'fp32' and baseline['device'] == 'cpu' and baseline['split'] == 'test'
        assert baseline['targets'] == candidate['targets'] and baseline['utf8_bytes'] == candidate['utf8_bytes']
        assert baseline['evaluator_sha256'] == candidate['evaluator_sha256']
        assert baseline['tokenizer_sha256'] == candidate['tokenizer_sha256']
    assert candidate['checkpoint_sha256'] == freeze['checkpoint_sha256'] == sha(run_dir/'checkpoint.pt')
    assert candidate['precision'] == 'fp32' and candidate['device'] == 'cpu' and candidate['split'] == 'test'
    assert candidate['targets'] == 428405 and candidate['utf8_bytes'] == 1292013
    assert sha(ROOT/'evaluate.py') == candidate['evaluator_sha256']
    assert sha(ROOT/'push147_signed_cache.py') == candidate['implementation_sha256']
    for name, asset in audit['assets'].items():
        assert sha(run_dir/'checkpoint.pt' if name == 'checkpoint.pt' else ROOT/name) == asset['sha256']
    mean_baseline = (baseline_before['seconds']+baseline_after['seconds'])/2
    ratio = candidate['seconds']/mean_baseline
    faster_ratio = candidate['seconds']/min(baseline_before['seconds'],baseline_after['seconds'])
    ram_gib = resources['peak_process_working_set_bytes']/2**30
    old_test = 1.4815738339536177
    control = next(row for row in selection['candidates'] if row['name'] == 'control')
    unsigned = next(row for row in selection['candidates'] if row['name'] == 'adaptive-cache')
    output_root = ROOT.parents[2]/'outputs/MP1_push147_tested_candidate'
    output_root.mkdir(parents=True,exist_ok=True)
    destination = output_root/'RESULTS.json'
    if destination.exists():
        raise FileExistsError('Do not overwrite a completed result record')
    result = dict(status='Frozen full-test scoring complete; resource results below apply to recorded runs, not a universal timing guarantee',
        result_recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        selected_run=winner['selected_run'], checkpoint_sha256=freeze['checkpoint_sha256'],
        implementation='push147_signed_cache',
        validation_bpb_cpu_fp32=audit['measurement']['bpb'],
        validation_bpb_gpu_fp32=winner['validation_bpb_gpu_fp32'],
        previous_candidate_validation_bpb_cpu_fp32=1.4646415447870387,
        previous_candidate_test_bpb_cpu_fp32=old_test,
        test_bpb_cpu_fp32=candidate['bpb'],test_candidate_scoring_passes=1,
        test_targets=candidate['targets'],test_utf8_bytes=candidate['utf8_bytes'],
        test_bpb_improvement=old_test-candidate['bpb'],
        test_relative_improvement_percent=(old_test-candidate['bpb'])/old_test*100,
        reached_1_45_test_bpb=candidate['bpb']<=1.45,
        temperature=winner['temperature'],statistical_weight=winner['mkn_weight'],
        cache_theta=15*math.exp(winner['best_training_snapshot_scalars']['cache_log_theta_delta']),
        cache_distance_coefficient=winner['best_training_snapshot_scalars']['cache_recency_strength'],
        parameters=audit['parameters'],
        comparisons=selection['candidates'],
        interpretation=dict(parent_to_control_validation_improvement=selection['parent_validation_bpb_gpu_fp32']-control['validation_bpb_gpu_fp32'],
            signed_vs_control_validation_improvement=control['validation_bpb_gpu_fp32']-winner['validation_bpb_gpu_fp32'],
            signed_vs_unsigned_validation_improvement=unsigned['validation_bpb_gpu_fp32']-winner['validation_bpb_gpu_fp32'],
            caveat='Single-seed results. Most improvement is continuation, not the two scalars. Rank changes initialization, learning rate and layout and is not an isolated architecture ablation.'),
        training=dict(inherited_targets=212992000,new_targets_on_selected_lineage=65536000,
            selected_total_targets=winner['selected_lineage_targets'],
            new_targets_for_all_runs=selection['all_new_search_targets'],runs=4,steps_per_run=8000,
            batch_size=32,context=256,seed=149,training_precision='bf16',
            total_loop_seconds=selection['total_training_loop_seconds'],
            loop_time_scope='Training plus checkpoint validations; excludes interpreter/device/data-loading setup, initialization and human/agent work',
            calibration_seconds=selection['total_calibration_seconds'],
            calibration_trials=60,raw_ema_validation_choices=36,
            statistical_tables='Reused frozen train-derived tables; no new table fitting or statistical corpus scan this round',
            rank_ancestry='Two parent paths with verified common dense ancestor; shared 98304000 targets counted once',
            original_plan_amended=True,auxiliary_loss_trained=False,
            external_text_or_pretrained_weights=False),
        assets=dict(bytes=audit['inference_assets_bytes'],mib=audit['inference_assets_mib'],limit_mib=64,**{'pass':audit['asset_limit_pass']}),
        validation_audit_ram=dict(bytes=audit['peak_process_working_set_bytes'],gib=audit['peak_ram_gib'],scope=audit['memory_scope']),
        full_test_ram=dict(bytes=resources['peak_process_working_set_bytes'],gib=ram_gib,scope=resources['includes'],limit_gib=4,**{'pass':ram_gib<=4}),
        full_test_timing=dict(protocol='Independent baseline-before, candidate, baseline-after processes',
            device='cpu',precision='fp32',threads=4,affinity_mask=15,
            candidate_seconds=candidate['seconds'],baseline_before_seconds=baseline_before['seconds'],
            baseline_after_seconds=baseline_after['seconds'],mean_baseline_seconds=mean_baseline,
            ratio_to_mean_baseline=ratio,ratio_to_faster_baseline=faster_ratio,limit_ratio=5,
            pass_in_recorded_full_test_run=ratio<=5),
        validation_timing=freeze['validation_timing'],
        prior_round_timing_variability='push146 included one validation pair above 5x; previous package remains intact and current evidence retains its freeze summary',
        all_three_resource_limits_pass_in_recorded_full_test_run=ratio<=5 and ram_gib<=4 and audit['asset_limit_pass'],
        contract_checks_all_pass=True,fixed_file_checks_all_pass=True,
        unittest_discovery=selection['unit_tests'],
        test_used_for_tuning=False,test_used_for_final_scoring_after_freeze=True,
        prior_test_exposure_disclosed=True,evaluation_network_access=False,
        prior_test_history_is_exhaustive=False,
        prior_test_exposure_scope='Freeze record lists selected recent prior scores, not every historical experiment. Earlier project variants were also tested; this project must not be described as having never seen test outcomes. This round developed and selected using train/validation only.',
        old_final_and_previous_candidates_overwritten=False,website_updated=False,
        complete_from_scratch_training_reproduction_bundle=False,
        scoring_reproduction_with_original_course_data_supported=True,
        missing_for_course_submission='Full <=10-page final report, finalized student identity, immutable code/checkpoint publication links and website submission; historical parent checkpoints are not in this scoring overlay.',
        result_script_sha256=sha(__file__))
    destination.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
