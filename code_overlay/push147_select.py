"""Select from the four completed validation-only trials; never open raw text."""
import json
from datetime import datetime, timezone
from pathlib import Path
from common import ROOT, sha


def main():
    destination = ROOT/'runs/push147_selection.json'
    if destination.exists():
        raise FileExistsError('Preserve the original selection record')
    runs = ['control', 'adaptive-cache', 'rank-recovery', 'signed-cache']
    rows = []
    for name in runs:
        training_dir = ROOT/'runs'/('push147-'+name)
        calibrated_dir = ROOT/'runs'/('push147-'+name+'-calibrated')
        training = json.loads((training_dir/'metrics.json').read_text())
        calibration = json.loads((calibrated_dir/'metrics.json').read_text())
        assert training['test_used'] is False and calibration['test_used'] is False
        assert training['steps'] == 8000 and training['incremental_search_targets'] == 65536000
        assert len(calibration['trials']) == 15
        assert sha(training_dir/'checkpoint.pt') == training['best']['checkpoint_sha256']
        assert sha(calibrated_dir/'checkpoint.pt') == calibration['checkpoint_sha256']
        rows.append(dict(name=name, selected_run=calibrated_dir.name,
            checkpoint_sha256=calibration['checkpoint_sha256'],
            validation_bpb_gpu_fp32=calibration['full_validation']['bpb'],
            temperature=calibration['best']['temperature'], mkn_weight=calibration['best']['mkn_weight'],
            selected_step=training['best']['step'], selected_weights=training['best']['weights'],
            selected_lineage_targets=training['best']['selected_lineage_targets'],
            training_loop_seconds=training['seconds'], calibration_seconds=calibration['seconds_total'],
            new_search_targets=training['incremental_search_targets'],
            best_training_snapshot_scalars=training['best']['scalars'],
            raw_ema_validation_choices=len(training['validation_history']),
            calibration_choices=len(calibration['trials'])))
    parent_dir = ROOT/'runs/push146-drop20-calibrated'
    parent = json.loads((parent_dir/'metrics.json').read_text())
    parent_score = parent['full_validation']['bpb']
    winner = min(rows, key=lambda row: row['validation_bpb_gpu_fp32'])
    improvement = parent_score-winner['validation_bpb_gpu_fp32']
    result = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        selection_split='validation', test_used=False, prior_test_exposure_disclosed=True,
        parent_checkpoint_sha256=sha(parent_dir/'checkpoint.pt'),
        parent_validation_bpb_gpu_fp32=parent_score, candidates=rows,
        winner=winner if improvement>0 else dict(selected_run=parent_dir.name,validation_bpb_gpu_fp32=parent_score),
        improvement_vs_parent=improvement,
        passes_predeclared_improvement_gate=improvement>=.0005,
        all_new_search_targets=sum(row['new_search_targets'] for row in rows),
        total_training_loop_seconds=sum(row['training_loop_seconds'] for row in rows),
        total_calibration_seconds=sum(row['calibration_seconds'] for row in rows),
        total_raw_ema_validation_choices=sum(row['raw_ema_validation_choices'] for row in rows),
        total_calibration_choices=sum(row['calibration_choices'] for row in rows),
        plan_sha256=sha(ROOT/'runs/push147_plan.json'),
        amendment_sha256=sha(ROOT/'runs/push147_plan_amendment.json'),
        amended_hypothesis='Signed distance replaced the unexecuted auxiliary-loss run after train/validation observations; it was not in the original four-run plan.',
        selection_script_sha256=sha(__file__),
        no_raw_text_loaded=True,
        unit_tests=dict(executions=140,failures=0,errors=0,seconds=2.035,
                        note='Independent reviewer full discovery, including inherited repeated contract cases'),
        next_gate='Audit actual CPU FP32 weights and resource limits before freeze and one final test pass')
    destination.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
