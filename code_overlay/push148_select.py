"""Evidence-based selection of at most five push148 validation-only runs.

No model is instantiated and no dataset text is opened. Partial/failed runs are
retained, counts are recorded occurrences (never deduplicated unique tokens),
and an invalid hash/lineage/accounting record prevents a selection decision.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from common import ROOT, sha

RUNS = ('control-single', 'control-dual', 'rdrop010', 'rdrop050', 'muon')
PARENT_RUN = 'push147-signed-cache-calibrated'
PARENT_VIEW_TARGETS = 278528000
MIN_GAIN = .0005
GRID = {(t, w) for t in (1.2, 1.25, 1.3, 1.35, 1.4) for w in (.1, .15, .2)}


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def natural(value, name):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f'Invalid {name}')
    return value


def finite(value, name):
    require(isinstance(value, (int, float)) and math.isfinite(value), f'Invalid {name}')
    return float(value)


def run_accounting(training, name, declared, parent_sha):
    require(training.get('test_used') is False, 'Training does not certify test_used=false')
    require(training['parent_sha256'] == parent_sha, 'Training parent hash mismatch')
    require(training['parent_train_targets'] == PARENT_VIEW_TARGETS, 'Unexpected parent lineage target count')
    planned = natural(training['steps'], 'planned steps')
    require(planned <= declared['steps'], 'Run exceeds its predeclared step cap')
    history = training.get('history', [])
    history_steps = [natural(row['step'], 'history step') for row in history]
    require(history_steps == sorted(history_steps), 'Training history is not chronological')
    legacy = name == 'control-single' and 'completed_steps' not in training
    if legacy:
        completed = history_steps[-1] if history_steps else 0
        views = 1
        require('dropout_views_per_sample' not in training or training['dropout_views_per_sample'] == 1, 'Legacy control has unexpected views')
        status = 'completed_inferred_from_history' if completed == planned else 'partial_inferred_from_history'
    else:
        completed = natural(training['completed_steps'], 'completed steps')
        views = natural(training['dropout_views_per_sample'], 'dropout views')
        status = training['status']
        require(status in ('completed', 'failed', 'partial', 'running'), 'Unknown run status')
        require(status != 'completed' or completed == planned, 'Completed status contradicts completed steps')
    batch = natural(training['batch_size'], 'batch size')
    require(views == declared['views'] and batch == declared['batch_size'], 'Batch/view recipe differs from plan')
    require(0 <= completed <= planned and (not history_steps or history_steps[-1] <= completed), 'Impossible completed step count')
    view_per_step, label_per_step = batch*views*256, batch*256
    require(view_per_step == 8192, 'Processed view targets per step changed')
    processed, sampled = completed*view_per_step, completed*label_per_step
    require(training['incremental_search_targets'] == processed, 'Recorded processed-view target count mismatch')
    if not legacy:
        require(training['incremental_sampled_label_targets'] == sampled, 'Recorded sampled-label target count mismatch')
    choices = training.get('validation_history', [])
    require(all(natural(row['step'], 'validation step') <= completed for row in choices), 'Validation comes after recorded completed work')
    require(all(row['weights'] in ('raw', 'ema') for row in choices), 'Unknown validation snapshot kind')
    seconds = finite(training['seconds'], 'training seconds')
    require(seconds >= 0, 'Negative training duration')
    return dict(planned_step_cap=planned, completed_steps=completed, recorded_status=status,
                legacy_driver_accounting=legacy, dropout_views_per_sample=views, batch_size=batch,
                processed_view_targets_per_step=view_per_step, sampled_label_occurrences_per_step=label_per_step,
                new_search_processed_view_targets=processed, new_search_sampled_label_occurrences=sampled,
                training_loop_seconds=seconds, raw_ema_validation_choices=len(choices),
                raw_validation_choices=sum(row['weights'] == 'raw' for row in choices),
                ema_validation_choices=sum(row['weights'] == 'ema' for row in choices),
                failed_attempt_cost_note=training.get('failed_attempt_cost_note'),
                failure_type=training.get('failure_type'), failure_message=training.get('failure_message'),
                status_note='running is only a last-recorded status, not a claim that its process completed' if status == 'running' else None)


def calibration_count(calibration):
    trials = calibration['trials']
    require(len(trials) == 15, 'A calibrated candidate must contain all 15 declared settings')
    require({(row['temperature'], row['mkn_weight']) for row in trials} == GRID, 'Calibration grid differs from plan')
    return len(trials)


def require_terminal_evidence(training_dir, training, row):
    status = row['recorded_status']
    if status in ('completed', 'completed_inferred_from_history'):
        return dict(kind=status)
    if status == 'failed':
        require(bool(training.get('failure_type')) and isinstance(training.get('failure_message'), str), 'Failed status lacks an explicit recorded exception')
        return dict(kind='recorded_failure', failure_type=training['failure_type'])
    # A stale running/partial log is NOT proof a run was intentionally stopped.
    # An optional explicit termination record must bind to these exact metrics.
    path = training_dir/'termination.json'
    require(path.exists(), 'Running/partial run lacks explicit stop/cancel evidence; refuse premature selection')
    record = read_json(path)
    require(record.get('status') in ('stopped', 'cancelled', 'partial'), 'Unknown explicit termination status')
    require(record.get('completed_steps') == row['completed_steps'], 'Termination step count mismatch')
    require(record.get('metrics_sha256') == sha(training_dir/'metrics.json'), 'Termination record does not bind current metrics')
    require(bool(record.get('reason')), 'Termination record lacks a reason')
    return dict(kind='explicit_termination_record', path=str(path), sha256=sha(path), **record)


def collect_selection(root, unit_test_executions):
    root = Path(root)
    natural(unit_test_executions, 'unit-test executions')
    require(unit_test_executions > 0, 'Supply the actual nonzero executed unit-test count')
    plan_path = root/'runs/push148_plan.json'
    plan = read_json(plan_path)
    require([row['name'] for row in plan['runs']] == list(RUNS), 'Plan does not contain exactly the five declared runs')
    require(plan['max_training_runs'] == 5 and plan['calibration']['max_settings'] == 75, 'Unexpected search caps')
    parent_dir = root/'runs'/PARENT_RUN
    parent = read_json(parent_dir/'metrics.json')
    parent_sha = sha(parent_dir/'checkpoint.pt')
    require(parent_sha == parent['checkpoint_sha256'] == plan['parent_sha256'], 'Parent checkpoint hash mismatch')
    require(plan['parent_train_targets'] == PARENT_VIEW_TARGETS, 'Plan parent target lineage changed')
    require(parent.get('test_used') is False, 'Parent selection metadata does not certify validation-only selection')
    parent_score = finite(parent['full_validation']['bpb'], 'parent full-validation BPB')
    rows, integrity_errors = [], []
    for name, declared in zip(RUNS, plan['runs']):
        train_dir = root/'runs'/('push148-'+name)
        calibration_dir = root/'runs'/('push148-'+name+'-calibrated')
        row = dict(name=name, selected_run=calibration_dir.name, selection_eligible=False,
                   evidence_errors=[], calibration_choices=0, calibration_seconds=None)
        rows.append(row)
        try:
            training = read_json(train_dir/'metrics.json')
            row.update(run_accounting(training, name, declared, parent_sha))
            row['training_metrics_sha256'] = sha(train_dir/'metrics.json')
            row['terminal_evidence'] = require_terminal_evidence(train_dir, training, row)
            if not (calibration_dir/'metrics.json').exists():
                partial_trials = calibration_dir/'trials.json'
                if partial_trials.exists():
                    entries = read_json(partial_trials)
                    require(isinstance(entries, list) and len(entries) <= 15, 'Partial calibration exceeds declared settings')
                    row['calibration_choices'] = len(entries)
                require(row['recorded_status'] not in ('completed', 'completed_inferred_from_history'),
                        'Completed training lacks its calibration result; refuse selection until evidence is complete')
                row['selection_rejection'] = 'No completed calibration metrics/full-scoring result; this run cannot be selected.'
                continue
            calibration = read_json(calibration_dir/'metrics.json')
            row['calibration_choices'] = calibration_count(calibration)
            row['calibration_seconds'] = finite(calibration['seconds_total'], 'calibration seconds')
            require(row['calibration_seconds'] >= 0, 'Negative calibration duration')
            require(calibration.get('test_used') is False, 'Calibration does not certify test_used=false')
            best = training['best']
            trained_sha = sha(train_dir/'checkpoint.pt')
            require(trained_sha == best['checkpoint_sha256'], 'Training checkpoint hash mismatch')
            require(calibration['parent_sha256'] == trained_sha, 'Calibration parent hash mismatch')
            output_sha = sha(calibration_dir/'checkpoint.pt')
            require(output_sha == calibration['checkpoint_sha256'], 'Calibrated checkpoint hash mismatch')
            selected_step = natural(best['step'], 'selected step')
            require(selected_step <= row['completed_steps'], 'Selected checkpoint exceeds completed updates')
            require(best['weights'] in ('raw', 'ema'), 'Unknown selected snapshot kind')
            require(any(item['step'] == selected_step and item['weights'] == best['weights'] for item in training['validation_history']), 'Selected snapshot absent from validation history')
            lineage = natural(best['selected_lineage_targets'], 'selected lineage targets')
            require(lineage == PARENT_VIEW_TARGETS+selected_step*row['processed_view_targets_per_step'], 'Selected lineage target count mismatch')
            selected_labels = selected_step*row['sampled_label_occurrences_per_step']
            if 'new_selected_sampled_label_targets' in best:
                require(best['new_selected_sampled_label_targets'] == selected_labels, 'Selected label occurrences mismatch')
            full_score = finite(calibration['full_validation']['bpb'], 'full-validation BPB')
            shortcut = finite(calibration['best']['validation_bpb'], 'shortcut BPB')
            require(abs(full_score-shortcut) < 1e-6, 'Shortcut was not confirmed by actual full prediction')
            choice = (calibration['best']['temperature'], calibration['best']['mkn_weight'])
            require(choice in GRID, 'Chosen calibration is outside declared grid')
            require(calibration.get('precision') == 'fp32' and str(calibration.get('device', '')).startswith('cuda'), 'Expected GPU FP32 full-validation confirmation')
            row.update(selection_eligible=True, checkpoint_sha256=output_sha,
                       training_checkpoint_sha256=trained_sha, calibration_metrics_sha256=sha(calibration_dir/'metrics.json'),
                       validation_bpb_gpu_fp32=full_score, temperature=choice[0], mkn_weight=choice[1],
                       selected_step=selected_step, selected_weights=best['weights'],
                       selected_lineage_targets=lineage, selected_new_processed_view_targets=lineage-PARENT_VIEW_TARGETS,
                       selected_new_sampled_label_occurrences=selected_labels,
                       new_search_targets=row['new_search_processed_view_targets'],
                       best_training_snapshot_scalars=best.get('scalars', {}))
        except (KeyError, TypeError, ValueError, OSError) as error:
            row['evidence_errors'].append(str(error))
            row['selection_rejection'] = 'Invalid or incomplete integrity/accounting evidence; selection refused.'
            integrity_errors.append(dict(name=name, error=str(error)))
    require(sum(row['calibration_choices'] for row in rows) <= 75, 'Calibration budget exceeded')
    eligible = [row for row in rows if row['selection_eligible']]
    candidate = min(eligible, key=lambda row: row['validation_bpb_gpu_fp32']) if eligible else None
    improvement = parent_score-candidate['validation_bpb_gpu_fp32'] if candidate else None
    parent_winner = dict(selected_run=PARENT_RUN, checkpoint_sha256=parent_sha,
                         validation_bpb_gpu_fp32=parent_score, selected_lineage_targets=PARENT_VIEW_TARGETS,
                         temperature=parent['best']['temperature'], mkn_weight=parent['best']['mkn_weight'],
                         selected_step=None, selected_weights=None, best_training_snapshot_scalars=None,
                         selected_metadata_note='Previous frozen parent retained; its training snapshot metadata is not rederived by this selector.',
                         selection_reason='No new candidate cleared the fixed 0.0005 validation-improvement gate.')
    passed = candidate is not None and improvement >= MIN_GAIN and not integrity_errors
    winner = None if integrity_errors or not eligible else candidate if passed else parent_winner
    accounted = [row for row in rows if 'new_search_processed_view_targets' in row]
    result = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        selection_split='validation', test_used=False, prior_test_exposure_disclosed=True,
        parent_checkpoint_sha256=parent_sha, parent_validation_bpb_gpu_fp32=parent_score,
        parent_lineage_processed_view_targets=PARENT_VIEW_TARGETS, candidates=rows,
        best_new_candidate=candidate, winner=winner,
        selection_status='refused' if winner is None else 'new_candidate' if passed else 'retain_parent',
        selection_integrity_errors=integrity_errors,
        refusal_reason='No valid complete calibrated candidate result exists.' if not eligible else None,
        improvement_vs_parent=improvement, minimum_validation_improvement=MIN_GAIN,
        passes_predeclared_improvement_gate=passed,
        all_new_search_processed_view_targets=sum(row['new_search_processed_view_targets'] for row in accounted),
        all_new_search_targets=sum(row['new_search_processed_view_targets'] for row in accounted),
        all_new_search_sampled_label_occurrences=sum(row['new_search_sampled_label_occurrences'] for row in accounted),
        total_training_loop_seconds=sum(row['training_loop_seconds'] for row in accounted),
        total_recorded_calibration_seconds=sum(row['calibration_seconds'] for row in rows if row['calibration_seconds'] is not None),
        total_calibration_seconds=sum(row['calibration_seconds'] for row in rows if row['calibration_seconds'] is not None),
        calibration_time_fully_recorded=all(row['calibration_seconds'] is not None for row in rows),
        total_raw_ema_validation_choices=sum(row['raw_ema_validation_choices'] for row in accounted),
        total_raw_validation_choices=sum(row['raw_validation_choices'] for row in accounted),
        total_ema_validation_choices=sum(row['ema_validation_choices'] for row in accounted),
        total_calibration_choices=sum(row['calibration_choices'] for row in rows),
        all_training_accounting_present=len(accounted) == len(RUNS),
        accounting_note='Counts are completed processed-view target occurrences and sampled-label position occurrences, NOT deduplicated unique tokens. Failed-attempt or unrecorded work is excluded and cannot be inferred from planned caps.',
        plan_sha256=sha(plan_path), selection_script_sha256=sha(__file__), no_raw_text_loaded=True,
        unit_tests=dict(executions=unit_test_executions, source='Required caller-supplied final measured execution count; selector does not rerun tests.'),
        next_gate='Audit actual CPU FP32 weights and resources before any freeze or separately authorized test.')
    return result


def write_selection(destination, result):
    # Exclusive creation, including refused records: existing evidence is never overwritten.
    with Path(destination).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit-test-executions', required=True, type=int)
    args = parser.parse_args()
    destination = ROOT/'runs/push148_selection.json'
    if destination.exists():
        raise FileExistsError('Preserve the original selection record')
    result = collect_selection(ROOT, args.unit_test_executions)
    write_selection(destination, result)
    print(json.dumps(result, indent=2), flush=True)
    if result['selection_status'] == 'refused':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
