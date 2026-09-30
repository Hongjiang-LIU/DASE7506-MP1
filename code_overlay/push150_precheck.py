"""Sequential, fresh-process validation-only feasibility screen, GPU idle."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--dense', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    baseline = ROOT/'runs/baseline-s17/checkpoint.pt'
    candidates = {'control': args.control, 'expanded': args.expanded, 'dense': args.dense}
    plan_path = ROOT/'runs/push150_phase_a_plan.json'
    plan = json.loads(plan_path.read_text())
    report = dict(status='running', created_at_utc=datetime.now(timezone.utc).isoformat(),
                  phase_plan_sha256=sha(plan_path), script_sha256=sha(__file__),
                  protocol='One fresh CPU process per pass; baseline before and after three initializers',
                  device='cpu', precision='fp32', threads=4, affinity_mask=15,
                  split='validation', test_used=False, measurements=[],
                  warning='Pre-training feasibility only; does not replace final actual-weight audit and timing')
    destination = args.run_dir/'summary.json'
    started = time.perf_counter()

    def save():
        report['seconds_total'] = time.perf_counter()-started
        destination.write_text(json.dumps(report, indent=2)+'\n')

    save()
    try:
        for name, checkpoint in [('baseline-before', baseline), *candidates.items(), ('baseline-after', baseline)]:
            output = args.run_dir/f'{name}.json'
            command = [sys.executable, '-B', str(ROOT/'measure_validation.py'),
                       '--checkpoint', str(checkpoint.resolve()), '--repeats', '1',
                       '--split', 'validation', '--affinity', '15', '--output', str(output.resolve())]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            if completed.returncode:
                report['failed_process'] = dict(name=name, returncode=completed.returncode,
                                                stdout=completed.stdout, stderr=completed.stderr)
                raise RuntimeError(f'{name} scoring failed')
            measured = json.loads(output.read_text())
            if measured['checkpoint_sha256'] != sha(checkpoint):
                raise ValueError('Checkpoint changed during scoring')
            row = dict(name=name, **measured)
            report['measurements'].append(row)
            save()
            print(json.dumps(dict(name=name, measurement=measured['repeats'][0],
                                  checkpoint_bytes=measured['checkpoint_bytes'],
                                  peak_ram_gib=measured['peak_process_working_set_bytes']/2**30)), flush=True)
        rows = {row['name']:row for row in report['measurements']}
        baseline_seconds = (rows['baseline-before']['median_scoring_seconds']+rows['baseline-after']['median_scoring_seconds'])/2
        report['baseline_mean_seconds'] = baseline_seconds
        report['candidates'] = {}
        for name in candidates:
            row = rows[name]
            ratio = row['median_scoring_seconds']/baseline_seconds
            score_difference = abs(row['repeats'][0]['bpb']-plan['parent_validation_bpb_cpu_fp32'])
            feasible = (ratio <= 5 and row['peak_process_working_set_bytes'] <= 4*2**30
                        and row['checkpoint_bytes'] < 64*2**20 and score_difference < 1e-5)
            report['candidates'][name] = dict(cpu_ratio=ratio, parent_bpb_absolute_difference=score_difference,
                                             preliminary_feasible=feasible,
                                             training_authorized_by_phase_a=feasible and name != 'dense')
        report['status'] = 'completed'
        save()
        print(json.dumps(report['candidates'], indent=2), flush=True)
    except BaseException as error:
        report.update(status='failed', failure_type=type(error).__name__, failure_message=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
