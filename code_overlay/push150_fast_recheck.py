"""Prespecified paired timing repair; never edit/discard the unstable screen."""
import json
from datetime import datetime, timezone
from pathlib import Path
import statistics
import subprocess
import sys
import time
from common import ROOT, sha


def main():
    parent = ROOT/'runs/push150-fast-dense'
    prepared = json.loads((parent/'prepared.json').read_text())
    original = parent/'cpu-screen/summary.json'
    old = json.loads(original.read_text())
    baselines = [row['median_scoring_seconds'] for row in old['measurements'] if row['name'].startswith('baseline')]
    if max(baselines)/min(baselines) < 2:
        raise ValueError('Declared repair rationale does not match original timing evidence')
    plan_path = ROOT/'runs/push150_phase_b_timing_repair_plan.json'
    plan = json.loads(plan_path.read_text())
    variants = {row['microbatch']:row for row in prepared['variants']}
    directory = parent/'paired-recheck'
    directory.mkdir(exist_ok=False)
    started = time.perf_counter()
    report = dict(status='running', started_at_utc=datetime.now(timezone.utc).isoformat(),
                  plan_sha256=sha(plan_path), original_screen_sha256=sha(original),
                  prepared_sha256=sha(parent/'prepared.json'), script_sha256=sha(__file__),
                  source_screen_eligibility_withheld=True, measurements=[], test_used=False,
                  device='cpu', precision='fp32', threads=4, affinity_mask=15,
                  extra_worst_case_gate=False)
    destination = directory/'summary.json'

    def save():
        report['seconds_total'] = time.perf_counter()-started
        destination.write_text(json.dumps(report, indent=2)+'\n')

    save()
    try:
        for order in plan['orders']:
            repeat = order['repeat']
            for mb in order['variants']:
                variant = variants[mb]
                candidate = Path(variant['checkpoint'])
                if sha(candidate) != variant['checkpoint_sha256'] or sha(ROOT/'push150_fast.py') != variant['implementation_sha256']:
                    raise ValueError('Same-config timing repair cannot change weights or source')
                for name in order['within_pair']:
                    checkpoint = ROOT/'runs/baseline-s17/checkpoint.pt' if name == 'baseline' else candidate
                    output = directory/f'mb{mb}-repeat{repeat}-{name}.json'
                    process = subprocess.run([sys.executable,'-B',str(ROOT/'measure_validation.py'),
                        '--checkpoint',str(checkpoint),'--repeats','1','--split','validation',
                        '--affinity','15','--output',str(output)],cwd=ROOT,capture_output=True,text=True)
                    if process.returncode:
                        report['failed_process'] = dict(microbatch=mb, repeat=repeat, name=name,
                            returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
                        raise RuntimeError('A fresh timing process failed')
                    measured = json.loads(output.read_text())
                    row = dict(microbatch=mb, repeat=repeat, name=name, **measured)
                    report['measurements'].append(row)
                    save()
                    print(json.dumps(dict(microbatch=mb,repeat=repeat,name=name,
                                          seconds=measured['median_scoring_seconds'],bpb=measured['repeats'][0]['bpb'])),flush=True)
        report['candidates'] = []
        for mb, variant in variants.items():
            values = {name:[row for row in report['measurements'] if row['microbatch']==mb and row['name']==name]
                      for name in ('baseline','candidate')}
            times = {name:[row['median_scoring_seconds'] for row in rows] for name,rows in values.items()}
            medians = {name:statistics.median(items) for name,items in times.items()}
            ratio = medians['candidate']/medians['baseline']
            differences = [abs(row['repeats'][0]['bpb']-1.4472849220037163) for row in values['candidate']]
            memory = max(row['peak_process_working_set_bytes'] for row in values['candidate'])
            assets = max(row['checkpoint_bytes'] for row in values['candidate'])
            eligible = max(differences)<1e-5 and ratio<=5 and memory<=4*2**30 and assets<64*2**20
            report['candidates'].append(dict(**variant,times=times,medians=medians,cpu_ratio=ratio,
                pair_ratios=[c/b for b,c in zip(times['baseline'],times['candidate'])],
                max_candidate_min_baseline_ratio=max(times['candidate'])/min(times['baseline']),
                bpb_max_difference=max(differences),peak_ram_gib=memory/2**30,
                eligible_for_training=eligible))
        eligible = [row for row in report['candidates'] if row['eligible_for_training']]
        report['winner'] = min(eligible,key=lambda row:row['medians']['candidate']) if eligible else None
        report['status'] = 'completed'
        save()
        print(json.dumps(dict(winner=report['winner']),indent=2),flush=True)
    except BaseException as error:
        report.update(status='failed',failure_type=type(error).__name__,failure_message=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
