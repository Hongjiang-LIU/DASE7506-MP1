"""Three prespecified alternating fresh-process validation timing pairs.

Diagnostic only: retain conflicting two-resident-model benchmarks, never replace
or overwrite them. Each scorer process contains one model, like the official CLI.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from common import ROOT,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True,type=Path)
    p.add_argument('--run-dir',required=True,type=Path)
    args=p.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=False)
    candidates={'baseline':ROOT/'runs/baseline-s17/checkpoint.pt','candidate':args.candidate.resolve()}
    rows=[]
    started=time.perf_counter()
    for repeat in range(3):
        order=['baseline','candidate'] if repeat%2==0 else ['candidate','baseline']
        for name in order:
            destination=args.run_dir/f'{name}-{repeat}.json'
            result=subprocess.run([sys.executable,str(ROOT/'measure_validation.py'),
                '--checkpoint',str(candidates[name]),'--repeats','1','--split','validation',
                '--affinity','15','--output',str(destination)],cwd=ROOT,capture_output=True,text=True)
            if result.returncode:
                raise RuntimeError(result.stdout+'\n'+result.stderr)
            data=json.loads(destination.read_text())
            row=dict(name=name,repeat=repeat,**data['repeats'][0],
                peak_process_working_set_bytes=data['peak_process_working_set_bytes'])
            rows.append(row)
            (args.run_dir/'partial.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(row),flush=True)
    medians={name:statistics.median(r['seconds'] for r in rows if r['name']==name) for name in candidates}
    report=dict(protocol='Three alternating pairs, one fresh CPU process per model per pass',
        diagnostic_reason='Investigate discrepancy with two-model-resident paired timings; those failures retained',
        measurements=rows,median_seconds=medians,cpu_time_ratio=medians['candidate']/medians['baseline'],
        conservative_time_ratio=max(r['seconds'] for r in rows if r['name']=='candidate')/min(r['seconds'] for r in rows if r['name']=='baseline'),
        peak_process_working_set_bytes=max(r['peak_process_working_set_bytes'] for r in rows),
        candidate_sha256=sha(args.candidate),baseline_sha256=sha(candidates['baseline']),
        evaluator_sha256=sha(ROOT/'evaluate.py'),script_sha256=sha(__file__),
        split='validation',test_used=False,device='cpu',precision='fp32',threads=4,affinity_mask=15,
        seconds_total=time.perf_counter()-started)
    (args.run_dir/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
