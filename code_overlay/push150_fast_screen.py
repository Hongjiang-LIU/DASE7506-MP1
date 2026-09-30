"""Prepare three equivalent dense runtime wrappers, then CPU validation screen."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, sha


PARENT_SHA = '3eb057e46038d52c75803c64bab103545251a8be310158bdb004e98b556dc899'
MICROBATCHES = (8, 16, 32)


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def prepare(args):
    import torch
    from common import make_model
    torch.set_num_threads(2)
    if sha(args.parent) != PARENT_SHA:
        raise ValueError('Expected the declared untrained dense expansion of push149')
    plan_path = ROOT/'runs/push150_phase_b_fast_plan.json'
    plan = json.loads(plan_path.read_text())
    if [row['cpu_feature_microbatch'] for row in plan['settings']] != list(MICROBATCHES):
        raise ValueError('Runtime settings differ from the declared screen')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    rows = []
    started = time.perf_counter()
    for microbatch in MICROBATCHES:
        candidate = copy.deepcopy(parent)
        candidate['implementation'] = 'push150_fast'
        candidate['config']['cpu_feature_microbatch'] = microbatch
        candidate['runtime_parent_sha256'] = PARENT_SHA
        candidate['runtime_selection'] = 'Three declared equivalent FP32 CPU implementations; no learned weight changes'
        model, implementation_sha = make_model('push150_fast', candidate['config'], torch.device('cpu'))
        model.load_state_dict(candidate['model'], strict=True)
        model.eval()
        candidate['config'] = copy.deepcopy(model.config)
        for name, value in model.state_dict().items():
            if not torch.equal(value, parent['model'][name]):
                raise ValueError(f'Runtime wrapper changed a parameter/buffer: {name}')
        if set(model.state_dict()) != set(parent['model']):
            raise ValueError('Runtime wrapper introduced a persistent inference asset')
        directory = args.run_dir/f'mb{microbatch}'
        directory.mkdir()
        checkpoint = directory/'checkpoint.pt'
        torch.save(candidate, checkpoint)
        rows.append(dict(microbatch=microbatch, checkpoint=str(checkpoint.resolve()),
                         checkpoint_sha256=sha(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
                         implementation_sha256=implementation_sha, unchanged_model_tensors=True))
    report = dict(status='prepared', created_at_utc=datetime.now(timezone.utc).isoformat(),
                  phase_plan_sha256=sha(plan_path), parent_sha256=PARENT_SHA,
                  inherited_train_targets=parent['train_tokens'], new_train_targets=0,
                  variants=rows, script_sha256=sha(__file__), data_text_loaded=False, test_used=False,
                  seconds=time.perf_counter()-started)
    write_new(args.run_dir/'prepared.json', report)
    print(json.dumps(report, indent=2), flush=True)


def screen(args):
    prepared = json.loads((args.run_dir/'prepared.json').read_text())
    if prepared['script_sha256'] != sha(__file__):
        raise ValueError('Runtime screen source changed after preparation')
    plan_path = ROOT/'runs/push150_phase_b_fast_plan.json'
    if prepared['phase_plan_sha256'] != sha(plan_path):
        raise ValueError('Runtime plan changed')
    directory = args.run_dir/'cpu-screen'
    directory.mkdir(exist_ok=False)
    report = dict(status='running', started_at_utc=datetime.now(timezone.utc).isoformat(),
                  prepared_sha256=sha(args.run_dir/'prepared.json'), measurements=[], test_used=False,
                  device='cpu', precision='fp32', threads=4, affinity_mask=15,
                  selection='Fastest correct resource-feasible runtime, never minimum BPB')
    destination = directory/'summary.json'
    started = time.perf_counter()

    def save():
        report['seconds_total'] = time.perf_counter()-started
        destination.write_text(json.dumps(report, indent=2)+'\n')

    save()
    try:
        baseline = ROOT/'runs/baseline-s17/checkpoint.pt'
        variants = {f"mb{row['microbatch']}":row for row in prepared['variants']}
        sequence = [('baseline-before', baseline)] + [(name,Path(row['checkpoint'])) for name,row in variants.items()] + [('baseline-after', baseline)]
        for name, checkpoint in sequence:
            if name in variants:
                if (sha(checkpoint) != variants[name]['checkpoint_sha256'] or
                        sha(ROOT/'push150_fast.py') != variants[name]['implementation_sha256']):
                    raise ValueError('Prepared runtime source/checkpoint changed')
            output = directory/f'{name}.json'
            process = subprocess.run([sys.executable,'-B',str(ROOT/'measure_validation.py'),
                '--checkpoint',str(checkpoint.resolve()),'--repeats','1','--split','validation',
                '--affinity','15','--output',str(output.resolve())],cwd=ROOT,capture_output=True,text=True)
            if process.returncode:
                report['failed_process'] = dict(name=name, returncode=process.returncode,
                                                stdout=process.stdout, stderr=process.stderr)
                raise RuntimeError(f'Runtime screen failed: {name}')
            measured = json.loads(output.read_text())
            report['measurements'].append(dict(name=name, **measured))
            save()
            print(json.dumps(dict(name=name, measurement=measured['repeats'][0])),flush=True)
        rows = {row['name']:row for row in report['measurements']}
        reference = (rows['baseline-before']['median_scoring_seconds']+rows['baseline-after']['median_scoring_seconds'])/2
        report['baseline_mean_seconds'] = reference
        report['candidates'] = []
        for name, original in variants.items():
            row = rows[name]
            ratio = row['median_scoring_seconds']/reference
            difference = abs(row['repeats'][0]['bpb']-1.4472849220037163)
            correct = difference < 1e-5
            feasible = ratio <= 5 and row['peak_process_working_set_bytes'] <= 4*2**30 and row['checkpoint_bytes'] < 64*2**20
            report['candidates'].append(dict(name=name, **original, seconds=row['median_scoring_seconds'],
                                             bpb_difference=difference,cpu_ratio=ratio,
                                             numerically_equivalent=correct, preliminary_resource_feasible=feasible,
                                             eligible_for_training=correct and feasible))
        eligible = [row for row in report['candidates'] if row['eligible_for_training']]
        report['winner'] = min(eligible,key=lambda row:row['seconds']) if eligible else None
        report['status'] = 'completed'
        save()
        print(json.dumps(dict(candidates=report['candidates'],winner=report['winner']),indent=2),flush=True)
    except BaseException as error:
        report.update(status='failed',failure_type=type(error).__name__,failure_message=str(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','screen'),required=True)
    parser.add_argument('--parent',type=Path,default=ROOT/'runs/push150-init-dense/checkpoint.pt')
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/push150-fast-dense')
    args = parser.parse_args()
    (prepare if args.phase == 'prepare' else screen)(args)


if __name__ == '__main__':
    main()
