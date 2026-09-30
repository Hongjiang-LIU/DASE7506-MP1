"""One-shot post-freeze official CPU evaluation, with an immutable launch ledger.

This program must not run during development. It opens no corpus itself; only
after all freeze checks pass does it invoke the unchanged official evaluator.
Any existing output directory prevents automatic retry, including after failure.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
BASELINE_SHA = 'b7fdb67c8ef14b5166c0de1f17cf306f34800b41dfe97da856c9812cdb1f9701'
BASELINE_MODEL_SHA = '6cce142a31d396f017319543f50d70f30fa6da740729af01c8baed13c6f671f9'
SCORER_SHA = '128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d'
TOKENIZER_SHA = '020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(not path.name.startswith('wikitext_'), 'Wrapper must not read/hash corpus text')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    require(not Path(path).name.startswith('wikitext_'), 'Metadata only')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def child(root, relative):
    name = str(relative).replace('\\', '/')
    require(name and not name.startswith('/') and ':' not in name and
            all(part not in ('', '.', '..') for part in name.split('/')), 'Unsafe relative path')
    resolved = (Path(root)/name).resolve()
    require(resolved.is_relative_to(Path(root).resolve()), 'Path escaped root')
    return resolved


def check_measurement(measured, resource, checkpoint_sha, implementation_sha):
    expected = dict(protocol='7506-mp1-wt2-v2', split='test', precision='fp32',
        checkpoint_sha256=checkpoint_sha, implementation_sha256=implementation_sha,
        evaluator_sha256=SCORER_SHA, tokenizer_sha256=TOKENIZER_SHA,
        device='cpu', device_name='CPU', targets=428405, utf8_bytes=1292013)
    require(all(measured.get(key) == value for key, value in expected.items()), 'Official result identity/counts differ')
    for key in ('bpb', 'nll_nats', 'seconds'):
        require(isinstance(measured.get(key), (float, int)) and math.isfinite(measured[key]) and
                measured[key] > 0, 'Nonfinite/nonpositive official measurement: '+key)
    require(abs(measured['bpb']-measured['nll_nats']/math.log(2)/1292013) < 1e-10,
            'BPB does not match unchanged denominator')
    require(resource.get('cpu_affinity_mask') == 15 and
            resource.get('peak_process_working_set_bytes', 0) > 0, 'Missing CPU affinity/peak RAM evidence')


def verify_freeze(root, selection_path):
    root = Path(root)
    selection = read(selection_path)
    require(selection.get('selection_split') == 'validation' and selection.get('test_used') is False,
            'Selection must precede test and use validation only')
    require(selection.get('all_declared_phases_accounted') is True and
            selection.get('passes_predeclared_improvement_gate') is True and
            not selection.get('selection_integrity_errors'), 'Incomplete/invalid overall selection')
    winner = selection['winner']
    require(re.fullmatch(r'push150-[a-z0-9-]+', winner['selected_run']), 'Expected a new push150 winner')
    run_dir = child(root/'runs', winner['selected_run'])
    freeze_path = run_dir/'candidate_frozen.json'
    freeze = read(freeze_path)
    audit_path = run_dir/'audit.json'
    audit = read(audit_path)
    require(freeze.get('selection_record_sha256') == sha(selection_path) and
            freeze.get('selected_validation_record') == winner, 'Frozen selection changed')
    require(freeze.get('final_test_runner_sha256') == sha(root/'push150_final_test.py'),
            'Final test runner differs from the frozen protocol')
    require(freeze.get('official_test_wrapper_sha256') == sha(root/'official_resource_run.py'),
            'Official CPU/thread/affinity wrapper changed after freeze')
    require(freeze.get('all_declared_phases_accounted') is True and
            freeze.get('test_bpb_at_freeze') is None and
            freeze.get('this_predictor_test_scored_before_freeze') is False and
            freeze.get('test_used_for_tuning') is False, 'Invalid pre-test freeze state')
    require(freeze.get('audit_sha256') == sha(audit_path) and audit.get('test_used') is False,
            'Audit changed or used test')
    require(audit.get('asset_limit_pass') is True and audit.get('ram_limit_pass') is True,
            'Resource audit not passed')
    for group in ('contract_checks', 'fixed_file_checks'):
        require(audit.get(group) and all(value is True for value in audit[group].values()), 'Audit contract failed')
    timing_path = child(root, freeze['timing_record'])
    require(sha(timing_path) == freeze['timing_record_sha256'], 'Timing changed after freeze')
    timing = read(timing_path)
    require(timing.get('test_used') is False and timing.get('cpu_time_ratio', math.inf) <= 5 and
            timing.get('candidate_sha256') == winner['checkpoint_sha256'], 'Final fresh timing not passed')
    require(sha(run_dir/'checkpoint.pt') == winner['checkpoint_sha256'] == freeze['checkpoint_sha256'],
            'Candidate changed after selection/freeze')
    total = 0
    for name, asset in audit['assets'].items():
        path = run_dir/'checkpoint.pt' if name == 'checkpoint.pt' else child(root, name)
        require(sha(path) == asset['sha256'] and path.stat().st_size == asset['bytes'], 'Audited asset changed: '+name)
        total += asset['bytes']
    require(total == audit['inference_assets_bytes'] and total <= 64*2**20, 'Inference asset budget failed')
    require(sha(root/'evaluate.py') == SCORER_SHA and sha(root/'data/tokenizer.json') == TOKENIZER_SHA,
            'Fixed scorer/tokenizer changed')
    baseline = root/'runs/baseline-s17/checkpoint.pt'
    require(sha(baseline) == BASELINE_SHA, 'Fixed baseline changed')
    require(sha(root/'model.py') == BASELINE_MODEL_SHA, 'Fixed baseline model implementation changed')
    require(sha(root/(freeze['implementation']+'.py')) == freeze['implementation_sha256'], 'Inference source changed')
    return dict(selection=selection, freeze=freeze, run_dir=run_dir, freeze_path=freeze_path,
                audit=audit, baseline=baseline)


def write_new(path, payload):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=ROOT/'runs/push150_selection.json')
    parser.add_argument('--run-dir', type=Path, default=ROOT/'runs/push150-final-test')
    args = parser.parse_args()
    require(args.run_dir.resolve() == (ROOT/'runs/push150-final-test').resolve(),
            'This round has exactly one fixed final-test ledger directory')
    require(not args.run_dir.exists(), 'Do not retry/overwrite a final-test run; inspect existing evidence')
    evidence = verify_freeze(ROOT, args.selection)
    winner = evidence['selection']['winner']
    specs = [dict(name='baseline-before', checkpoint=evidence['baseline'], sha256=BASELINE_SHA),
             dict(name='candidate', checkpoint=evidence['run_dir']/'checkpoint.pt', sha256=winner['checkpoint_sha256']),
             dict(name='baseline-after', checkpoint=evidence['baseline'], sha256=BASELINE_SHA)]
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    identity = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        selection_sha256=sha(args.selection), freeze_sha256=sha(evidence['freeze_path']),
        wrapper_sha256=sha(__file__), official_wrapper_sha256=sha(ROOT/'official_resource_run.py'),
        evaluator_sha256=SCORER_SHA, tokenizer_sha256=TOKENIZER_SHA,
        candidate_sha256=winner['checkpoint_sha256'], baseline_sha256=BASELINE_SHA,
        planned_passes=[dict(name=row['name'], checkpoint=str(row['checkpoint']), sha256=row['sha256']) for row in specs],
        candidate_pass_cap=1, device='cpu', precision='fp32', threads=4, affinity_mask=15,
        failure_policy='Preserve launch/output/failure evidence; no automatic retry or predictor changes.')
    write_new(args.run_dir/'launch_plan.json', identity)
    rows = []
    try:
        for index, spec in enumerate(specs):
            verify_freeze(ROOT, args.selection)
            require(sha(__file__) == identity['wrapper_sha256'] and
                    sha(ROOT/'official_resource_run.py') == identity['official_wrapper_sha256'], 'Test wrapper changed')
            output = args.run_dir/(spec['name']+'.json')
            command = [sys.executable, '-B', str(ROOT/'official_resource_run.py'), '--checkpoint',
                       str(spec['checkpoint']), '--split', 'test', '--output', str(output)]
            write_new(args.run_dir/(spec['name']+'.started.json'), dict(
                index=index, name=spec['name'], started_at_utc=datetime.now(timezone.utc).isoformat(),
                launch_plan_sha256=sha(args.run_dir/'launch_plan.json'), command=command,
                checkpoint_sha256=spec['sha256']))
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            write_new(args.run_dir/(spec['name']+'.process.json'), dict(returncode=process.returncode,
                stdout=process.stdout, stderr=process.stderr, completed_at_utc=datetime.now(timezone.utc).isoformat()))
            require(process.returncode == 0, 'Official scoring process failed: '+spec['name'])
            measured, resource = read(output), read(output.with_suffix('.resources.json'))
            implementation_sha = (evidence['freeze']['implementation_sha256'] if spec['name'] == 'candidate'
                                  else BASELINE_MODEL_SHA)
            check_measurement(measured, resource, spec['sha256'], implementation_sha)
            row = dict(name=spec['name'], output=output.name, output_sha256=sha(output),
                resources_sha256=sha(output.with_suffix('.resources.json')), **measured,
                peak_ram_gib=resource['peak_process_working_set_bytes']/2**30)
            rows.append(row)
            print(json.dumps(row), flush=True)
        verify_freeze(ROOT, args.selection)
        baselines = [row['seconds'] for row in rows if row['name'].startswith('baseline')]
        candidate = rows[1]
        report = dict(status='completed', completed_at_utc=datetime.now(timezone.utc).isoformat(),
            launch_plan_sha256=sha(args.run_dir/'launch_plan.json'), measurements=rows,
            candidate_scoring_process_launches=1, baseline_scoring_process_launches=2,
            test_bpb=candidate['bpb'], cpu_ratio_to_baseline_mean=candidate['seconds']/statistics.mean(baselines),
            cpu_ratio_to_faster_baseline=candidate['seconds']/min(baselines),
            peak_ram_gib=candidate['peak_ram_gib'], inference_assets_mib=evidence['audit']['inference_assets_mib'],
            limits_pass=candidate['seconds']/statistics.mean(baselines) <= 5 and
                candidate['peak_ram_gib'] <= 4 and evidence['audit']['inference_assets_mib'] <= 64,
            no_post_test_selection=True, seconds_total=time.perf_counter()-started)
        write_new(args.run_dir/'summary.json', report)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        write_new(args.run_dir/'failure.json', dict(status='failed', failure_type=type(error).__name__,
            failure_message=str(error), measurements_completed=rows, seconds_total=time.perf_counter()-started,
            no_automatic_retry=True, partial_test_exposure_must_be_disclosed=True))
        raise


if __name__ == '__main__':
    main()
