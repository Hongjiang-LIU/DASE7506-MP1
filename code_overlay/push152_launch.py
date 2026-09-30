"""Fixed push152 process ledger with live output and ten-archive preservation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from push151_launch import sha, read, write_new, preserve

ROOT = Path(__file__).resolve().parent
PLAN_SHA = 'd15cd6276ace1894a37db7c005453c61acd2e5acb7e934cb6790f3860aa06b51'
COMMANDS = {
    'initialize': ('cpu', ['push152_init.py','--parent','runs/push151-combination/checkpoint.pt',
        '--run-dir','runs/push152-expanded-init','--plan','runs/push152_plan.json']),
    'preflight-audit': ('cpu', ['push152_preflight.py','audit']),
    'preflight-timing': ('cpu', ['push145_fresh_benchmark.py','--candidate','runs/push152-expanded-init/checkpoint.pt',
        '--run-dir','runs/push152-expanded-preflight/timing']),
    'preflight-authorize': ('cpu', ['push152_preflight.py','authorize']),
    'optimizer-keep': ('gpu', ['push152_resume.py','optimizer-keep','--plan','runs/push152_plan.json']),
    'optimizer-reset': ('gpu', ['push152_resume.py','optimizer-reset','--plan','runs/push152_plan.json']),
    'control': ('gpu', ['push152_train.py','--branch','control','--plan','runs/push152_plan.json']),
    'auxiliary': ('gpu', ['push152_train.py','--branch','auxiliary','--plan','runs/push152_plan.json']),
    'expanded': ('gpu', ['push152_train.py','--branch','expanded','--plan','runs/push152_plan.json']),
}


def preserve_ten():
    previous = preserve()
    newest = ROOT.parents[2]/'outputs/MP1_push151_tested_candidate.zip'
    record = dict(name=newest.name, bytes=37455872,
        sha256='5a826454a8f6d0e177ecafbe3bf3860c749875ec33db44504765bb113528861f')
    if newest.stat().st_size != record['bytes'] or sha(newest) != record['sha256']:
        raise ValueError('Preserve the tenth immutable scoring package')
    return [*previous, record]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=tuple(COMMANDS))
    args = parser.parse_args()
    plan_path = ROOT/'runs/push152_plan.json'
    if sha(plan_path) != PLAN_SHA:
        raise ValueError('Fixed plan changed')
    plan = read(plan_path)
    if sha(ROOT/plan['parent']['path']) != plan['parent']['sha256']:
        raise ValueError('Frozen parent changed')
    protected = preserve_ten()
    device, arguments = COMMANDS[args.phase]
    python = ROOT/('.venv-gpu' if device == 'gpu' else '.venv')/'Scripts/python.exe'
    command = [str(python), '-B', str(ROOT/arguments[0]), *arguments[1:]]
    directory = ROOT/'runs/push152-launches'
    directory.mkdir(exist_ok=True)
    started_path = directory/(args.phase+'.started.json')
    process_path = directory/(args.phase+'.process.json')
    log_path = directory/(args.phase+'.output.log')
    if any(path.exists() for path in (started_path, process_path, log_path)):
        raise FileExistsError('Preserve previous phase launch and output; no silent retries')
    source_sha = sha(ROOT/arguments[0])
    write_new(started_path, dict(phase=args.phase, command=command, cwd=str(ROOT),
        plan_sha256=PLAN_SHA, script_sha256=source_sha, launcher_sha256=sha(__file__),
        started_at_utc=datetime.now(timezone.utc).isoformat(), test_used=False, preserved_archives=protected))
    started = time.perf_counter()
    returncode, failure = None, None
    try:
        with log_path.open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1)
            for line in process.stdout:
                log.write(line); log.flush()
                print(line, end='', flush=True)
            returncode = process.wait()
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error),
            note='Child may still exist on interrupted orchestration; inspect state/process before any retry.')
        raise
    finally:
        write_new(process_path, dict(phase=args.phase, returncode=returncode, failure=failure,
            started_record_sha256=sha(started_path), output_sha256=sha(log_path) if log_path.exists() else None,
            seconds_total=time.perf_counter()-started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
            test_used=False, script_unchanged=sha(ROOT/arguments[0]) == source_sha))
    preserve_ten()
    if sha(ROOT/arguments[0]) != source_sha:
        raise ValueError('Phase source changed during execution')
    if returncode:
        raise SystemExit(returncode)


if __name__ == '__main__':
    main()
