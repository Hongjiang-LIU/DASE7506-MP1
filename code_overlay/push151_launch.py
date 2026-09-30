"""Record exact fixed push151 commands and protect all nine previous packages."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
PLAN_SHA = '3ba6539a20ba74c7989b9733e755d48bd7529ae9446785e918ceb65f049063aa'
OLD_PRESERVE_SHA = 'd7c78ffe615ca37e0d6f71538105f6a09f3f7827488144c3d2ef1ae4b2ccb7c9'
NEWEST_ZIP_SHA = '2d2954750c619e7842deebeba7a11c419e23c6db6a69d3a62c49e5a8e9dcfb13'
COMMANDS = {
    'average-build': ['push151_average.py','build','--plan','runs/push151_plan.json','--run-dir','runs/push151-average'],
    'average-score': ['push151_average.py','score','--plan','runs/push151_plan.json','--run-dir','runs/push151-average','--device','cuda:0'],
    'statistics-build': ['push151_statistics.py','build'],
    'statistics-score': ['push151_statistics.py','score','--device','cuda'],
}


def sha(path):
    path = Path(path)
    if path.name.startswith('wikitext_'):
        raise ValueError('Launcher cannot read any corpus text')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, record):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')


def preserve():
    source = ROOT/'runs/push150_preserved_packages.json'
    if sha(source) != OLD_PRESERVE_SHA:
        raise ValueError('Original preservation manifest changed')
    rows = read(source)['archives'] + [dict(name='MP1_push150_tested_candidate.zip',bytes=35049812,sha256=NEWEST_ZIP_SHA)]
    if len(rows) != 9:
        raise ValueError('Expected all nine previous output ZIPs')
    for row in rows:
        path = PROJECT/'outputs'/row['name']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('A protected package changed: '+row['name'])
    return rows


def verify_inputs():
    plan_path = ROOT/'runs/push151_plan.json'
    if sha(plan_path) != PLAN_SHA:
        raise ValueError('Frozen plan changed')
    plan = read(plan_path)
    for item in [plan['parent'], *plan['phase_a']['snapshots'].values()]:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen source checkpoint changed')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['preflight', *COMMANDS])
    args = parser.parse_args()
    plan = verify_inputs()
    protected = preserve()
    directory = ROOT/'runs/push151-launches'
    directory.mkdir(exist_ok=True)
    if args.phase == 'preflight':
        write_new(directory/'preflight.json', dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            plan_sha256=PLAN_SHA, launcher_sha256=sha(__file__), preserved_archives=protected,
            parent_and_three_snapshot_hashes_verified=True, test_text_accessed=False,
            prior_test_exposure_disclosed=True, new_gradient_targets=0))
        print('Preflight passed; nine archives and all four checkpoint hashes unchanged.')
        return
    preflight = read(directory/'preflight.json')
    if preflight['plan_sha256'] != PLAN_SHA or preflight['launcher_sha256'] != sha(__file__):
        raise ValueError('Preflight/launcher changed')
    python = ROOT/'.venv-gpu/Scripts/python.exe'
    script_name, *arguments = COMMANDS[args.phase]
    command = [str(python), '-B', str(ROOT/script_name), *arguments]
    start_path = directory/(args.phase+'.started.json')
    write_new(start_path, dict(phase=args.phase, started_at_utc=datetime.now(timezone.utc).isoformat(),
        command=command, cwd=str(ROOT), plan_sha256=PLAN_SHA, preflight_sha256=sha(directory/'preflight.json'),
        launcher_sha256=sha(__file__), driver_sha256=sha(ROOT/script_name), test_used=False))
    started = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    write_new(directory/(args.phase+'.process.json'), dict(phase=args.phase,
        started_record_sha256=sha(start_path), returncode=process.returncode,
        completed_at_utc=datetime.now(timezone.utc).isoformat(), seconds_total=time.perf_counter()-started,
        stdout=process.stdout, stderr=process.stderr, test_used=False))
    preserve()
    print(process.stdout, end='', flush=True)
    if process.stderr:
        print(process.stderr, file=sys.stderr, end='', flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
