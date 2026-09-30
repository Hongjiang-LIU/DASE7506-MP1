"""Exclusive, fixed-rank audit/timing process ledger; never score test data."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import subprocess
import sys
import time

from push152_launch import ROOT, PLAN_SHA, preserve_ten
from push151_launch import sha, read, write_new

SELECTION = ROOT/'runs/push152_selection.json'
TIMING_SHA = '5e0a5173dd00a1b3519877172d1dcb3d195a2a106a17cd6256d7b55cb78bb99e'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('audit','timing'))
    parser.add_argument('--rank', type=int, choices=(0,1,2), default=0)
    args = parser.parse_args()
    require(sha(ROOT/'runs/push152_plan.json') == PLAN_SHA, 'Fixed plan changed')
    require(not (ROOT/'runs/push152_candidate_frozen.json').exists() and
            not (ROOT/'runs/push152-final-test').exists(), 'Resource exploration must precede freeze/test')
    preserve_ten()
    selection = read(SELECTION)
    require(selection['plan_sha256'] == PLAN_SHA and selection['test_used'] is False and
            selection['status'] == 'validation_qualified_candidates_need_resource_audit' and
            selection['all_declared_runs_accounted'] is True, 'Selection incomplete or unqualified')
    candidates = selection['resource_audit_candidates']
    require(0 < len(candidates) <= 3 and args.rank < len(candidates), 'Rank not in fixed resource menu')
    candidate = candidates[args.rank]
    checkpoint = (ROOT/candidate['checkpoint']).resolve()
    require(checkpoint.is_relative_to(ROOT/'runs') and
            sha(checkpoint) == candidate['checkpoint_sha256'], 'Candidate identity differs')
    # Metadata-only precheck avoids consuming an exclusive candidate audit
    # attempt when a rank is accidentally requested out of order.
    work = ROOT.parents[1]
    sys.path.insert(0, str(work))
    spec = importlib.util.spec_from_file_location('_push152_resource_audit', work/'push152_audit.py')
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    plan = read(ROOT/'runs/push152_plan.json')
    auditor.verify_plan(plan)
    checked_rank, checked = auditor.verify_selection(selection,checkpoint.relative_to(ROOT).as_posix(),
                                                     candidate['checkpoint_sha256'],plan)
    require(checked_rank == args.rank and checked == candidate, 'Requested rank differs from checked selection')
    auditor.verify_prior_ranks(ROOT, selection, args.rank, {})
    directory = ROOT/f'runs/push152-resource-rank{args.rank}'
    directory.mkdir(exist_ok=True)
    if args.phase == 'audit':
        script = ROOT.parents[1]/'push152_audit.py'
        arguments = ['--code-root', str(ROOT), '--checkpoint', str(checkpoint),
                     '--output', str(directory/'audit.json')]
    else:
        audit = read(directory/'audit.json')
        require(not (directory/'audit.failure.json').exists() and
                audit['status'] == 'completed' and audit['asset_limit_pass'] is True and
                audit['ram_limit_pass'] is True and audit['precision_diagnostic']['fixed_diagnostic_passes'] is True and
                audit['all_limits_pass_not_claimed_without_separate_paired_timing'] is True and
                audit['plan_sha256'] == PLAN_SHA and audit['selection_sha256'] == sha(SELECTION) and
                audit['resource_rank'] == args.rank and audit['checkpoint_sha256'] == sha(checkpoint) and
                audit['audit_script_sha256'] == sha(ROOT.parents[1]/'push152_audit.py'),
                'Timing requires a completed successful audit of this exact rank')
        require(not (directory/'timing').exists() and not (directory/'timing.failure.json').exists(),
                'Preserve every prior timing attempt')
        script = ROOT/'push145_fresh_benchmark.py'
        require(sha(script) == TIMING_SHA, 'Timing source changed')
        arguments = ['--candidate', str(checkpoint), '--run-dir', str(directory/'timing')]
    command = [str(ROOT/'.venv/Scripts/python.exe'), '-B', str(script), *arguments]
    started_path = directory/(args.phase+'.launch.started.json')
    process_path = directory/(args.phase+'.launch.process.json')
    log_path = directory/(args.phase+'.launch.output.log')
    require(not any(path.exists() for path in (started_path,process_path,log_path)), 'No silent launch retry')
    source_sha, launcher_sha = sha(script), sha(__file__)
    selection_sha, checkpoint_sha = sha(SELECTION), sha(checkpoint)
    write_new(started_path, dict(phase=args.phase, resource_rank=args.rank, command=command,
        plan_sha256=PLAN_SHA, selection_sha256=selection_sha, checkpoint_sha256=checkpoint_sha,
        script_sha256=source_sha, launcher_sha256=launcher_sha, test_used=False,
        started_at_utc=datetime.now(timezone.utc).isoformat()))
    started = time.perf_counter()
    code, failure, process = None, None, None
    try:
        with log_path.open('x',encoding='utf-8') as log:
            process = subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace',bufsize=1)
            for line in process.stdout:
                log.write(line); log.flush()
                print(line,end='',flush=True)
            code = process.wait()
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error),
            note='Interrupted orchestration requires manual child-tree review; no automatic retry or rank fallback.')
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            code = process.returncode
        raise
    finally:
        exited = process is None or process.poll() is not None
        write_new(process_path, dict(phase=args.phase, resource_rank=args.rank,
            started_record_sha256=sha(started_path), returncode=code, failure=failure, test_used=False,
            child_process_exited=exited,
            child_tree_review_required=failure is not None,
            output_sha256=sha(log_path) if log_path.exists() else None,
            seconds_total=time.perf_counter()-started, completed_at_utc=datetime.now(timezone.utc).isoformat()))
        # A normal nonzero return means the benchmark's synchronous scorer call
        # returned. An interrupted parent may leave scorer descendants, so its
        # process evidence alone never creates an automatic rank-skip record.
        if args.phase == 'timing' and failure is None and code not in (None,0):
            require(exited, 'Do not allow a later rank while the prior timing child still runs')
            require(not (directory/'timing/summary.json').exists(),
                    'Completed timing summary exists; preserve exception but do not relabel resources')
            write_new(directory/'timing.failure.json', dict(status='failed', plan_sha256=PLAN_SHA,
                selection_sha256=selection_sha, checkpoint_sha256=checkpoint_sha,
                resource_rank=args.rank,test_used=False,no_automatic_retry=True,
                failure_type=failure['type'] if failure else 'NonzeroProcessExit',
                failure_message=failure['message'] if failure else 'Timing process returned '+str(code),
                started_record=started_path.relative_to(ROOT).as_posix(),started_record_sha256=sha(started_path),
                process_record=process_path.relative_to(ROOT).as_posix(),process_record_sha256=sha(process_path)))
    require(sha(script) == source_sha and sha(__file__) == launcher_sha and
            sha(SELECTION) == selection_sha and sha(checkpoint) == checkpoint_sha,
            'Frozen source/selection/candidate changed during resource execution')
    preserve_ten()
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    main()
