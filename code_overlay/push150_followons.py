"""Sequential declared B/C/D work after the complete Phase-A selection.

One child at a time owns the GPU. Every launch, exit and stdout is retained,
including failures before a child trainer creates its own metrics. This driver
never loads corpus text, scores test, retries failed runs, or changes a recipe.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from push150_final_test import ROOT, child, read, require, sha, write_new
from push150_oof_prepare import source_graph


def declared_commands(parent, parent_sha):
    cpu = str(ROOT/'.venv/Scripts/python.exe')
    gpu = str(ROOT/'.venv-gpu/Scripts/python.exe')
    prefix = ['--parent', str(parent)]
    gate = prefix+['--expected-parent-sha256', parent_sha, '--run-dir', 'runs/push150-gate']
    commands = [('gate-build', [cpu,'-B','push150_gate_train.py','--phase','build',*gate,'--device','cpu']),
        ('oof-prepare',[cpu,'-B','push150_oof_prepare.py',*prefix,'--expected-parent-sha256',parent_sha])]
    for phase in ('extract','train','score'):
        commands.append(('gate-'+phase,[gpu,'-B','push150_gate_train.py','--phase',phase,*gate,'--device','cuda']))
    commands.extend([
        ('suffix-init',[cpu,'-B','push150_suffix_init.py',*prefix,'--expected-parent-sha256',parent_sha,
                        '--run-dir','runs/push150-suffix-init']),
        ('suffix-control',[gpu,'-B','push150_train.py',*prefix,'--run-dir','runs/push150-suffix-control',
            '--steps','4000','--eval-every','1000','--seed','191','--schedule','cosine','--warmup','100','--decay-steps','1000']),
        ('suffix-control-calibrate',[gpu,'-B','push146_calibrate.py','--parent',
            'runs/push150-suffix-control/checkpoint.pt','--run-dir','runs/push150-suffix-control-calibrated']),
        ('suffix-joint',[gpu,'-B','push150_suffix_train.py','--parent',
            'runs/push150-suffix-init/checkpoint.pt','--run-dir','runs/push150-suffix-joint']),
        ('suffix-joint-calibrate',[gpu,'-B','push150_suffix_calibrate.py','--parent',
            'runs/push150-suffix-joint/checkpoint.pt','--run-dir','runs/push150-suffix-joint-calibrated'])])
    for objective in ('control','joint'):
        name = 'oof-'+objective
        commands.extend([
            (name,[gpu,'-B','push150_oof_train.py',*prefix,'--objective',objective,
                '--run-dir','runs/push150-'+name,'--prepared-dir','runs/push150-oof-data',
                '--phase-a-selection','runs/push150_phase_a_selection.json']),
            (name+'-calibrate',[gpu,'-B','push146_calibrate.py','--parent',
                'runs/push150-'+name+'/checkpoint.pt','--run-dir','runs/push150-'+name+'-calibrated'])])
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit-evidence', type=Path, required=True)
    args = parser.parse_args()
    selection_path = ROOT/'runs/push150_phase_a_selection.json'
    selection = read(selection_path)
    require(selection.get('status') == 'phase_a_choice_complete' and selection.get('test_used') is False and
            selection.get('evidence_errors') == [] and selection.get('all_training_accounting_verified') is True,
            'All declared Phase-A runs must be complete and verified first')
    units = read(args.unit_evidence)
    require(units.get('successful') is True and units.get('tests_run',0)>0 and
            not units['failures'] and not units['errors'], 'Require successful current synthetic unit suite')
    winner = selection['winner']
    parent = child(ROOT/'runs',winner['selected_run'])/'checkpoint.pt'
    require(sha(parent) == winner['checkpoint_sha256'], 'Selected Phase-A parent changed')
    commands = declared_commands(parent,winner['checkpoint_sha256'])
    # Pin the full execution graph, not unrelated reporting scripts still being
    # developed while training runs. Dynamic model roots are explicit.
    sources = source_graph(ROOT, ['push150_followons.py','push150_gate_train.py',
        'push150_oof_prepare.py','push150_suffix_init.py','push150_train.py','push146_calibrate.py',
        'push150_suffix_train.py','push150_suffix_calibrate.py','push150_oof_train.py',
        winner['implementation']+'.py'])
    for name, digest in units['source_sha256'].items():
        if name in sources:
            require(sources[name] == digest, 'Tested runtime source changed before launch: '+name)
    for required in ('push150_gate.py','push150_gate_train.py','push150_suffix.py',
                     'push150_suffix_train.py','push150_oof_prepare.py','push150_oof_train.py'):
        require(required in units['source_sha256'], 'Required implementation absent from unit suite: '+required)
    plans = {path.relative_to(ROOT).as_posix():sha(path) for path in sorted((ROOT/'runs').glob('push150_phase_*_plan.json'))}
    directory = ROOT/'runs/push150_followon_launches'
    directory.mkdir(parents=True,exist_ok=False)
    ledger = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        phase_a_selection_sha256=sha(selection_path),parent_sha256=winner['checkpoint_sha256'],
        source_sha256=sources,plan_sha256=plans,unit_evidence_path=str(args.unit_evidence.resolve()),
        unit_evidence_sha256=sha(args.unit_evidence),commands=[dict(name=name,command=command) for name,command in commands],
        gpu_policy='One child process at a time; no CPU resource benchmark overlaps training',test_used=False)
    write_new(directory/'plan.json',ledger)
    records=[]
    for index,(name,command) in enumerate(commands):
        started=time.perf_counter()
        require(sha(selection_path)==ledger['phase_a_selection_sha256'] and
                sha(parent)==ledger['parent_sha256'],'Pinned Phase-A parent/selection changed')
        for source,digest in sources.items():
            require(sha(ROOT/source)==digest,'Pinned source changed: '+source)
        for plan,digest in plans.items():
            require(sha(ROOT/plan)==digest,'Pinned phase plan changed: '+plan)
        write_new(directory/(name+'.started.json'),dict(index=index,name=name,command=command,
            started_at_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(directory/'plan.json'),test_used=False))
        print(json.dumps(dict(launch=name,index=index,command=command)),flush=True)
        try:
            with (directory/(name+'.output.log')).open('x',encoding='utf-8') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,encoding='utf-8',errors='replace',bufsize=1)
                for line in process.stdout:
                    log.write(line);log.flush()
                    print(line,end='',flush=True)
                returncode=process.wait()
            record=dict(index=index,name=name,returncode=returncode,seconds_total=time.perf_counter()-started,
                completed_at_utc=datetime.now(timezone.utc).isoformat(),test_used=False,
                output_log_sha256=sha(directory/(name+'.output.log')),
                failure_cost_note=None if returncode==0 else 'Inspect child metrics for completed work; partial setup/failed work may be unquantified.')
            write_new(directory/(name+'.process.json'),record)
            records.append(record)
            require(returncode==0,'Child failed; no automatic retry: '+name)
        except BaseException as error:
            write_new(directory/'failure.json',dict(name=name,index=index,failure_type=type(error).__name__,
                failure_message=str(error),seconds_current_launch=time.perf_counter()-started,
                completed_launches=records,partial_files_preserved=True,test_used=False,no_automatic_retry=True))
            raise
    write_new(directory/'summary.json',dict(status='completed',launches=records,
        seconds_total=sum(row['seconds_total'] for row in records),test_used=False,
        parent_sha256=winner['checkpoint_sha256'],plan_sha256=sha(directory/'plan.json')))


if __name__=='__main__':
    main()
