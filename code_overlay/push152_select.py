"""Read-only fixed-menu push152 validation selection and cost accounting.

Standard-library metadata/SHA checks only. Never import a model, load weights,
read any corpus, score, calibrate, average, or inspect a final-test result.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PLAN = 'runs/push152_plan.json'
PLAN_SHA = 'd15cd6276ace1894a37db7c005453c61acd2e5acb7e934cb6790f3860aa06b51'
PARENT_SHA = 'c72cffff6b4152492518473f46a274b53272f59dacb80072cd9e1497d2e0bdf5'
OLD_STATE_SHA = '9ef09a43f4e7703e498ec6801b85738d042b5fd721f2770a8e1622d935ca0d5d'
IMPLEMENTATION_SHA = 'd0340092e2a2882113533f05f4668c48b721ee4416842910c2ba2ba1758e1d45'
SCORER_SHA = '128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d'
DRIVERS = {
    'A': ('push152_resume.py', 'acf14022eeb1e28fd003c0315db078e1d3196fc542df89175fd9aaeab4cb7ad3'),
    'B': ('push152_train.py', '7044375f667bffc320e928d93ba38ee369b2172b7bb8b1476cbac9ef913d2b4c')}
LAUNCHER_SHA = 'fb8b0789f793e668c463db348c4b01e78923c8e35454dffc8f07af359f3e3f51'
PREFLIGHT_SHA = '72e55de2b8041f1974264ea95ce49ca89689998d729d0c73ae88ccffa1365a36'
DATA_SHA = {'data/tokenizer.json': '020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e',
    'data/manifest.json': 'f4790bb599d9fcd04e48263e685335e2f62d9bca0324b08061a10c38984f4fe4',
    'data/wikitext_train.txt': 'c58d555a94ada0882d1e8c0b3dc7ba68e1c269da5ad9e7a057300245d130c4e8',
    'data/wikitext_validation.txt': '714a24a5de7459876f7cfdc64837e551dd7cde2a51f95b763ee083fff273e307'}
NAMES = ('optimizer-keep', 'optimizer-reset', 'control', 'auxiliary', 'expanded')
LINEAGE, TARGETS_PER_STEP, MIN_GAIN = 638976000, 8192, .001
PROTOCOL = '7506-mp1-wt2-v2'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, positive=True):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def sha(path):
    path = Path(path)
    require(not path.name.lower().startswith('wikitext_') and
        not any(part.lower().startswith('push152-final-test') for part in path.parts), 'Corpus/final-test access prohibited')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    require(Path(path).suffix == '.json', 'Only JSON metadata may be decoded')
    require(not any(part.lower().startswith('push152-final-test') for part in Path(path).parts), 'No final-test evidence selection')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def child(root, name):
    name = str(name).replace('\\', '/')
    require(name and ':' not in name and not name.startswith('/') and
        all(part not in ('', '.', '..') for part in name.split('/')), 'Unsafe relative evidence path')
    path = (Path(root)/name).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Evidence escaped root')
    return path


def utc(value):
    stamp = datetime.fromisoformat(value)
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0, 'Explicit UTC time required')
    return stamp


def menu(phase):
    if phase == 'A':
        return [(0, 'raw'), (0, 'ema99')] + [(step, weight) for step in (2000,4000,6000,8000)
            for weight in ('raw','ema99','ema999')]
    require(phase == 'B', 'Unknown phase')
    return [(0, 'raw')] + [(step, weight) for step in (4000,8000,12000,16000,20000) for weight in ('raw','ema')]


def ranks(name):
    rank = 176 if name == 'expanded' else 144
    return dict(qkv=160, proj=0, gate=rank, value=rank, out=rank)


def score_check(row):
    require(row.get('targets') == 376599 and row.get('utf8_bytes') == 1148007, 'Incomplete validation split')
    require(all(finite(row.get(key)) for key in ('bpb','nll_nats','token_ppl','seconds')), 'Invalid scoring quantity')
    require(abs(row['bpb']-row['nll_nats']/math.log(2)/1148007) <= 1e-10, 'Wrong BPB denominator')


def verify_recipe(record, phase, name, plan, parent_sha):
    identity, recipe = record['identity'], record['recipe']
    require(identity['plan_sha256'] == record['plan_sha256'] == PLAN_SHA and identity['recipe'] == recipe,
        'Plan/recipe identity differs')
    require(record['parent_sha256'] == parent_sha and record['parent_train_targets'] == LINEAGE and
        record['test_used'] is False and record['source_splits'] == ['train','validation'] and
        record['training_implementation'] == record['inference_implementation'] == 'push150_expand' and
        record['implementation_sha256'] == IMPLEMENTATION_SHA and record['no_statistical_tables_in_training'] is True,
        'Wrong parent/source/implementation')
    require(record['source_sha256'] == identity['source_sha256'] and record['source_sha256'] and
        record['data_sha256'] == identity['data_sha256'] == DATA_SHA and
        record['source_sha256'].get(DRIVERS[phase][0]) == DRIVERS[phase][1], 'Frozen source/data identity differs')
    require(identity['implementation'] == 'push150_expand' and identity['runtime']['cpu_threads'] == 4 and
        identity['runtime']['device'] == 'cuda:0' and identity['runtime']['cuda_device_count'] == 1,
        'GPU training runtime differs')
    require(record['parameters'] == 5597331+(681984 if name == 'expanded' else 0) and
        record['processed_view_targets_per_step'] == record['sampled_label_targets_per_step'] == TARGETS_PER_STEP,
        'Wrong parameter/target accounting')
    train = recipe['training_config']; evaluation = recipe['validation_config']
    shared_config = dict(vocab=2048, width=320, heads=8, depth=6, context=256, ffn_hidden=864,
        dropout=.2, push150_ranks=ranks(name))
    require(all(train.get(key) == value and evaluation.get(key) == value for key,value in shared_config.items()),
        'Undeclared architecture/dropout')
    require(train['temperature'] == 1.15 and not any(key.startswith(('ngram_', 'mkn_')) for key in train) and
        all(evaluation.get(key) == value for key,value in dict(temperature=1.3,mkn_weight=.15,mkn_order=5,discount_scale=.7).items()),
        'Changed training objective tables or fixed validation calibration')
    common = dict(optimizer='muon', batch_size=32, context=256, training_precision='bf16',
        training_temperature=1.15, validation_temperature=1.3, validation_mkn_weight=.15, dropout=.2, gradient_clip=1.)
    require(all(recipe.get(key) == value for key,value in common.items()), 'Shared recipe changed')
    if phase == 'A':
        expected = dict(phase='A', phase_name=name, steps=8000, eval_every=2000, schedule='constant',
            learning_rates=plan['phase_a']['learning_rates'], optimizer_state_kept=(name == 'optimizer-keep'),
            ema_decays=dict(ema99=.99,ema999=.999), validation_precision='fp32',
            validation_choices_per_completed_run=14, additional_calibration_choices=0,
            omitted_unused_oof_target_lookup=True, objective='pure neural-plus-pointer target NLL')
        require(all(recipe.get(key) == value for key,value in expected.items()) and
            identity['phase_name'] == name and identity['fork_training_state_sha256'] == OLD_STATE_SHA and
            identity['statistics_parent_sha256'] == PARENT_SHA and record['fork_training_state_sha256'] == OLD_STATE_SHA,
            'Complete-state fork recipe differs')
        require(record['cached_target_probability_lookups'] == 0 and record['new_statistics_scans'] == 0 and
            record['same_ancestor_not_summed'] is True, 'Unexpected Phase-A learning/cache cost')
    else:
        expected = dict(branch=name, steps=20000, eval_every=4000, seed=193, schedule='cosine', warmup=200,
            decay_steps=4000, lr_floor=.1, backbone_lr=7.5e-6, head_lr=7.5e-5, scalar_lr=.00025,
            muon_lr=.0005, ema_decay=.99, validation_choices=11, all_validation_snapshots_saved=True,
            auxiliary_weight=.1 if name == 'auxiliary' else 0., dropout_views_per_sample=1,
            adamw_betas=[.9,.95], backbone_weight_decay=.1, pointer_weight_decay=.01, scalar_weight_decay=0.,
            muon_momentum=.95, muon_ns_steps=5)
        require(all(recipe.get(key) == value for key,value in expected.items()) and
            identity['branch'] == record['branch'] == name and identity['parent_sha256'] == parent_sha and
            identity['statistical_reference_sha256'] == PARENT_SHA, 'Fresh-optimizer branch recipe differs')


def check_metrics(record, phase, name, plan, parent_sha):
    """Pure metadata gate; artifact bytes/source pins are checked by collect()."""
    require(record['status'] in ('completed','failed'), 'Run is still pending/running, not selectable')
    verify_recipe(record, phase, name, plan, parent_sha)
    completed, discarded = record['completed_steps'], record['discarded_completed_steps']
    total_steps = 8000 if phase == 'A' else 20000
    require(type(completed) is int and type(discarded) is int and 0 <= completed <= total_steps and discarded >= 0,
        'Invalid completed/discarded update count')
    targets = (completed+discarded)*TARGETS_PER_STEP
    require(record['incremental_search_targets'] == record['incremental_sampled_label_targets'] == targets and
        record['lineage_new_targets'] == completed*TARGETS_PER_STEP and finite(record['seconds']), 'Training cost differs')
    history = record['validation_history']; expected = menu(phase)
    require([(row['step'],row['weights']) for row in history] == expected[:len(history)] and len(history) <= len(expected),
        'Validation menu has an extra, duplicate, missing or reordered choice')
    if record['status'] == 'completed':
        require(completed == total_steps and len(history) == len(expected), 'Completed run lacks its entire fixed menu')
    else:
        require(isinstance(record.get('failure_type'),str) and isinstance(record.get('failure_message'),str),
            'Failed metrics must preserve failure details')
    snapshots = record['snapshots' if phase == 'A' else 'validation_snapshots']
    require(len(snapshots) == len(history), 'Snapshot evidence and scored history differ')
    candidates = []
    run_dir = plan['phase_a' if phase == 'A' else 'phase_b']['variants'][name]['run_dir']
    for row, snapshot in zip(history, snapshots):
        score_check(row)
        step, weight, session = row['step'], row['weights'], row['session']
        require(type(session) is int and session >= 0 and step <= completed, 'Invalid snapshot step/session')
        filename = (f'session{session:03d}-step{step:08d}-{weight}.pt' if phase == 'A' else
                    f'validation-session{session:03d}-step{step:08d}-{weight}.pt')
        require(row['checkpoint_file'] == snapshot['path'] == filename and
            row['checkpoint_sha256'] == snapshot['sha256'] and snapshot['step'] == step and snapshot['weights'] == weight,
            'Scored snapshot identity differs')
        lineage = LINEAGE+step*TARGETS_PER_STEP
        measured = dict(row)
        identity_fields = dict(protocol=PROTOCOL, split='validation', precision='fp32', device='cuda:0',
            checkpoint_sha256=row['checkpoint_sha256'], implementation='push150_expand',
            implementation_sha256=IMPLEMENTATION_SHA, evaluator_sha256=SCORER_SHA,
            tokenizer_sha256=DATA_SHA['data/tokenizer.json'], test_used=False)
        if phase == 'A':
            require(row['selected_lineage_targets'] == lineage and all(row.get(key) == value
                for key,value in identity_fields.items() if key != 'implementation'), 'Phase-A scored identity differs')
        measured.update(identity_fields)
        gain = plan['parent']['validation_bpb_gpu_fp32']-row['bpb']
        candidates.append(dict(phase=phase,name=name,run_dir=run_dir,checkpoint=run_dir+'/'+filename,
            checkpoint_sha256=row['checkpoint_sha256'],step=step,weights=weight,session=session,
            selected_lineage_targets=lineage,parameters=record['parameters'],expected_ranks=ranks(name),
            validation_bpb_gpu_fp32=row['bpb'],measured=measured,improvement_vs_parent=gain,
            passes_validation_improvement_gate=gain >= MIN_GAIN,cpu_resource_status='not_yet_audited',
            scoring_identity_source='Actual scored row' if phase == 'A' else
                'Score quantities from actual row; protocol fields from pinned Phase-B scorer call/recipe'))
    if history:
        best = min(history, key=lambda row: row['bpb'])
        require(all(record['best'].get(key) == value for key,value in best.items()) and
            record['best']['selected_lineage_targets'] == LINEAGE+best['step']*TARGETS_PER_STEP,
            'Best checkpoint does not correspond to the earliest minimum retained validation score')
    discarded_val = record.get('discarded_validation_passes', 0) if phase == 'A' else len(record['discarded_validation_history'])
    require(type(discarded_val) is int and discarded_val >= 0, 'Invalid discarded validation cost')
    if phase == 'A':
        coins = record['count_block_coins_including_discarded']
        require(coins['drawn'] == coins['kept']+coins['dropped'] and
            coins['drawn'] >= completed+discarded and record['omitted_unused_oof_target_lookups'] == coins['drawn']*TARGETS_PER_STEP,
            'Inherited coin/omitted lookup accounting differs')
        if record['status'] == 'completed':
            require(record['count_block_coins']['drawn'] == completed, 'Completed trajectory has extra/missing count coins')
        require(record['validation_completed_records'] >= len(history)+discarded_val and
            record['validation_attempts_recorded'] >= record['validation_completed_records'], 'Validation attempt accounting differs')
        scored_calls = record['validation_completed_records']
    else:
        require(record['completed_validation_calls'] == len(history)+discarded_val and
            record['mixed_nll_target_exposures'] == targets and
            record['auxiliary_neural_nll_target_exposures'] == (targets if name == 'auxiliary' else 0),
            'Phase-B objective/validation exposure cost differs')
        scored_calls = record['completed_validation_calls']
    costs = dict(completed_steps=completed,discarded_completed_steps=discarded,
        incremental_search_targets=targets,incremental_sampled_label_targets=targets,
        lineage_new_targets=completed*TARGETS_PER_STEP,completed_validation_calls=scored_calls,
        retained_validation_choices=len(history),discarded_validation_calls=discarded_val,seconds=record['seconds'],
        auxiliary_neural_nll_target_exposures=targets if name == 'auxiliary' else 0,
        partial_failed_work_unquantified=(record['status']=='failed'))
    return candidates if record['status'] == 'completed' else [], costs


def check_terminal(terminal, phase, name, authorization, costs=None):
    expected = dict(phase=phase,name=name,plan_sha256=PLAN_SHA,driver_sha256=DRIVERS[phase][1],
        test_used=False,no_running_processes=True)
    require(all(terminal.get(key) == value for key,value in expected.items()) and
        terminal.get('status') in ('skipped','failed') and isinstance(terminal.get('reason'),str) and terminal['reason'],
        'Missing explicit terminal disposition')
    for key in ('completed_steps','discarded_completed_steps','incremental_search_targets','lineage_new_targets'):
        require(type(terminal.get(key)) is int and terminal[key] >= 0, 'Terminal work count missing')
    require(terminal['incremental_search_targets'] == (terminal['completed_steps']+terminal['discarded_completed_steps'])*TARGETS_PER_STEP
        and terminal['lineage_new_targets'] == terminal['completed_steps']*TARGETS_PER_STEP and
        type(terminal.get('partial_failed_work_unquantified')) is bool, 'Terminal cost arithmetic differs')
    if terminal['status'] == 'skipped':
        require(phase == 'B' and name == 'expanded' and authorization['training_authorized'] is False and
            terminal['completed_steps'] == terminal['discarded_completed_steps'] == 0 and
            terminal['metrics_sha256'] is None and terminal['partial_failed_work_unquantified'] is False,
            'Only the failed-resource expansion may be skipped without training')
    else:
        require(isinstance(terminal.get('failure_type'),str) and isinstance(terminal.get('failure_message'),str)
            and terminal['partial_failed_work_unquantified'] is True, 'Failure/partial-work disclosure missing')
        if costs is not None:
            require(all(terminal[key] == costs[key] for key in
                ('completed_steps','discarded_completed_steps','incremental_search_targets','lineage_new_targets')),
                'Failure terminal and actual metrics costs disagree')
        else:
            require(terminal['metrics_sha256'] is None and terminal['completed_steps'] ==
                terminal['discarded_completed_steps'] == 0, 'Missing metrics for claimed updates')


def rank_candidates(candidates):
    order = {'raw':0,'ema':1,'ema99':1,'ema999':2}
    qualified = [row for row in candidates if row['passes_validation_improvement_gate']]
    return sorted(qualified, key=lambda row: (row['validation_bpb_gpu_fp32'], NAMES.index(row['name']),
        row['step'], order[row['weights']]))


def endpoint_comparisons(candidates):
    endpoints = {(row['name'],row['weights']):row for row in candidates
        if row['step'] == (8000 if row['phase']=='A' else 20000)}
    result = dict(phase_a={},phase_b={},note='Matched final-budget endpoints, not per-run best/early-stopped values.')
    for weight in ('raw','ema99','ema999'):
        keep, reset = endpoints.get(('optimizer-keep',weight)), endpoints.get(('optimizer-reset',weight))
        result['phase_a'][weight] = dict(available=bool(keep and reset),
            keep_bpb=keep['validation_bpb_gpu_fp32'] if keep else None,
            reset_bpb=reset['validation_bpb_gpu_fp32'] if reset else None,
            reset_minus_keep_bpb=reset['validation_bpb_gpu_fp32']-keep['validation_bpb_gpu_fp32'] if keep and reset else None)
    for weight in ('raw','ema'):
        control = endpoints.get(('control',weight))
        for name in ('auxiliary','expanded'):
            other = endpoints.get((name,weight))
            result['phase_b'][name+'-'+weight] = dict(available=bool(control and other),
                control_bpb=control['validation_bpb_gpu_fp32'] if control else None,
                candidate_bpb=other['validation_bpb_gpu_fp32'] if other else None,
                candidate_minus_control_bpb=other['validation_bpb_gpu_fp32']-control['validation_bpb_gpu_fp32'] if control and other else None)
    return result


def pin(root, relative, expected, evidence):
    path = child(root, relative)
    require(isinstance(expected,str) and len(expected) == 64 and sha(path) == expected, 'Artifact/source changed: '+relative)
    require(relative not in evidence or evidence[relative] == expected, 'Conflicting evidence digest')
    evidence[relative] = expected
    return path


def source_map(root, values, evidence, sources):
    for name, digest in values.items():
        require(name.endswith('.py'), 'Expected executable Python source')
        pin(root, name, digest, evidence)
        require(name not in sources or sources[name] == digest, 'Different frozen source across runs')
        sources[name] = digest


def check_launch(root, phase, name, status, evidence):
    base = 'runs/push152-launches/'+name
    started_path, process_path = child(root, base+'.started.json'), child(root, base+'.process.json')
    started, process = read(started_path), read(process_path)
    driver, driver_sha = DRIVERS[phase]
    argv = ([name,'--plan',PLAN] if phase == 'A' else ['--branch',name,'--plan',PLAN])
    command = [str(root/'.venv-gpu/Scripts/python.exe'),'-B',str(root/driver),*argv]
    require(started.get('phase') == name and started['command'] == command and Path(started['cwd']).resolve() == root and
        started['plan_sha256'] == PLAN_SHA and started['script_sha256'] == driver_sha and
        started['launcher_sha256'] == LAUNCHER_SHA and started['test_used'] is False,
        'Training launch command/source identity differs')
    require(process['phase'] == name and process['started_record_sha256'] == sha(started_path) and
        process['script_unchanged'] is True and process['test_used'] is False and finite(process['seconds_total']) and
        utc(started['started_at_utc']) <= utc(process['completed_at_utc']), 'Training process evidence differs')
    if status == 'completed':
        require(process['returncode'] == 0 and process['failure'] is None, 'Completed metrics need a successful process exit')
    else:
        require(process['returncode'] != 0 or process['failure'] is not None, 'Failure disposition conflicts with successful process')
    for suffix in ('.started.json','.process.json'):
        pin(root, base+suffix, sha(child(root,base+suffix)), evidence)
    if process['output_sha256'] is not None:
        pin(root, base+'.output.log', process['output_sha256'], evidence)
    return process['seconds_total']


def collect(root=ROOT):
    root = Path(root).resolve(); evidence, sources = {}, {}
    plan_path = pin(root, PLAN, PLAN_SHA, evidence); plan = read(plan_path)
    require(plan['selection']['minimum_improvement'] == MIN_GAIN and plan['selection']['maximum_training_runs'] == 5
        and plan['parent']['sha256'] == PARENT_SHA and plan['parent']['selected_lineage_targets'] == LINEAGE,
        'Fixed selection menu/gain/parent changed')
    require(set(plan['phase_a']['variants']) == set(NAMES[:2]) and set(plan['phase_b']['variants']) == set(NAMES[2:]),
        'Unexpected declared run menu')
    pin(root, plan['parent']['path'], PARENT_SHA, evidence)
    pin(root, plan['phase_a']['state'], OLD_STATE_SHA, evidence)
    source_map(root, {driver:digest for driver,digest in DRIVERS.values()}, evidence, sources)
    source_map(root, {'push152_launch.py':LAUNCHER_SHA,'push152_preflight.py':PREFLIGHT_SHA}, evidence, sources)
    auth_name = plan['expansion']['authorization']; auth_path = child(root,auth_name)
    pin(root, auth_name, sha(auth_path), evidence); authorization = read(auth_path)
    require(authorization['status'] == 'completed' and authorization['plan_sha256'] == PLAN_SHA and
        authorization['test_used'] is False and type(authorization['training_authorized']) is bool and
        authorization['all_limits_pass'] is authorization['training_authorized'] and
        authorization['no_layout_fallback'] is True and authorization['script_sha256'] == PREFLIGHT_SHA,
        'Completed expansion preflight decision required')
    for field in ('audit','timing_summary'):
        pin(root, authorization[field], authorization[field+'_sha256'], evidence)
    candidates, runs = [], []
    validation_config_reference = None
    for index, name in enumerate(NAMES):
        phase = 'A' if index < 2 else 'B'
        run_dir = plan['phase_a' if phase == 'A' else 'phase_b']['variants'][name]['run_dir']
        require(run_dir == 'runs/push152-'+name, 'Run directory differs')
        directory = child(root, run_dir); metrics_path = directory/'metrics.json'; terminal_path = directory/'terminal.json'
        record = read(metrics_path) if metrics_path.is_file() else None
        terminal = read(terminal_path) if terminal_path.is_file() else None
        if record is None:
            require(terminal is not None, 'Unaccounted run: '+name)
            check_terminal(terminal,phase,name,authorization)
            if terminal['status'] == 'skipped':
                require(terminal['authorization_sha256'] == evidence[auth_name] and
                    not child(root,'runs/push152-launches/'+name+'.started.json').exists(), 'Skipped branch was launched or mis-authorized')
                process_seconds = 0.
            else:
                process_seconds = check_launch(root,phase,name,'failed',evidence)
            costs = dict(completed_steps=0,discarded_completed_steps=0,incremental_search_targets=0,
                incremental_sampled_label_targets=0,lineage_new_targets=0,completed_validation_calls=0,
                retained_validation_choices=0,discarded_validation_calls=0,seconds=0.,
                auxiliary_neural_nll_target_exposures=0,partial_failed_work_unquantified=terminal['partial_failed_work_unquantified'])
            rows, status = [], terminal['status']
        else:
            pin(root, run_dir+'/metrics.json', sha(metrics_path), evidence)
            parent_sha = authorization['initializer_checkpoint_sha256'] if name == 'expanded' else PARENT_SHA
            rows, costs = check_metrics(record,phase,name,plan,parent_sha)
            status = record['status']
            source_map(root,record['source_sha256'],evidence,sources)
            current_config = {key:value for key,value in record['recipe']['validation_config'].items() if key != 'push150_ranks'}
            if validation_config_reference is None:
                validation_config_reference = current_config
            require(current_config == validation_config_reference, 'Inference settings/statistical shapes changed across runs')
            process_seconds = check_launch(root,phase,name,status,evidence)
            if status == 'completed':
                require(terminal is None, 'Completed run has a conflicting terminal disposition')
                state_name = run_dir+'/train_state.pt'
                pin(root,state_name,sha(child(root,state_name)),evidence)
                if name == 'expanded':
                    require(authorization['training_authorized'] is True, 'Expansion trained despite failed feasibility')
            else:
                require(terminal is not None and terminal['metrics_sha256'] == sha(metrics_path), 'Failed run needs bound explicit disposition')
                check_terminal(terminal,phase,name,authorization,costs)
            pins = record['identity']['pinned_input_sha256' if phase == 'A' else 'resource_authorization_sha256']
            for path_name,digest in pins.items():
                path = Path(path_name)
                if path.is_absolute():
                    require(path.resolve().is_relative_to(root), 'Input pin outside model workspace')
                    path_name = path.resolve().relative_to(root).as_posix()
                pin(root,path_name,digest,evidence)
            for original in record['validation_history']:
                checkpoint = run_dir+'/'+original['checkpoint_file']
                pin(root,checkpoint,original['checkpoint_sha256'],evidence)
                if phase == 'A':
                    basename = Path(original['checkpoint_file']).stem
                    measured_name = run_dir+'/'+basename+'-validation.json'
                    pin(root,measured_name,sha(child(root,measured_name)),evidence)
                    require(read(child(root,measured_name)) == original, 'Actual exclusive validation record differs')
                    pin(root,run_dir+'/'+basename+'.validation.window-nll.npy',original['window_nll_sha256'],evidence)
                    start_name = run_dir+'/validation-started-'+basename+'.json'
                    pin(root,start_name,sha(child(root,start_name)),evidence)
                    start = read(child(root,start_name))
                    require(start['plan_sha256'] == PLAN_SHA and start['test_used'] is False and
                        start['checkpoint_sha256'] == original['checkpoint_sha256'], 'Validation launch identity differs')
            if record.get('best'):
                pin(root,run_dir+'/checkpoint.pt',record['best']['scoring_checkpoint_sha256'],evidence)
        if terminal is not None:
            pin(root,run_dir+'/terminal.json',sha(terminal_path),evidence)
        # Retain all small branch evidence, including failed attempts/replays.
        if directory.is_dir():
            for path in sorted(directory.glob('*.json')):
                pin(root,path.relative_to(root).as_posix(),sha(path),evidence)
        candidates.extend(rows)
        runs.append(dict(phase=phase,name=name,run_dir=run_dir,status=status,costs=costs,
            process_seconds=process_seconds,terminal=terminal,selected_snapshots_eligible=(status=='completed')))
    qualified = rank_candidates(candidates)
    known_targets = sum(row['costs']['incremental_search_targets'] for row in runs)
    require(known_targets <= plan['maximum_new_training_target_occurrences'], 'Recorded target budget exceeded frozen plan')
    source_map(root,{'push152_select.py':sha(root/'push152_select.py')},evidence,sources)
    result = dict(status='validation_qualified_candidates_need_resource_audit' if qualified else 'no_new_validation_qualified_candidate',
        round='push152',created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=PLAN_SHA,
        script_sha256=sha(root/'push152_select.py'),selection_split='validation',test_used=False,
        prior_test_exposure_disclosed=True,parent=plan['parent'],all_declared_runs_accounted=True,
        all_completed_runs_have_exact_predeclared_validation_menu=True,minimum_improvement=MIN_GAIN,
        maximum_resource_candidates=3,tie_order=dict(run_order=list(NAMES),step='ascending',weights=['raw','ema99_or_ema','ema999']),
        run_records=runs,candidates=candidates,validation_qualified_in_rank_order=qualified,resource_audit_candidates=qualified[:3],
        resource_audit_directories=['runs/push152-resource-rank'+str(index) for index in range(min(3,len(qualified)))],
        same_budget_endpoint_comparisons=endpoint_comparisons(candidates),
        costs=dict(new_neural_gradient_target_occurrences=known_targets,
            new_sampled_label_target_occurrences=sum(row['costs']['incremental_sampled_label_targets'] for row in runs),
            discarded_update_target_occurrences=sum(row['costs']['discarded_completed_steps'] for row in runs)*TARGETS_PER_STEP,
            completed_training_validation_calls=sum(row['costs']['completed_validation_calls'] for row in runs),
            retained_validation_choices=len(candidates),
            auxiliary_neural_nll_target_exposures=sum(row['costs']['auxiliary_neural_nll_target_exposures'] for row in runs),
            shared_parent_targets_per_selected_lineage=LINEAGE,shared_parent_not_added_as_new_training=True,
            training_seconds=sum(row['costs']['seconds'] for row in runs),
            process_wall_seconds=sum(row['process_seconds'] for row in runs),
            partial_failed_work_unquantified=any(row['costs']['partial_failed_work_unquantified'] for row in runs),
            preflight_validation_passes=6,preflight_costs_bound_by_evidence=True,
            interpretation='EMA choices do not add gradient targets; auxiliary two-term exposure is not doubled sampled data.'),
        no_new_calibration_or_averaging=True,final_cpu_resource_audit_and_freeze_required=bool(qualified),
        source_sha256=sources,evidence_sha256=evidence)
    # Last-moment pin check prevents a concurrent source/run mutation from being hidden.
    for name,digest in evidence.items():
        require(sha(child(root,name)) == digest, 'Selection evidence changed during collection: '+name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/push152_selection.json')
    args = parser.parse_args()
    require(args.output.resolve() == (ROOT/'runs/push152_selection.json').resolve(), 'Fixed exclusive selection output only')
    require(not args.output.exists(), 'Never overwrite an existing selection')
    report = collect()
    write_new(args.output,report)
    print(json.dumps(dict(status=report['status'],eligible_snapshots=len(report['validation_qualified_in_rank_order']),
        resource_candidates=len(report['resource_audit_candidates']),costs=report['costs']),indent=2),flush=True)


if __name__ == '__main__':
    main()
