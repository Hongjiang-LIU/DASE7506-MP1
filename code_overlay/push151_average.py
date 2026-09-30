"""Two predeclared self-trained weight averages; build and score are separate.

No optimizer, new training, test access, input memoization, prediction ensemble,
temperature grid, or count-table averaging. Score uses full validation once per
candidate, including the unchanged parent reference. Existing files are never
replaced, including after a failed or partial attempt.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from common import ROOT, PROTOCOL, make_model, setup
from evaluate import score
from push150_mkn_sweep import load_one_split


PARENT_SHA = 'afaf5f98c4c2148942454bb9295a41f7d41209b804c28db96676a5d9886eb9c7'
PLAN_SHA = '3ba6539a20ba74c7989b9733e755d48bd7529ae9446785e918ceb65f049063aa'
IMPLEMENTATION = 'push150_expand'
PARENT_TARGETS = 638976000
SHARED_BASE_TARGETS = 540672000
FIXED_CONFIG = dict(temperature=1.3, mkn_weight=.15, mkn_order=5, discount_scale=.7)
SNAPSHOTS = {
    'ema_10000': dict(step=10000, weights='ema', sha256='cd746e94ebf97684b411e956f3f6d35f52377fd9ec8ed5842eec5ed5500abef0'),
    'raw_12000': dict(step=12000, weights='raw', sha256='2d735180b7c68ece230374e90ae823dee4f2c23d918bf8e45ac2365cd674533f'),
    'ema_12000': dict(step=12000, weights='ema', sha256='ff7487be56b1c7d522252748dcf9e29d2f41a85737da21c8e7b995d55a9f57b8'),
}
CANDIDATES = [
    dict(name='tail-ema-average', sources=[dict(snapshot='ema_10000', weight=.5), dict(snapshot='ema_12000', weight=.5)]),
    dict(name='last-raw-ema-average', sources=[dict(snapshot='raw_12000', weight=.5), dict(snapshot='ema_12000', weight=.5)]),
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(not path.name.startswith('wikitext_'), 'This metadata hasher cannot open corpus text')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    require(not Path(path).name.startswith('wikitext_'), 'Metadata only')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def child(root, name):
    name = str(name).replace('\\', '/')
    require(name and not name.startswith('/') and ':' not in name and
            all(part not in ('', '.', '..') for part in name.split('/')), 'Unsafe relative path')
    result = (Path(root)/name).resolve()
    require(result.is_relative_to(Path(root).resolve()), 'Input escapes root')
    return result


def write_json(path, record):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')


def save_checkpoint(path, checkpoint):
    with Path(path).open('xb') as stream:
        torch.save(checkpoint, stream)


def validate_plan(plan):
    parent, phase = plan['parent'], plan['phase_a']
    require(parent['sha256'] == PARENT_SHA and parent['implementation'] == IMPLEMENTATION and
            parent['selected_lineage_targets'] == PARENT_TARGETS, 'Wrong declared parent')
    require(phase['control_name'] == 'control' and phase['fixed_config'] == FIXED_CONFIG and
            phase['allowed_snapshot_config_differences'] == ['temperature'], 'Changed averaging protocol')
    require(phase['candidates'] == CANDIDATES and set(phase['snapshots']) == set(SNAPSHOTS),
            'Only the two fixed averages and three fixed snapshots are allowed')
    for name, expected in SNAPSHOTS.items():
        require(all(phase['snapshots'][name][key] == value for key, value in expected.items()),
                'Changed declared snapshot: '+name)
    return phase


def source_hashes(root=ROOT):
    root = Path(root)
    pending = [Path(__file__).name, IMPLEMENTATION+'.py']
    found = {}
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = child(root, name); found[name] = sha(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                       [entry.name for entry in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                local = module.replace('.', '/')+'.py'
                if child(root, local).is_file():
                    pending.append(local)
    return dict(sorted(found.items()))


def verify_pins(pins):
    for path, expected in pins.items():
        require(sha(path) == expected, 'Pinned source/input changed: '+str(path))


def same_tensor(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)


def parameter_groups(model):
    """Group every parameter name by Parameter identity, including tied aliases."""
    groups = {}
    for name, value in model.named_parameters(remove_duplicate=False):
        require(value.dtype == torch.float32, 'Only ordinary FP32 parameters may be averaged')
        groups.setdefault(id(value), []).append(name)
    require(model.token.weight is model.head.weight, 'Embedding and output head must remain tied')
    require(any(set(names) == {'token.weight', 'head.weight'} for names in groups.values()), 'Unexpected embedding alias group')
    return list(groups.values())


def average_state(parent_state, states, model, weights=(.5, .5)):
    require(len(states) == 2 and tuple(weights) == (.5, .5), 'Exactly the predeclared arithmetic half average is allowed')
    expected = model.state_dict()
    require(set(parent_state) == set(expected) and all(set(state) == set(expected) for state in states), 'Model state key sets differ')
    groups = parameter_groups(model)
    parameter_names = {name for names in groups for name in names}
    for name, reference in expected.items():
        for state in [parent_state, *states]:
            value = state[name]
            require(torch.is_tensor(value) and value.shape == reference.shape and value.dtype == reference.dtype,
                    'Shape or dtype differs: '+name)
            if value.is_floating_point():
                require(bool(torch.isfinite(value).all()), 'Nonfinite checkpoint tensor: '+name)
        if name not in parameter_names:
            require(all(same_tensor(parent_state[name], state[name]) for state in states), 'Non-parameter buffer differs: '+name)
    for names in groups:
        for state in [parent_state, *states]:
            require(all(same_tensor(state[names[0]], state[alias]) for alias in names), 'Tied parameter values disagree')
    result = {name: value.detach().cpu().clone() for name, value in parent_state.items() if name not in parameter_names}
    for names in groups:
        # CPU FP64 accumulation prevents intermediate overflow; the final
        # learned parameters are rounded once to ordinary FP32, never deployed
        # as FP64 and never treated as an inference-time ensemble.
        value = (states[0][names[0]].detach().cpu().double()*.5 +
                 states[1][names[0]].detach().cpu().double()*.5).float()
        require(bool(torch.isfinite(value).all()), 'Nonfinite parameter average')
        for alias in names:
            result[alias] = value
    model.load_state_dict(result, strict=True)
    require(model.token.weight is model.head.weight, 'Averaging broke the tied embedding')
    return result


def verify_checkpoints(parent, snapshots):
    require(parent['protocol'] == PROTOCOL and parent['implementation'] == IMPLEMENTATION and
            parent['train_tokens'] == PARENT_TARGETS, 'Parent protocol/implementation/lineage differs')
    require(all(parent['config'].get(key) == value for key, value in FIXED_CONFIG.items()) and parent['config'].get('mkn_shapes'),
            'Parent fixed inference configuration differs')
    require(set(snapshots) == set(SNAPSHOTS), 'Missing snapshot')
    base_hashes, recipes, ancestries = [], [], []
    for name, snap in snapshots.items():
        expected = SNAPSHOTS[name]
        require(snap['protocol'] == PROTOCOL and snap['implementation'] == IMPLEMENTATION and
                snap['selected_step'] == expected['step'] and snap['weights'] == expected['weights'] and
                snap['train_tokens'] == SHARED_BASE_TARGETS+expected['step']*8192,
                'Snapshot step/type/lineage differs: '+name)
        before = {key: value for key, value in parent['config'].items() if key != 'temperature'}
        after = {key: value for key, value in snap['config'].items() if key != 'temperature'}
        require(before == after and snap['config']['temperature'] == 1.25, 'Snapshot architecture/statistics/config differs')
        base_hashes.append(snap['parent_sha256']); recipes.append(snap['training_recipe']); ancestries.append(snap.get('ancestry', {}))
    require(len(set(base_hashes)) == 1 and all(recipe == recipes[0] for recipe in recipes) and
            all(ancestry == ancestries[0] for ancestry in ancestries), 'Snapshots are not the same shared training trajectory')
    require(recipes[0]['seed'] == 211 and recipes[0]['steps'] == 12000, 'Different D-control trajectory')
    last = snapshots['ema_12000']['model']
    require(set(parent['model']) == set(last) and all(same_tensor(value, last[name]) for name, value in parent['model'].items()),
            'The calibrated parent is not exactly the last EMA weights/buffers')


def make_average(parent, sources, model, specification, provenance):
    result = copy.deepcopy(parent)
    state = average_state(parent['model'], [sources[item['snapshot']]['model'] for item in specification['sources']], model)
    result.update(model=state, config=copy.deepcopy(parent['config']), parent_sha256=PARENT_SHA,
        train_tokens=PARENT_TARGETS, selected_step=None, weights='fixed_half_weight_average',
        selection='predeclared train-derived parameter average; validation-only selection pending',
        new_training_targets=0, inference_architecture_unchanged_from_parent=True,
        averaging=dict(name=specification['name'],sources=copy.deepcopy(specification['sources']),
            provenance=copy.deepcopy(provenance),shared_trajectory_union_targets=PARENT_TARGETS,
            ancestor_targets_not_summed=True,nonparameter_buffers_from_parent_bitwise_unchanged=True,
            tied_embedding_preserved=True,accumulation_dtype='float64',saved_parameter_dtype='float32'))
    result.setdefault('ancestry', {})['push151_weight_averaging'] = copy.deepcopy(result['averaging'])
    return result


def identity_inputs(args):
    require(sha(args.plan) == PLAN_SHA, 'The frozen push151 pre-execution plan changed')
    plan = read(args.plan); phase = validate_plan(plan)
    sources = source_hashes()
    pins = {args.plan.resolve(): sha(args.plan)}
    pins.update({ROOT/name: digest for name, digest in sources.items()})
    parent_path = child(ROOT, plan['parent']['path']); pins[parent_path] = PARENT_SHA
    paths = {name: child(ROOT, record['path']) for name, record in phase['snapshots'].items()}
    pins.update({paths[name]: SNAPSHOTS[name]['sha256'] for name in paths})
    verify_pins(pins)
    return plan, phase, sources, pins, parent_path, paths


def run_build(args):
    require(not args.run_dir.exists(), 'Preserve an existing build, including partial failures')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); built = []
    try:
        torch.set_num_threads(2)
        plan, phase, sources, pins, parent_path, paths = identity_inputs(args)
        parent = torch.load(parent_path, map_location='cpu', weights_only=True)
        snapshots = {name: torch.load(path, map_location='cpu', weights_only=True) for name, path in paths.items()}
        verify_checkpoints(parent, snapshots)
        model, implementation_sha = make_model(IMPLEMENTATION, parent['config'], torch.device('cpu'))
        model.load_state_dict(parent['model'], strict=True)
        provenance = dict(plan_sha256=sha(args.plan),parent_sha256=PARENT_SHA,source_sha256=sources,
            snapshot_sha256={name: SNAPSHOTS[name]['sha256'] for name in paths})
        identity = dict(status='declared_build',test_used=False,**provenance,plan_path=str(args.plan.resolve()),
            parent_path=plan['parent']['path'],snapshot_paths={name: phase['snapshots'][name]['path'] for name in paths},
            implementation=IMPLEMENTATION,implementation_sha256=implementation_sha,fixed_config=FIXED_CONFIG,
            new_training_targets=0,shared_trajectory_union_targets=PARENT_TARGETS)
        write_json(args.run_dir/'identity.json', identity)
        built.append(dict(name='control',checkpoint=plan['parent']['path'],checkpoint_sha256=PARENT_SHA,
                          checkpoint_bytes=parent_path.stat().st_size,original_parent_reference=True))
        for spec in CANDIDATES:
            candidate = make_average(parent, snapshots, model, spec, provenance)
            directory = args.run_dir/spec['name']; directory.mkdir()
            path = directory/'checkpoint.pt'; save_checkpoint(path, candidate)
            verify_pins(pins)
            built.append(dict(name=spec['name'],checkpoint=path.resolve().relative_to(ROOT.resolve()).as_posix(),
                checkpoint_sha256=sha(path),checkpoint_bytes=path.stat().st_size,
                original_parent_reference=False,new_training_targets=0,train_tokens=PARENT_TARGETS,
                sources=spec['sources']))
        verify_pins(pins)
        report = dict(status='completed',phase='build',test_used=False,identity_sha256=sha(args.run_dir/'identity.json'),
            plan_sha256=sha(args.plan),source_sha256=sources,candidates=built,new_training_targets=0,
            shared_trajectory_union_targets=PARENT_TARGETS,unique_ancestry_not_multiplied=True,
            buffers_and_tied_parameters_verified=True,seconds_total=time.perf_counter()-started)
        write_json(args.run_dir/'build.json', report)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        write_json(args.run_dir/'build_failure.json', dict(status='failed',failure_type=type(error).__name__,
            failure_message=str(error),candidates_built=built,test_used=False,new_training_targets=0,
            seconds_total=time.perf_counter()-started,no_automatic_retry=True))
        raise


def run_score(args):
    started_path = args.run_dir/'score_started.json'
    require(not started_path.exists() and not (args.run_dir/'score.json').exists() and
            not (args.run_dir/'score_failure.json').exists(), 'Do not retry a complete or partial scoring attempt')
    plan, phase, sources, pins, parent_path, paths = identity_inputs(args)
    build, identity = read(args.run_dir/'build.json'), read(args.run_dir/'identity.json')
    require(build['status'] == 'completed' and build['test_used'] is False and build['source_sha256'] == sources and
            build['identity_sha256'] == sha(args.run_dir/'identity.json') and build['plan_sha256'] == sha(args.plan) and
            identity['plan_sha256'] == sha(args.plan) and identity['source_sha256'] == sources, 'Build/source/plan binding differs')
    require([item['name'] for item in build['candidates']] == ['control']+[item['name'] for item in CANDIDATES], 'Changed scoring candidates')
    pins[args.run_dir/'build.json'] = sha(args.run_dir/'build.json')
    pins[args.run_dir/'identity.json'] = sha(args.run_dir/'identity.json')
    for item in build['candidates']:
        pins[child(ROOT, item['checkpoint'])] = item['checkpoint_sha256']
    verify_pins(pins)
    write_json(started_path, dict(started_at_utc=datetime.now(timezone.utc).isoformat(),build_sha256=sha(args.run_dir/'build.json'),
        plan_sha256=sha(args.plan),device=args.device,precision='fp32',split='validation',test_used=False,
        declared_candidate_order=[item['name'] for item in build['candidates']],maximum_full_validation_passes=3))
    started = time.perf_counter(); rows = []
    try:
        device, precision = setup(args.device, 'fp32', 4)
        require(device.type == 'cuda', 'This predeclared development score requires GPU FP32')
        ids, metadata = load_one_split('validation')
        require(metadata['tokens']-1 == 376599 and metadata['utf8_bytes'] == 1148007, 'Validation counts differ')
        tokens = torch.tensor(ids, dtype=torch.long)
        for index, item in enumerate(build['candidates']):
            verify_pins(pins)
            path = child(ROOT, item['checkpoint'])
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            require(checkpoint['implementation'] == IMPLEMENTATION and checkpoint['train_tokens'] == PARENT_TARGETS and
                    all(checkpoint['config'].get(key) == value for key, value in FIXED_CONFIG.items()), 'Scoring candidate config differs')
            model, implementation_sha = make_model(IMPLEMENTATION, checkpoint['config'], device)
            model.load_state_dict(checkpoint['model'], strict=True); model.eval()
            directory = args.run_dir/item['name']; directory.mkdir(exist_ok=True)
            write_json(directory/'validation_started.json', dict(index=index,checkpoint_sha256=item['checkpoint_sha256'],
                score_started_sha256=sha(started_path),test_used=False,split='validation'))
            measured = score(model, tokens, metadata['utf8_bytes'], device, 'fp32')
            losses = measured.pop('window_nll_nats')
            require(measured['targets'] == 376599 and math.isfinite(measured['bpb']) and measured['bpb'] > 0, 'Invalid complete validation result')
            loss_path = directory/'validation.window-nll.npy'
            with loss_path.open('xb') as stream:
                np.save(stream, np.asarray(losses, dtype=np.float64), allow_pickle=False)
            row = dict(name=item['name'],checkpoint_sha256=item['checkpoint_sha256'],**measured,
                protocol=PROTOCOL,split='validation',precision='fp32',device=str(device),implementation=IMPLEMENTATION,
                implementation_sha256=implementation_sha,evaluator_sha256=sources['evaluate.py'],
                tokenizer_sha256=metadata['tokenizer_sha256'],test_used=False,
                window_nll_sha256=sha(loss_path),new_training_targets=0,selected_lineage_targets=PARENT_TARGETS)
            write_json(directory/'validation.json', row); rows.append(row)
            del model, checkpoint
        verify_pins(pins)
        result = dict(status='completed',phase='score',selection_split='validation',test_used=False,precision='fp32',
            plan_sha256=sha(args.plan),build_sha256=sha(args.run_dir/'build.json'),source_sha256=sources,
            validation=metadata,candidates=rows,best=min(rows,key=lambda row:row['bpb']),
            full_validation_passes=len(rows),new_training_targets=0,shared_trajectory_union_targets=PARENT_TARGETS,
            no_temperature_or_statistics_grid=True,final_cpu_resource_audit_pending=True,
            seconds_total=time.perf_counter()-started)
        write_json(args.run_dir/'score.json', result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        write_json(args.run_dir/'score_failure.json', dict(status='failed',failure_type=type(error).__name__,
            failure_message=str(error),measurements_completed=rows,test_used=False,
            seconds_total=time.perf_counter()-started,no_automatic_retry=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'score'])
    parser.add_argument('--plan', type=Path, default=ROOT/'runs/push151_plan.json')
    parser.add_argument('--run-dir', type=Path, default=ROOT/'runs/push151-average')
    parser.add_argument('--device', default='cuda:0', choices=['cuda:0'])
    args = parser.parse_args()
    require(args.run_dir.resolve().is_relative_to((ROOT/'runs').resolve()), 'Outputs must stay in code/runs')
    (run_build if args.command == 'build' else run_score)(args)


if __name__ == '__main__':
    main()
