"""Checkpoint-only FFN rank expansion of the fixed push151 predictor.

No corpus/tokenizer loading, prediction, scoring, optimization or timing. Old
FFN factors are retained exactly; only new left rows are random and new right
columns are zero. QKV and already-dense attention outputs are copied, not folded.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import torch

from common import ROOT, PROTOCOL
from lowrank import LowRankLinear
from push150_expand import ExpandedSignedCache, projection_modules, validate_layout


PARENT_SHA256 = 'c72cffff6b4152492518473f46a274b53272f59dacb80072cd9e1497d2e0bdf5'
PLAN_SHA256 = 'd15cd6276ace1894a37db7c005453c61acd2e5acb7e934cb6790f3860aa06b51'
SOURCE_LAYOUT = dict(qkv=160, proj=0, gate=144, value=144, out=144)
TARGET_LAYOUT = dict(qkv=160, proj=0, gate=176, value=176, out=176)
FFN_ROLES = frozenset(('gate', 'value', 'out'))
SEED = 193
LINEAGE = 638976000
PARAMETER_DELTA = 681984
IMPLEMENTATION_SHA256 = 'd0340092e2a2882113533f05f4668c48b721ee4416842910c2ba2ba1758e1d45'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(not path.name.lower().startswith('wikitext_'), 'Initializer never reads/hashes corpus text')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    require(Path(path).suffix.lower() == '.json', 'Only metadata JSON is accepted')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def save_new(path, value):
    with Path(path).open('xb') as stream:
        torch.save(value, stream)


def validate_plan(plan):
    parent, expansion = plan['parent'], plan['expansion']
    require(parent['path'] == 'runs/push151-combination/checkpoint.pt' and
            parent['sha256'] == PARENT_SHA256 and parent['implementation'] == 'push150_expand' and
            parent['implementation_sha256'] == IMPLEMENTATION_SHA256 and
            parent['selected_lineage_targets'] == LINEAGE, 'Plan parent identity differs')
    require(expansion['source_layout'] == SOURCE_LAYOUT and expansion['target_layout'] == TARGET_LAYOUT and
            expansion['seed'] == SEED and expansion['expected_parameter_increase'] == PARAMETER_DELTA and
            expansion['run_dir'] == 'runs/push152-expanded-init', 'Plan expansion differs')
    require(plan['shared']['mkn_weight'] == .15 and plan['shared']['mkn_order'] == 5 and
            plan['shared']['discount_scale'] == .7 and plan['shared']['validation_temperature'] == 1.3,
            'Plan inference configuration differs')


def source_hashes(root=ROOT):
    root = Path(root).resolve()
    pending = [root/'push152_init.py', root/'push150_expand.py']
    found = {}
    while pending:
        path = pending.pop().resolve()
        require(path.is_relative_to(root), 'Initializer source graph escaped code root')
        name = path.relative_to(root).as_posix()
        if name in found:
            continue
        found[name] = sha(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                       [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                local = root/(module.replace('.', '/')+'.py')
                if local.is_file():
                    pending.append(local)
    return dict(sorted(found.items()))


def verify_sources(pins, root=ROOT):
    root = Path(root).resolve()
    for name, digest in pins.items():
        path = (root/name).resolve()
        require(path.is_relative_to(root) and path.suffix == '.py' and sha(path) == digest,
                'Initializer source changed: '+name)


def checked_tensor(original, name, expected):
    value = original.get(name)
    require(isinstance(value, torch.Tensor) and value.dtype == expected.dtype and value.shape == expected.shape,
            'Missing, wrong-dtype or wrong-shape source tensor: '+name)
    require(not value.is_floating_point() or bool(torch.isfinite(value).all()), 'Nonfinite source tensor: '+name)
    return value.detach().cpu().clone()


@torch.no_grad()
def initialize_from_parent(parent, *, target_ffn_rank=176, seed=SEED):
    """Generic tiny-testable helper; production CLI fixes layout, seed and SHA."""
    require(parent.get('implementation') == 'push150_expand', 'Expected current mixed-layout ExpandedSignedCache')
    config = copy.deepcopy(parent['config'])
    original = parent['model']
    require(isinstance(original, dict), 'Source state must be a dictionary')
    source = config.get('push150_ranks')
    hidden = config.get('ffn_hidden', math.ceil((8*config['width']/3)/16)*16)
    source = validate_layout(source, config['width'], hidden)
    require(isinstance(source, dict) and source.get('proj') == 0 and source.get('qkv', 0) > 0,
            'Source requires factored QKV and already-dense projection')
    require(len({source.get(name) for name in FFN_ROLES}) == 1 and source.get('gate', 0) > 0,
            'Source requires one positive common FFN rank')
    require(type(target_ffn_rank) is int and target_ffn_rank > source['gate'], 'FFN rank must strictly grow')
    require(type(seed) is int and 0 <= seed < 2**63, 'Invalid explicit initialization seed')
    target = dict(source, **dict.fromkeys(FFN_ROLES, target_ffn_rank))
    # The original constructor owns width/hidden/layout validation.
    config['push150_ranks'] = target
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ExpandedSignedCache(config)
    state = model.state_dict()
    modules = list(projection_modules(model))
    prefixes = tuple(prefix+'.' for prefix, _, _ in modules)
    used = set()
    for name, expected in state.items():
        if name.startswith(prefixes):
            continue
        state[name] = checked_tensor(original, name, expected)
        used.add(name)
    require(torch.equal(state['token.weight'], state['head.weight']), 'Source tied embedding/output disagree')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    changes = []
    for prefix, role, module in modules:
        if role not in FFN_ROLES:
            names = [prefix+'.left.weight', prefix+'.right.weight'] if isinstance(module, LowRankLinear) else [prefix+'.weight']
            for name in names:
                state[name] = checked_tensor(original, name, state[name])
                used.add(name)
            changes.append(dict(projection=prefix, role=role, operation='bitwise factor copy' if role=='qkv'
                                else 'bitwise already-dense copy; no GEMM folding',
                                source_rank=source[role], target_rank=target[role]))
            continue
        require(isinstance(module, LowRankLinear), 'FFN target must stay factored')
        old_rank = source[role]
        left_name, right_name = prefix+'.left.weight', prefix+'.right.weight'
        inputs, outputs = module.left.in_features, module.right.out_features
        left = checked_tensor(original, left_name, torch.empty((old_rank, inputs), dtype=torch.float32))
        right = checked_tensor(original, right_name, torch.empty((outputs, old_rank), dtype=torch.float32))
        new_left = torch.empty((target_ffn_rank, inputs), dtype=torch.float32)
        new_left[:old_rank].copy_(left)
        new_left[old_rank:].copy_(torch.randn(target_ffn_rank-old_rank, inputs, generator=generator)*.02)
        new_right = torch.zeros((outputs, target_ffn_rank), dtype=torch.float32)
        new_right[:, :old_rank].copy_(right)
        state[left_name], state[right_name] = new_left, new_right
        used.update((left_name, right_name))
        changes.append(dict(projection=prefix, role=role, inputs=inputs, outputs=outputs,
            source_rank=old_rank, target_rank=target_ffn_rank, operation='old factors exact; new left Gaussian .02; new right zero'))
    require(used == set(original), 'Unaccounted source tensors: '+str(sorted(set(original)-used)))
    state['head.weight'] = state['token.weight']
    model.load_state_dict(state, strict=True)
    require(model.token.weight is model.head.weight, 'Embedding/output parameter tying lost')
    old_parameters = sum(original[name].numel() for name, _ in model.named_parameters())
    new_parameters = sum(value.numel() for value in model.parameters())
    return model, dict(source_layout=source, target_layout=target, seed=seed,
        projections=changes, parent_parameters=old_parameters, parameters=new_parameters,
        parameter_delta=new_parameters-old_parameters, parameter_payload_delta_bytes=4*(new_parameters-old_parameters),
        statistics_buffers_preserved_bitwise=sum(name.startswith('ngram.') for name in original),
        all_non_ffn_tensors_preserved_bitwise=True, dense_projection_copied_without_folding=True,
        token_head_tying_preserved=True, function_preservation='Algebraically identical; changed GEMM rank may change FP32 rounding',
        prediction_or_scoring_performed=False, training_performed=False, new_training_targets=0)


def initialized_checkpoint(parent, parent_sha256, *, target_ffn_rank=176, seed=SEED):
    require(parent.get('protocol') == PROTOCOL, 'Unexpected course protocol')
    targets = parent.get('train_tokens')
    require(type(targets) is int and targets >= 0, 'Missing nonnegative ancestral target count')
    require(isinstance(parent_sha256, str) and len(parent_sha256) == 64 and
            all(c in '0123456789abcdef' for c in parent_sha256), 'Invalid actual parent SHA-256')
    model, report = initialize_from_parent(parent, target_ffn_rank=target_ffn_rank, seed=seed)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    state['head.weight'] = state['token.weight']
    checkpoint = copy.deepcopy(parent)
    ancestry = dict(initialization_kind='single inherited push151 predictor; FFN-only zero-output rank expansion',
        parent_sha256=parent_sha256, inherited_processed_view_targets=targets,
        shared_inherited_targets_counted_once=targets, new_training_targets=0,
        prior_ancestry=copy.deepcopy(parent.get('ancestry', {})))
    checkpoint.update(implementation='push150_expand', config=copy.deepcopy(model.config), model=state,
        train_tokens=targets, parent_sha256=parent_sha256, seed=seed, selected_step=None,
        weights='function-preserving FFN initialization; not a trained/selected snapshot',
        ancestry=ancestry, initialization_ancestry=ancestry, push152_initialization=copy.deepcopy(report),
        selection='Initialization only; no new training, validation/test or resource acceptance')
    report.update(parent_sha256=parent_sha256, implementation='push150_expand',
        inherited_processed_view_targets=targets, shared_inherited_targets_counted_once=targets)
    return checkpoint, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT/'runs/push151-combination/checkpoint.pt')
    parser.add_argument('--run-dir', type=Path, default=ROOT/'runs/push152-expanded-init')
    parser.add_argument('--plan', type=Path, default=ROOT/'runs/push152_plan.json')
    args = parser.parse_args()
    require(args.run_dir.resolve() == (ROOT/'runs/push152-expanded-init').resolve(), 'Only the fixed exclusive initializer directory is allowed')
    require(not args.run_dir.exists(), 'Never overwrite a previous initialization attempt')
    require(args.plan.resolve() == (ROOT/'runs/push152_plan.json').resolve() and sha(args.plan) == PLAN_SHA256,
            'Fixed pre-execution plan changed')
    plan = read(args.plan)
    validate_plan(plan)
    require(args.parent.resolve() == (ROOT/'runs/push151-combination/checkpoint.pt').resolve(), 'Only the fixed push151 parent is allowed')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    pins = source_hashes()
    write_new(args.run_dir/'started.json', dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=PLAN_SHA256, parent_sha256=PARENT_SHA256, source_sha256=pins,
        test_used=False, data_text_loaded=False))
    try:
        torch.set_num_threads(2)
        require(sha(args.parent) == PARENT_SHA256 and sha(ROOT/'push150_expand.py') == IMPLEMENTATION_SHA256,
                'Fixed parent or inference implementation changed')
        parent = torch.load(args.parent, map_location='cpu', weights_only=True)
        require(parent['config']['push150_ranks'] == SOURCE_LAYOUT and parent['train_tokens'] == LINEAGE,
                'Production parent layout/lineage differs')
        require(all(parent['config'].get(key) == value for key,value in
                    dict(temperature=1.3,mkn_weight=.15,mkn_order=5,discount_scale=.7).items()),
                'Production parent inference configuration differs')
        checkpoint, report = initialized_checkpoint(parent, PARENT_SHA256)
        require(report['target_layout'] == TARGET_LAYOUT and report['parameter_delta'] == PARAMETER_DELTA and
                report['statistics_buffers_preserved_bitwise'] == 21, 'Declared production capacity/statistics differ')
        checkpoint['push152_initialization'].update(plan_sha256=PLAN_SHA256, source_sha256=pins)
        destination = args.run_dir/'checkpoint.pt'
        save_new(destination, checkpoint)
        parent_audit_path = args.parent.parent/'audit.json'
        require(sha(parent_audit_path) == plan['parent']['audit_sha256'], 'Fixed parent asset audit changed')
        audit = read(parent_audit_path)
        require(audit['checkpoint_sha256'] == PARENT_SHA256 and audit['assets']['checkpoint.pt']['sha256'] == PARENT_SHA256,
                'Parent audited asset identity differs')
        assets = copy.deepcopy(audit['assets'])
        for name, entry in assets.items():
            path = args.parent if name == 'checkpoint.pt' else ROOT/name.replace('\\', '/')
            require(path.resolve().is_relative_to(ROOT.resolve()) and sha(path) == entry['sha256'] and
                    path.stat().st_size == entry['bytes'], 'Parent runtime asset changed: '+name)
        assets['checkpoint.pt'] = dict(bytes=destination.stat().st_size, sha256=sha(destination))
        estimated_assets = sum(entry['bytes'] for entry in assets.values())
        require(estimated_assets <= 64*2**20, 'Estimated unchanged-code inference assets exceed 64MiB')
        verify_sources(pins)
        require(sha(args.parent) == PARENT_SHA256 and sha(args.plan) == PLAN_SHA256, 'Parent/plan changed during initialization')
        report.update(status='completed', plan_sha256=PLAN_SHA256, source_sha256=pins,
            initializer_sha256=sha(__file__), implementation_sha256=IMPLEMENTATION_SHA256,
            parent_checkpoint=str(args.parent.resolve()), parent_audit_sha256=sha(parent_audit_path),
            checkpoint_sha256=sha(destination), checkpoint_bytes=destination.stat().st_size,
            estimated_inference_assets_bytes=estimated_assets, estimated_inference_assets_mib=estimated_assets/2**20,
            inference_asset_estimate=assets, resource_acceptance_not_claimed=True,
            test_used=False, source_splits=[], data_text_loaded=False, prior_test_exposure_disclosed=True)
        write_new(args.run_dir/'initialization.json', report)
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    except BaseException as error:
        write_new(args.run_dir/'failure.json', dict(status='failed', failure_type=type(error).__name__,
            failure_message=str(error), plan_sha256=PLAN_SHA256, parent_sha256=PARENT_SHA256,
            source_sha256=pins, test_used=False, data_text_loaded=False,
            no_automatic_retry=True, partial_files_preserved=True))
        raise


if __name__ == '__main__':
    main()
