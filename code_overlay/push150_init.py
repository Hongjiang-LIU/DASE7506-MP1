"""Function-preserving initialization from the owned push149 low-rank parent.

This tool reads checkpoints only and never loads or scores corpus text. It
performs no SGD, SVD, calibration, selection, or benchmark. Algebraic equality
does not imply bitwise FP32 equality after a factored projection is folded.
"""
import argparse
import copy
import json
from pathlib import Path

import torch

from common import PROTOCOL, sha
from lowrank import LowRankLinear
from push150_expand import ExpandedSignedCache, LAYOUTS, projection_modules


EXPECTED_PARENT_SHA256 = '97696b7fe5e0e0f5c7573f24efd43e86579c48a29a9e995b8c03829cbaf26867'


@torch.no_grad()
def initialize_from_parent(parent, *, rank_layout=None, layout='expanded', seed=173):
    """Return (model, diagnostics), without writing a checkpoint or scoring.

All unchanged tensors retain their exact dtype/value. Positive target ranks
must be at least the source rank; new left rows are seeded Gaussian values,
and new right columns are zero so added capacity is initially silent but can
receive a first-step right-factor gradient. This includes FFN out, whose left
factor input dimension is the FFN hidden size, not the model width.
"""
    if parent.get('implementation') != 'push147_signed_cache':
        raise ValueError('Expected the existing signed-cache low-rank parent')
    if layout not in LAYOUTS:
        raise ValueError('Unknown initialization layout')
    config = copy.deepcopy(parent['config'])
    config['push150_ranks'] = copy.deepcopy(LAYOUTS[layout] if rank_layout is None else rank_layout)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ExpandedSignedCache(config)
    original = parent['model']
    state = model.state_dict()
    if not torch.equal(original['head.weight'], original['token.weight']):
        raise ValueError('Parent tied embedding/output values disagree')
    modules = list(projection_modules(model))
    prefixes = tuple(prefix+'.' for prefix, _, _ in modules)
    used = set()
    for name, target in state.items():
        if name.startswith(prefixes):
            continue
        source = original.get(name)
        if source is None or source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError(f'Missing, wrong-shape or wrong-dtype parent tensor: {name}')
        state[name] = source.detach().cpu().clone()
        used.add(name)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    diagnostics = []
    for prefix, name, module in modules:
        left_name, right_name = prefix+'.left.weight', prefix+'.right.weight'
        if left_name not in original or right_name not in original:
            raise ValueError(f'Expected factorized parent projection: {prefix}')
        left, right = original[left_name].detach().cpu(), original[right_name].detach().cpu()
        inputs = module.left.in_features if isinstance(module, LowRankLinear) else module.in_features
        outputs = module.right.out_features if isinstance(module, LowRankLinear) else module.out_features
        if (left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[1] or
                left.shape[1] != inputs or right.shape[0] != outputs or
                left.dtype != torch.float32 or right.dtype != torch.float32):
            raise ValueError(f'Incompatible FP32 factor shapes: {prefix}')
        old_rank = left.shape[0]
        if isinstance(module, LowRankLinear):
            new_rank = module.left.out_features
            if new_rank < old_rank:
                raise ValueError(f'Cannot shrink rank while preserving the parent function: {prefix}')
            new_left = torch.empty((new_rank, inputs), dtype=torch.float32)
            new_left[:old_rank].copy_(left)
            if new_rank > old_rank:
                new_left[old_rank:].copy_(torch.randn(new_rank-old_rank, inputs, generator=rng)*.02)
            new_right = torch.zeros((outputs, new_rank), dtype=torch.float32)
            new_right[:, :old_rank].copy_(right)
            state[left_name], state[right_name] = new_left, new_right
            operation = 'factor_copy' if new_rank == old_rank else 'zero-output rank expansion'
        else:
            new_rank = 0
            state[prefix+'.weight'] = right@left
            operation = 'dense FP32 right @ left'
        used.update((left_name, right_name))
        diagnostics.append(dict(projection=prefix, role=name, inputs=inputs, outputs=outputs,
                                source_rank=old_rank, target_rank=new_rank, operation=operation))
    if used != set(original):
        raise ValueError(f'Unaccounted parent tensors: {sorted(set(original)-used)}')
    state['head.weight'] = state['token.weight']
    model.load_state_dict(state, strict=True)
    return model, dict(layout=layout, push150_ranks=copy.deepcopy(model.config['push150_ranks']),
        seed=seed, initialization='Exact factor retention/zero-output expansion; dense folding algebraically equivalent, not guaranteed bitwise FP32 equivalent',
        projections=diagnostics, parameters=sum(value.numel() for value in model.parameters()),
        statistics='All original ngram tensors and complete configuration metadata retained unchanged',
        new_training_targets=0, evaluated=False, selected=False)


def initialized_checkpoint(parent, parent_sha256, *, rank_layout=None, layout='expanded', seed=173):
    if parent.get('protocol') != PROTOCOL:
        raise ValueError('Unexpected parent course protocol')
    targets = parent.get('train_tokens')
    if isinstance(targets, bool) or not isinstance(targets, int) or targets < 0:
        raise ValueError('Parent must record its nonnegative inherited target count')
    if not isinstance(parent_sha256, str) or len(parent_sha256) != 64 or any(c not in '0123456789abcdef' for c in parent_sha256):
        raise ValueError('Expected a recorded SHA-256 of the actual parent file')
    model, report = initialize_from_parent(parent, rank_layout=rank_layout, layout=layout, seed=seed)
    state = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    state['head.weight'] = state['token.weight']
    ancestry = dict(initialization_kind='single-parent function-preserving capacity change',
        parent_sha256=parent_sha256, inherited_processed_view_targets=targets,
        shared_inherited_targets_counted_once=targets, new_training_targets=0,
        prior_ancestry=copy.deepcopy(parent.get('ancestry', {})),
        note='One inherited chain, not two additive parents; occurrence counts are not unique corpus tokens')
    checkpoint = dict(protocol=PROTOCOL, implementation='push150_expand', config=copy.deepcopy(model.config),
        model=state, train_tokens=targets, parent_sha256=parent_sha256, seed=seed,
        ancestry=ancestry, initialization_ancestry=ancestry, initialization_layout=layout,
        selection='Initialization only: no validation/test selection and no new SGD',
        training_config=copy.deepcopy(parent.get('training_config')))
    report.update(parent_sha256=parent_sha256, inherited_processed_view_targets=targets,
                  shared_inherited_targets_counted_once=targets, initialization_ancestry=ancestry,
                  implementation='push150_expand')
    return checkpoint, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--layout', choices=tuple(LAYOUTS), default='expanded')
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--expected-parent-sha256', default=EXPECTED_PARENT_SHA256)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError('Do not overwrite an initialization directory')
    parent_sha = sha(args.parent)
    if parent_sha != args.expected_parent_sha256:
        raise ValueError('Actual parent file does not match the declared expected hash')
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    checkpoint, report = initialized_checkpoint(parent, parent_sha, layout=args.layout, seed=args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    destination = args.run_dir/'checkpoint.pt'
    torch.save(checkpoint, destination)
    report.update(checkpoint_sha256=sha(destination), checkpoint_bytes=destination.stat().st_size,
        initializer_sha256=sha(__file__), implementation_sha256=sha(Path(__file__).with_name('push150_expand.py')),
        parent_path=str(args.parent), test_used=False, source_splits=[], data_text_loaded=False)
    with (args.run_dir/'initialization.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
