"""Zero-SGD, zero-bias initialization from the declared Phase-A winner only.

No corpus text is read or hashed. The CLI requires the parent's SHA to be
pinned by the caller after Phase-A validation selection, before this runs.
"""
import argparse
import copy
import json
from pathlib import Path

import torch

from common import PROTOCOL, sha
from push150_expand import PROJECTIONS
from push150_suffix import SuffixSignedCache, SUFFIX_PARAMETER_NAMES


@torch.no_grad()
def initialize_from_parent(parent, seed=191):
    if parent.get('implementation') not in ('push150_expand', 'push147_signed_cache'):
        raise ValueError('Expected the Phase-A model or its unchanged original signed-cache parent')
    config = copy.deepcopy(parent['config'])
    if 'push150_ranks' not in config:
        config['push150_ranks'] = dict.fromkeys(PROJECTIONS, config.get('lowrank_rank', 128))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SuffixSignedCache(config)
    original = parent['model']
    state = model.state_dict()
    names = set(SUFFIX_PARAMETER_NAMES)
    if set(original) != set(state)-names:
        raise ValueError('Parent keys differ by more than the four declared new suffix scalars')
    if not torch.equal(original['head.weight'], original['token.weight']):
        raise ValueError('Parent tied embedding/output values disagree')
    for name, source in original.items():
        if source.shape != state[name].shape or source.dtype != state[name].dtype:
            raise ValueError(f'Parent tensor shape or dtype changed: {name}')
        state[name] = source.detach().cpu().clone()
    state['head.weight'] = state['token.weight']
    model.load_state_dict(state, strict=True)
    return model, dict(seed=seed, added_parameter_names=list(SUFFIX_PARAMETER_NAMES),
        added_parameters=4, new_training_targets=0, evaluated=False, selected=False,
        initialization='Every inherited tensor copied exactly; only four 0D FP32 suffix biases added at zero',
        parameters=sum(value.numel() for value in model.parameters()),
        statistics='All inherited ngram tensors and complete configuration metadata retained unchanged')


def initialized_checkpoint(parent, parent_sha256, seed=191):
    if parent.get('protocol') != PROTOCOL:
        raise ValueError('Unexpected parent course protocol')
    targets = parent.get('train_tokens')
    if isinstance(targets, bool) or not isinstance(targets, int) or targets < 0:
        raise ValueError('Parent must record its nonnegative inherited target count')
    if (not isinstance(parent_sha256, str) or len(parent_sha256) != 64
            or any(c not in '0123456789abcdef' for c in parent_sha256)):
        raise ValueError('Expected the actual Phase-A parent SHA-256')
    model, report = initialize_from_parent(parent, seed=seed)
    state = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    state['head.weight'] = state['token.weight']
    ancestry = dict(initialization_kind='single-parent zero-initialized causal suffix biases',
        parent_sha256=parent_sha256, parent_implementation=parent['implementation'],
        inherited_processed_view_targets=targets, shared_inherited_targets_counted_once=targets,
        new_training_targets=0, prior_ancestry=copy.deepcopy(parent.get('ancestry', {})),
        note='Same Phase-A selected weights as the matched control, not B-gate or fast-dense weights; occurrence counts are not unique corpus tokens')
    checkpoint = dict(protocol=PROTOCOL, implementation='push150_suffix', config=copy.deepcopy(model.config),
        model=state, train_tokens=targets, parent_sha256=parent_sha256, seed=seed,
        ancestry=ancestry, initialization_ancestry=ancestry,
        selection='Zero-bias initialization only; no training or validation/test selection',
        training_config=copy.deepcopy(parent.get('training_config')))
    report.update(parent_sha256=parent_sha256, parent_implementation=parent['implementation'],
        inherited_processed_view_targets=targets, shared_inherited_targets_counted_once=targets,
        initialization_ancestry=ancestry, implementation='push150_suffix')
    return checkpoint, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--expected-parent-sha256', required=True)
    parser.add_argument('--seed', default=191, type=int)
    args = parser.parse_args()
    if args.seed != 191:
        raise ValueError('Phase C declares only initialization seed 191')
    if args.run_dir.exists():
        raise FileExistsError('Do not overwrite a suffix initialization directory')
    parent_sha = sha(args.parent)
    if parent_sha != args.expected_parent_sha256:
        raise ValueError('Actual parent does not match the pinned Phase-A winner SHA')
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    checkpoint, report = initialized_checkpoint(parent, parent_sha, seed=args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    destination = args.run_dir/'checkpoint.pt'
    torch.save(checkpoint, destination)
    report.update(checkpoint_sha256=sha(destination), checkpoint_bytes=destination.stat().st_size,
        initializer_sha256=sha(__file__), implementation_sha256=sha(Path(__file__).with_name('push150_suffix.py')),
        parent_path=str(args.parent), test_used=False, source_splits=[], data_text_loaded=False)
    with (args.run_dir/'initialization.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
