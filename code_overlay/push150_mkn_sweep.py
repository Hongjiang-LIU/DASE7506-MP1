"""Nine declared MKN discount choices; TRAIN build, VALIDATION-only selection.

Build and score are separate explicit commands. No common.load_data call, test
text access, neural optimization, tokenizer change, or architecture change.
All new files are exclusive-create; failed/partial evidence is never replaced.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from common import PROTOCOL, ROOT, make_model, setup, sha, windows
from evaluate import score
from push145_mkn_build import effective_counts, make_tables
from push145_mkn_select import target_probabilities


SCALES = (.40, .55, .70)
WEIGHTS = (.10, .15, .20)
TEMPERATURE = 1.25
ORDER = 5
PARENT_SHA256 = '97696b7fe5e0e0f5c7573f24efd43e86579c48a29a9e995b8c03829cbaf26867'
IMPLEMENTATION = 'push147_signed_cache'
SOURCE_FILES = (
    'push150_mkn_sweep.py', 'push145_mkn_build.py', 'push145_mkn_select.py',
    'push145_mkn.py', 'push147_signed_cache.py', 'push147_cache.py',
    'push145_cache_hybrid.py', 'push145_cache.py', 'lowrank.py', 'refined.py',
    'student.py', 'efficient.py', 'ngram_hybrid.py', 'common.py', 'evaluate.py',
)


def write_json_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def save_checkpoint_new(path, value):
    with Path(path).open('xb') as stream:
        torch.save(value, stream)


def source_hashes():
    return {name: sha(ROOT/name) for name in SOURCE_FILES}


def load_one_split(split, root=ROOT):
    """Whitelist checked before any I/O; never enumerate/read the test corpus."""
    if split not in ('train', 'validation'):
        raise ValueError('Only train or validation may be loaded.')
    root = Path(root)
    manifest_path = root/'data/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    text_path = root/'data'/f'wikitext_{split}.txt'
    tokenizer_path = root/'data/tokenizer.json'
    raw = text_path.read_bytes()
    text_hash = hashlib.sha256(raw).hexdigest()
    tokenizer_hash = sha(tokenizer_path)
    if text_hash != manifest['sha256'][text_path.name]:
        raise ValueError(f'Changed {split} text.')
    if tokenizer_hash != manifest['sha256'][tokenizer_path.name]:
        raise ValueError('Changed tokenizer.')
    ids = np.asarray(Tokenizer.from_file(str(tokenizer_path)).encode(raw.decode('utf-8')).ids,
                     dtype=np.int64)
    if len(ids) < ORDER+1:
        raise ValueError('Too few tokens.')
    return ids, dict(split=split, text_sha256=text_hash, utf8_bytes=len(raw),
                     tokens=len(ids), tokenizer_sha256=tokenizer_hash,
                     manifest_sha256=sha(manifest_path))


def load_parent(path):
    actual_sha = sha(path)
    if actual_sha != PARENT_SHA256:
        raise ValueError('Parent differs from the predeclared push149 winner.')
    parent = torch.load(path, map_location='cpu', weights_only=True)
    if parent.get('protocol') != PROTOCOL or parent.get('implementation') != IMPLEMENTATION:
        raise ValueError('Unexpected parent protocol or implementation.')
    config = parent['config']
    if (config.get('temperature') != TEMPERATURE or config.get('mkn_weight') != .15
            or config.get('discount_scale') != .7 or config.get('mkn_order') != ORDER
            or not config.get('mkn_shapes')):
        raise ValueError('Parent differs from the declared temperature/statistics configuration.')
    return parent, actual_sha


def statistics_from_state(state):
    return {name.removeprefix('ngram.'): value for name, value in state.items()
            if name.startswith('ngram.')}


def assert_exact_statistics(state, rebuilt):
    original = statistics_from_state(state)
    if len(original) != 21 or set(original) != set(rebuilt):
        raise AssertionError('Expected exactly the same 21 statistical buffers.')
    for name, value in rebuilt.items():
        expected = original[name]
        if value.dtype != expected.dtype or value.shape != expected.shape or not torch.equal(value, expected):
            raise AssertionError(f'Rebuilt .70 statistic differs: {name}')
    return dict(buffers=21, shapes_dtypes_values_bitwise_equal=True)


def rebuild_variants(ids, parent_state):
    """Reuse the unmodified original count/discount/pruning implementation."""
    effective = effective_counts(np.asarray(ids, dtype=np.int64), maximum=ORDER)
    variants = {}
    for scale in SCALES:
        tables, stats = make_tables(effective, scale=scale, minimum=2, trigram_minimum=1)
        variants[scale] = (tables, stats)
    parity = assert_exact_statistics(parent_state, variants[.70][0])
    return variants, parity


def score_grid(neural, counts_by_scale, targets, byte_count):
    neural, targets = np.asarray(neural), np.asarray(targets)
    valid = targets != -100
    if byte_count <= 0 or not valid.any() or neural.shape != targets.shape:
        raise ValueError('Invalid scoring shape, targets, or byte count.')
    if set(counts_by_scale) != set(SCALES):
        raise ValueError('The three predeclared count distributions are required.')
    for probabilities in (neural, *counts_by_scale.values()):
        values = np.asarray(probabilities)
        if values.shape != targets.shape or not np.isfinite(values[valid]).all():
            raise ValueError('Invalid target probabilities.')
        if np.any(values[valid] <= 0) or np.any(values[valid] > 1.000001):
            raise ValueError('Target probabilities must be positive and at most one.')
    trials = []
    for scale, weight in itertools.product(SCALES, WEIGHTS):
        mixed = (1-weight)*neural+weight*counts_by_scale[scale]
        nll = float(-np.log(mixed[valid].astype(np.float64)).sum())
        trials.append(dict(discount_scale=scale, mkn_order=ORDER, mkn_weight=weight,
                           temperature=TEMPERATURE, validation_nll_nats=nll,
                           validation_bpb=nll/math.log(2)/byte_count))
    return trials, min(trials, key=lambda item: item['validation_bpb'])


def candidate_from_parent(parent, tables, best):
    if (best['discount_scale'] not in SCALES or best['mkn_weight'] not in WEIGHTS
            or best['temperature'] != TEMPERATURE or best['mkn_order'] != ORDER):
        raise ValueError('Choice is outside the declared grid.')
    candidate = copy.deepcopy(parent)
    if set(tables) != set(statistics_from_state(parent['model'])):
        raise AssertionError('Statistics key set changed.')
    for name, tensor in tables.items():
        old = parent['model']['ngram.'+name]
        if tensor.shape != old.shape or tensor.dtype != old.dtype:
            raise AssertionError('Discount-only changes cannot alter table shapes/dtypes.')
        candidate['model']['ngram.'+name] = tensor.clone()
    candidate['config'].update(discount_scale=best['discount_scale'], mkn_weight=best['mkn_weight'])
    assert_neural_and_config_unchanged(parent, candidate)
    return candidate


def assert_neural_and_config_unchanged(parent, candidate):
    if candidate['implementation'] != parent['implementation']:
        raise AssertionError('Inference implementation changed.')
    if set(parent['model']) != set(candidate['model']):
        raise AssertionError('Model state key set changed.')
    for name, tensor in parent['model'].items():
        if not name.startswith('ngram.'):
            other = candidate['model'][name]
            if tensor.dtype != other.dtype or tensor.shape != other.shape or not torch.equal(tensor, other):
                raise AssertionError(f'Neural parameter/buffer changed: {name}')
    allowed = {'discount_scale', 'mkn_weight'}
    before = {k: v for k, v in parent['config'].items() if k not in allowed}
    after = {k: v for k, v in candidate['config'].items() if k not in allowed}
    if before != after:
        raise AssertionError('An undeclared inference setting changed.')


def run_build(args):
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(2)
    parent, parent_sha = load_parent(args.parent)
    sources = source_hashes()
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                parent_path=str(args.parent.resolve()), parent_sha256=parent_sha,
                implementation=IMPLEMENTATION, scales=list(SCALES), mkn_weights=list(WEIGHTS),
                temperature=TEMPERATURE, mkn_order=ORDER, total_choices=9,
                minimum_count_by_order={'2': 1, '3': 1, '4': 2, '5': 2},
                rationale='Explore below the old .70 discount-scale boundary; not motivated by test scores.',
                count_source='supplied train text only', selection_split='validation',
                new_neural_training_targets=0, test_used=False, prior_test_exposure_disclosed=True,
                source_sha256=sources)
    write_json_new(args.run_dir/'experiment_plan.json', plan)
    ids, data = load_one_split('train')
    variants, parity = rebuild_variants(ids, parent['model'])
    table_dir = args.run_dir/'tables'
    table_dir.mkdir()
    records = []
    for scale in SCALES:
        tables, stats = variants[scale]
        path = table_dir/f'tables-scale{scale:.2f}.pt'
        save_checkpoint_new(path, tables)
        records.append(dict(scale=scale, path=str(path.relative_to(args.run_dir)),
                            sha256=sha(path), bytes=path.stat().st_size, stats=stats))
    report = dict(status='completed', phase='build', parent_sha256=parent_sha,
                  source_sha256=sources, train=data, variants=records, control_scale_parity=parity,
                  count_scan_token_occurrences=len(ids), new_neural_training_targets=0,
                  inherited_train_targets=parent.get('train_tokens'), test_used=False,
                  prior_test_exposure_disclosed=True, seconds_total=time.perf_counter()-started)
    write_json_new(args.run_dir/'build.json', report)
    return report


@torch.no_grad()
def run_score(args):
    started = time.perf_counter()
    for name in ('trials.json', 'metrics.json', 'checkpoint.pt', 'validation.window-nll.npy', 'score_started.json'):
        if (args.run_dir/name).exists():
            raise FileExistsError(f'Preserve previous/partial score evidence: {name}')
    plan = json.loads((args.run_dir/'experiment_plan.json').read_text(encoding='utf-8'))
    build = json.loads((args.run_dir/'build.json').read_text(encoding='utf-8'))
    parent, parent_sha = load_parent(args.parent)
    sources = source_hashes()
    if (build['status'] != 'completed' or build['parent_sha256'] != parent_sha
            or plan['parent_sha256'] != parent_sha or sources != build['source_sha256']
            or sources != plan['source_sha256'] or plan['scales'] != list(SCALES)
            or plan['mkn_weights'] != list(WEIGHTS) or plan['temperature'] != TEMPERATURE
            or plan['total_choices'] != 9):
        raise ValueError('Build/plan/source/parent provenance mismatch.')
    records = build['variants']
    if len(records) != 3 or [row['scale'] for row in records] != list(SCALES):
        raise ValueError('Expected exactly three table variants in declared order.')
    tables_by_scale = {}
    for record in records:
        relative = Path(record['path'])
        if relative.is_absolute() or relative.drive or '..' in relative.parts:
            raise ValueError('Unsafe table path.')
        path = args.run_dir/relative
        if sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('A train-built table changed.')
        tables_by_scale[record['scale']] = torch.load(path, map_location='cpu', weights_only=True)
    parity = assert_exact_statistics(parent['model'], tables_by_scale[.70])
    write_json_new(args.run_dir/'score_started.json', dict(
        started_utc=datetime.now(timezone.utc).isoformat(), parent_sha256=parent_sha,
        experiment_plan_sha256=sha(args.run_dir/'experiment_plan.json'),
        build_sha256=sha(args.run_dir/'build.json'), requested_device=args.device,
        precision='fp32', source_sha256=sources, test_used=False))
    device, _ = setup(args.device, 'fp32', 2)
    ids, validation = load_one_split('validation')
    if (validation['tokenizer_sha256'] != build['train']['tokenizer_sha256']
            or validation['manifest_sha256'] != build['train']['manifest_sha256']):
        raise ValueError('Data manifest/tokenizer changed between build and score.')
    tokens = torch.from_numpy(ids)
    model, implementation_sha = make_model(IMPLEMENTATION, copy.deepcopy(parent['config']), device)
    model.load_state_dict(parent['model'], strict=True)
    model.eval()
    xs, ys, ns = [], [], []
    for x, y in windows(tokens):
        # Full target-free neural/cache distribution; only the scorer gathers y.
        neural = model.neural_and_cache(x.to(device))
        target = y.clamp_min(0).unsqueeze(-1).to(device)
        ns.append(neural.gather(-1, target).squeeze(-1).cpu().numpy())
        xs.append(x.numpy()); ys.append(y.numpy())
    x, y, neural = map(np.concatenate, (xs, ys, ns))
    count_targets = {}
    for scale, tables in tables_by_scale.items():
        arrays = {name: value.numpy() for name, value in tables.items()}
        count_targets[scale] = target_probabilities(arrays, x, y, maximum=ORDER)
    trials, best = score_grid(neural, count_targets, y, validation['utf8_bytes'])
    write_json_new(args.run_dir/'trials.json', trials)
    candidate = candidate_from_parent(parent, tables_by_scale[best['discount_scale']], best)
    model.config.update(candidate['config'])
    # load_state_dict clears the train-only bigram memoization via its hook.
    model.load_state_dict(candidate['model'], strict=True)
    measured = score(model, tokens, validation['utf8_bytes'], device, 'fp32')
    losses = measured.pop('window_nll_nats')
    with (args.run_dir/'validation.window-nll.npy').open('xb') as stream:
        np.save(stream, np.asarray(losses, dtype=np.float64))
    difference = abs(measured['bpb']-best['validation_bpb'])
    provenance = dict(parent_sha256=parent_sha, all_table_variants=records,
                      experiment_plan_sha256=sha(args.run_dir/'experiment_plan.json'),
                      build_sha256=sha(args.run_dir/'build.json'), source_sha256=sources,
                      train=build['train'], validation=validation, choices=trials,
                      full_validation=measured, new_neural_training_targets=0,
                      count_scan_token_occurrences=build['count_scan_token_occurrences'],
                      build_seconds=build['seconds_total'], score_seconds=time.perf_counter()-started,
                      test_used=False, prior_test_exposure_disclosed=True)
    report = dict(status='confirmed' if difference < 1e-6 else 'confirmation_failed',
                  best=best, trials=trials, full_validation=measured,
                  shortcut_full_bpb_absolute_difference=difference,
                  shortcut_matches_full_predictor=difference < 1e-6,
                  parent_sha256=parent_sha, parent_path=str(args.parent.resolve()),
                  implementation=IMPLEMENTATION, implementation_sha256=implementation_sha,
                  precision='fp32', device=str(device), threads=2,
                  control_scale_parity=parity, neural_weights_unchanged=True,
                  validation_target_occurrences=int((y != -100).sum()),
                  provenance=provenance, new_training_targets=0, test_used=False,
                  prior_test_exposure_disclosed=True)
    if difference < 1e-6:
        candidate['mkn_sweep_provenance'] = provenance
        candidate['mkn_sweep_parent_sha256'] = parent_sha
        candidate['mkn_train_tables_sha256'] = next(row['sha256'] for row in records if row['scale'] == best['discount_scale'])
        candidate['selection'] = 'Nine declared train-only MKN discount/mixture choices; full validation confirmation'
        destination = args.run_dir/'checkpoint.pt'
        save_checkpoint_new(destination, candidate)
        reloaded = torch.load(destination, map_location='cpu', weights_only=True)
        assert_neural_and_config_unchanged(parent, reloaded)
        if reloaded['config'] != candidate['config']:
            raise AssertionError('Serialized candidate config changed.')
        for name, value in model.state_dict().items():
            expected = reloaded['model'][name]
            if value.dtype != expected.dtype or not torch.equal(value.cpu(), expected):
                raise AssertionError(f'Saved candidate differs from fully scored model: {name}')
        report.update(checkpoint_sha256=sha(destination), checkpoint_bytes=destination.stat().st_size,
                      saved_candidate_exactly_matches_full_scored_model=True)
    report['seconds_total'] = time.perf_counter()-started
    write_json_new(args.run_dir/'metrics.json', report)
    if difference >= 1e-6:
        raise AssertionError('Shortcut/full prediction mismatch; no candidate checkpoint was published.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('build', 'score'))
    parser.add_argument('--parent', type=Path, default=ROOT/'runs/push149-muon-low-calibrated/checkpoint.pt')
    parser.add_argument('--run-dir', type=Path, default=ROOT/'runs/push150-mkn')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    try:
        result = run_build(args) if args.phase == 'build' else run_score(args)
    except BaseException as error:
        if args.run_dir.is_dir():
            path = args.run_dir/f'{args.phase}_failure.json'
            if not path.exists():
                write_json_new(path, dict(phase=args.phase, failure_type=type(error).__name__,
                    message=str(error), recorded_utc=datetime.now(timezone.utc).isoformat(),
                    test_used=False, prior_test_exposure_disclosed=True))
        raise
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
