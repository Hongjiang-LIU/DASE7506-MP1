"""Prepare TRAIN-only cross-fitted target probabilities for declared Phase D.

The N x 4 FP32 training cache is not an inference asset. At global input i it
contains probabilities of train[i+1] for maximum orders 2/3/4/5. A random
window must reset its first three columns to 0/1/2; no pre-window context is
allowed. Ten Phase-B tables are reused, never rebuilt or selected here.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from common import PROTOCOL, ROOT
from ngram_hybrid import prefix_codes
from push145_mkn import MKNTables
from push150_crossfit import contiguous_folds
from push150_mkn_sweep import load_one_split, write_json_new, assert_exact_statistics


PLAN_SHA256 = '65800e86fac769df299f3794914a287189274adbb3442e56fa9e7e24367ec25a'
GATE_PLAN_SHA256 = '96be93fd8bc1f2199042a57956e3d79eb7b55c48c606db098b1ecf4a808fe90b'
MAXIMUM_ORDERS = (2, 3, 4, 5)
PARITY_LENGTHS = (1, 2, 3, 256)
PARITY_ATOL, PARITY_RTOL = 1e-7, 1e-5


def sha_file(path):
    """Metadata/assets only; raw train is read solely by load_one_split."""
    path = Path(path)
    if path.name.startswith('wikitext_'):
        raise ValueError('Do not read/hash raw corpus through the evidence helper.')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def child(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or relative.drive or '..' in relative.parts:
        raise ValueError('Unsafe evidence path.')
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Evidence path escapes its root.')
    return path


def source_graph(root, seeds):
    """Recursively pin every existing local Python import, plus dynamic roots.

The selected architecture and every recorded B-build source are explicit
roots because common.make_model uses importlib rather than a static import.
External installed libraries are recorded separately, not vendored sources.
"""
    root, pending, result = Path(root).resolve(), list(seeds), {}
    while pending:
        name = pending.pop()
        path = child(root, name)
        if path.suffix != '.py' or not path.is_file():
            raise ValueError('Expected an existing local Python source: '+str(name))
        key = path.relative_to(root).as_posix()
        if key in result:
            continue
        result[key] = sha_file(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module] if node.module else []
                if node.level:
                    raise ValueError('Relative local imports need explicit graph handling.')
            else:
                continue
            for module in modules:
                local = root/(module.replace('.', '/')+'.py')
                if local.is_file():
                    pending.append(local.relative_to(root).as_posix())
    return dict(sorted(result.items()))


def window_columns(context=256):
    if not isinstance(context, int) or context < 1 or context > 256:
        raise ValueError('Context must be an integer in 1..256.')
    return np.minimum(np.arange(context, dtype=np.int64), 3)


def valid_start_ranges(folds, context=256):
    """Half-open start ranges [fold.start, fold.end-context), never a join."""
    window_columns(context)
    ranges, previous = [], 0
    for row in folds:
        start, end = row['start'], row['end']
        if (isinstance(start, bool) or isinstance(end, bool) or
                not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)) or
                start != previous or end-start <= context):
            raise ValueError('Folds must be contiguous and each contain a full input/target window.')
        ranges.append((start, end-context))
        previous = end
    if not ranges:
        raise ValueError('Missing fold intervals.')
    return np.asarray(ranges, dtype=np.int64)


def sample_window_starts(rng, batch_size, folds, context=256):
    """Uniform over legal starts (not uniform over unequal-size folds)."""
    if not isinstance(rng, np.random.Generator) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('Expected a NumPy generator and positive batch size.')
    ranges = valid_start_ranges(folds, context)
    lengths = ranges[:, 1]-ranges[:, 0]
    cumulative = lengths.cumsum()
    flat = rng.integers(0, cumulative[-1], size=batch_size, dtype=np.int64)
    fold = np.searchsorted(cumulative, flat, side='right')
    before = np.r_[0, cumulative[:-1]]
    return ranges[fold, 0]+flat-before[fold]


def gather_window_probabilities(probabilities, starts, context=256, max_order=5):
    """Gather with an explicit independent-window order reset.

Caller must choose starts from valid_start_ranges. The zero fold-end sentinel
also causes a rejected gather if an otherwise in-bounds window crosses a fold.
"""
    if max_order not in MAXIMUM_ORDERS:
        raise ValueError('Maximum order must be 2, 3, 4 or 5.')
    values, starts = np.asarray(probabilities), np.asarray(starts)
    if values.ndim != 2 or values.shape[1] != 4 or values.dtype != np.float32:
        raise ValueError('Expected an N x 4 FP32 cache.')
    if starts.ndim != 1 or starts.dtype.kind not in 'iu' or not len(starts):
        raise ValueError('Expected nonempty integer starts.')
    positions = starts.astype(np.int64)[:, None]+np.arange(context, dtype=np.int64)[None, :]
    columns = np.minimum(window_columns(context), max_order-2)
    if positions.min() < 0 or positions.max() >= len(values):
        raise ValueError('Window is outside the training cache.')
    result = values[positions, columns[None, :]]
    if not np.isfinite(result).all() or np.any(result <= 0) or np.any(result > 1.000001):
        raise ValueError('Invalid probability or window crossed a fold-end sentinel.')
    return result


def fold_target_probabilities(arrays, ids):
    """One target-only lookup per order, without an N x vocabulary array."""
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) < 2 or np.any((ids < 0) | (ids >= 2048)):
        raise ValueError('Expected at least two valid token IDs in one fold.')
    x, target = ids[:-1].astype(np.int64)[None, :], ids[1:].astype(np.int64)
    result = np.asarray(arrays['unigram'], dtype=np.float32)[target].copy()
    columns = np.zeros((len(ids), 4), dtype=np.float32)
    for order in MAXIMUM_ORDERS:
        contexts = np.asarray(arrays[f'contexts_{order}'])
        if len(contexts):
            codes, valid = prefix_codes(x, order-1)
            positions = np.searchsorted(contexts, codes)
            safe = np.minimum(positions, len(contexts)-1)
            valid = valid & (positions < len(contexts)) & (contexts[safe] == codes)
            offsets = np.asarray(arrays[f'offsets_{order}'])
            keys = (np.repeat(contexts, np.diff(offsets)) << 11) | arrays[f'words_{order}']
            if not len(keys):
                raise ValueError('A stored context has no continuation entries.')
            queries = (codes << 11) | target
            positions_key = np.searchsorted(keys, queries)
            safe_key = np.minimum(positions_key, len(keys)-1)
            found = (positions_key < len(keys)) & (keys[safe_key] == queries)
            mass = np.where(found, arrays[f'masses_{order}'][safe_key], np.float32(0))
            result = np.where(valid, result*arrays[f'backoffs_{order}'][safe]+mass, result).astype(np.float32)
        columns[:-1, order-2] = result
    if not np.isfinite(columns).all() or np.any(columns[:-1] <= 0) or np.any(columns[:-1] > 1.000001):
        raise ValueError('Precomputed target probabilities are invalid.')
    # No legal label after the last token of a held-out region, even when the
    # global train corpus has another fold immediately afterwards.
    return columns


@torch.no_grad()
def parity_check(buffers, ids, probabilities, progress=None):
    """Fixed fold-first/middle/last windows and 1/2/3/256-token prefixes."""
    if len(ids) < 257:
        raise ValueError('Parity requires a complete 257-token train window.')
    tables = MKNTables({name: list(value.shape) for name, value in buffers.items()})
    tables.load_state_dict(buffers, strict=True)
    tables.eval()
    starts = sorted({0, (len(ids)-257)//2, len(ids)-257})
    maximum, comparisons, forward_targets = 0., 0, 0
    for start in starts:
        for length in PARITY_LENGTHS:
            x = torch.from_numpy(np.asarray(ids[start:start+length], dtype=np.int64).copy())[None, :]
            y = torch.from_numpy(np.asarray(ids[start+1:start+length+1], dtype=np.int64).copy())[None, :]
            for order in MAXIMUM_ORDERS:
                if progress is not None:
                    progress['parity_forward_target_occurrences_started'] += length
                expected = tables.probabilities(x, max_order=order).gather(-1, y.unsqueeze(-1)).squeeze(-1).numpy()
                actual = gather_window_probabilities(probabilities, np.asarray([start], dtype=np.int64), length, order)
                np.testing.assert_allclose(actual, expected, atol=PARITY_ATOL, rtol=PARITY_RTOL)
                maximum = max(maximum, float(np.max(np.abs(actual-expected))))
                comparisons += 1; forward_targets += length
                if progress is not None:
                    progress['parity_forward_target_occurrences_completed'] += length
    return dict(passes=True, window_starts_local=starts, prefix_lengths=list(PARITY_LENGTHS),
                maximum_orders=list(MAXIMUM_ORDERS), comparisons=comparisons,
                full_distribution_forward_target_occurrences=forward_targets,
                max_absolute_difference=maximum, atol=PARITY_ATOL, rtol=PARITY_RTOL)


def verified_record(root, record):
    path = child(root, record['path'])
    if path.stat().st_size != record['bytes'] or sha_file(path) != record['sha256']:
        raise ValueError('Referenced artifact differs: '+str(path))
    return path


def check_inputs(args):
    plan_path = ROOT/'runs/push150_phase_d_oof_plan.json'
    if sha_file(plan_path) != PLAN_SHA256:
        raise ValueError('Predeclared Phase-D plan changed.')
    if sha_file(args.parent) != args.expected_parent_sha256:
        raise ValueError('Actual parent does not match pinned parent SHA.')
    selected = read_json(args.phase_a_selection)
    winner = selected.get('winner') or {}
    if (selected.get('status') != 'phase_a_choice_complete' or selected.get('test_used') is not False
            or selected.get('evidence_errors') or winner.get('checkpoint_sha256') != args.expected_parent_sha256
            or selected.get('all_training_accounting_verified') is not True):
        raise ValueError('Expected the completed, integrity-checked Phase-A winner.')
    if (sha_file(ROOT/'runs/push150_phase_a_plan.json') != selected.get('plan_sha256')
            or sha_file(ROOT/'push150_phase_a_report.py') != selected.get('report_script_sha256')):
        raise ValueError('Phase-A selection plan/source binding changed.')
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    if (parent.get('protocol') != PROTOCOL or parent.get('implementation') not in ('push147_signed_cache','push150_expand')
            or parent.get('train_tokens') != winner.get('selected_lineage_targets')
            or parent['config'].get('context') != 256 or parent['config'].get('vocab') != 2048
            or parent['config'].get('mkn_order') != 5):
        raise ValueError('Phase-A parent metadata/configuration differs.')
    identity_path, build_path = args.gate_run_dir/'identity.json', args.gate_run_dir/'build.json'
    identity, build = read_json(identity_path), read_json(build_path)
    if (identity.get('parent_sha256') != args.expected_parent_sha256 or identity.get('test_used') is not False
            or identity.get('plan_sha256') != GATE_PLAN_SHA256
            or identity.get('parent_discount_scale') != parent['config']['discount_scale']
            or identity.get('parent_temperature') != parent['config']['temperature']
            or identity.get('parent_mkn_weight') != parent['config']['mkn_weight']
            or identity.get('inherited_processed_targets') != parent['train_tokens']):
        raise ValueError('B-build identity does not belong to this Phase-A parent.')
    if (build.get('status') != 'completed' or build.get('phase') != 'build' or build.get('test_used') is not False
            or build.get('identity_sha256') != sha_file(identity_path)
            or build.get('full_train_21_buffer_parity') != dict(buffers=21, shapes_dtypes_values_bitwise_equal=True)):
        raise ValueError('B-build is not complete or lacks full-table parity.')
    if sha_file(ROOT/'runs/push150_phase_b_gate_plan.json') != GATE_PLAN_SHA256:
        raise ValueError('B-build plan changed.')
    for name, expected in identity['source_sha256'].items():
        if sha_file(child(ROOT, name)) != expected:
            raise ValueError('Frozen B source changed: '+name)
    roots = [Path(__file__).name, 'push150_phase_a_report.py', parent['implementation']+'.py', *identity['source_sha256']]
    source_hashes = source_graph(ROOT, roots)
    full_path = verified_record(args.gate_run_dir, build['full_table'])
    assert_exact_statistics(parent['model'], torch.load(full_path, map_location='cpu', weights_only=True))
    evidence = dict(plan_sha256=PLAN_SHA256, parent_sha256=args.expected_parent_sha256,
        parent_path=str(args.parent.resolve()), parent_implementation=parent['implementation'],
        parent_lineage_targets=parent['train_tokens'], parent_config=parent['config'],
        phase_a_selection_path=str(args.phase_a_selection.resolve()), phase_a_selection_sha256=sha_file(args.phase_a_selection),
        phase_a_plan_sha256=selected['plan_sha256'], phase_a_report_script_sha256=selected['report_script_sha256'],
        gate_run_dir=str(args.gate_run_dir.resolve()), gate_identity_sha256=sha_file(identity_path),
        gate_build_sha256=sha_file(build_path), gate_plan_sha256=GATE_PLAN_SHA256,
        full_train_table=build['full_table'], source_sha256=source_hashes,
        source_graph_policy='Recursive local static imports plus explicit parent and recorded B-build roots',
        runtime=dict(torch=torch.__version__, numpy=np.__version__, device='cpu', threads=2),
        test_used=False, only_count_component_is_out_of_fold=True, neural_parent_saw_all_train=True)
    return evidence, build


def prepare(args):
    # Exclusive run ownership also preserves evidence if input verification fails.
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    progress = dict(folds_completed=0, target_probability_queries_started=0, target_probability_queries_completed=0,
                    parity_forward_target_occurrences_started=0, parity_forward_target_occurrences_completed=0)
    write_json_new(args.run_dir/'prepare_started.json', dict(started_at_utc=utc_now(), parent=str(args.parent),
        expected_parent_sha256=args.expected_parent_sha256, plan_sha256=PLAN_SHA256, test_used=False))
    try:
        torch.set_num_threads(2)
        identity, build = check_inputs(args)
        ids, train = load_one_split('train')
        if train != build['train']:
            raise ValueError('Supplied train identity differs from B-build.')
        expected_folds = contiguous_folds(len(ids))
        if len(build['folds']) != 10:
            raise ValueError('Expected exactly ten reused OOF tables.')
        identity['train'] = train
        identity['fold_table_sha256'] = [row['sha256'] for row in build['folds']]
        write_json_new(args.run_dir/'identity.json', identity)
        probabilities = np.zeros((len(ids), 4), dtype=np.float32)
        fold_reports = []
        for index, ((start, end), row) in enumerate(zip(expected_folds, build['folds'])):
            included = [list(bounds) for fold, bounds in enumerate(expected_folds) if fold != index]
            if (row['fold'] != index or row['excluded_token_interval'] != [start,end]
                    or row['included_token_intervals'] != included):
                raise ValueError('B fold partition differs from the declared contiguous partition.')
            table_path = verified_record(args.gate_run_dir, row)
            buffers = torch.load(table_path, map_location='cpu', weights_only=True)
            if len(buffers) != 21:
                raise ValueError('Expected all 21 statistical buffers.')
            arrays = {name:value.numpy() for name,value in buffers.items()}
            queries = 4*(end-start-1)
            progress['target_probability_queries_started'] += queries
            probabilities[start:end] = fold_target_probabilities(arrays, ids[start:end])
            progress['target_probability_queries_completed'] += queries
            parity = parity_check(buffers, ids[start:end], probabilities[start:end], progress)
            report = dict(fold=index, start=start, end=end, valid_random_starts=end-start-256,
                          sentinel_input_position=end-1, table_path=row['path'], table_sha256=row['sha256'],
                          table_bytes=row['bytes'], target_probability_queries=queries,
                          queried_label_positions=end-start-1, parity=parity)
            write_json_new(args.run_dir/f'fold-{index:02d}.json', report)
            fold_reports.append(report); progress['folds_completed'] += 1
            del buffers, arrays
        valid_start_ranges(fold_reports)
        path = args.run_dir/'train_oof_probabilities.npy'
        with path.open('xb') as stream:
            np.save(stream, probabilities, allow_pickle=False)
        # Recheck pinned evidence after work, without reloading any raw corpus.
        if (sha_file(args.parent) != identity['parent_sha256']
                or sha_file(args.phase_a_selection) != identity['phase_a_selection_sha256']
                or sha_file(args.gate_run_dir/'identity.json') != identity['gate_identity_sha256']
                or sha_file(args.gate_run_dir/'build.json') != identity['gate_build_sha256']
                or sha_file(ROOT/'runs/push150_phase_a_plan.json') != identity['phase_a_plan_sha256']
                or sha_file(ROOT/'runs/push150_phase_d_oof_plan.json') != PLAN_SHA256):
            raise ValueError('Pinned evidence changed during preparation.')
        for name, expected in identity['source_sha256'].items():
            if sha_file(child(ROOT, name)) != expected:
                raise ValueError('Pinned source changed during preparation: '+name)
        for row in build['folds']:
            verified_record(args.gate_run_dir, row)
        verified_record(args.gate_run_dir, build['full_table'])
        seconds = time.perf_counter()-started
        metadata = dict(**identity, status='completed', completed_at_utc=utc_now(),
            identity_sha256=sha_file(args.run_dir/'identity.json'),
            array=dict(path=path.name, sha256=sha_file(path), bytes=path.stat().st_size,
                       shape=list(probabilities.shape), dtype='float32', maximum_orders=list(MAXIMUM_ORDERS)),
            folds=fold_reports, window_context=256, window_target_span=257,
            window_columns='column=min(within_window_position,3); maximum_order=min(5,t+2)',
            valid_random_starts=sum(row['valid_random_starts'] for row in fold_reports),
            valid_label_positions=sum(row['queried_label_positions'] for row in fold_reports),
            invalid_label_positions=[end-1 for _,end in expected_folds],
            costs=dict(target_probability_queries=progress['target_probability_queries_completed'],
                       parity_forward_target_occurrences=progress['parity_forward_target_occurrences_completed'],
                       new_count_scan_token_occurrences=0, reused_table_scan_cost_counted_in_phase_b_only=True,
                       new_neural_training_targets=0, new_gate_training_targets=0, seconds_total=seconds),
            parity=dict(passes=True, fold_checks=len(fold_reports), atol=PARITY_ATOL, rtol=PARITY_RTOL,
                        maximum_absolute_difference=max(row['parity']['max_absolute_difference'] for row in fold_reports)),
            training_intermediate_only=True, include_in_inference_assets=False,
            prior_test_exposure_disclosed=True, no_validation_or_test_arrays=True)
        metadata['costs']['seconds_total'] = time.perf_counter()-started
        write_json_new(args.run_dir/'metadata.json', metadata)
        return metadata
    except BaseException as error:
        write_json_new(args.run_dir/'prepare_failure.json', dict(status='failed', failed_at_utc=utc_now(),
            failure_type=type(error).__name__, failure_message=str(error), progress=progress,
            elapsed_seconds=time.perf_counter()-started, test_used=False, partial_files_preserved=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--expected-parent-sha256', required=True)
    parser.add_argument('--phase-a-selection', type=Path, default=ROOT/'runs/push150_phase_a_selection.json')
    parser.add_argument('--gate-run-dir', type=Path, default=ROOT/'runs/push150-gate')
    parser.add_argument('--run-dir', type=Path, default=ROOT/'runs/push150-oof-data')
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
