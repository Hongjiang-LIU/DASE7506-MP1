"""Four explicit, auditable phases for the predeclared 57-parameter OOF gate.

build: train-only CPU tables; extract: frozen neural train features; train:
six epochs on seven-column train arrays; score: five validation-only choices.
Never call common.load_data, read test text, or optimize the neural parent.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from common import PROTOCOL, ROOT, make_model, setup, sha, windows
from evaluate import score
from push145_mkn_build import effective_counts, make_tables
from push150_mkn_sweep import load_one_split, write_json_new, save_checkpoint_new, assert_exact_statistics
from push150_crossfit import contiguous_folds, complete_window_starts, effective_counts_segments, fold_windows
from push150_gate import (AdaptiveMKNCache, AuxiliaryMKNTables, TinyMixtureGate,
                          gate_features, model_from_parent, assert_frozen_parent,
                          FEATURE_NAMES, NORMALIZATION_EPSILON)


PLAN_SHA256 = '96be93fd8bc1f2199042a57956e3d79eb7b55c48c606db098b1ecf4a808fe90b'
OFFSETS = (-.5, 0., .5, 1.)
SEED, EPOCHS, BATCH_SIZE = 181, 6, 8192
LEARNING_RATE, WEIGHT_DECAY = .001, .0001
SOURCES = ('push150_gate_train.py', 'push150_gate.py', 'push150_crossfit.py',
           'push150_mkn_sweep.py', 'push145_mkn_build.py', 'push145_mkn_select.py',
           'push145_mkn.py', 'push150_expand.py', 'push147_signed_cache.py',
           'push147_cache.py', 'push145_cache_hybrid.py', 'push145_cache.py',
           'lowrank.py', 'efficient.py', 'refined.py', 'student.py',
           'ngram_hybrid.py', 'common.py', 'evaluate.py')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sources():
    return {name: sha(ROOT/name) for name in SOURCES}


def load_parent(args):
    expected = args.expected_parent_sha256
    if len(expected) != 64 or any(letter not in '0123456789abcdef' for letter in expected):
        raise ValueError('Provide a lowercase SHA-256 of the preselected phase-A parent.')
    if sha(args.parent) != expected:
        raise ValueError('Parent does not match the explicitly pinned phase-A winner.')
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    config = parent['config']
    if (parent.get('protocol') != PROTOCOL or parent.get('implementation') not in
            ('push147_signed_cache', 'push150_expand') or config.get('context') != 256
            or config.get('vocab') != 2048 or config.get('mkn_order') != 5
            or not config.get('mkn_shapes') or not 0 < config.get('mkn_weight', 0) < .5
            or not 0 < config.get('discount_scale', 0) < 2):
        raise ValueError('Parent is incompatible with the declared gate experiment.')
    if isinstance(parent.get('train_tokens'), bool) or not isinstance(parent.get('train_tokens'), int):
        raise ValueError('Inherited training cost is missing.')
    return parent


def identity_for(args, parent):
    if sha(args.plan) != PLAN_SHA256:
        raise ValueError('The predeclared gate plan changed.')
    return dict(plan_sha256=PLAN_SHA256, parent_sha256=args.expected_parent_sha256,
                parent_path=str(args.parent.resolve()), parent_implementation=parent['implementation'],
                parent_temperature=parent['config']['temperature'],
                parent_mkn_weight=parent['config']['mkn_weight'],
                parent_discount_scale=parent['config']['discount_scale'],
                inherited_processed_targets=parent['train_tokens'], source_sha256=sources(),
                seed=SEED, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY, offsets=list(OFFSETS), features=list(FEATURE_NAMES),
                test_used=False, prior_test_exposure_disclosed=True,
                neural_in_sample_bias_disclosed=True, only_count_component_is_out_of_fold=True)


def begin_phase(args, identity):
    if args.phase == 'build':
        args.run_dir.mkdir(parents=True, exist_ok=False)
        write_json_new(args.run_dir/'identity.json', identity)
    else:
        previous = json.loads((args.run_dir/'identity.json').read_text(encoding='utf-8'))
        if previous != identity:
            raise ValueError('Pinned plan, parent, sources, or fixed recipe changed.')
    marker = args.run_dir/f'{args.phase}_started.json'
    write_json_new(marker, dict(started_utc=utc_now(), identity_sha256=sha(args.run_dir/'identity.json'),
                               phase=args.phase, device=args.device, test_used=False))
    args.phase_started = True


def load_record(args, phase):
    path = args.run_dir/f'{phase}.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('status') != 'completed' or record.get('identity_sha256') != sha(args.run_dir/'identity.json'):
        raise ValueError(f'Incomplete or unrelated {phase} evidence.')
    return record


def verified_path(args, record):
    relative = Path(record['path'])
    if relative.is_absolute() or relative.drive or '..' in relative.parts:
        raise ValueError('Unsafe intermediate path.')
    path = args.run_dir/relative
    if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
        raise ValueError(f'Intermediate hash/size changed: {relative}')
    return path


def file_record(path, run_dir, **extra):
    return dict(path=path.relative_to(run_dir).as_posix(), sha256=sha(path),
                bytes=path.stat().st_size, **extra)


def finish_phase(args, report):
    report.update(status='completed', phase=args.phase, completed_utc=utc_now(),
                  identity_sha256=sha(args.run_dir/'identity.json'), test_used=False,
                  prior_test_exposure_disclosed=True)
    write_json_new(args.run_dir/f'{args.phase}.json', report)
    return report


def run_build(args, parent):
    ids, data = load_one_split('train')
    scale = parent['config']['discount_scale']
    args.progress.update(train_tokens=len(ids), count_scan_tokens_started=len(ids))
    complete, complete_stats = make_tables(effective_counts(ids), scale, minimum=2, trigram_minimum=1)
    parity = assert_exact_statistics(parent['model'], complete)
    args.progress['count_scan_tokens_completed'] = len(ids)
    table_dir = args.run_dir/'tables'; table_dir.mkdir()
    full_path = table_dir/'full-train-parity.pt'
    save_checkpoint_new(full_path, complete)
    full_record = file_record(full_path, args.run_dir, stats=complete_stats)
    del complete
    folds = contiguous_folds(len(ids))
    records = []
    for index, (start, end) in enumerate(folds):
        included = [ids[left:right] for fold, (left, right) in enumerate(folds) if fold != index]
        token_cost = sum(len(part) for part in included)
        args.progress['count_scan_tokens_started'] += token_cost
        effective = effective_counts_segments(included)
        tables, statistics = make_tables(effective, scale, minimum=2, trigram_minimum=1)
        path = table_dir/f'fold-{index:02d}.pt'
        save_checkpoint_new(path, tables)
        retained_windows = len(complete_window_starts(start, end))
        record = file_record(path, args.run_dir, fold=index, excluded_token_interval=[start, end],
            included_token_intervals=[[left, right] for fold, (left, right) in enumerate(folds) if fold != index],
            count_scan_token_occurrences=token_cost, stats=statistics,
            complete_training_windows=retained_windows, training_positions=retained_windows*256,
            unused_within_fold_target_positions=(end-start-1)-retained_windows*256)
        write_json_new(table_dir/f'fold-{index:02d}.json', record)
        records.append(record)
        args.progress['count_scan_tokens_completed'] += token_cost
        args.progress['fold_tables_completed'] = index+1
        del effective, tables, included
    return finish_phase(args, dict(train=data, full_table=full_record, folds=records,
        full_train_21_buffer_parity=parity, retained_training_positions=sum(row['training_positions'] for row in records),
        excluded_cross_fold_target_positions=len(folds)-1,
        ten_crossfit_count_scan_token_occurrences=sum(row['count_scan_token_occurrences'] for row in records),
        parity_count_scan_token_occurrences=len(ids), new_neural_training_targets=0,
        seconds_total=time.perf_counter()-args.started))


@torch.no_grad()
def initial_prediction_parity(parent, model, x, device):
    reference, _ = make_model(parent['implementation'], copy.deepcopy(parent['config']), device)
    reference.load_state_dict(parent['model'], strict=True)
    reference.eval()
    before, after = reference.predict_log_probs(x.to(device)), model.predict_log_probs(x.to(device))
    maximum = float((before-after).abs().max().cpu())
    del reference
    return dict(maximum_full_log_probability_difference=maximum,
                tolerance=2e-5, passes=maximum < 2e-5,
                note='FP32 scatter/interpolation arithmetic may differ in rounding; no weight/config change.')


@torch.no_grad()
def run_extract(args, parent):
    build = load_record(args, 'build')
    ids, data = load_one_split('train')
    if data != build['train']:
        raise ValueError('Training text/tokenizer/manifest changed after build.')
    full = torch.load(verified_path(args, build['full_table']), map_location='cpu', weights_only=True)
    assert_exact_statistics(parent['model'], full)
    del full
    device, _ = setup(args.device, 'fp32', 2)
    model = model_from_parent(parent, SEED).to(device).eval()
    first = build['folds'][0]['excluded_token_interval']
    x, _ = next(fold_windows(ids, first, batch_size=4))
    args.progress['initialization_parity_neural_forward_targets_started'] = 2*x.numel()
    parity = initial_prediction_parity(parent, model, x, device)
    args.progress['initialization_parity_neural_forward_targets_completed'] = 2*x.numel()
    write_json_new(args.run_dir/'extract_initialization_parity.json', parity)
    if not parity['passes']:
        raise AssertionError('Zero-head full prediction differs from the frozen parent.')
    directory = args.run_dir/'train_features'; directory.mkdir()
    records = []
    for fold in build['folds']:
        buffers = torch.load(verified_path(args, fold), map_location='cpu', weights_only=True)
        table = AuxiliaryMKNTables({key: list(value.shape) for key, value in buffers.items()})
        table.load_state_dict(buffers, strict=True)
        parts = []
        for x, y in fold_windows(ids, fold['excluded_token_interval']):
            args.progress['neural_extraction_target_occurrences_started'] = args.progress.get('neural_extraction_target_occurrences_started', 0)+x.numel()
            neural, pointer = model.neural_and_pointer(x.to(device))
            args.progress['neural_extraction_target_occurrences'] = args.progress.get('neural_extraction_target_occurrences', 0)+x.numel()
            counts, aux = table.probabilities_and_aux(x)
            features = gate_features(neural, pointer, aux.to(device)).cpu().numpy()
            neural_target = neural.gather(-1, y.to(device)[..., None]).squeeze(-1).cpu().numpy()
            count_target = counts.gather(-1, y[..., None]).squeeze(-1).numpy()
            rows = np.concatenate((features, neural_target[..., None], count_target[..., None]), axis=-1).reshape(-1, 7)
            if rows.dtype != np.float32 or not np.isfinite(rows).all() or np.any(rows[:, 5:] <= 0):
                raise ValueError('Invalid train-only feature/target-probability rows.')
            parts.append(rows)
        joined = np.concatenate(parts)
        if len(joined) != fold['training_positions']:
            raise AssertionError('Incomplete or crossed fold extraction windows.')
        path = directory/f'fold-{fold["fold"]:02d}.npy'
        with path.open('xb') as stream:
            np.save(stream, joined)
        record = file_record(path, args.run_dir, fold=fold['fold'], rows=len(joined), columns=7,
            dtype='float32', statistics_table_sha256=fold['sha256'])
        records.append(record)
        write_json_new(directory/f'fold-{fold["fold"]:02d}.json', record)
        args.progress['feature_folds_completed'] = len(records)
        del table, buffers, parts, joined, neural, counts
    assert_frozen_parent(parent, model)
    return finish_phase(args, dict(build_sha256=sha(args.run_dir/'build.json'), train=data,
        features=records, feature_column_names=list(FEATURE_NAMES)+['neural_target_probability','oof_count_target_probability'],
        retained_training_positions=sum(row['rows'] for row in records), new_neural_training_targets=0,
        neural_extraction_target_occurrences=args.progress['neural_extraction_target_occurrences'],
        initialization_parity_neural_forward_target_occurrences=args.progress['initialization_parity_neural_forward_targets_completed'],
        initialization_parity=parity, neural_and_full_statistics_unchanged=True,
        device=str(device), precision='fp32', seconds_total=time.perf_counter()-args.started))


def train_target_loss(head, rows):
    weight = head(rows[:, :5])
    mixed = (1-weight)*rows[:, 5]+weight*rows[:, 6]
    return -mixed.clamp_min(1e-30).log().mean()


def train_normalization(rows):
    values = np.asarray(rows)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('Expected nonempty, finite seven-column train rows.')
    mean = values[:, :5].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values[:, :5].std(axis=0, dtype=np.float64).clip(NORMALIZATION_EPSILON).astype(np.float32)
    return mean, std


def run_train(args, parent):
    extracted = load_record(args, 'extract')
    if extracted['build_sha256'] != sha(args.run_dir/'build.json'):
        raise ValueError('Build evidence changed after extraction.')
    arrays = []
    for record in extracted['features']:
        array = np.load(verified_path(args, record), allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (record['rows'], 7):
            raise ValueError('Unexpected training feature array shape or dtype.')
        arrays.append(array)
    rows = np.concatenate(arrays); del arrays
    if len(rows) != extracted['retained_training_positions'] or np.any(rows[:, 5:] <= 0) or np.any(rows[:, 5:] > 1.000001):
        raise ValueError('Invalid retained target probabilities.')
    mean, std = train_normalization(rows)
    device, _ = setup(args.device, 'fp32', 2)
    torch.manual_seed(SEED)
    head = TinyMixtureGate(parent['config']['mkn_weight']).to(device)
    with torch.no_grad():
        head.feature_mean.copy_(torch.from_numpy(mean).to(device))
        head.feature_std.copy_(torch.from_numpy(std).to(device))
    if sum(value.numel() for value in head.parameters()) != 57:
        raise AssertionError('Unexpected gate parameter count.')
    save_checkpoint_new(args.run_dir/'initial_head.pt', {key: value.detach().cpu().clone() for key, value in head.state_dict().items()})
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sampler = torch.Generator(device='cpu').manual_seed(SEED)
    data = torch.from_numpy(rows).to(device)
    epoch_records = []
    for epoch in range(1, EPOCHS+1):
        permutation = torch.randperm(len(rows), generator=sampler)
        total_loss, epoch_targets = 0., 0
        for start in range(0, len(rows), BATCH_SIZE):
            batch = data[permutation[start:start+BATCH_SIZE].to(device)]
            optimizer.zero_grad(set_to_none=True)
            loss = train_target_loss(head, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite gate training loss.')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach().cpu())*len(batch)
            epoch_targets += len(batch)
            args.progress['gate_target_occurrences'] = args.progress.get('gate_target_occurrences', 0)+len(batch)
            args.progress['optimizer_updates'] = args.progress.get('optimizer_updates', 0)+1
        record = dict(epoch=epoch, mean_train_nll=total_loss/epoch_targets,
                      target_occurrences=epoch_targets, cumulative_gate_target_occurrences=args.progress['gate_target_occurrences'])
        write_json_new(args.run_dir/f'train_epoch_{epoch:02d}.json', record)
        epoch_records.append(record)
    head_path = args.run_dir/'trained_head.pt'
    save_checkpoint_new(head_path, dict(model={key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
                                       parent_sha256=args.expected_parent_sha256, plan_sha256=PLAN_SHA256))
    return finish_phase(args, dict(extract_sha256=sha(args.run_dir/'extract.json'),
        trained_head=file_record(head_path, args.run_dir), parameters=57, epochs=epoch_records,
        retained_training_positions=len(rows), gate_target_occurrences=args.progress['gate_target_occurrences'],
        optimizer_updates=args.progress['optimizer_updates'], new_neural_training_targets=0,
        normalization=dict(source='train feature rows ONLY', mean=mean.tolist(), std=std.tolist(), epsilon=NORMALIZATION_EPSILON),
        seed=SEED, device=str(device), precision='fp32', seconds_total=time.perf_counter()-args.started))


def choice_rows(totals, byte_count):
    if byte_count <= 0 or len(totals) != 5:
        raise ValueError('Expected five complete-validation totals.')
    names = ['unchanged_parent']+[f'gate_delta_{offset:+g}' for offset in OFFSETS]
    offsets = [None]+list(OFFSETS)
    return [dict(name=name, offset=offset, validation_nll_nats=float(total),
                 validation_bpb=float(total)/math.log(2)/byte_count)
            for name, offset, total in zip(names, offsets, totals)]


@torch.no_grad()
def run_score(args, parent):
    trained = load_record(args, 'train')
    extracted = load_record(args, 'extract')
    build = load_record(args, 'build')
    if (trained['extract_sha256'] != sha(args.run_dir/'extract.json')
            or extracted['build_sha256'] != sha(args.run_dir/'build.json')):
        raise ValueError('Prior phase evidence changed.')
    head = torch.load(verified_path(args, trained['trained_head']), map_location='cpu', weights_only=True)
    if head['parent_sha256'] != args.expected_parent_sha256 or head['plan_sha256'] != PLAN_SHA256:
        raise ValueError('Unrelated trained head.')
    ids, validation = load_one_split('validation')
    for key in ('manifest_sha256', 'tokenizer_sha256'):
        if validation[key] != build['train'][key]:
            raise ValueError('Manifest/tokenizer changed between phases.')
    device, _ = setup(args.device, 'fp32', 2)
    model = model_from_parent(parent, SEED).to(device).eval()
    model.mixture_gate.load_state_dict(head['model'], strict=True)
    tokens = torch.from_numpy(ids)
    totals = [0.]*5
    target_count = 0
    for x, y in windows(tokens):
        x, y = x.to(device), y.to(device)
        neural, pointer = model.neural_and_pointer(x)
        counts, aux = model.ngram.probabilities_and_aux(x)
        features = gate_features(neural, pointer, aux)
        target = y.clamp_min(0)[..., None]
        neural_target = neural.gather(-1, target).squeeze(-1)
        count_target = counts.gather(-1, target).squeeze(-1)
        valid = y != -100
        weights = [parent['config']['mkn_weight']]+[model.mixture_gate(features, offset) for offset in OFFSETS]
        for index, weight in enumerate(weights):
            mixed = (1-weight)*neural_target+weight*count_target
            totals[index] += float(-mixed[valid].double().clamp_min(1e-30).log().sum().cpu())
        target_count += int(valid.sum().cpu())
        args.progress['validation_screen_target_occurrences'] = target_count
    choices = choice_rows(totals, validation['utf8_bytes'])
    write_json_new(args.run_dir/'choices.json', choices)
    winner = min(choices, key=lambda row: row['validation_bpb'])
    assert_frozen_parent(parent, model)
    if winner['offset'] is None:
        confirmed, _ = make_model(parent['implementation'], copy.deepcopy(parent['config']), device)
        confirmed.load_state_dict(parent['model'], strict=True)
        confirmed.eval()
    else:
        model.config['push150_gate_offset'] = winner['offset']
        confirmed = model
    measured = score(confirmed, tokens, validation['utf8_bytes'], device, 'fp32')
    losses = measured.pop('window_nll_nats')
    with (args.run_dir/'validation.window-nll.npy').open('xb') as stream:
        np.save(stream, np.asarray(losses, dtype=np.float64))
    difference = abs(measured['bpb']-winner['validation_bpb'])
    confirmation = dict(winner=winner, full_validation=measured,
                        shortcut_full_bpb_absolute_difference=difference, passes=difference < 1e-6)
    write_json_new(args.run_dir/'confirmation.json', confirmation)
    if difference >= 1e-6:
        raise AssertionError('Winner shortcut differs from the full target-free predictor; no candidate published.')
    provenance = dict(identity=json.loads((args.run_dir/'identity.json').read_text(encoding='utf-8')),
        build_sha256=sha(args.run_dir/'build.json'), extract_sha256=sha(args.run_dir/'extract.json'),
        train_sha256=sha(args.run_dir/'train.json'), train_text=build['train'], validation_text=validation,
        trained_head_sha256=trained['trained_head']['sha256'], choices=choices, full_validation=measured,
        gate_target_occurrences=trained['gate_target_occurrences'], new_neural_training_targets=0,
        neural_extraction_target_occurrences=extracted['neural_extraction_target_occurrences'],
        initialization_parity_neural_forward_target_occurrences=extracted['initialization_parity_neural_forward_target_occurrences'],
        ten_crossfit_count_scan_token_occurrences=build['ten_crossfit_count_scan_token_occurrences'],
        parity_count_scan_token_occurrences=build['parity_count_scan_token_occurrences'],
        build_seconds=build['seconds_total'], extract_seconds=extracted['seconds_total'], train_seconds=trained['seconds_total'],
        validation_gradient=False, test_used=False, prior_test_exposure_disclosed=True,
        neural_in_sample_bias_disclosed=True, training_arrays_in_deployment=False)
    destination = args.run_dir/'checkpoint.pt'
    if winner['offset'] is None:
        # Identical original bytes, not a repacked checkpoint with new ancestry.
        with args.parent.open('rb') as source, destination.open('xb') as target:
            while chunk := source.read(1024*1024):
                target.write(chunk)
        if sha(destination) != args.expected_parent_sha256:
            raise AssertionError('Original parent copy changed.')
        selected_lineage = parent['train_tokens']
    else:
        selected_lineage = parent['train_tokens']+trained['gate_target_occurrences']
        candidate = copy.deepcopy(parent)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        state['head.weight'] = state['token.weight']
        candidate.update(implementation='push150_gate', config=copy.deepcopy(model.config), model=state,
            train_tokens=selected_lineage, gate_parent_sha256=args.expected_parent_sha256,
            gate_training_provenance=provenance,
            selection='Five declared full-validation choices; frozen base weights, train-only cross-fit 57-parameter gate')
        save_checkpoint_new(destination, candidate)
        saved = torch.load(destination, map_location='cpu', weights_only=True)
        if saved['config'] != model.config:
            raise AssertionError('Serialized gate configuration changed.')
        for key, value in model.state_dict().items():
            if not torch.equal(value.detach().cpu(), saved['model'][key]):
                raise AssertionError(f'Saved weights differ from the fully scored predictor: {key}')
    report = dict(best=winner, choices=choices, full_validation=measured,
        checkpoint_sha256=sha(destination), checkpoint_bytes=destination.stat().st_size,
        selected_is_parent=winner['offset'] is None, selected_lineage_targets=selected_lineage,
        selected_gate_offset=winner['offset'], gate_target_occurrences=trained['gate_target_occurrences'],
        new_neural_training_targets=0, validation_split='validation',
        parent_sha256=args.expected_parent_sha256, improvement_vs_gate_parent=choices[0]['validation_bpb']-winner['validation_bpb'],
        shortcut_matches_full_predictor=True, shortcut_full_bpb_absolute_difference=difference,
        base_neural_and_statistics_weights_unchanged=True, validation_target_occurrences=target_count,
        device=str(device), precision='fp32', all_phase_provenance=provenance,
        seconds_total=time.perf_counter()-args.started)
    completed = finish_phase(args, report)
    write_json_new(args.run_dir/'metrics.json', completed)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('build', 'extract', 'train', 'score'))
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--expected-parent-sha256', required=True)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--plan', type=Path, default=ROOT/'runs/push150_phase_b_gate_plan.json')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    args.started, args.progress, args.phase_started = time.perf_counter(), {}, False
    try:
        if args.phase == 'build' and args.device != 'cpu':
            raise ValueError('Cross-fit table building is CPU only.')
        torch.set_num_threads(2)
        parent = load_parent(args)
        identity = identity_for(args, parent)
        begin_phase(args, identity)
        result = {'build': run_build, 'extract': run_extract, 'train': run_train, 'score': run_score}[args.phase](args, parent)
    except BaseException as error:
        if args.phase_started:
            failure = args.run_dir/f'{args.phase}_failure.json'
            if not failure.exists():
                write_json_new(failure, dict(phase=args.phase, failure_type=type(error).__name__,
                    message=str(error), recorded_utc=utc_now(), progress=args.progress,
                    elapsed_seconds=time.perf_counter()-args.started, test_used=False,
                    prior_test_exposure_disclosed=True))
        raise
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
