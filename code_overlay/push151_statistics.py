"""Fixed push151 statistics experiment: train-only construction, validation selection.

No test text access, neural optimization, changed inference code or calibration.
Every phase writes a unique start/result/failure record; previous runs stay intact.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from common import ROOT, PROTOCOL, make_model, setup, sha
from evaluate import score
from push145_mkn_build import effective_counts, make_tables
from push150_mkn_sweep import load_one_split, assert_exact_statistics


PLAN_SHA = '3ba6539a20ba74c7989b9733e755d48bd7529ae9446785e918ceb65f049063aa'
PLAN_PATH = ROOT/'runs/push151_plan.json'
RUN = ROOT/'runs/push151-statistics'
PARENT_SHA = 'afaf5f98c4c2148942454bb9295a41f7d41209b804c28db96676a5d9886eb9c7'
SCORER_SHA = '128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d'
IMPLEMENTATION_SHA = 'd0340092e2a2882113533f05f4668c48b721ee4416842910c2ba2ba1758e1d45'
FIXED_CONFIG = dict(temperature=1.3, mkn_weight=.15, mkn_order=5, discount_scale=.7)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    require(not Path(path).name.startswith('wikitext_'), 'Metadata reader cannot open corpus text')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def child(name, root=ROOT):
    name = str(name).replace('\\', '/')
    require(name and ':' not in name and not name.startswith('/') and
            all(part not in ('', '.', '..') for part in name.split('/')), 'Unsafe relative path')
    destination = (Path(root)/name).resolve()
    require(destination.is_relative_to(Path(root).resolve()), 'Path escaped root')
    return destination


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def save_new(path, value):
    with Path(path).open('xb') as stream:
        torch.save(value, stream)


def sources():
    pending = [ROOT/'push151_statistics.py', ROOT/'push151_prune.py', ROOT/'push150_expand.py']
    found = {}
    while pending:
        path = pending.pop().resolve()
        name = path.relative_to(ROOT).as_posix()
        if name in found:
            continue
        found[name] = sha(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                     else [entry.name for entry in node.names] if isinstance(node, ast.Import) else [])
            for module in names:
                local = ROOT/(module.replace('.', '/')+'.py')
                if local.is_file():
                    pending.append(local)
    return dict(sorted(found.items()))


def load_plan():
    require(sha(PLAN_PATH) == PLAN_SHA, 'Frozen plan changed')
    plan = read(PLAN_PATH)
    require(plan['parent']['sha256'] == PARENT_SHA and
            plan['phase_b']['extra_budgets'] == [0, 8388608], 'Unexpected plan')
    require(sha(ROOT/'evaluate.py') == SCORER_SHA and
            sha(ROOT/'push150_expand.py') == IMPLEMENTATION_SHA, 'Fixed scoring source changed')
    return plan


def load_parent(plan):
    path = child(plan['parent']['path'])
    require(sha(path) == PARENT_SHA, 'Fixed parent changed')
    parent = torch.load(path, map_location='cpu', weights_only=True)
    require(parent['protocol'] == PROTOCOL and parent['implementation'] == 'push150_expand' and
            parent['train_tokens'] == 638976000, 'Parent protocol/architecture/ancestry differs')
    require(all(parent['config'].get(key) == value for key, value in FIXED_CONFIG.items()),
            'Parent temperature/statistics differs')
    return parent


def table_tensors(checkpoint):
    return {name.removeprefix('ngram.'): tensor for name, tensor in checkpoint['model'].items()
            if name.startswith('ngram.')}


def tensor_bytes(tables):
    return sum(tensor.numel()*tensor.element_size() for tensor in tables.values())


def assert_neural_unchanged(parent, candidate):
    require(parent['implementation'] == candidate['implementation'] and
            parent['protocol'] == candidate['protocol'] and
            parent['train_tokens'] == candidate['train_tokens'], 'Identity/ancestry changed')
    require(set(parent['model']) == set(candidate['model']), 'State keys changed')
    for name, tensor in parent['model'].items():
        if not name.startswith('ngram.'):
            other = candidate['model'][name]
            require(tensor.dtype == other.dtype and tensor.shape == other.shape and torch.equal(tensor, other),
                    'Non-statistical weight/buffer changed: '+name)
    before = {key: value for key, value in parent['config'].items() if key != 'mkn_shapes'}
    after = {key: value for key, value in candidate['config'].items() if key != 'mkn_shapes'}
    require(before == after, 'Only table shapes may change in inference config')


def validate_tables(parent_tables, tables, budget):
    require(len(tables) == 21 and set(tables) == set(parent_tables), 'Need exactly 21 statistical buffers')
    require(tensor_bytes(tables) <= budget, 'Statistical tensor budget exceeded')
    for name, value in tables.items():
        require(value.ndim == 1 and value.dtype == parent_tables[name].dtype and value.device.type == 'cpu',
                'Bad statistical buffer dtype/shape/device: '+name)
        if name == 'unigram' or name.endswith('_2'):
            require(torch.equal(value, parent_tables[name]), 'Fixed unigram/bigram changed: '+name)
        if value.is_floating_point():
            require(bool(torch.isfinite(value).all()) and bool((value >= 0).all()), 'Invalid probability buffer')
    for order in range(2, 6):
        contexts, offsets, words, masses, backoffs = [tables[f'{kind}_{order}'].numpy()
                                                    for kind in ('contexts','offsets','words','masses','backoffs')]
        require(len(offsets) == len(contexts)+1 and offsets[0] == 0 and offsets[-1] == len(words) == len(masses),
                'Invalid sparse offsets')
        require(len(backoffs) == len(contexts) and np.all(np.diff(contexts) > 0) and
                np.all(np.diff(offsets) > 0), 'Duplicate/unsorted/empty contexts')
        require(np.all((words >= 0) & (words < 2048)) and np.all(backoffs <= 1), 'Invalid probability/key range')
        if len(contexts):
            totals = np.add.reduceat(masses.astype(np.float64), offsets[:-1]) + backoffs.astype(np.float64)
            require(np.max(np.abs(totals-1)) < 2e-6, 'Explicit mass plus backoff does not sum to one')
            keys = (np.repeat(contexts, np.diff(offsets)) << 11) | words.astype(np.int64)
            require(np.all(np.diff(keys) > 0), 'Duplicate/unsorted exact entries')


def build():
    from push151_prune import build_pruned_candidates, raw_context_counts_from_ids
    plan = load_plan()
    parent = load_parent(plan)
    RUN.mkdir(exist_ok=False)
    pinned = sources()
    started = time.perf_counter()
    write_new(RUN/'build-started.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=PLAN_SHA, parent_sha256=PARENT_SHA, source_sha256=pinned, test_used=False))
    torch.set_num_threads(2)
    ids, train = load_one_split('train')
    effective = effective_counts(ids, maximum=5)
    rebuilt, original_stats = make_tables(effective, scale=.7, minimum=2, trigram_minimum=1)
    parity = assert_exact_statistics(parent['model'], rebuilt)
    raw_contexts = raw_context_counts_from_ids(ids)
    outputs = build_pruned_candidates(effective, raw_contexts, rebuilt, extra_budgets=(0,8388608), scale=.7)
    require(len(outputs) == 2, 'Unexpected candidate count')
    control_bytes = tensor_bytes(rebuilt)
    records = [dict(name='control', checkpoint=plan['parent']['path'], checkpoint_sha256=PARENT_SHA,
        checkpoint_bytes=child(plan['parent']['path']).stat().st_size, statistical_tensor_bytes=control_bytes)]
    for name, extra, result in zip(plan['phase_b']['names'], plan['phase_b']['extra_budgets'], outputs):
        tables, report = result
        validate_tables(rebuilt, tables, control_bytes+extra)
        destination = RUN/name
        destination.mkdir()
        save_new(destination/'tables.pt', tables)
        candidate = copy.deepcopy(parent)
        candidate['config']['mkn_shapes'] = {key: list(value.shape) for key, value in tables.items()}
        candidate['model'].update({'ngram.'+key: tensor.clone() for key, tensor in tables.items()})
        candidate['push151_provenance'] = dict(phase='B', method='train-only single-correction information proxy',
            plan_sha256=PLAN_SHA, parent_sha256=PARENT_SHA, sources=pinned, train=train,
            new_gradient_targets=0, inherited_shared_lineage_targets=638976000,
            table_sha256=sha(destination/'tables.pt'), extra_tensor_byte_budget=extra, build_report=report,
            test_used=False, prior_test_exposure_disclosed=True)
        candidate['mkn_train_tables_sha256'] = sha(destination/'tables.pt')
        candidate['selection'] = 'push151 predeclared train-only information-proxy table candidate; not yet selected'
        assert_neural_unchanged(parent, candidate)
        save_new(destination/'checkpoint.pt', candidate)
        reloaded = torch.load(destination/'checkpoint.pt', map_location='cpu', weights_only=True)
        assert_neural_unchanged(parent, reloaded)
        require(all(torch.equal(reloaded['model']['ngram.'+key], value) for key,value in tables.items()),
                'Serialized statistics changed')
        records.append(dict(name=name, checkpoint=(destination/'checkpoint.pt').relative_to(ROOT).as_posix(),
            checkpoint_sha256=sha(destination/'checkpoint.pt'), checkpoint_bytes=(destination/'checkpoint.pt').stat().st_size,
            tables_sha256=sha(destination/'tables.pt'), statistical_tensor_bytes=tensor_bytes(tables),
            statistical_tensor_budget_bytes=control_bytes+extra, report=report))
    require(sources() == pinned and sha(PLAN_PATH) == PLAN_SHA, 'Build sources/plan changed')
    report = dict(status='completed', phase='build', plan_sha256=PLAN_SHA, parent_sha256=PARENT_SHA,
        source_sha256=pinned, train=train, control_parity=parity, original_statistics=original_stats,
        candidates=records, count_source_token_occurrences=len(ids), new_gradient_targets=0,
        count_cost_note='One train encoding. Shared effective-count computation for orders1..5 and raw history histograms for3..5; all priorities and two budgets reuse these statistics. This token count is not a claim of only one logical counting operation.',
        inherited_shared_lineage_targets=638976000, test_used=False, prior_test_exposure_disclosed=True,
        seconds_total=time.perf_counter()-started)
    write_new(RUN/'build.json', report)
    return report


def measure_checkpoint(path, expected_sha, tokens, byte_count, device, destination):
    require(sha(path) == expected_sha, 'Candidate changed before scoring')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    require(checkpoint['protocol'] == PROTOCOL and checkpoint['implementation'] == 'push150_expand',
            'Unexpected predictor protocol/implementation')
    require(all(checkpoint['config'].get(key) == value for key,value in FIXED_CONFIG.items()), 'Undeclared calibration')
    model, implementation_sha = make_model(checkpoint['implementation'], copy.deepcopy(checkpoint['config']), device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    measured = score(model, tokens, byte_count, device, 'fp32')
    losses = measured.pop('window_nll_nats')
    require(measured['targets'] == 376599 and measured['utf8_bytes'] == 1148007 and
            math.isfinite(measured['bpb']), 'Incomplete/nonfinite full validation score')
    with destination.with_suffix('.window-nll.npy').open('xb') as stream:
        np.save(stream, np.asarray(losses, dtype=np.float64))
    measured.update(checkpoint=path.relative_to(ROOT).as_posix(), checkpoint_sha256=expected_sha,
        implementation_sha256=implementation_sha, evaluator_sha256=sha(ROOT/'evaluate.py'),
        device=str(device), precision='fp32', split='validation', test_used=False)
    write_new(destination, measured)
    require(sha(path) == expected_sha, 'Candidate changed while scoring')
    return measured


def run_score(device_name):
    load_plan()
    build_report = read(RUN/'build.json')
    require(build_report['status'] == 'completed' and build_report['plan_sha256'] == PLAN_SHA and
            build_report['source_sha256'] == sources(), 'Build/source/plan mismatch')
    pinned = sources()
    started = time.perf_counter()
    write_new(RUN/'score-started.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=PLAN_SHA, build_sha256=sha(RUN/'build.json'), source_sha256=pinned, test_used=False))
    device, _ = setup(device_name, 'fp32', 2)
    ids, validation = load_one_split('validation')
    tokens = torch.from_numpy(ids)
    rows = []
    for record in build_report['candidates']:
        measured = measure_checkpoint(child(record['checkpoint']), record['checkpoint_sha256'],
                                      tokens, validation['utf8_bytes'], device, RUN/(record['name']+'.validation.json'))
        rows.append(dict(name=record['name'], **measured))
        print(json.dumps(rows[-1]), flush=True)
    require(abs(rows[0]['bpb']-load_plan()['parent']['validation_bpb_gpu_fp32']) < 1e-6,
            'Fresh fixed control differs from prior full FP32 validation')
    require(sources() == pinned, 'Scoring sources changed')
    report = dict(status='completed', phase='score', plan_sha256=PLAN_SHA,
        build_sha256=sha(RUN/'build.json'), source_sha256=pinned, validation=validation,
        candidates=rows, best=min(rows,key=lambda row:row['bpb']), new_gradient_targets=0,
        test_used=False, prior_test_exposure_disclosed=True, seconds_total=time.perf_counter()-started)
    write_new(RUN/'score.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('build','score'))
    parser.add_argument('--device', default='cuda', choices=('cuda','cpu'))
    args = parser.parse_args()
    try:
        result = build() if args.phase == 'build' else run_score(args.device)
    except BaseException as error:
        failure = RUN/(args.phase+'-failure.json')
        if RUN.is_dir() and not failure.exists():
            write_new(failure, dict(status='failed', phase=args.phase, failure_type=type(error).__name__,
                failure_message=str(error), recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                test_used=False, no_automatic_retry=True))
        raise
    print(json.dumps({key:value for key,value in result.items() if key not in ('source_sha256','original_statistics','candidates')},
                     indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
