"""Read-only Phase A/B/C evidence selection, never a freeze or test authority.

Only JSON, Python source and CPU checkpoint evidence are inspected. Training
feature .npy files may be hashed, never decoded; corpus text is never opened or
hashed. Missing/failed/resumed work retains its observed evidence but refuses
automatic selection. No predictor is instantiated and no split is scored.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch
import push150_phase_a_report as a


ROOT = Path(__file__).resolve().parent
require, natural, finite = a.require, a.natural, a.finite
read, child, checkpoint, same_tensors = a.read, a.child, a.checkpoint, a.same_tensors
PLAN_SHA = {
    'gate': '96be93fd8bc1f2199042a57956e3d79eb7b55c48c606db098b1ecf4a808fe90b',
    'suffix': 'ae460ed88b4b3e4f0101a2dd837d6aa77873610cf5d991cad07f40c4590ae38d',
    'fast': '6105ab5485051a4e3085f555d5a221d7fd85e7f4cd7be42e87ebf6c717991928',
    'repair': 'b864308cd43c295461e80f35aa6cd4161e2366ba1b555d51cd643d453fc503df',
}
NEW_SOURCES = {
    'push150_crossfit.py': 'f7e32454fed14ac04fbf13175a3e4384973c964d6ac319bd47be118cd0afe42f',
    'push150_gate.py': 'f53b247c567d734cb295ae44de72d05600c9195e68b46b246e9eca9416ef7ce2',
    'push150_gate_train.py': '705493516ab734589a1f48e38c5eba715fd29b12be0406e83d643c450adace81',
    'push150_suffix.py': '8337bffc2d138eefe8b27be1237217ee3499c9d5127c0efcf8139ea99f4060c8',
    'push150_suffix_init.py': '88a508fcec4815a099ddd92ac0fd7d06c8fcd03fde162b1ceb520994f88d36fd',
    'push150_suffix_train.py': 'c377748d2b14b8eed6da261a2be80cdf70d22ba332ccb8ee6afad6e07821323c',
    'push150_suffix_calibrate.py': 'b68789897735410887f046cf0ce3030dd7228f80f349046aced6d0d50426b795',
    'push150_fast.py': '46ff0662aacafdb5bc1ecd6561d60678299100a835c7f1fd0f830e2366dfff9d',
    'push150_fast_screen.py': '244fd3aa18764f1af5ac1acba4f0e4301f14a10ae5262fa0170b1c49b08583b4',
    'push150_fast_recheck.py': '9a3ca2514a3da79520a29312bb3a5b14c320f8799d7f73cdb95cf42679eb0d6c',
}
FROZEN_SOURCES = dict(a.TRAIN_SOURCES, **a.MKN_SOURCES, **NEW_SOURCES)
FROZEN_SOURCES.update({'push150_init.py': a.INIT_SHA, 'push146_calibrate.py': a.CALIBRATE_SHA})
GATE_SOURCE_NAMES = (
    'push150_gate_train.py', 'push150_gate.py', 'push150_crossfit.py', 'push150_mkn_sweep.py',
    'push145_mkn_build.py', 'push145_mkn_select.py', 'push145_mkn.py', 'push150_expand.py',
    'push147_signed_cache.py', 'push147_cache.py', 'push145_cache_hybrid.py', 'push145_cache.py',
    'lowrank.py', 'efficient.py', 'refined.py', 'student.py', 'ngram_hybrid.py', 'common.py', 'evaluate.py')
FEATURES = ['neural_cache_max_logit', 'pointer_gate', 'high_order_mass',
            'highest_order_divided_by_5', 'log1p_highest_context_branches']
OFFSETS = [-.5, 0., .5, 1.]
SUFFIX_NAMES = {f'cache_suffix_bias_{order}' for order in range(1, 5)}
TRAIN_TOKENS, TRAIN_BYTES = 3613343, 10951562
ERRORS = (OSError, ValueError, KeyError, TypeError, RuntimeError, AssertionError)


def sha(path):
    """Explicitly refuse corpus text, even when only a hash was requested."""
    path = Path(path)
    if path.suffix.lower() == '.npy':
        require(path.parent.name == 'train_features', 'Only derived train feature arrays may be hashed')
    else:
        a.allowed(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(directory, record):
    path = child(directory, record['path'])
    require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'],
            'Derived evidence file changed: '+str(path))
    return path


def raw_checkpoint(path):
    a.allowed(path)
    require(Path(path).suffix == '.pt', 'Expected .pt evidence')
    return torch.load(path, map_location='cpu', weights_only=True)


def near(first, second, label, tolerance=1e-10):
    require(abs(finite(first, label)-finite(second, label)) <= tolerance, label+' mismatch')


def source_graph(root, entrypoints):
    """Read ASTs without importing any training/inference module."""
    pending, result = list(entrypoints), {}
    while pending:
        name = pending.pop()
        if name in result:
            continue
        path = child(root, name)
        require(name in FROZEN_SOURCES, 'Unpinned local dependency: '+name)
        result[name] = sha(path)
        require(result[name] == FROZEN_SOURCES[name], 'Frozen source changed: '+name)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                       [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                dependency = module.replace('.', '/')+'.py'
                if child(root, dependency).is_file():
                    pending.append(dependency)
    return dict(sorted(result.items()))


def training_sources(root, implementation, suffix):
    entries = ['push150_suffix_train.py' if suffix else 'push150_train.py',
               implementation+'.py', 'push148_muon.py']
    if suffix:
        entries += ['push150_suffix_init.py', 'push150_suffix_calibrate.py']
    return source_graph(root, entries)


def verify_data(record, split):
    tokens, byte_count = (TRAIN_TOKENS, TRAIN_BYTES) if split == 'train' else (376600, 1148007)
    expected = dict(split=split, tokens=tokens, utf8_bytes=byte_count,
        text_sha256=a.DATA_SHA['data/wikitext_'+split+'.txt'],
        tokenizer_sha256=a.DATA_SHA['data/tokenizer.json'], manifest_sha256=a.DATA_SHA['data/manifest.json'])
    require(record == expected, 'Recorded '+split+' provenance differs; no text is read by this verifier')


def verify_units(root, paths):
    records = []
    for path in paths:
        record = read(path)
        require(record['successful'] is True and not record['failures'] and not record['errors'] and
                not record['skipped'] and record['test_text_loaded'] is False, 'Unit evidence did not pass cleanly')
        require(natural(record['tests_run'], 'unit executions') > 0, 'No executed unit tests')
        for name, digest in record['source_sha256'].items():
            require(sha(child(root, name)) == digest, 'Unit-tested source changed: '+name)
        for name, digest in record['unit_source_sha256'].items():
            require(sha(child(root/'tests', name)) == digest, 'Unit test source changed: '+name)
        records.append(dict(path=str(path), sha256=sha(path), record=record))
    return dict(evidence=records, execution_count=sum(item['record']['tests_run'] for item in records),
        execution_count_is_not_unique_tests=True,
        this_report_covered=any('push150_phase_bc_report.py' in item['record']['source_sha256'] and
                               'test_push150_phase_bc_report.py' in item['record']['unit_source_sha256'] for item in records))


def verify_phase_a(root, row):
    path = root/'runs/push150_phase_a_selection.json'
    saved = read(path)
    row.update(sha256=sha(path), record=saved)
    require(saved['status'] == 'phase_a_choice_complete' and not saved['evidence_errors'] and
            saved['test_used'] is False and saved['all_training_accounting_verified'] is True,
            'Phase A did not finish with intact evidence')
    require(all(item.get('status') in ('calibrated','skipped_cpu_precheck') for item in saved['candidates']
                if item.get('kind') == 'training'), 'Failed/unfinished Phase A training requires explicit reconciliation')
    # Re-run the *evidence checker*, not a predictor, against all bound artifacts.
    checked = a.collect_report(root, saved['unit_test_executions'])
    expected, actual = copy.deepcopy(saved), copy.deepcopy(checked)
    expected.pop('recorded_at_utc', None); actual.pop('recorded_at_utc', None)
    require(expected == actual, 'Phase A artifacts no longer reproduce its evidence report')
    candidates = [copy.deepcopy(item) for item in saved['candidates'] if item['selection_eligible']]
    winner = min(candidates, key=lambda item: finite(item['validation_bpb_gpu_fp32'], 'Phase A BPB'))
    require(saved['winner'] == winner, 'Phase A winner is not its eligible minimum')
    parent_path = child(root/'runs', winner['selected_run'])/'checkpoint.pt'
    require(sha(parent_path) == winner['checkpoint_sha256'], 'Phase A winner checkpoint changed')
    parent = checkpoint(parent_path)
    require(parent['train_tokens'] == winner['selected_lineage_targets'], 'Phase A parent cost differs')
    row['revalidated_without_scoring'] = True
    for item in candidates:
        item['phase'] = 'A'
    return candidates, parent, parent_path, winner


def expected_suffix_recipe(parent_config, suffix):
    result = a.expected_recipe({'schedule': 'cosine'}, parent_config)
    result.update(steps=4000, eval_every=1000, seed=191, warmup=100, decay_steps=1000,
                  tail_snapshot_start_step=3000)
    if suffix:
        result['objective'] = 'train-only neural+pointer+causal-suffix NLL; no statistical table loss'
    return result


def verify_suffix_history(metrics, parent_targets):
    require(metrics['status'] == 'completed' and metrics['completed_steps'] == 4000,
            'A declared suffix/control run is incomplete or failed')
    require(not metrics['resume_history'] and metrics['discarded_completed_steps'] == 0,
            'Resume requires manual merged-validation and replay-cost accounting')
    history = metrics['validation_history']
    expected = [(0, 'raw')]+[(step, label) for step in (1000, 2000, 3000, 4000) for label in ('raw', 'ema')]
    require([(item['step'], item['weights']) for item in history] == expected, 'Expected exactly nine ordered raw/EMA choices')
    for item in history:
        require(item['session'] == 0, 'Resumed validation cannot be silently discarded')
        a.verify_measurement(item)
    best = min(history, key=lambda item: item['bpb'])
    require(all(metrics['best'][key] == best[key] for key in ('step', 'weights', 'bpb')), 'Best is not minimum validation snapshot')
    require(metrics['best']['selected_lineage_targets'] == parent_targets+best['step']*8192,
            'Selected suffix lineage differs')
    return best


def verify_suffix_init(root, parent, parent_sha, row):
    directory = root/'runs/push150-suffix-init'
    report = read(directory/'initialization.json')
    row.update(report=report, report_sha256=sha(directory/'initialization.json'))
    path = directory/'checkpoint.pt'; digest = sha(path); initial = checkpoint(path)
    require(report['parent_sha256'] == parent_sha and initial['parent_sha256'] == parent_sha,
            'Suffix initializer is not descended from the selected Phase A winner')
    require(report['checkpoint_sha256'] == digest and report['checkpoint_bytes'] == path.stat().st_size,
            'Suffix initializer checkpoint changed')
    for field, value in dict(seed=191, added_parameters=4, new_training_targets=0, evaluated=False,
                             selected=False, test_used=False, data_text_loaded=False, source_splits=[]).items():
        require(report[field] == value, 'Initializer field differs: '+field)
    require(report['initializer_sha256'] == NEW_SOURCES['push150_suffix_init.py'] and
            report['implementation_sha256'] == NEW_SOURCES['push150_suffix.py'], 'Initializer source differs')
    require(initial['implementation'] == 'push150_suffix' and initial['train_tokens'] == parent['train_tokens'],
            'Initializer architecture or cost differs')
    expected_config = copy.deepcopy(parent['config'])
    expected_config.setdefault('push150_ranks', dict.fromkeys(('qkv', 'proj', 'gate', 'value', 'out'), parent['config']['lowrank_rank']))
    expected_config.update(push150_suffix_orders=[1, 2, 3, 4], push150_suffix_bounds=[-4., 4.])
    require(initial['config'] == expected_config, 'Suffix initializer changed parent configuration')
    require(set(initial['model'])-set(parent['model']) == SUFFIX_NAMES, 'Unexpected initializer tensors')
    same_tensors(parent['model'], {key: value for key, value in initial['model'].items() if key not in SUFFIX_NAMES})
    for name in SUFFIX_NAMES:
        value = initial['model'][name]
        require(value.dtype == torch.float32 and value.ndim == 0 and value.item() == 0., 'Suffix scalar is not zero-initialized FP32')
    ancestry = initial['ancestry']
    require(ancestry == initial['initialization_ancestry'] == report['initialization_ancestry'] and
            ancestry['parent_sha256'] == parent_sha and ancestry['prior_ancestry'] == parent.get('ancestry', {}) and
            ancestry['shared_inherited_targets_counted_once'] == parent['train_tokens'] and
            ancestry['inherited_processed_view_targets'] == parent['train_tokens'] and ancestry['new_training_targets'] == 0,
            'Initializer ancestry differs')
    row.update(checkpoint_sha256=digest, verified=True)
    return initial, digest


def verify_calibration(root, directory, parent, parent_sha, suffix, row):
    metrics = read(directory/'metrics.json'); trials = read(directory/'trials.json')
    plan = read(directory/'experiment_plan.json')
    row.update(calibration_metrics_sha256=sha(directory/'metrics.json'), calibration_choices=len(trials),
               calibration_seconds=finite(metrics['seconds_total'], 'calibration seconds'))
    implementation = parent['implementation']
    require(metrics['trials'] == trials and len(trials) == 15 and
            [(r['temperature'], r['mkn_weight']) for r in trials] ==
            [(t, w) for t in (1.2, 1.25, 1.3, 1.35, 1.4) for w in (.1, .15, .2)], 'Undeclared calibration grid')
    for trial in trials:
        near(trial['validation_nll_nats']/math.log(2)/1148007, trial['validation_bpb'], 'Calibration NLL/BPB')
    require(metrics['best'] == min(trials, key=lambda r: finite(r['validation_bpb'], 'calibration BPB')), 'Calibration best is not minimum')
    require(metrics['parent_sha256'] == parent_sha and metrics['parent_config'] == parent['config'], 'Calibration parent differs')
    # The frozen older control calibrator predates this metadata field. Its
    # lineage is still checked against the actual unchanged calibrated CP.
    if suffix:
        require(metrics['inherited_processed_view_targets'] == parent['train_tokens'], 'Calibration lineage differs')
    require(metrics['test_used'] is False and metrics['new_training_targets'] == 0 and
            metrics['implementation'] == implementation and metrics['precision'] == 'fp32' and
            metrics['device'] == 'cuda:0' and metrics['threads'] == 2, 'Calibration protocol differs')
    expected_script = NEW_SOURCES['push150_suffix_calibrate.py'] if suffix else a.CALIBRATE_SHA
    for field, value in dict(script_sha256=expected_script, implementation_sha256=FROZEN_SOURCES[implementation+'.py'],
        evaluator_sha256=a.TRAIN_SOURCES['evaluate.py'], tokenizer_sha256=a.DATA_SHA['data/tokenizer.json'],
        validation_sha256=a.DATA_SHA['data/wikitext_validation.txt'], validation_target_occurrences=376599,
        validation_utf8_bytes=1148007).items():
        require(metrics[field] == value, 'Calibration field differs: '+field)
    full = a.verify_measurement(metrics['full_validation'])
    difference = abs(full-metrics['best']['validation_bpb'])
    require(difference < 1e-6 and metrics['shortcut_matches_full_predictor'] is True, 'Full target-free confirmation failed')
    near(difference, metrics['shortcut_full_bpb_absolute_difference'], 'Calibration confirmation difference', 1e-12)
    require(plan['parent_sha256'] == parent_sha and plan['trials'] == 15 and plan['test_used'] is False and
            plan['temperatures'] == [1.2, 1.25, 1.3, 1.35, 1.4] and plan['mkn_weights'] == [.1, .15, .2] and
            plan['split'] == 'validation' and plan['script_sha256'] == expected_script, 'Calibration plan differs')
    if suffix:
        expected_sources = training_sources(root, implementation, True)
        require(metrics['source_sha256'] == plan['source_sha256'] == expected_sources, 'Suffix calibration source graph differs')
    path = directory/'checkpoint.pt'; digest = sha(path); calibrated = checkpoint(path)
    require(digest == metrics['checkpoint_sha256'] and path.stat().st_size == metrics['checkpoint_bytes'], 'Calibrated checkpoint changed')
    require(calibrated['calibration_parent_sha256'] == parent_sha and calibrated['implementation'] == implementation and
            calibrated['train_tokens'] == parent['train_tokens'], 'Calibrated ancestry differs')
    same_tensors(parent['model'], calibrated['model'])
    config = copy.deepcopy(parent['config']); config.update(temperature=metrics['best']['temperature'], mkn_weight=metrics['best']['mkn_weight'])
    require(calibrated['config'] == config, 'Calibration changed more than temperature/mixture weight')
    row.update(status='calibrated', selection_eligible=True, selected_run=directory.name,
        checkpoint_sha256=digest, selected_lineage_targets=calibrated['train_tokens'], implementation=implementation,
        validation_bpb_gpu_fp32=full, temperature=config['temperature'], mkn_weight=config['mkn_weight'])


def verify_suffix_run(root, name, parent, parent_sha, suffix, row):
    directory = root/'runs'/('push150-'+name)
    metrics = read(directory/'metrics.json')
    require(not list(directory.glob('metrics-before-resume-*.json')), 'Archived resume evidence requires manual merged accounting')
    best = verify_suffix_history(metrics, parent['train_tokens'])
    recipe = expected_suffix_recipe(parent['config'], suffix)
    implementation = parent['implementation']; sources = training_sources(root, implementation, suffix)
    identity = dict(parent_sha256=parent_sha, implementation=implementation, recipe=recipe,
                    source_sha256=sources, data_sha256=a.DATA_SHA, runtime=a.RUNTIME)
    require(metrics['identity'] == identity and metrics['recipe'] == recipe, 'Phase C recipe/runtime/source/parent differs')
    for key, value in dict(parent_sha256=parent_sha, parent_train_targets=parent['train_tokens'],
        training_implementation=implementation, inference_implementation=implementation, seed=191, steps=4000,
        eval_every=1000, batch_size=32, dropout_views_per_sample=1, processed_view_targets_per_step=8192,
        sampled_label_targets_per_step=8192, no_statistical_tables_in_training=True, source_splits=['train','validation'],
        test_used=False, prior_test_exposure_disclosed=True, optimizer='muon', training_precision='bf16', ema_decay=.99,
        training_config=recipe['training_config'], validation_config=recipe['validation_config'],
        source_sha256=sources, data_sha256=a.DATA_SHA, implementation_sha256=sources[implementation+'.py']).items():
        require(metrics[key] == value, 'Phase C training field differs: '+key)
    require(metrics['optimizer_initialization'] == 'Fresh optimizers and EMA from scoring parent; old optimizer states unavailable',
            'Optimizer restart disclosure differs')
    trained_path = directory/'checkpoint.pt'; digest = sha(trained_path)
    require(digest == metrics['best']['scoring_checkpoint_sha256'], 'Canonical checkpoint SHA differs from scoring SHA')
    archive_path = child(directory, metrics['best']['checkpoint_file'])
    require(sha(archive_path) == metrics['best']['checkpoint_sha256'], 'Archived best checkpoint changed')
    trained, archived = checkpoint(trained_path), checkpoint(archive_path)
    same_tensors(trained['model'], archived['model'])
    require(trained['config'] == archived['config'] == recipe['validation_config'], 'Canonical/archive evaluation config differs')
    require(trained['implementation'] == implementation and trained['parent_sha256'] == parent_sha and
            trained['selected_step'] == best['step'] and trained['weights'] == best['weights'] and
            trained['train_tokens'] == parent['train_tokens']+best['step']*8192 and
            trained['training_recipe'] == recipe and trained['ancestry'] == parent.get('ancestry', {}), 'Scoring checkpoint lineage differs')
    same_tensors(parent['model'], trained['model'], 'ngram.')
    state_path = directory/'train_state.pt'; state = raw_checkpoint(state_path)
    require(state['format_version'] == 1 and state['identity'] == identity and state['step'] == 4000 and
            state['log'] == metrics and len(state['optimizers']) == 2, 'Final complete training state differs')
    expected_keys = {key for key in trained['model'] if not key.startswith('ngram.')}
    require(set(state['raw']) == set(state['ema']) == expected_keys, 'Full state lacks raw/EMA model tensors')
    require(set(state['rng']) == {'python','numpy','torch_cpu','sampler','torch_cuda'} and
            len(state['rng']['torch_cuda']) == 1 and all(opt['state'] for opt in state['optimizers']), 'Optimizer/RNG state is incomplete')
    expected_tail = [(step,label) for step in (3000,4000) for label in ('raw','ema')]
    require([(item['step'],item['weights']) for item in metrics['tail_snapshots']] == expected_tail,
            'Complete declared tail snapshots are missing')
    for item in metrics['tail_snapshots']:
        path = child(directory,item['path'])
        require(sha(path) == item['sha256'], 'Retained tail checkpoint changed')
        tail = checkpoint(path)
        require(tail['parent_sha256'] == parent_sha and tail['selected_step'] == item['step'] and
                tail['weights'] == item['weights'] and tail['training_recipe'] == recipe and
                tail['train_tokens'] == parent['train_tokens']+item['step']*8192,
                'Retained tail checkpoint metadata differs')
        same_tensors(tail['model'],parent['model'],'ngram.')
        if item['step'] == 4000:
            same_tensors(state[item['weights']],{k:v for k,v in tail['model'].items() if not k.startswith('ngram.')})
    if suffix:
        for name in SUFFIX_NAMES:
            require(state['raw'][name].ndim == state['ema'][name].ndim == 0 and
                    state['raw'][name].dtype == state['ema'][name].dtype == torch.float32, 'Scalar missing from full state')
    row.update(training_checkpoint_sha256=digest, archive_checkpoint_sha256=metrics['best']['checkpoint_sha256'],
        train_state_sha256=sha(state_path), selected_step=best['step'], selected_weights=best['weights'],
        training_metrics_sha256=sha(directory/'metrics.json'), source_sha256=sources, runtime=a.RUNTIME)
    del state, archived
    verify_calibration(root, directory.with_name(directory.name+'-calibrated'), trained, digest, suffix, row)
    row['accounting_verified'] = True


def fold_specifications(length=TRAIN_TOKENS):
    blocks = length//256
    boundaries = [(index*blocks//10)*256 for index in range(10)]+[length]
    bounds = [[boundaries[i], boundaries[i+1]] for i in range(10)]
    return [dict(fold=i, excluded_token_interval=[start,end],
        included_token_intervals=[value for j,value in enumerate(bounds) if j != i],
        count_scan_token_occurrences=length-(end-start),
        complete_training_windows=len(range(start,end-256,256)),
        training_positions=len(range(start,end-256,256))*256,
        unused_within_fold_target_positions=(end-start-1)-len(range(start,end-256,256))*256)
        for i,(start,end) in enumerate(bounds)]


def verify_gate_training_counts(build, extracted, trained):
    specifications = fold_specifications()
    require(len(build['folds']) == 10, 'Gate needs ten OOF folds')
    for actual, expected in zip(build['folds'], specifications):
        require(all(actual[key] == value for key,value in expected.items()), 'OOF split/window/count accounting differs')
    positions = sum(row['training_positions'] for row in specifications)
    require(build['retained_training_positions'] == extracted['retained_training_positions'] ==
            trained['retained_training_positions'] == positions, 'Gate retained labels differ')
    require(build['parity_count_scan_token_occurrences'] == TRAIN_TOKENS and
            build['ten_crossfit_count_scan_token_occurrences'] == 9*TRAIN_TOKENS and
            build['excluded_cross_fold_target_positions'] == 9, 'OOF count-scan costs differ')
    require(extracted['neural_extraction_target_occurrences'] == positions and
            extracted['initialization_parity_neural_forward_target_occurrences'] == 2*4*256, 'Frozen neural extraction cost differs')
    require(trained['parameters'] == 57 and trained['seed'] == 181 and trained['gate_target_occurrences'] == 6*positions and
            trained['optimizer_updates'] == 6*((positions+8191)//8192), 'Six-epoch gate training cost differs')
    require([row['epoch'] for row in trained['epochs']] == list(range(1,7)), 'Gate epoch choices differ')
    for row in trained['epochs']:
        require(row['target_occurrences'] == positions and row['cumulative_gate_target_occurrences'] == row['epoch']*positions,
                'Gate epoch cost differs')
        require(finite(row['mean_train_nll'], 'gate NLL') >= 0, 'Negative gate training NLL')
    for record in (build, extracted, trained):
        require(record['new_neural_training_targets'] == 0, 'Unexpected neural optimization in gate branch')
    return positions


def verify_gate(root, directory, parent, parent_path, parent_sha, row):
    identity = read(directory/'identity.json'); identity_sha = sha(directory/'identity.json')
    sources = {name:FROZEN_SOURCES[name] for name in GATE_SOURCE_NAMES}
    expected_identity = dict(plan_sha256=PLAN_SHA['gate'], parent_sha256=parent_sha,
        parent_path=str(parent_path.resolve()), parent_implementation=parent['implementation'],
        parent_temperature=parent['config']['temperature'], parent_mkn_weight=parent['config']['mkn_weight'],
        parent_discount_scale=parent['config']['discount_scale'], inherited_processed_targets=parent['train_tokens'],
        source_sha256=sources, seed=181, epochs=6, batch_size=8192, learning_rate=.001, weight_decay=.0001,
        offsets=OFFSETS, features=FEATURES, test_used=False, prior_test_exposure_disclosed=True,
        neural_in_sample_bias_disclosed=True, only_count_component_is_out_of_fold=True)
    require(identity == expected_identity, 'Gate identity is not bound to the Phase A winner and frozen recipe')
    records = {}
    for phase in ('build','extract','train','score'):
        record = read(directory/(phase+'.json')); started = read(directory/(phase+'_started.json'))
        require(record['status'] == 'completed' and record['phase'] == phase and record['identity_sha256'] == identity_sha and
                record['test_used'] is False and started['identity_sha256'] == identity_sha and started['phase'] == phase,
                'Gate phase incomplete or detached from identity: '+phase)
        require(not (directory/(phase+'_failure.json')).exists(), 'Failed gate phase requires manual reconciliation')
        records[phase] = record
    build, extracted, trained, scored = [records[name] for name in ('build','extract','train','score')]
    require(extracted['build_sha256'] == sha(directory/'build.json') and trained['extract_sha256'] == sha(directory/'extract.json'),
            'Gate phase dependency changed')
    verify_data(build['train'], 'train'); require(extracted['train'] == build['train'], 'Gate train data differs')
    positions = verify_gate_training_counts(build, extracted, trained)
    require(all(record['precision'] == 'fp32' and record['device'] in ('cpu','cuda:0') for record in (extracted,trained)),
            'Gate feature/training arithmetic differs from the declared FP32 phases')
    require(build['full_train_21_buffer_parity'] == dict(buffers=21, shapes_dtypes_values_bitwise_equal=True), 'Full train table parity absent')
    table = raw_checkpoint(artifact(directory, build['full_table']))
    same_tensors({key.removeprefix('ngram.'):value for key,value in parent['model'].items() if key.startswith('ngram.')}, table)
    for fold in build['folds']:
        artifact(directory, fold)
        require(read(directory/'tables'/f"fold-{fold['fold']:02d}.json") == fold, 'OOF table sidecar differs')
    require(len(extracted['features']) == 10 and extracted['feature_column_names'] == FEATURES+
            ['neural_target_probability','oof_count_target_probability'], 'Gate features differ')
    for feature, fold in zip(extracted['features'], build['folds']):
        require(feature['fold'] == fold['fold'] and feature['rows'] == fold['training_positions'] and feature['columns'] == 7 and
                feature['dtype'] == 'float32' and feature['statistics_table_sha256'] == fold['sha256'], 'Feature-to-OOF-table binding differs')
        artifact(directory, feature)
        require(read(directory/'train_features'/f"fold-{fold['fold']:02d}.json") == feature, 'Feature sidecar differs')
    parity = extracted['initialization_parity']
    require(parity == read(directory/'extract_initialization_parity.json') and parity['passes'] is True and
            parity['tolerance'] == 2e-5 and finite(parity['maximum_full_log_probability_difference'], 'initial parity') < 2e-5,
            'Initial constant gate did not reproduce parent')
    require(extracted['neural_and_full_statistics_unchanged'] is True, 'Frozen neural/statistics assertion missing')
    normalization = trained['normalization']
    require(normalization['source'] == 'train feature rows ONLY' and normalization['epsilon'] == 1e-6 and
            len(normalization['mean']) == len(normalization['std']) == 5, 'Normalization provenance differs')
    for mean, std in zip(normalization['mean'], normalization['std']):
        finite(mean, 'feature mean'); require(finite(std, 'feature std') >= 1e-6-1e-12, 'Invalid feature std')
    for epoch in trained['epochs']:
        require(read(directory/f"train_epoch_{epoch['epoch']:02d}.json") == epoch, 'Gate epoch sidecar differs')
    head = raw_checkpoint(artifact(directory, trained['trained_head']))
    require(head['parent_sha256'] == parent_sha and head['plan_sha256'] == PLAN_SHA['gate'], 'Trained gate head parent differs')
    shapes = {'feature_mean':(5,), 'feature_std':(5,), 'network.0.weight':(8,5),
              'network.0.bias':(8,), 'network.2.weight':(1,8), 'network.2.bias':(1,)}
    require(set(head['model']) == set(shapes), 'Gate head state keys differ')
    for name, shape in shapes.items():
        value = head['model'][name]
        require(tuple(value.shape) == shape and value.dtype == torch.float32 and bool(torch.isfinite(value).all()), 'Invalid gate head tensor '+name)
    require(head['model']['feature_mean'].tolist() == normalization['mean'] and
            head['model']['feature_std'].tolist() == normalization['std'], 'Saved normalization differs')
    initial_head = raw_checkpoint(directory/'initial_head.pt')
    require(set(initial_head) == set(head['model']) and not bool(initial_head['network.2.weight'].count_nonzero()), 'Initial gate output is not constant')
    fraction = parent['config']['mkn_weight']/.5
    near(initial_head['network.2.bias'].item(), math.log(fraction/(1-fraction)), 'Initial gate bias', 1e-6)
    same_tensors({k:v for k,v in initial_head.items() if k.startswith('feature_')},
                 {k:v for k,v in head['model'].items() if k.startswith('feature_')})
    choices = scored['choices']
    require(choices == read(directory/'choices.json') and len(choices) == 5 and
            [choice['offset'] for choice in choices] == [None]+OFFSETS, 'Gate offset selection differs')
    for choice in choices:
        near(choice['validation_nll_nats']/math.log(2)/1148007, choice['validation_bpb'], 'Gate screen NLL/BPB')
    winner = min(choices, key=lambda c: finite(c['validation_bpb'], 'gate BPB'))
    require(scored['best'] == winner and scored['selected_gate_offset'] == winner['offset'] and
            scored['selected_is_parent'] is (winner['offset'] is None), 'Gate did not select minimum of five')
    near(scored['improvement_vs_gate_parent'],choices[0]['validation_bpb']-winner['validation_bpb'], 'Gate improvement')
    full = a.verify_measurement(scored['full_validation']); difference = abs(full-winner['validation_bpb'])
    require(scored['precision'] == 'fp32' and scored['device'] == 'cuda:0' and scored['validation_split'] == 'validation' and
            scored['validation_target_occurrences'] == 376599, 'Gate selection was not full GPU FP32 validation')
    require(difference < 1e-6 and scored['shortcut_matches_full_predictor'] is True and
            scored['base_neural_and_statistics_weights_unchanged'] is True, 'Gate full confirmation failed')
    near(difference, scored['shortcut_full_bpb_absolute_difference'], 'Gate confirmation difference', 1e-12)
    confirmation = read(directory/'confirmation.json')
    require(confirmation['winner'] == winner and confirmation['full_validation'] == scored['full_validation'] and
            confirmation['passes'] is True, 'Gate confirmation sidecar differs')
    require(read(directory/'metrics.json') == scored, 'Gate final metrics differs from score record')
    provenance = scored['all_phase_provenance']
    verify_data(provenance['validation_text'], 'validation')
    require(provenance['identity'] == identity and provenance['train_text'] == build['train'] and
            provenance['choices'] == choices and provenance['full_validation'] == scored['full_validation'] and
            provenance['validation_gradient'] is False and provenance['training_arrays_in_deployment'] is False and
            provenance['test_used'] is False and provenance['neural_in_sample_bias_disclosed'] is True, 'Gate provenance differs')
    for phase in ('build','extract','train'):
        require(provenance[phase+'_sha256'] == sha(directory/(phase+'.json')), 'Gate provenance phase SHA differs')
        near(provenance[phase+'_seconds'], records[phase]['seconds_total'], 'Gate phase seconds')
    for key, value in dict(gate_target_occurrences=6*positions, new_neural_training_targets=0,
        neural_extraction_target_occurrences=positions, initialization_parity_neural_forward_target_occurrences=2048,
        ten_crossfit_count_scan_token_occurrences=9*TRAIN_TOKENS, parity_count_scan_token_occurrences=TRAIN_TOKENS,
        trained_head_sha256=trained['trained_head']['sha256']).items():
        require(provenance[key] == value, 'Gate provenance cost differs: '+key)
    path = directory/'checkpoint.pt'; digest = sha(path); selected = checkpoint(path)
    require(digest == scored['checkpoint_sha256'] and path.stat().st_size == scored['checkpoint_bytes'] and
            scored['parent_sha256'] == parent_sha and scored['gate_target_occurrences'] == 6*positions and
            scored['new_neural_training_targets'] == 0, 'Gate selected checkpoint/cost differs')
    lineage = parent['train_tokens']+(0 if winner['offset'] is None else 6*positions)
    require(selected['train_tokens'] == scored['selected_lineage_targets'] == lineage, 'Gate selected lineage differs')
    if winner['offset'] is None:
        require(digest == parent_sha, 'Unchanged gate parent winner was repacked or changed')
    else:
        require(selected['implementation'] == 'push150_gate' and selected['gate_parent_sha256'] == parent_sha and
                selected['gate_training_provenance'] == provenance, 'Gate checkpoint ancestry differs')
        same_tensors(parent['model'], {k:v for k,v in selected['model'].items() if not k.startswith('mixture_gate.')})
        same_tensors(head['model'], {k.removeprefix('mixture_gate.'):v for k,v in selected['model'].items() if k.startswith('mixture_gate.')})
        config = copy.deepcopy(parent['config'])
        config.setdefault('push150_ranks', dict.fromkeys(('qkv','proj','gate','value','out'), parent['config']['lowrank_rank']))
        config['push150_gate_offset'] = winner['offset']
        require(selected['config'] == config, 'Gate changed an inherited config')
    row.update(status='confirmed', accounting_verified=True, selection_eligible=True, selected_run=directory.name,
        checkpoint_sha256=digest, validation_bpb_gpu_fp32=full, selected_lineage_targets=lineage,
        implementation=selected['implementation'], selected_is_parent=winner['offset'] is None,
        parent_checkpoint_sha256=parent_sha, identity_sha256=identity_sha, phases=records,
        new_neural_training_targets=0, gate_label_occurrences=6*positions,
        neural_extraction_target_occurrences=positions, initialization_parity_neural_forward_target_occurrences=2048,
        count_scan_token_occurrences=10*TRAIN_TOKENS, validation_screen_choices=5,
        count_only_out_of_fold=True, neural_in_sample_bias_disclosed=True)


def verify_fast_measurement(row, digest, implementation_sha):
    require(row['split'] == 'validation' and row['threads'] == 4 and row['cpu_affinity_mask'] == 15 and
            row['torch_version'] == '2.7.1+cpu' and row['checkpoint_sha256'] == digest and
            row['implementation_sha256'] == implementation_sha and row['evaluator_sha256'] == a.TRAIN_SOURCES['evaluate.py'] and
            len(row['repeats']) == 1, 'Fast CPU protocol differs')
    a.verify_measurement(row['repeats'][0])
    near(row['median_scoring_seconds'], row['repeats'][0]['seconds'], 'Single-repeat CPU time')


def verify_repaired_timings(repair):
    rows = repair['measurements']
    sequence = [(repeat, mb, name) for repeat, batches, names in
                ((0,(8,16,32),('baseline','candidate')), (1,(32,16,8),('candidate','baseline')),
                 (2,(16,32,8),('baseline','candidate'))) for mb in batches for name in names]
    require([(r['repeat'],r['microbatch'],r['name']) for r in rows] == sequence, 'Expected all 18 ordered repair measurements')
    require(len(repair['candidates']) == 3 and [r['microbatch'] for r in repair['candidates']] == [8,16,32], 'Repair candidates differ')
    for candidate in repair['candidates']:
        mb = candidate['microbatch']
        grouped = {name:[r for r in rows if r['microbatch'] == mb and r['name'] == name] for name in ('baseline','candidate')}
        times = {name:[r['median_scoring_seconds'] for r in values] for name,values in grouped.items()}
        medians = {name:statistics.median(values) for name,values in times.items()}
        ratio = medians['candidate']/medians['baseline']
        difference = max(abs(r['repeats'][0]['bpb']-1.4472849220037163) for r in grouped['candidate'])
        memory = max(r['peak_process_working_set_bytes'] for r in grouped['candidate'])
        assets = max(r['checkpoint_bytes'] for r in grouped['candidate'])
        eligible = ratio <= 5 and difference < 1e-5 and memory <= 4*2**30 and assets < 64*2**20
        require(candidate['times'] == times and candidate['medians'] == medians and candidate['eligible_for_training'] is eligible,
                'Repair ratios/resource eligibility do not reproduce measurements')
        near(candidate['cpu_ratio'], ratio, 'Repair CPU ratio')
        near(candidate['bpb_max_difference'], difference, 'Repair BPB difference')
        near(candidate['peak_ram_gib'], memory/2**30, 'Repair RAM')
        require(candidate['pair_ratios'] == [c/b for b,c in zip(times['baseline'],times['candidate'])], 'Pair diagnostics differ')
        near(candidate['max_candidate_min_baseline_ratio'], max(times['candidate'])/min(times['baseline']), 'Diagnostic worst-case ratio')
        require(not eligible and ratio > 5, 'This report only supports the declared failed-fast skip branch')
    require(repair['winner'] is None, 'Failed repaired screen cannot authorize dense training')


def verify_fast(root, row):
    directory = root/'runs/push150-fast-dense'
    prepared = read(directory/'prepared.json'); initial = read(directory/'cpu-screen/summary.json')
    repair = read(directory/'paired-recheck/summary.json')
    row.update(preparation=prepared, initial_drift_screen=initial, paired_repair=repair,
        prepared_sha256=sha(directory/'prepared.json'), initial_screen_sha256=sha(directory/'cpu-screen/summary.json'),
        repair_sha256=sha(directory/'paired-recheck/summary.json'))
    require(prepared['status'] == 'prepared' and prepared['phase_plan_sha256'] == PLAN_SHA['fast'] and
            prepared['parent_sha256'] == '3eb057e46038d52c75803c64bab103545251a8be310158bdb004e98b556dc899' and
            prepared['inherited_train_targets'] == a.PARENT_TARGETS and prepared['new_train_targets'] == 0 and
            prepared['script_sha256'] == NEW_SOURCES['push150_fast_screen.py'], 'Dense wrapper preparation differs')
    require(initial['prepared_sha256'] == repair['prepared_sha256'] == row['prepared_sha256'] and
            repair['original_screen_sha256'] == row['initial_screen_sha256'] and repair['plan_sha256'] == PLAN_SHA['repair'] and
            repair['script_sha256'] == NEW_SOURCES['push150_fast_recheck.py'], 'Timing repair evidence links differ')
    for record in (initial, repair):
        require(record['status'] == 'completed' and record['test_used'] is False and record['device'] == 'cpu' and
                record['precision'] == 'fp32' and record['threads'] == 4 and record['affinity_mask'] == 15, 'CPU screen incomplete/protocol changed')
    require(repair['source_screen_eligibility_withheld'] is True and repair['extra_worst_case_gate'] is False,
            'Drift repair must withhold initial eligibility and use only the declared acceptance gate')
    variants = {item['microbatch']:item for item in prepared['variants']}
    require(set(variants) == {8,16,32}, 'Expected three prepared runtimes')
    original_path = root/'runs/push150-init-dense/checkpoint.pt'
    require(sha(original_path) == prepared['parent_sha256'], 'Dense original parent changed')
    original = checkpoint(original_path)
    for mb, variant in variants.items():
        path = directory/f'mb{mb}/checkpoint.pt'
        require(sha(path) == variant['checkpoint_sha256'] and path.stat().st_size == variant['checkpoint_bytes'] and
                variant['implementation_sha256'] == NEW_SOURCES['push150_fast.py'], 'Prepared fast artifact changed')
        converted = checkpoint(path); same_tensors(original['model'], converted['model'])
        require(converted['implementation'] == 'push150_fast' and converted['train_tokens'] == a.PARENT_TARGETS and
                converted['config']['cpu_feature_microbatch'] == mb, 'Fast conversion metadata differs')
    require([r['name'] for r in initial['measurements']] == ['baseline-before','mb8','mb16','mb32','baseline-after'], 'Original drift screen was discarded or changed')
    row['raw_measurement_evidence'] = []
    for screen, subdirectory in ((initial,'cpu-screen'), (repair,'paired-recheck')):
        for measured in screen['measurements']:
            name = measured['name']
            is_base = name.startswith('baseline')
            mb = measured.get('microbatch', int(name[2:]) if name.startswith('mb') else None)
            variant = None if is_base else variants[mb]
            verify_fast_measurement(measured, a.BASELINE_SHA if is_base else variant['checkpoint_sha256'],
                                    a.BASELINE_IMPL_SHA if is_base else NEW_SOURCES['push150_fast.py'])
            filename = f"mb{mb}-repeat{measured['repeat']}-{name}.json" if subdirectory == 'paired-recheck' else name+'.json'
            raw = read(directory/subdirectory/filename)
            require({k:v for k,v in measured.items() if k not in ('name','repeat','microbatch')} == raw, 'Timing raw record differs from summary')
            row['raw_measurement_evidence'].append(dict(path=subdirectory+'/'+filename,
                sha256=sha(directory/subdirectory/filename)))
    baseline_times = [r['median_scoring_seconds'] for r in initial['measurements'] if r['name'].startswith('baseline')]
    require(max(baseline_times)/min(baseline_times) > 2, 'Original drift rationale does not match evidence')
    verify_repaired_timings(repair)
    decision_path = directory/'training_decision.json'; decision = read(decision_path)
    require(decision['decision'] == 'skipped' and decision['repair_summary'] ==
            'runs/push150-fast-dense/paired-recheck/summary.json' and decision['repair_summary_sha256'] == row['repair_sha256'] and
            decision['conditional_training_run_dir'] == 'runs/push150-fast-dense-cosine' and
            decision['new_training_targets'] == 0 and decision['training_started'] is False and
            decision['validation_calibration_choices'] == 0 and decision['test_used'] is False,
            'Conditional dense training lacks a bound zero-work skip decision')
    require(not child(root, decision['conditional_training_run_dir']).exists(), 'Dense training directory exists despite skip decision')
    row.update(training_decision=decision, training_decision_sha256=sha(decision_path))
    row.update(status='skipped_after_repaired_cpu_failure', accounting_verified=True,
        original_eligibility_accepted=False, repaired_cpu_passes=18, original_cpu_passes=5,
        new_neural_training_targets=0, new_search_sampled_label_occurrences=0,
        diagnostic_worst_case_is_not_an_acceptance_gate=True,
        seconds=prepared['seconds']+initial['seconds_total']+repair['seconds_total'])


def retain_observed_costs(root, gate_directory):
    """Best-effort evidence capture runs BEFORE prerequisites can fail."""
    result = {'suffix_runs':[], 'gate_phases':{}, 'fast_records':{}, 'errors':[]}
    for name in ('suffix-control','suffix-joint'):
        directory = root/'runs'/('push150-'+name)
        row = dict(name=name, accounting_verified=False, selection_eligible=False)
        result['suffix_runs'].append(row)
        try:
            if (directory/'metrics.json').exists():
                metrics = read(directory/'metrics.json')
                row.update(a.training_accounting(metrics), training_metrics_sha256=sha(directory/'metrics.json'),
                    recorded_training_metrics=metrics)
                row['resume_archives'] = [dict(path=p.name,sha256=sha(p),record=read(p)) for p in sorted(directory.glob('metrics-before-resume-*.json'))]
            calibration = directory.with_name(directory.name+'-calibrated')
            for filename in ('experiment_plan.json','trials.json','metrics.json'):
                if (calibration/filename).exists():
                    row.setdefault('observed_calibration', {})[filename] = dict(sha256=sha(calibration/filename), record=read(calibration/filename))
        except ERRORS as error:
            row['observation_error'] = str(error)
    for phase in ('build','extract','train','score'):
        row = {}
        result['gate_phases'][phase] = row
        for suffix in ('.json','_started.json','_failure.json'):
            path = gate_directory/(phase+suffix)
            if path.exists():
                try:
                    row[phase+suffix] = dict(sha256=sha(path), record=read(path))
                except ERRORS as error:
                    result['errors'].append(str(error))
    for relative in ('prepared.json','cpu-screen/summary.json','paired-recheck/summary.json','training_decision.json'):
        path = root/'runs/push150-fast-dense'/relative
        if path.exists():
            try:
                result['fast_records'][relative] = dict(sha256=sha(path), record=read(path))
            except ERRORS as error:
                result['errors'].append(str(error))
    return result


def select_candidates(candidates, original_parent_bpb, errors):
    if errors:
        return dict(status='refused', winner=None, passes_predeclared_improvement_gate=False)
    eligible = [row for row in candidates if row.get('selection_eligible')]
    require(bool(eligible), 'No eligible validation candidates')
    winner = min(eligible, key=lambda row: finite(row['validation_bpb_gpu_fp32'], 'candidate BPB'))
    improvement = original_parent_bpb-winner['validation_bpb_gpu_fp32']
    return dict(status='phase_abc_choice_complete', winner=copy.deepcopy(winner),
                improvement_vs_original149=improvement, passes_predeclared_improvement_gate=improvement >= .0005)


def collect_report(root=ROOT, unit_paths=None, gate_directory=None):
    root = Path(root); gate_directory = Path(gate_directory or root/'runs/push150-gate')
    unit_paths = unit_paths or [root/'runs/push150_unit_tests_all_phases.json']
    result = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(), phase='A+B+C',
        status='refused', winner=None, candidates=[], evidence_errors=[], test_used=False,
        no_raw_text_loaded_or_hashed=True, prior_test_exposure_disclosed=True,
        minimum_validation_improvement=.0005, passes_predeclared_improvement_gate=False,
        selection_split='validation', selection_device='cuda:0', selection_precision='fp32',
        preliminary_phase_choice_only=True, final_cpu_audit_completed=False,
        test_authorized_by_this_report=False, complete_submission=False,
        report_script_sha256=sha(__file__), required_frozen_sources=FROZEN_SOURCES,
        observed_evidence=retain_observed_costs(root, gate_directory))

    def attempt(name, function):
        try:
            return function()
        except ERRORS as error:
            result['evidence_errors'].append(dict(name=name, error=str(error)))
            return None

    def global_evidence():
        for name, digest in FROZEN_SOURCES.items():
            require(sha(root/name) == digest, 'Frozen source changed: '+name)
        result['plans'] = {}
        for key, filename in (('gate','push150_phase_b_gate_plan.json'), ('suffix','push150_phase_c_suffix_plan.json'),
                              ('fast','push150_phase_b_fast_plan.json'), ('repair','push150_phase_b_timing_repair_plan.json')):
            path = root/'runs'/filename
            require(sha(path) == PLAN_SHA[key], 'Predeclared plan changed: '+key)
            result['plans'][key] = dict(sha256=PLAN_SHA[key], record=read(path))
        result['units'] = verify_units(root, unit_paths)

    attempt('sources_plans_units', global_evidence)
    phase_a = result.setdefault('phase_a', {})
    selected_a = attempt('phase_a', lambda: verify_phase_a(root, phase_a))
    fast = result.setdefault('fast_dense', {})
    attempt('fast_dense', lambda: verify_fast(root, fast))
    gate_row = dict(name='gate', phase='B', kind='train_only_gate', selection_eligible=False, accounting_verified=False)
    result['candidates'].append(gate_row)
    suffix_rows = []
    for name in ('suffix-control','suffix-joint'):
        observed = next(row for row in result['observed_evidence']['suffix_runs'] if row['name'] == name)
        row = {key:copy.deepcopy(value) for key,value in observed.items() if key not in ('recorded_training_metrics','observed_calibration')}
        row.update(phase='C', kind='neural_training', selection_eligible=False, accounting_verified=False)
        result['candidates'].append(row); suffix_rows.append(row)
    if selected_a is not None:
        candidates, parent, parent_path, winner = selected_a
        result['candidates'] = candidates+result['candidates']
        result.update(phase_a_winner_checkpoint_sha256=winner['checkpoint_sha256'],
            phase_a_selection_sha256=phase_a['sha256'], phase_a_parent_lineage_targets=parent['train_tokens'],
            original149_validation_bpb_gpu_fp32=phase_a['record']['parent_validation_bpb_gpu_fp32'])
        attempt('gate', lambda: verify_gate(root, gate_directory, parent, parent_path, winner['checkpoint_sha256'], gate_row))
        init_row = result.setdefault('suffix_initialization', {})
        initialized = attempt('suffix_init', lambda: verify_suffix_init(root, parent, winner['checkpoint_sha256'], init_row))
        attempt('suffix_control', lambda: verify_suffix_run(root, 'suffix-control', parent, winner['checkpoint_sha256'], False, suffix_rows[0]))
        if initialized is not None:
            initial, digest = initialized
            attempt('suffix_joint', lambda: verify_suffix_run(root, 'suffix-joint', initial, digest, True, suffix_rows[1]))
    else:
        result['evidence_errors'].append(dict(name='dependent_branches', error='Gate/C parent binding cannot be established until Phase A passes'))
    result.update(select_candidates(result['candidates'], result.get('original149_validation_bpb_gpu_fp32', 0.), result['evidence_errors']))
    a_record = phase_a.get('record', {})
    c_targets = sum(row.get('new_search_sampled_label_occurrences',0) for row in suffix_rows)
    gate_training = result['observed_evidence']['gate_phases']['train'].get('train.json', {}).get('record', {})
    gate_failure = result['observed_evidence']['gate_phases']['train'].get('train_failure.json', {}).get('record', {})
    gate_labels = gate_training.get('gate_target_occurrences', gate_failure.get('progress', {}).get('gate_target_occurrences',0))
    result['cost_summary'] = dict(accounting_verified=not result['evidence_errors'],
        phase_a_new_neural_view_targets=a_record.get('all_new_search_processed_view_targets',0),
        phase_c_new_neural_view_targets=c_targets, phase_c_new_label_occurrences=c_targets,
        gate_new_neural_training_targets=0, gate_label_occurrences=gate_labels,
        all_recorded_new_gradient_label_occurrences=a_record.get('all_new_search_sampled_label_occurrences',0)+c_targets+gate_labels,
        shared_original149_ancestry_targets=a.PARENT_TARGETS,
        ancestry_is_not_multiplied_by_number_of_trials=True,
        scope='Occurrences, not unique tokens. Branch lineage is separate from whole-search work. Gate extraction/count scans and runtime screens are separate non-gradient work in their full evidence records. Missing/failed costs remain partial, never imputed zero as verified.',
        prior_experiment_costs_not_exhaustively_reconstructed=True)
    result['next_action'] = 'Preliminary validation choice only. Complete actual final CPU/RAM/asset/causality audit before any separate freeze/test decision.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit-evidence', type=Path, action='append', help='May repeat; counts test executions, not unique tests')
    parser.add_argument('--gate-run-dir', type=Path, default=ROOT/'runs/push150-gate')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/push150_phase_bc_selection.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing report; choose a new exclusive output path')
    torch.set_num_threads(2)
    result = collect_report(ROOT, args.unit_evidence, args.gate_run_dir)
    a.write_report(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    if result['status'] == 'refused':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
