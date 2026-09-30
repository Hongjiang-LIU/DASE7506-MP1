"""Genuine full-state fork: matched fixed-low-LR optimizer keep/reset trials.

Only supplied train learns; validation selects. The old raw parameters retain
their matching optimizer moments, never moments attached to an averaged scoring
checkpoint. A separate current checkpoint supplies only fixed inference tables.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from common import ROOT, PROTOCOL, autocast, make_model, setup, sha
from evaluate import score
from push148_train import build_optimizers
from push150_crossfit import contiguous_folds
from push150_oof_prepare import valid_start_ranges
from push150_oof_train import (cpu_clone, model_state, capture_rng, restore_rng,
    sample_starts_and_coin, training_config, empty_coin_counts, add_coin,
    retain_discarded_coins, atomic_json_save, atomic_torch_save)


STATE_FORMAT = 2
OLD_STATE_SHA = '9ef09a43f4e7703e498ec6801b85738d042b5fd721f2770a8e1622d935ca0d5d'
STATISTICS_PARENT_SHA = 'c72cffff6b4152492518473f46a274b53272f59dacb80072cd9e1497d2e0bdf5'
PLAN_SHA = 'd15cd6276ace1894a37db7c005453c61acd2e5acb7e934cb6790f3860aa06b51'
ANCESTOR_TARGETS, OLD_STEP, STEPS = 638976000, 12000, 8000
BATCH_SIZE, CONTEXT, TARGETS_PER_STEP, EVAL_EVERY = 32, 256, 8192, 2000
RATES = dict(muon=5e-5, backbone=7.5e-7, pointer=7.5e-6, scalar=2.5e-5)
EMA_DECAYS = dict(ema99=.99, ema999=.999)
FIXED_VALIDATION = dict(temperature=1.3, mkn_weight=.15, mkn_order=5, discount_scale=.7)
RUN_NAMES = ('optimizer-keep', 'optimizer-reset')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def save_new(payload, path):
    with Path(path).open('xb') as stream:
        torch.save(payload, stream)


def child(root, name):
    relative = Path(name)
    path = (root/relative).resolve()
    require(not relative.is_absolute() and not relative.drive and '..' not in relative.parts and
        path.is_relative_to(root.resolve()), 'Unsafe relative artifact path')
    require(path.name.lower() != 'wikitext_test.txt', 'Test access is prohibited')
    return path


def source_digests(implementation='push150_expand', root=ROOT):
    root = root.resolve()
    pending = [root/'push152_resume.py', root/(implementation+'.py')]
    found = {}
    while pending:
        path = pending.pop().resolve()
        name = path.relative_to(root).as_posix()
        if name in found:
            continue
        found[name] = sha(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                [entry.name for entry in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                local = root/(module.replace('.', '/')+'.py')
                if local.is_file():
                    pending.append(local)
    return dict(sorted(found.items()))


def control_loss(model, ids, targets):
    require(not hasattr(model, 'ngram'), 'Training must not contain inference count tables')
    probabilities = model.neural_and_cache(ids)
    selected = probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
    return -selected.clamp_min(1e-30).log().mean()


def parameter_kind(name, parameter):
    if name.startswith('blocks.') and parameter.ndim == 2:
        return 'muon'
    if parameter.ndim == 0:
        return 'scalar'
    return 'pointer' if name.startswith('pointer_') else 'backbone'


def set_constant_rates(model, optimizers):
    """After loading moments, replace only the declared LR/base-LR values."""
    named = {id(parameter): (name, parameter_kind(name, parameter)) for name, parameter in model.named_parameters()}
    seen = []
    layout = []
    for optimizer in optimizers:
        groups = []
        for group in optimizer.param_groups:
            values = [named[id(parameter)] for parameter in group['params']]
            kinds = {kind for _, kind in values}
            require(len(kinds) == 1, 'Optimizer group mixes distinct declared learning rates')
            kind = next(iter(kinds))
            group['lr'] = group['base_lr'] = RATES[kind]
            groups.append(dict(kind=kind, names=[name for name, _ in values], lr=RATES[kind]))
            seen.extend(id(parameter) for parameter in group['params'])
        layout.append(dict(optimizer=type(optimizer).__name__, groups=groups))
    require(len(seen) == len(set(seen)) == len(named), 'Optimizer parameter coverage changed')
    return layout


def validate_optimizer_layout(saved, optimizers):
    require(len(saved) == len(optimizers), 'Optimizer count changed')
    for previous, current in zip(saved, optimizers):
        expected = current.state_dict()['param_groups']
        require(len(previous['param_groups']) == len(expected), 'Optimizer group count changed')
        require([group['params'] for group in previous['param_groups']] == [group['params'] for group in expected],
            'Optimizer parameter order changed')


def fork_from_old_state(state, model, averages, optimizers, sampler, phase_name, cuda=False):
    """Fork old raw/EMA/RNG; this is not a resume under the old identity."""
    require(phase_name in RUN_NAMES and set(averages) == set(EMA_DECAYS), 'Unknown fork or EMA menu')
    require(state.get('format_version') == 1 and state.get('step') == OLD_STEP and
        state['log']['completed_steps'] == OLD_STEP and state['log']['status'] == 'completed',
        'Expected complete original 12000-step training state')
    require(state['identity']['implementation'] == 'push150_expand' and
        state['identity']['recipe']['objective'] == 'control' and
        state['identity']['recipe']['training_config'] == model.config,
        'The old optimizer does not match this exact raw training architecture')
    require(state['log']['parent_train_targets']+OLD_STEP*TARGETS_PER_STEP == ANCESTOR_TARGETS,
        'Shared ancestor cost differs')
    require(not any(key.startswith('ngram.') for key in state['raw']) and
        torch.equal(state['raw']['head.weight'], state['raw']['token.weight']) and
        torch.equal(state['ema']['head.weight'], state['ema']['token.weight']), 'Invalid neural-only tied state')
    validate_optimizer_layout(state['optimizers'], optimizers)
    model.load_state_dict(state['raw'], strict=True)
    for averaged in averages.values():
        averaged.load_state_dict(state['ema'], strict=True)
        averaged.eval().requires_grad_(False)
    if phase_name == 'optimizer-keep':
        for optimizer, previous in zip(optimizers, state['optimizers']):
            optimizer.load_state_dict(cpu_clone(previous))
    else:
        for optimizer in optimizers:
            require(not optimizer.state, 'Reset branch must start with fresh empty optimizer state')
    layout = set_constant_rates(model, optimizers)
    model.train(state['model_training'])
    # Must be last: constructors, optimizer restoration and loading may consume RNG.
    restore_rng(state['rng'], sampler, cuda)
    return layout


@torch.no_grad()
def update_averages(model, averages):
    require(set(averages) == set(EMA_DECAYS), 'EMA menu changed')
    current = dict(model.named_parameters())
    for label, averaged in averages.items():
        targets = dict(averaged.named_parameters())
        require(targets.keys() == current.keys(), 'EMA parameter order differs')
        for name, parameter in current.items():
            targets[name].lerp_(parameter, 1.-EMA_DECAYS[label])


def pack_state(model, averages, optimizers, sampler, identity, step, log, cuda=False):
    return dict(format_version=STATE_FORMAT, identity=copy.deepcopy(identity), step=step,
        raw=model_state(model), averages={label: model_state(value) for label, value in averages.items()},
        optimizers=[cpu_clone(value.state_dict()) for value in optimizers], model_training=model.training,
        rng=capture_rng(sampler, cuda), log=copy.deepcopy(log))


def restore_state(state, model, averages, optimizers, sampler, identity, cuda=False):
    require(state.get('format_version') == STATE_FORMAT and state.get('identity') == identity,
        'New trajectory resume identity changed')
    step = state['step']
    require(type(step) is int and 0 <= step <= identity['recipe']['steps'] and
        state['log']['completed_steps'] == step and set(state['averages']) == set(EMA_DECAYS),
        'Incomplete new trajectory state')
    validate_optimizer_layout(state['optimizers'], optimizers)
    model.load_state_dict(state['raw'], strict=True)
    for label, averaged in averages.items():
        averaged.load_state_dict(state['averages'][label], strict=True)
        averaged.eval().requires_grad_(False)
    for optimizer, previous in zip(optimizers, state['optimizers']):
        optimizer.load_state_dict(previous)
    set_constant_rates(model, optimizers)
    model.train(state['model_training'])
    restore_rng(state['rng'], sampler, cuda)
    return step, copy.deepcopy(state['log'])


def validation_choices(step, model, averages):
    return [('raw', model), ('ema99', averages['ema99'])] + ([] if step == 0 else [('ema999', averages['ema999'])])


def make_snapshot(model, statistics_parent, identity, recipe, step, label):
    state = model_state(model)
    require(not any(key.startswith('ngram.') for key in state), 'Training statistics contamination')
    state.update({key: cpu_clone(value) for key, value in statistics_parent['model'].items() if key.startswith('ngram.')})
    return dict(protocol=PROTOCOL, implementation='push150_expand', config=copy.deepcopy(statistics_parent['config']),
        model=state, train_tokens=ANCESTOR_TARGETS+step*TARGETS_PER_STEP, selected_step=step, weights=label,
        parent_sha256=STATISTICS_PARENT_SHA, fork_training_state_sha256=OLD_STATE_SHA,
        training_config=copy.deepcopy(model.config), training_recipe=copy.deepcopy(recipe),
        plan_sha256=identity['plan_sha256'], source_sha256=copy.deepcopy(identity['source_sha256']),
        ancestry=dict(inherited=copy.deepcopy(statistics_parent.get('ancestry', {})),
            shared_ancestor_targets=ANCESTOR_TARGETS, new_training_targets=step*TARGETS_PER_STEP,
            fork_state_and_scoring_parent_share_one_training_trajectory=True,
            note='Old raw/EMA optimizer trajectory and current averaged scoring parent share ancestry; never sum them.'),
        selection='Fixed-menu full validation only; no calibration grid',
        inference_architecture_unchanged_from_parent=True, inference_tables_unchanged_from_statistics_parent=True)


def validate_fold_geometry(folds, token_count):
    expected = contiguous_folds(token_count)
    require(len(folds) == len(expected), 'Inherited fold count differs')
    for index, (record, (start, end)) in enumerate(zip(folds, expected)):
        require(record['fold'] == index and record['start'] == start and record['end'] == end and
            record['valid_random_starts'] == end-start-CONTEXT, 'Inherited fold-contained sampler geometry differs')
    return torch.from_numpy(valid_start_ranges(folds, context=CONTEXT).copy())


def verify_plan(plan, phase_name):
    phase, shared = plan['phase_a'], plan['shared']
    require(plan['round'] == 'push152' and plan['parent']['sha256'] == STATISTICS_PARENT_SHA and
        plan['parent']['implementation'] == 'push150_expand' and
        plan['parent']['selected_lineage_targets'] == ANCESTOR_TARGETS, 'Wrong fixed statistics parent')
    require(phase['state_sha256'] == OLD_STATE_SHA and phase['source_step'] == OLD_STEP and
        phase['source_lineage_targets'] == ANCESTOR_TARGETS and phase['steps'] == STEPS and
        phase['eval_every'] == EVAL_EVERY and phase['learning_rates'] == RATES and
        phase['schedule'] == 'constant' and set(phase['variants']) == set(RUN_NAMES), 'Phase-A recipe changed')
    require(phase['ema']['existing_decay'] == .99 and phase['ema']['new_decay'] == .999,
        'EMA decay menu changed')
    expected = dict(training_split='train', selection_split='validation', context=CONTEXT,
        batch_size=BATCH_SIZE, targets_per_step=TARGETS_PER_STEP, training_precision='bf16',
        validation_precision='fp32', validation_device='cuda', training_temperature=1.15,
        validation_temperature=1.3, dropout=.2, mkn_weight=.15, mkn_order=5, discount_scale=.7,
        gradient_clip=1., optimizer='muon', adamw_betas=[.9,.95], backbone_weight_decay=.1,
        pointer_weight_decay=.01, scalar_weight_decay=0., muon_momentum=.95, muon_ns_steps=5,
        cpu_threads=4, no_calibration_grid=True)
    require(all(shared.get(key) == value for key, value in expected.items()), 'Shared training/evaluation recipe changed')
    require(phase_name in RUN_NAMES and phase['variants'][phase_name]['keep_optimizer_state'] is
        (phase_name == 'optimizer-keep'), 'Optimizer fork choice disagrees with plan')
    require(plan['external_training_data_or_weights'] is False and plan['evaluation_network_access'] is False,
        'Forbidden external sources')
    return phase['variants'][phase_name]


def runtime_record(device):
    return dict(torch=str(torch.__version__), numpy=np.__version__, python=platform.python_version(),
        cuda=torch.version.cuda, device=str(device), device_name=torch.cuda.get_device_name(device),
        cuda_device_count=torch.cuda.device_count(), cpu_threads=torch.get_num_threads())


def load_allowed_data(old_state):
    manifest_path = ROOT/'data/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    tokenizer_path = ROOT/'data/tokenizer.json'
    data_hashes = {name: sha(ROOT/name) for name in ('data/manifest.json', 'data/tokenizer.json')}
    require(data_hashes['data/tokenizer.json'] == manifest['sha256']['tokenizer.json'], 'Fixed tokenizer changed')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    data = {}
    for split in ('train', 'validation'):
        path = ROOT/'data'/('wikitext_'+split+'.txt')
        digest = sha(path)
        require(digest == manifest['sha256'][path.name], 'Changed supplied '+split+' text')
        raw = path.read_bytes()
        data_hashes['data/'+path.name] = digest
        data[split] = (torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids, dtype=torch.long), len(raw))
    require(data_hashes == old_state['identity']['data_sha256'], 'Continuation data differs from the original trajectory')
    require(len(data['validation'][0])-1 == 376599 and data['validation'][1] == 1148007,
        'Fixed validation token/byte counts differ')
    return data, data_hashes


def run(args):
    plan_path = args.plan.resolve()
    require(sha(plan_path) == PLAN_SHA, 'Pre-execution push152 plan changed')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    variant = verify_plan(plan, args.phase_name)
    args.run_dir = child(ROOT, variant['run_dir'])
    state_path = args.run_dir/'train_state.pt'
    if args.resume:
        require(args.resume.resolve() == state_path.resolve() and state_path.is_file(),
            'Resume requires this new run directory\'s exact full train_state.pt')
    else:
        args.run_dir.mkdir(parents=True, exist_ok=False)
    args._run_directory_confirmed = True
    old_state_path = child(ROOT, plan['phase_a']['state'])
    parent_path = child(ROOT, plan['parent']['path'])
    require(sha(old_state_path) == OLD_STATE_SHA and sha(parent_path) == STATISTICS_PARENT_SHA,
        'Fixed fork state/statistics parent changed')
    # The global phase gate is completion of expansion preflight, not necessarily
    # permission to train the expansion branch. Parent-shape A runs remain legal
    # if the expansion fails its resource gate and is explicitly skipped.
    authorization_path = child(ROOT, plan['expansion']['authorization'])
    authorization = json.loads(authorization_path.read_text(encoding='utf-8'))
    require(authorization.get('status') == 'completed' and authorization.get('plan_sha256') == PLAN_SHA and
        type(authorization.get('training_authorized')) is bool and authorization.get('test_used') is False and
        authorization.get('no_layout_fallback') is True and
        authorization.get('script_sha256') == sha(ROOT/'push152_preflight.py'),
        'Expansion feasibility must be explicitly concluded before any GPU training')
    pins = {str(plan_path): PLAN_SHA, str(old_state_path): OLD_STATE_SHA,
        str(parent_path): STATISTICS_PARENT_SHA, str(authorization_path): sha(authorization_path)}
    pins[str(ROOT/'push152_preflight.py')] = authorization['script_sha256']
    for name in ('audit', 'timing_summary'):
        evidence = child(ROOT, authorization[name])
        require(sha(evidence) == authorization[name+'_sha256'], 'Preflight evidence changed: '+name)
        pins[str(evidence)] = authorization[name+'_sha256']
    device, precision = setup('cuda', 'bf16', 4)
    old_state = torch.load(old_state_path, map_location='cpu', weights_only=True)
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    require(parent['protocol'] == PROTOCOL and parent['implementation'] == 'push150_expand' and
        parent['train_tokens'] == ANCESTOR_TARGETS and
        all(parent['config'].get(key) == value for key, value in FIXED_VALIDATION.items()),
        'Statistics parent protocol/configuration/lineage changed')
    require(parent['config']['push150_ranks'] == dict(qkv=160, proj=0, gate=144, value=144, out=144),
        'Phase-A must use the unchanged current rank144 architecture')
    require(sum(key.startswith('ngram.') for key in parent['model']) == 21,
        'The declared twenty-one fixed statistical buffers are required')
    config = training_config(parent['config'], .2)
    model, implementation_sha = make_model('push150_expand', config, device)
    require(implementation_sha == plan['parent']['implementation_sha256'], 'Inference implementation changed')
    require(sum(parameter.numel() for parameter in model.parameters()) == old_state['log']['parameters'],
        'Full-state fork changed the learned-parameter count')
    averages = {label: copy.deepcopy(model).eval().requires_grad_(False) for label in EMA_DECAYS}
    evaluator, _ = make_model('push150_expand', parent['config'], device)
    evaluator.eval().requires_grad_(False)
    optimizers, muon_names = build_optimizers(model, 'muon', RATES['backbone'], RATES['pointer'], RATES['scalar'], RATES['muon'])
    require(muon_names == old_state['log']['muon_parameter_names'], 'Old optimizer parameter grouping differs')
    runtime = runtime_record(device)
    require(runtime == old_state['identity']['runtime'], 'Complete-state fork requires the original numerical runtime')
    for name, digest in old_state['identity']['source_sha256'].items():
        path = child(ROOT, name)
        require(path.suffix == '.py' and sha(path) == digest, 'Original state source changed: '+name)
        pins[str(path)] = digest
    data, data_hashes = load_allowed_data(old_state)
    folds = copy.deepcopy(old_state['identity']['oof_preparation']['folds'])
    legal_ranges = validate_fold_geometry(folds, len(data['train'][0]))
    tokens = data['train'][0].to(device)
    sampler = torch.Generator(device='cpu')
    sources = source_digests()
    recipe = dict(phase='A', phase_name=args.phase_name, steps=STEPS, eval_every=EVAL_EVERY,
        optimizer='muon', optimizer_state_kept=(args.phase_name == 'optimizer-keep'), schedule='constant',
        learning_rates=copy.deepcopy(RATES), batch_size=BATCH_SIZE, context=CONTEXT,
        training_precision='bf16', validation_precision='fp32', training_temperature=1.15,
        validation_temperature=1.3, validation_mkn_weight=.15, dropout=.2,
        gradient_clip=1., ema_decays=copy.deepcopy(EMA_DECAYS),
        new_ema999_initialization='Saved historical EMA.99; not a resumed historical EMA.999',
        objective='pure neural-plus-pointer target NLL',
        sampler='Inherited uniform fold-contained 257-token starts and one unused count coin per minibatch',
        validation_choices_per_completed_run=14, additional_calibration_choices=0,
        omitted_unused_oof_target_lookup=True, neural_views_per_sample=1,
        training_config=copy.deepcopy(config), validation_config=copy.deepcopy(parent['config']))
    identity = dict(plan_sha256=PLAN_SHA, phase_name=args.phase_name, recipe=recipe,
        fork_training_state_sha256=OLD_STATE_SHA, statistics_parent_sha256=STATISTICS_PARENT_SHA,
        implementation='push150_expand', source_sha256=sources, data_sha256=data_hashes,
        pinned_input_sha256=pins, legal_fold_geometry=[{key: row[key] for key in ('fold', 'start', 'end', 'valid_random_starts')}
            for row in folds], runtime=runtime)
    completed_step, offset_seconds, session = 0, 0., 0
    log = dict(status='running', identity=copy.deepcopy(identity), recipe=copy.deepcopy(recipe),
        plan_sha256=PLAN_SHA, parent_sha256=STATISTICS_PARENT_SHA, parent_train_targets=ANCESTOR_TARGETS,
        fork_training_state_sha256=OLD_STATE_SHA, original_source_step=OLD_STEP,
        training_implementation='push150_expand', inference_implementation='push150_expand',
        inference_architecture_unchanged_from_parent=True, training_config=config, validation_config=parent['config'],
        steps=STEPS, eval_every=EVAL_EVERY, batch_size=BATCH_SIZE, processed_view_targets_per_step=TARGETS_PER_STEP,
        sampled_label_targets_per_step=TARGETS_PER_STEP, dropout_views_per_sample=1,
        parameters=sum(parameter.numel() for parameter in model.parameters()), optimizer='muon',
        muon_parameter_names=muon_names, source_sha256=sources, data_sha256=data_hashes,
        implementation_sha256=implementation_sha, no_statistical_tables_in_training=True,
        original_ancestry=copy.deepcopy(old_state['log'].get('ancestry', {})),
        same_ancestor_not_summed=True, source_splits=['train', 'validation'], test_used=False,
        prior_test_exposure_disclosed=True, history=[], validation_history=[], snapshots=[],
        resume_history=[], completed_steps=0, discarded_completed_steps=0, discarded_validation_passes=0,
        count_block_coins=empty_coin_counts(), discarded_count_block_coins=empty_coin_counts(),
        cached_target_probability_lookups=0, new_statistics_scans=0, seconds=0.,
        compute_difference_disclosure='Old pure-control OOF probabilities had no mathematical effect; their lookup is omitted, coin RNG consumption retained.')
    if args.resume:
        saved = torch.load(state_path, map_location='cpu', weights_only=True)
        completed_step, log = restore_state(saved, model, averages, optimizers, sampler, identity, cuda=True)
        disk = json.loads((args.run_dir/'metrics.json').read_text(encoding='utf-8'))
        require(disk['identity'] == identity, 'On-disk metrics identity differs from restored state')
        require(not (completed_step == STEPS and disk['status'] == 'completed'), 'Completed run cannot retrain or rescore')
        extra = max(0, disk.get('completed_steps', completed_step)-completed_step)
        discarded_coins = retain_discarded_coins(log, disk)
        log['discarded_completed_steps'] = max(log.get('discarded_completed_steps', 0), disk.get('discarded_completed_steps', 0))+extra
        log['discarded_validation_passes'] = max(log.get('discarded_validation_passes', 0),
            disk.get('discarded_validation_passes', 0))+max(0, len(disk['validation_history'])-len(log['validation_history']))
        session = max(len(log['resume_history']), len(disk.get('resume_history', [])))+1
        write_new(args.run_dir/f'metrics-before-resume-{session:03d}.json', disk)
        log['resume_history'] = copy.deepcopy(disk.get('resume_history', []))
        log['resume_history'].append(dict(session=session, resumed_from_step=completed_step,
            discarded_completed_steps=extra, discarded_count_block_coins=discarded_coins,
            previous_status=disk['status'], unknown_partial_update_work_not_estimated=True))
        offset_seconds = max(float(log.get('seconds', 0)), float(disk.get('seconds', 0)))
        if log.get('best'):
            best_path = args.run_dir/log['best']['checkpoint_file']
            require(sha(best_path) == log['best']['checkpoint_sha256'], 'Restored best snapshot changed')
            atomic_torch_save(torch.load(best_path, map_location='cpu', weights_only=True), args.run_dir/'checkpoint.pt')
            log['best']['scoring_checkpoint_sha256'] = sha(args.run_dir/'checkpoint.pt')
        log['status'] = 'running'
    else:
        layout = fork_from_old_state(old_state, model, averages, optimizers, sampler, args.phase_name, cuda=True)
        log['optimizer_parameter_layout'] = layout
        write_new(args.run_dir/'identity.json', identity)
        write_new(args.run_dir/'fork.json', dict(plan_sha256=PLAN_SHA, phase_name=args.phase_name,
            old_state_sha256=OLD_STATE_SHA, statistics_parent_sha256=STATISTICS_PARENT_SHA,
            old_step=OLD_STEP, new_step=0, inherited_targets=ANCESTOR_TARGETS, new_training_targets=0,
            optimizer_state_kept=(args.phase_name == 'optimizer-keep'), raw_from_training_state=True,
            both_emas_initialized_from_old_ema99=True, all_rng_restored_after_construction=True,
            old_identity_not_modified_or_impersonated=True, optimizer_parameter_layout=layout, test_used=False))
    del old_state
    started = time.perf_counter()

    def refresh_log():
        draws = {key: log['count_block_coins'][key]+log['discarded_count_block_coins'][key]
            for key in log['count_block_coins']}
        log.update(completed_steps=completed_step, count_block_coins_including_discarded=draws,
            incremental_search_targets=(completed_step+log['discarded_completed_steps'])*TARGETS_PER_STEP,
            incremental_sampled_label_targets=(completed_step+log['discarded_completed_steps'])*TARGETS_PER_STEP,
            lineage_new_targets=completed_step*TARGETS_PER_STEP,
            omitted_unused_oof_target_lookups=draws['drawn']*TARGETS_PER_STEP,
            validation_choices_in_retained_history=len(log['validation_history']),
            validation_attempts_recorded=len(list(args.run_dir.glob('validation-started-*.json'))),
            validation_completed_records=len(list(args.run_dir.glob('session*-validation.json'))),
            seconds=offset_seconds+time.perf_counter()-started,
            peak_cuda_allocated_bytes=max(log.get('peak_cuda_allocated_bytes', 0), torch.cuda.max_memory_allocated()))

    def check_sources():
        require(source_digests() == sources, 'New training source changed during the trajectory')
        for path, digest in pins.items():
            require(sha(Path(path)) == digest, 'Pinned source/input changed during training: '+path)
        for name, digest in data_hashes.items():
            require(sha(ROOT/name) == digest, 'Supplied train/validation identity changed')

    def save_state():
        check_sources()
        refresh_log()
        atomic_torch_save(pack_state(model, averages, optimizers, sampler, identity, completed_step, log, cuda=True), state_path)
        atomic_json_save(log, args.run_dir/'metrics.json')

    def validate(step):
        check_sources()
        for label, candidate in validation_choices(step, model, averages):
            basename = f'session{session:03d}-step{step:08d}-{label}'
            checkpoint = make_snapshot(candidate, parent, identity, recipe, step, label)
            snapshot_path = args.run_dir/(basename+'.pt')
            save_new(checkpoint, snapshot_path)
            cp_sha = sha(snapshot_path)
            write_new(args.run_dir/('validation-started-'+basename+'.json'), dict(step=step, weights=label,
                checkpoint=snapshot_path.name, checkpoint_sha256=cp_sha, session=session, test_used=False,
                plan_sha256=PLAN_SHA, started_at_utc=datetime.now(timezone.utc).isoformat()))
            evaluator.load_state_dict(checkpoint['model'], strict=True)
            measured = score(evaluator, *data['validation'], device, 'fp32')
            losses = measured.pop('window_nll_nats')
            require(math.isfinite(measured['bpb']) and measured['targets'] == 376599 and
                measured['utf8_bytes'] == 1148007, 'Incomplete or nonfinite full validation score')
            loss_path = args.run_dir/(basename+'.validation.window-nll.npy')
            with loss_path.open('xb') as stream:
                np.save(stream, np.asarray(losses, dtype=np.float64), allow_pickle=False)
            row = dict(step=step, weights=label, session=session, checkpoint_file=snapshot_path.name,
                checkpoint_sha256=cp_sha, selected_lineage_targets=checkpoint['train_tokens'],
                scalars={name: float(value.detach()) for name, value in candidate.named_parameters() if value.ndim == 0},
                protocol=PROTOCOL, split='validation', precision='fp32', device=str(device), test_used=False,
                implementation_sha256=implementation_sha, evaluator_sha256=sha(ROOT/'evaluate.py'),
                tokenizer_sha256=data_hashes['data/tokenizer.json'], window_nll_sha256=sha(loss_path), **measured)
            write_new(args.run_dir/(basename+'-validation.json'), row)
            log['validation_history'].append(row)
            log['snapshots'].append(dict(step=step, weights=label, path=snapshot_path.name, sha256=cp_sha))
            if measured['bpb'] < log.get('best', {}).get('bpb', float('inf')):
                atomic_torch_save(checkpoint, args.run_dir/'checkpoint.pt')
                log['best'] = dict(row, scoring_checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'))
            refresh_log()
            atomic_json_save(log, args.run_dir/'metrics.json')
            print(json.dumps(dict(validation=row)), flush=True)

    args._entered_training_state_machine = True
    try:
        if not args.resume:
            validate(0)
            save_state()
        for step in range(completed_step+1, STEPS+1):
            model.train()
            starts_cpu, keep_count = sample_starts_and_coin(sampler, legal_ranges)
            add_coin(log, keep_count)
            starts = starts_cpu.to(device)
            batch = tokens[starts[:, None]+torch.arange(CONTEXT+1, device=device)]
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            with autocast(device, precision):
                loss = control_loss(model, batch[:, :-1], batch[:, 1:])
            require(bool(torch.isfinite(loss)), 'Nonfinite pure neural-plus-pointer loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            for optimizer in optimizers:
                optimizer.step()
            model.project_cache_parameters()
            update_averages(model, averages)
            completed_step = step
            if step % 400 == 0:
                torch.cuda.synchronize()
                refresh_log()
                row = dict(step=step, loss=loss.item(), gradient_norm=float(norm), session=session,
                    seconds=log['seconds'], count_block_kept=keep_count, schedule_factor=1.)
                log['history'].append(row)
                atomic_json_save(log, args.run_dir/'metrics.json')
                print(json.dumps(row), flush=True)
            if step % EVAL_EVERY == 0:
                validate(step)
                save_state()
        require(len(log['validation_history']) == 14, 'Retained validation menu must have exactly fourteen choices')
        log['status'] = 'completed'
        save_state()
    except BaseException as error:
        refresh_log()
        log.update(status='failed', failure_type=type(error).__name__, failure_message=str(error),
            attempted_step=locals().get('step', 0),
            failed_attempt_cost_note='Completed updates counted; partially executed optimizer update/score may be unquantified. All exclusive records remain. No automatic retry.')
        atomic_json_save(log, args.run_dir/'metrics.json')
        write_new(args.run_dir/f'failure-session{session:03d}.json', dict(status='failed',
            failure_type=type(error).__name__, failure_message=str(error), completed_steps=completed_step,
            attempted_step=locals().get('step', 0), plan_sha256=PLAN_SHA, test_used=False,
            metrics_sha256=sha(args.run_dir/'metrics.json'), recorded_at_utc=datetime.now(timezone.utc).isoformat()))
        raise
    print(json.dumps(dict(best=log['best'], seconds=log['seconds'], train_state=str(state_path))), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase_name', choices=RUN_NAMES)
    parser.add_argument('--plan', type=Path, default=ROOT/'runs/push152_plan.json')
    parser.add_argument('--resume', type=Path, help='Existing full state in this new run directory only')
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run(args)
    except BaseException as error:
        if getattr(args, '_run_directory_confirmed', False) and not getattr(args, '_entered_training_state_machine', False):
            index = 1
            while (args.run_dir/f'preflight-failure-{index:03d}.json').exists():
                index += 1
            write_new(args.run_dir/f'preflight-failure-{index:03d}.json', dict(status='failed', stage='preflight',
                phase_name=args.phase_name, plan_sha256=PLAN_SHA, failure_type=type(error).__name__,
                failure_message=str(error), new_completed_neural_updates=0, test_used=False,
                note='No update or validation call in this preflight; existing state and partial files preserved.'))
        raise


if __name__ == '__main__':
    main()
