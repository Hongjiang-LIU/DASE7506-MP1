"""Phase-D matched control / OOF joint training; inference stays unchanged.

Only a complete training state may resume optimizer momentum. A scoring parent
starts new optimizers and EMA explicitly. No test corpus is loaded or hashed.
"""
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from common import ROOT, PROTOCOL, autocast, make_model, setup, sha
from evaluate import score
from push148_train import build_optimizers
from push150_crossfit import contiguous_folds
from push150_oof_prepare import valid_start_ranges, gather_window_probabilities


BATCH_SIZE, CONTEXT, TARGETS_PER_STEP = 32, 256, 8192
STATE_FORMAT = 1

PLAN_SHA256 = '65800e86fac769df299f3794914a287189274adbb3442e56fa9e7e24367ec25a'
TRAINING_COUNT_WEIGHT = .15


def validate_prepared_metadata(metadata, parent, parent_sha, selection, selection_sha, train_record):
    """Metadata-only compatibility checks; no model/scoring or text I/O."""
    if (metadata.get('status') != 'completed' or metadata.get('test_used') is not False
            or metadata.get('plan_sha256') != PLAN_SHA256):
        raise ValueError('OOF preparation is incomplete, used test, or used a different declared plan')
    if (selection.get('status') != 'phase_a_choice_complete' or selection.get('test_used') is not False
            or selection.get('evidence_errors') != []):
        raise ValueError('Phase-A selection is incomplete or has evidence errors')
    winner = selection['winner']
    if (winner['checkpoint_sha256'] != parent_sha or winner['selected_lineage_targets'] != parent['train_tokens']
            or winner['implementation'] != parent['implementation']):
        raise ValueError('Parent is not the exact declared Phase-A winner')
    if (metadata.get('parent_sha256') != parent_sha
            or metadata.get('parent_lineage_targets') != parent['train_tokens']
            or metadata.get('phase_a_selection_sha256') != selection_sha
            or metadata.get('train') != train_record or train_record.get('split') != 'train'):
        raise ValueError('OOF cache parent/lineage/selection/train provenance mismatch')
    record = metadata['array']
    if (record.get('path') != 'train_oof_probabilities.npy' or record.get('dtype') != 'float32'
            or record.get('shape') != [train_record['tokens'], 4]
            or record.get('maximum_orders') != [2,3,4,5]):
        raise ValueError('Unexpected OOF probability array schema')
    expected = contiguous_folds(train_record['tokens'])
    folds = metadata['folds']
    if len(folds) != len(expected):
        raise ValueError('OOF cache does not have the ten declared folds')
    for index, (fold, (start,end)) in enumerate(zip(folds,expected)):
        if (fold.get('fold') != index or fold.get('start') != start or fold.get('end') != end
                or fold.get('valid_random_starts') != end-start-CONTEXT):
            raise ValueError('OOF fold intervals or valid-start counts differ from the plan')
    if metadata.get('parity',{}).get('passes') is not True:
        raise ValueError('OOF probability parity has not explicitly passed')
    if not isinstance(metadata.get('costs'),dict) or not metadata.get('source_sha256'):
        raise ValueError('OOF preparation costs/sources missing')
    return folds


def load_prepared_cache(prepared_dir, parent, parent_sha, selection_path, train_record, plan_path):
    """Verify immutable training-only cache once, then memory-map probabilities.

    The saved inference checkpoint contains neither this array nor a loader for
    it. All referenced files are pinned into the training/resume identity.
    """
    prepared_dir = Path(prepared_dir).resolve()
    metadata_path = prepared_dir/'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    selection_path, plan_path = Path(selection_path).resolve(), Path(plan_path).resolve()
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if sha(plan_path) != PLAN_SHA256:
        raise ValueError('Changed predeclared Phase-D plan')
    folds = validate_prepared_metadata(metadata,parent,parent_sha,selection,sha(selection_path),train_record)
    identity_path = prepared_dir/'identity.json'
    if sha(identity_path) != metadata['identity_sha256']:
        raise ValueError('Preparation identity changed')
    record = metadata['array']
    array_path = prepared_dir/record['path']
    if (not array_path.resolve().is_relative_to(prepared_dir)
            or array_path.stat().st_size != record['bytes'] or sha(array_path) != record['sha256']):
        raise ValueError('OOF array path, size or SHA changed')
    for name, digest in metadata['source_sha256'].items():
        relative = Path(name)
        source = (ROOT/relative).resolve()
        if (relative.is_absolute() or relative.drive or '..' in relative.parts
                or not source.is_relative_to(ROOT.resolve()) or source.suffix != '.py'
                or sha(source) != digest):
            raise ValueError('OOF preparation source changed: '+name)
    array = np.load(array_path,mmap_mode='r',allow_pickle=False)
    if array.dtype != np.float32 or list(array.shape) != record['shape']:
        raise ValueError('OOF array shape/dtype disagrees with metadata')
    ranges = torch.from_numpy(valid_start_ranges(folds,context=CONTEXT).copy())
    pins = {str(path):sha(path) for path in (metadata_path,identity_path,array_path,selection_path,plan_path)}
    # Some preparation roots (B gate and the Phase-A report) are dynamic and
    # need not appear in this trainer's own import graph. Freeze those too.
    pins.update({str((ROOT/name).resolve()):digest for name,digest in metadata['source_sha256'].items()})
    provenance = dict(metadata_sha256=pins[str(metadata_path)],identity_sha256=metadata['identity_sha256'],
        array_sha256=record['sha256'],array_shape=record['shape'],array_dtype=record['dtype'],
        phase_a_selection_sha256=metadata['phase_a_selection_sha256'],plan_sha256=PLAN_SHA256,
        gate_identity_sha256=metadata['gate_identity_sha256'],gate_build_sha256=metadata['gate_build_sha256'],
        folds=copy.deepcopy(folds),shared_preparation_costs=copy.deepcopy(metadata['costs']),
        only_count_component_is_out_of_fold=True,neural_parent_previously_saw_all_training_text=True,
        reused_B_count_scans_counted_again_in_D=0,training_only_array_excluded_from_inference_assets=True)
    return array,ranges,provenance,pins


def sample_starts_and_coin(sampler, ranges, batch_size=BATCH_SIZE):
    """Uniform over legal starts, then exactly one shared minibatch count coin.

    CPU sampler state is already saved by the complete resume machinery. This
    function never consumes the global CPU/CUDA RNG used for neural dropout.
    """
    if (ranges.device.type != 'cpu' or ranges.dtype != torch.int64 or ranges.ndim != 2
            or ranges.shape[1] != 2 or len(ranges) == 0 or batch_size <= 0):
        raise ValueError('Expected nonempty CPU int64 half-open legal-start ranges')
    counts = ranges[:,1]-ranges[:,0]
    if torch.any(counts <= 0):
        raise ValueError('Every fold needs at least one legal full input/target window')
    cumulative = counts.cumsum(0)
    draws = torch.randint(int(cumulative[-1]),(batch_size,),generator=sampler)
    folds = torch.searchsorted(cumulative,draws,right=True)
    preceding = torch.cat([counts.new_zeros(1),cumulative[:-1]])
    starts = ranges[folds,0]+draws-preceding[folds]
    keep_count = bool(torch.randint(2,(),generator=sampler).item())
    return starts,keep_count


def mixture_target_loss(neural_targets, count_targets, objective, keep_count):
    """Normalized convex-mixture NLL; cached count probabilities are constants."""
    if objective not in ('control','joint') or not isinstance(keep_count,bool):
        raise ValueError('Unknown objective or count-dropout coin')
    if neural_targets.ndim != 2 or neural_targets.shape != count_targets.shape:
        raise ValueError('Expected matching batch-by-context target probabilities')
    probabilities = neural_targets.float()
    if objective == 'joint' and keep_count:
        counts = count_targets.detach().to(device=probabilities.device,dtype=torch.float32)
        probabilities = (1-TRAINING_COUNT_WEIGHT)*probabilities+TRAINING_COUNT_WEIGHT*counts
    # For control or a dropped count block this is the original normalized
    # neural+pointer probability, not .85 times that probability.
    return -probabilities.clamp_min(1e-30).log().mean()


def training_loss(model, ids, targets, count_targets, objective, keep_count):
    if hasattr(model,'ngram'):
        raise ValueError('OOF training model must not contain inference count tables')
    probabilities = model.neural_and_cache(ids)
    neural_targets = probabilities.gather(-1,targets.unsqueeze(-1)).squeeze(-1)
    return mixture_target_loss(neural_targets,count_targets,objective,keep_count)


def empty_coin_counts():
    return dict(drawn=0,kept=0,dropped=0)


def add_coin(log, keep_count):
    counts=log['count_block_coins']
    counts['drawn']+=1
    counts['kept' if keep_count else 'dropped']+=1


def retain_discarded_coins(log, disk):
    """Retain replayed/failed-attempt coins just like discarded SGD updates."""
    before=log['count_block_coins']
    extra={key:max(0,disk['count_block_coins'][key]-before[key]) for key in before}
    log['discarded_count_block_coins']={
        key:max(log.get('discarded_count_block_coins',{}).get(key,0),
                disk.get('discarded_count_block_coins',{}).get(key,0))+extra[key]
        for key in before}
    return extra



def training_config(parent_config, dropout):
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError('dropout must be finite and in [0,1)')
    result = {k:copy.deepcopy(v) for k,v in parent_config.items()
              if not k.startswith(('ngram_', 'mkn_')) and k != 'discount_scale'}
    result.update(temperature=1.15, dropout=dropout)
    return result


def schedule_factor(step, steps, schedule, warmup=200, decay_steps=4000, floor=.1):
    """Updates are numbered 1..steps; both schedules end exactly at floor.

    Cosine starts decaying at the first update (as in the old driver). WSD
    holds its peak until steps-decay_steps, then applies its cosine cooldown.
    Warmup is a multiplicative ramp shared by both routes.
    """
    if not 1 <= step <= steps or steps < 2 or not 0 <= warmup < steps:
        raise ValueError('Invalid step/total/warmup')
    if schedule not in ('cosine','wsd') or not 0 < floor <= 1:
        raise ValueError('Invalid schedule/floor')
    if not 1 <= decay_steps <= steps or (schedule == 'wsd' and warmup > steps-decay_steps):
        raise ValueError('WSD needs a non-overlapping warmup and cooldown')
    progress = ((step-1)/(steps-1) if schedule == 'cosine'
                else max(0., (step-(steps-decay_steps))/decay_steps))
    warm = min(1., step/warmup) if warmup else 1.
    return warm * (floor + (1-floor)*.5*(1+math.cos(math.pi*progress)))


def cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key:cpu_clone(item) for key,item in value.items()}
    if isinstance(value, list):
        return [cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def model_state(model):
    state = cpu_clone(model.state_dict())
    if model.head.weight is not model.token.weight:
        raise ValueError('This recipe requires the existing tied embedding/head')
    state['head.weight'] = state['token.weight']
    return state


def snapshot(model, parent, eval_config, parent_sha, recipe, step, weights):
    state = model_state(model)
    if any(key.startswith('ngram.') for key in state):
        raise ValueError('Statistics must not exist on the training model')
    state.update({key:cpu_clone(value) for key,value in parent['model'].items()
                  if key.startswith('ngram.')})
    return dict(protocol=PROTOCOL, implementation=parent['implementation'],
        config=copy.deepcopy(eval_config), model=state, seed=recipe['seed'],
        parent_sha256=parent_sha, train_tokens=parent['train_tokens']+step*TARGETS_PER_STEP,
        selected_step=step, weights=weights, training_config=copy.deepcopy(model.config),
        training_recipe=copy.deepcopy(recipe), ancestry=copy.deepcopy(parent.get('ancestry',{})),
        selection='full validation only; unchanged frozen train-derived statistics',
        inference_architecture_unchanged_from_parent=True)


def capture_rng(sampler, cuda=False):
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(),
        numpy=dict(generator=numpy_state[0], keys=numpy_state[1].tolist(),
                   position=numpy_state[2], has_gauss=numpy_state[3], cached_gaussian=numpy_state[4]),
        torch_cpu=torch.get_rng_state().clone(), sampler=sampler.get_state().clone(),
        torch_cuda=[value.cpu().clone() for value in torch.cuda.get_rng_state_all()] if cuda else [])


def restore_rng(state, sampler, cuda=False):
    if bool(state['torch_cuda']) != bool(cuda):
        raise ValueError('RNG device kind differs from saved state')
    random.setstate(state['python'])
    ns=state['numpy']
    np.random.set_state((ns['generator'],np.asarray(ns['keys'],dtype=np.uint32),
                         ns['position'],ns['has_gauss'],ns['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'].cpu())
    sampler.set_state(state['sampler'].cpu())
    if cuda:
        if len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from saved state')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])


def pack_train_state(model, averaged, optimizers, sampler, identity, step, log, cuda=False):
    return dict(format_version=STATE_FORMAT, identity=copy.deepcopy(identity), step=step,
        raw=model_state(model), ema=model_state(averaged),
        optimizers=[cpu_clone(optimizer.state_dict()) for optimizer in optimizers],
        model_training=model.training, rng=capture_rng(sampler,cuda), log=copy.deepcopy(log))


def restore_train_state(state, model, averaged, optimizers, sampler, identity, cuda=False):
    # Fail before mutating models or random generators when provenance changes.
    if state.get('format_version') != STATE_FORMAT or state.get('identity') != identity:
        raise ValueError('Resume identity mismatch: recipe, parent, sources, data or runtime changed')
    step=state['step']
    if not isinstance(step,int) or not 0 <= step <= identity['recipe']['steps']:
        raise ValueError('Invalid completed update count in training state')
    if len(state['optimizers']) != len(optimizers):
        raise ValueError('Optimizer count changed')
    if state['log']['completed_steps'] != step:
        raise ValueError('State and log completed counts disagree')
    model.load_state_dict(state['raw'],strict=True)
    averaged.load_state_dict(state['ema'],strict=True)
    for optimizer,saved in zip(optimizers,state['optimizers']):
        optimizer.load_state_dict(saved)
    model.train(state['model_training'])
    averaged.eval().requires_grad_(False)
    restore_rng(state['rng'],sampler,cuda)
    return step,copy.deepcopy(state['log'])


def source_digests(implementation, root=ROOT):
    """Pin recursively imported local Python sources, never corpus text.

    The configured implementation is dynamic, so it is an explicit root.
    Standard/external package imports are represented by runtime versions.
    """
    pending=[root/'push150_oof_train.py',root/(implementation+'.py'),root/'push148_muon.py',
             root/'push150_oof_prepare.py']
    found={}
    while pending:
        path=pending.pop().resolve()
        name=path.relative_to(root.resolve()).as_posix()
        if name in found:
            continue
        found[name]=sha(path)
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            imports=([node.module] if isinstance(node,ast.ImportFrom) and node.module
                     else [entry.name for entry in node.names] if isinstance(node,ast.Import) else [])
            for module in imports:
                local=root/(module.replace('.','/')+'.py')
                if local.is_file():
                    pending.append(local)
    return dict(sorted(found.items()))


def atomic_torch_save(payload,path):
    temporary=path.with_name(path.name+'.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def atomic_json_save(payload,path):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',required=True,type=Path)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--objective',required=True,choices=['control','joint'])
    parser.add_argument('--prepared-dir',required=True,type=Path)
    parser.add_argument('--phase-a-selection',required=True,type=Path)
    parser.add_argument('--plan',default=ROOT/'runs/push150_phase_d_oof_plan.json',type=Path)
    parser.add_argument('--resume',type=Path,help='Existing train_state.pt in the SAME run directory')
    parser.add_argument('--steps',default=12000,type=int)
    parser.add_argument('--eval-every',default=2000,type=int)
    parser.add_argument('--seed',default=211,type=int)
    parser.add_argument('--optimizer',choices=['adamw','muon'],default='muon')
    parser.add_argument('--schedule',choices=['cosine','wsd'],default='cosine')
    parser.add_argument('--warmup',default=200,type=int)
    parser.add_argument('--decay-steps',default=2000,type=int)
    parser.add_argument('--backbone-lr',default=.0000075,type=float)
    parser.add_argument('--head-lr',default=.000075,type=float)
    parser.add_argument('--scalar-lr',default=.00025,type=float)
    parser.add_argument('--muon-lr',default=.0005,type=float)
    parser.add_argument('--dropout',default=.20,type=float)
    return parser.parse_args()


def run(args):
    declared=dict(steps=12000,eval_every=2000,seed=211,optimizer='muon',schedule='cosine',
        warmup=200,decay_steps=2000,backbone_lr=.0000075,head_lr=.000075,
        scalar_lr=.00025,muon_lr=.0005,dropout=.20)
    if any(getattr(args,key) != value for key,value in declared.items()):
        raise ValueError('Requested recipe differs from the predeclared Phase-D trial')
    schedule_factor(1,args.steps,args.schedule,args.warmup,args.decay_steps)
    training_config({},args.dropout)
    if args.eval_every <= 0 or any(not math.isfinite(rate) or rate <= 0 for rate in
                                 (args.backbone_lr,args.head_lr,args.scalar_lr,args.muon_lr)):
        raise ValueError('Evaluation interval and learning rates must be positive')
    state_path=args.run_dir/'train_state.pt'
    if args.resume:
        if args.resume.resolve() != state_path.resolve() or not state_path.is_file():
            raise ValueError('Resume requires this run directory\'s existing train_state.pt')
    else:
        args.run_dir.mkdir(parents=True,exist_ok=False)
    args._run_directory_confirmed=True
    device,precision=setup('cuda','bf16',4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    parent=torch.load(args.parent,map_location='cpu',weights_only=True)
    if parent['protocol'] != PROTOCOL:
        raise ValueError('Parent protocol mismatch')
    parent_sha=sha(args.parent)
    implementation=parent['implementation']
    if implementation not in ('push147_signed_cache','push150_expand'):
        raise ValueError('Phase D must inherit the Phase-A model, not gate/suffix/fast branches')
    model,implementation_sha=make_model(implementation,training_config(parent['config'],args.dropout),device)
    model.load_state_dict({k:v for k,v in parent['model'].items() if not k.startswith('ngram.')},strict=True)
    if hasattr(model,'ngram') or any(k.startswith('ngram.') for k in model.state_dict()):
        raise ValueError('Training model contains statistical tables')
    averaged=copy.deepcopy(model).eval().requires_grad_(False)
    eval_config=copy.deepcopy(parent['config'])
    eval_config.update(temperature=1.25,mkn_weight=.15,dropout=args.dropout)
    evaluator,_=make_model(implementation,eval_config,device)
    evaluator.eval().requires_grad_(False)
    optimizers,muon_names=build_optimizers(model,args.optimizer,args.backbone_lr,args.head_lr,
                                        args.scalar_lr,args.muon_lr)
    sampler=torch.Generator().manual_seed(args.seed)
    tokenizer=Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    data={}
    data_hashes={name:sha(ROOT/name) for name in ('data/manifest.json','data/tokenizer.json')}
    if data_hashes['data/tokenizer.json'] != manifest['sha256']['tokenizer.json']:
        raise ValueError('Changed fixed tokenizer')
    for split in ('train','validation'):
        path=ROOT/'data'/f'wikitext_{split}.txt'
        digest=sha(path)
        if digest != manifest['sha256'][path.name]:
            raise ValueError(f'Changed allowed corpus split: {split}')
        raw=path.read_bytes()
        data_hashes['data/'+path.name]=digest
        data[split]=(torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids),len(raw))
    tokens=data['train'][0].to(device)
    train_record=dict(split='train',text_sha256=data_hashes['data/wikitext_train.txt'],
        utf8_bytes=data['train'][1],tokens=len(tokens),tokenizer_sha256=data_hashes['data/tokenizer.json'],
        manifest_sha256=data_hashes['data/manifest.json'])
    oof_array,legal_ranges,preparation,prepared_pins=load_prepared_cache(
        args.prepared_dir,parent,parent_sha,args.phase_a_selection,train_record,args.plan)
    if len(tokens) <= CONTEXT:
        raise ValueError('Training sequence is shorter than one input/target window')
    recipe=dict(steps=args.steps,eval_every=args.eval_every,seed=args.seed,optimizer=args.optimizer,
        schedule=args.schedule,warmup=args.warmup,decay_steps=args.decay_steps,lr_floor=.1,
        backbone_lr=args.backbone_lr,head_lr=args.head_lr,scalar_lr=args.scalar_lr,muon_lr=args.muon_lr,
        batch_size=BATCH_SIZE,context=CONTEXT,dropout_views_per_sample=1,dropout=args.dropout,
        training_temperature=1.15,validation_temperature=1.25,validation_mkn_weight=.15,
        ema_decay=.99,gradient_clip=1.,adamw_betas=[.9,.95],backbone_weight_decay=.1,
        pointer_weight_decay=.01,scalar_weight_decay=0.,muon_momentum=.95,muon_ns_steps=5,
        training_precision=precision,training_config=model.config,validation_config=eval_config,
        tail_snapshot_start_step=max(1,args.steps-args.decay_steps),
        objective=args.objective,training_count_weight=TRAINING_COUNT_WEIGHT,count_block_dropout=.5,
        count_block_dropout_granularity='one coin per complete minibatch; control consumes same coin',
        count_coin_rng='CPU sampler generator shared with legal-window start draws; no global dropout RNG use',
        sampler='uniform over all valid fold-contained 257-token starts',
        objective_description='pure neural+pointer NLL' if args.objective=='control' else
            'OOF convex-mixture NLL, with whole count block independently dropped per minibatch',
        declared_raw_ema_validation_choices=13,additional_calibration_choices=15)
    identity=dict(parent_sha256=parent_sha,implementation=implementation,recipe=copy.deepcopy(recipe),
        source_sha256=source_digests(implementation),data_sha256=data_hashes,
        oof_preparation=copy.deepcopy(preparation),prepared_file_sha256=copy.deepcopy(prepared_pins),
        runtime=dict(torch=str(torch.__version__),numpy=np.__version__,python=platform.python_version(),
                     cuda=torch.version.cuda,device=str(device),device_name=torch.cuda.get_device_name(device),
                     cuda_device_count=torch.cuda.device_count(),cpu_threads=torch.get_num_threads()))
    completed_step=0
    log=dict(status='running',parent_sha256=parent_sha,parent_train_targets=parent['train_tokens'],
        training_implementation=implementation,inference_implementation=implementation,
        inference_architecture_unchanged_from_parent=True,recipe=copy.deepcopy(recipe),
        training_config=model.config,validation_config=eval_config,seed=args.seed,steps=args.steps,
        eval_every=args.eval_every,batch_size=BATCH_SIZE,dropout_views_per_sample=1,
        processed_view_targets_per_step=TARGETS_PER_STEP,sampled_label_targets_per_step=TARGETS_PER_STEP,
        parameters=sum(p.numel() for p in model.parameters()),optimizer=args.optimizer,
        muon_parameter_names=muon_names,training_precision=precision,ema_decay=.99,
        no_statistical_tables_in_training=True,source_splits=['train','validation'],test_used=False,
        prior_test_exposure_disclosed=True,implementation_sha256=implementation_sha,
        source_sha256=identity['source_sha256'],data_sha256=data_hashes,identity=identity,
        optimizer_initialization='Fresh optimizers and EMA from scoring parent; old optimizer states unavailable',
        ancestry=copy.deepcopy(parent.get('ancestry',{})),history=[],validation_history=[],
        tail_snapshots=[],resume_history=[],discarded_completed_steps=0,completed_steps=0,
        count_block_coins=empty_coin_counts(),discarded_count_block_coins=empty_coin_counts(),
        oof_preparation=copy.deepcopy(preparation),objective=args.objective,
        seconds=0.,cost_note='Occurrence counts, not unique tokens; unrecorded interrupted attempts cannot be reconstructed')
    offset_seconds=0.
    session_index=0
    if args.resume:
        saved=torch.load(state_path,map_location='cpu',weights_only=True)
        completed_step,log=restore_train_state(saved,model,averaged,optimizers,sampler,identity,cuda=True)
        old_metrics_path=args.run_dir/'metrics.json'
        disk=json.loads(old_metrics_path.read_text()) if old_metrics_path.exists() else saved['log']
        if disk.get('identity') != identity:
            raise ValueError('On-disk metrics identity differs from resume state')
        if completed_step == args.steps and disk.get('status') == 'completed':
            raise ValueError('This run is already complete; no retraining or rescoring requested')
        extra=max(0,disk.get('completed_steps',completed_step)-completed_step)
        discarded_coins=retain_discarded_coins(log,disk)
        log['discarded_completed_steps']=max(log.get('discarded_completed_steps',0),
                                             disk.get('discarded_completed_steps',0))+extra
        session_index=max(len(log['resume_history']),len(disk.get('resume_history',[])))+1
        archive=args.run_dir/f'metrics-before-resume-{session_index:03d}.json'
        if archive.exists():
            raise FileExistsError('Refuse to overwrite resume evidence')
        atomic_json_save(disk,archive)
        if len(disk.get('resume_history',[])) > len(log['resume_history']):
            log['resume_history']=copy.deepcopy(disk['resume_history'])
        log['resume_history'].append(dict(session=session_index,resumed_from_step=completed_step,
            discarded_completed_steps=extra,discarded_count_block_coins=discarded_coins,
            previous_status=disk.get('status'),evidence=archive.name,
            note='Replayed updates count as extra search work; unknown partial attempt work remains unquantified'))
        offset_seconds=max(float(disk.get('seconds',0)),float(log.get('seconds',0)))
        best_path=args.run_dir/log['best']['checkpoint_file']
        if sha(best_path) != log['best']['checkpoint_sha256']:
            raise ValueError('Saved best scoring checkpoint changed')
        # Recover the best belonging to the restored state, not a checkpoint
        # written by an interrupted, later validation transaction.
        atomic_torch_save(torch.load(best_path,map_location='cpu',weights_only=True),args.run_dir/'checkpoint.pt')
        log['best']['scoring_checkpoint_sha256']=sha(args.run_dir/'checkpoint.pt')
        log['status']='running'
    started=time.perf_counter()

    def refresh_log():
        coin_total={key:log['count_block_coins'][key]+log['discarded_count_block_coins'][key]
                    for key in log['count_block_coins']}
        log.update(count_block_coins_including_discarded=coin_total,
            cached_target_probability_lookups=coin_total['drawn']*TARGETS_PER_STEP,
            joint_count_kept_target_occurrences=coin_total['kept']*TARGETS_PER_STEP if args.objective=='joint' else 0,
            validation_choices_in_retained_history=len(log['validation_history']),
            completed_steps=completed_step,
            incremental_search_targets=(completed_step+log['discarded_completed_steps'])*TARGETS_PER_STEP,
            incremental_sampled_label_targets=(completed_step+log['discarded_completed_steps'])*TARGETS_PER_STEP,
            lineage_new_targets=completed_step*TARGETS_PER_STEP,seconds=offset_seconds+time.perf_counter()-started,
            peak_cuda_allocated_bytes=max(log.get('peak_cuda_allocated_bytes',0),torch.cuda.max_memory_allocated()))

    def check_sources():
        if source_digests(implementation) != identity['source_sha256'] or sha(args.parent) != parent_sha:
            raise ValueError('Pinned source or parent changed during training')
        if any(sha(Path(path)) != digest for path,digest in prepared_pins.items()):
            raise ValueError('Pinned OOF preparation/cache/selection/plan changed during training')

    def save_state():
        check_sources()
        refresh_log()
        packed=pack_train_state(model,averaged,optimizers,sampler,identity,completed_step,log,cuda=True)
        atomic_torch_save(packed,state_path)
        atomic_json_save(log,args.run_dir/'metrics.json')

    def validate(step):
        check_sources()
        choices=[('raw',model)] if step == 0 else [('raw',model),('ema',averaged)]
        for label,candidate in choices:
            checkpoint=snapshot(candidate,parent,eval_config,parent_sha,recipe,step,label)
            evaluator.load_state_dict(checkpoint['model'],strict=True)
            measured=score(evaluator,*data['validation'],device,'fp32')
            measured.pop('window_nll_nats')
            if not math.isfinite(measured['bpb']):
                raise FloatingPointError('Nonfinite validation BPB')
            row=dict(step=step,weights=label,session=session_index,
                scalars={name:float(value.detach()) for name,value in candidate.named_parameters() if value.ndim==0},**measured)
            log['validation_history'].append(row)
            print(json.dumps(dict(validation=row)),flush=True)
            if step >= recipe['tail_snapshot_start_step']:
                tail_name=f'tail-session{session_index:03d}-step{step:08d}-{label}.pt'
                tail_path=args.run_dir/tail_name
                if tail_path.exists():
                    raise FileExistsError('Refuse to overwrite a saved tail snapshot')
                atomic_torch_save(checkpoint,tail_path)
                log['tail_snapshots'].append(dict(step=step,weights=label,path=tail_name,sha256=sha(tail_path)))
            if measured['bpb'] < log.get('best',{}).get('bpb',float('inf')):
                name=f'best-session{session_index:03d}-step{step:08d}-{label}.pt'
                path=args.run_dir/name
                if path.exists():
                    raise FileExistsError('Refuse to overwrite a saved best snapshot')
                atomic_torch_save(checkpoint,path)
                atomic_torch_save(checkpoint,args.run_dir/'checkpoint.pt')
                log['best']=dict(row,checkpoint_file=name,checkpoint_sha256=sha(path),
                    scoring_checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'),
                    selected_lineage_targets=checkpoint['train_tokens'])

    args._entered_training_state_machine=True
    try:
        if not args.resume:
            validate(0)
            save_state()
        for step in range(completed_step+1,args.steps+1):
            model.train()
            starts_cpu,keep_count=sample_starts_and_coin(sampler,legal_ranges)
            add_coin(log,keep_count)
            count_targets=torch.from_numpy(gather_window_probabilities(
                oof_array,starts_cpu.numpy(),context=CONTEXT)).to(device=device,dtype=torch.float32)
            starts=starts_cpu.to(device)
            batch=tokens[starts[:,None]+torch.arange(CONTEXT+1,device=device)]
            factor=schedule_factor(step,args.steps,args.schedule,args.warmup,args.decay_steps)
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group['lr']=group['base_lr']*factor
                optimizer.zero_grad(set_to_none=True)
            with autocast(device,precision):
                loss=training_loss(model,batch[:,:-1],batch[:,1:],count_targets,args.objective,keep_count)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at attempted step {step}')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            for optimizer in optimizers:
                optimizer.step()
            project=getattr(model,'project_cache_parameters',None)
            if project is not None:
                project()
            with torch.no_grad():
                for target,current in zip(averaged.parameters(),model.parameters()):
                    target.lerp_(current,.01)
            completed_step=step
            if step%400 == 0:
                torch.cuda.synchronize()
                refresh_log()
                row=dict(step=step,loss=loss.item(),gradient_norm=float(norm),schedule_factor=factor,
                         count_block_kept=keep_count,count_block_coins=copy.deepcopy(log['count_block_coins']),
                         seconds=log['seconds'],session=session_index)
                log['history'].append(row)
                print(json.dumps(row),flush=True)
                atomic_json_save(log,args.run_dir/'metrics.json')
            if step%args.eval_every == 0 or step == args.steps:
                validate(step)
                save_state()
        log['status']='completed'
        save_state()
    except BaseException as error:
        refresh_log()
        log.update(status='failed',failure_type=type(error).__name__,failure_message=str(error),
            attempted_step=locals().get('step',0),
            failed_attempt_cost_note='Completed updates are counted; failed partial work is unquantified. Last complete train_state remains resumable.')
        atomic_json_save(log,args.run_dir/'metrics.json')
        raise
    print(json.dumps(dict(best=log['best'],seconds=log['seconds'],train_state=str(state_path))),flush=True)


def main():
    args=parse_args()
    try:
        run(args)
    except BaseException as error:
        # Setup/cache/source failures before the inner training state machine
        # also need evidence. Do not write into an unrelated preexisting run
        # directory, or replace the richer failure log from the inner handler.
        if (getattr(args,'_run_directory_confirmed',False)
                and not getattr(args,'_entered_training_state_machine',False)):
            index=1
            while (args.run_dir/f'preflight-failure-{index:03d}.json').exists():
                index+=1
            report=dict(status='failed',stage='preflight',failure_type=type(error).__name__,
                failure_message=str(error),objective=args.objective,parent_path=str(args.parent),
                new_completed_neural_updates=0,test_used=False,partial_files_preserved=True,
                note='No optimizer updates or validation calls in this preflight attempt; prior resume costs remain in existing evidence.')
            with (args.run_dir/f'preflight-failure-{index:03d}.json').open('x',encoding='utf-8') as stream:
                json.dump(report,stream,indent=2,allow_nan=False)
                stream.write('\n')
        raise


if __name__=='__main__':
    main()
