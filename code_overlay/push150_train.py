"""Resumable train/validation-only continuation; scoring architecture unchanged.

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
from push147_train import training_loss
from push148_train import build_optimizers


BATCH_SIZE, CONTEXT, TARGETS_PER_STEP = 32, 256, 8192
STATE_FORMAT = 1


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
    pending=[root/'push150_train.py',root/(implementation+'.py'),root/'push148_muon.py']
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',required=True,type=Path)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--resume',type=Path,help='Existing train_state.pt in the SAME run directory')
    parser.add_argument('--steps',default=20000,type=int)
    parser.add_argument('--eval-every',default=2000,type=int)
    parser.add_argument('--seed',default=173,type=int)
    parser.add_argument('--optimizer',choices=['adamw','muon'],default='muon')
    parser.add_argument('--schedule',choices=['cosine','wsd'],default='cosine')
    parser.add_argument('--warmup',default=200,type=int)
    parser.add_argument('--decay-steps',default=4000,type=int)
    parser.add_argument('--backbone-lr',default=.0000075,type=float)
    parser.add_argument('--head-lr',default=.000075,type=float)
    parser.add_argument('--scalar-lr',default=.00025,type=float)
    parser.add_argument('--muon-lr',default=.0005,type=float)
    parser.add_argument('--dropout',default=.20,type=float)
    args=parser.parse_args()
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
    device,precision=setup('cuda','bf16',4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    parent=torch.load(args.parent,map_location='cpu',weights_only=True)
    if parent['protocol'] != PROTOCOL:
        raise ValueError('Parent protocol mismatch')
    parent_sha=sha(args.parent)
    implementation=parent['implementation']
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
        objective='train-only neural+pointer NLL; no statistical table loss')
    identity=dict(parent_sha256=parent_sha,implementation=implementation,recipe=copy.deepcopy(recipe),
        source_sha256=source_digests(implementation),data_sha256=data_hashes,
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
            discarded_completed_steps=extra,previous_status=disk.get('status'),evidence=archive.name,
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
        log.update(completed_steps=completed_step,
            incremental_search_targets=(completed_step+log['discarded_completed_steps'])*TARGETS_PER_STEP,
            incremental_sampled_label_targets=(completed_step+log['discarded_completed_steps'])*TARGETS_PER_STEP,
            lineage_new_targets=completed_step*TARGETS_PER_STEP,seconds=offset_seconds+time.perf_counter()-started,
            peak_cuda_allocated_bytes=max(log.get('peak_cuda_allocated_bytes',0),torch.cuda.max_memory_allocated()))

    def check_sources():
        if source_digests(implementation) != identity['source_sha256'] or sha(args.parent) != parent_sha:
            raise ValueError('Pinned source or parent changed during training')

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

    try:
        if not args.resume:
            validate(0)
            save_state()
        for step in range(completed_step+1,args.steps+1):
            model.train()
            starts=torch.randint(len(tokens)-CONTEXT,(BATCH_SIZE,),generator=sampler).to(device)
            batch=tokens[starts[:,None]+torch.arange(CONTEXT+1,device=device)]
            factor=schedule_factor(step,args.steps,args.schedule,args.warmup,args.decay_steps)
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group['lr']=group['base_lr']*factor
                optimizer.zero_grad(set_to_none=True)
            with autocast(device,precision):
                loss=training_loss(model,batch[:,:-1],batch[:,1:],0.)
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


if __name__=='__main__':
    main()
