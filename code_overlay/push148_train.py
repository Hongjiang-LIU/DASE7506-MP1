"""Train-only R-Drop / Muon trials; existing frozen inference architecture."""
import argparse
import copy
import json
import math
import time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from common import ROOT, setup, make_model, sha, autocast
from evaluate import score
from push147_train import train_config, training_loss, snapshot
from push148_losses import dual_view_loss


def build_optimizers(model, kind, backbone_lr, head_lr, scalar_lr, muon_lr):
    named = list(model.named_parameters())
    muon_names = {name for name, parameter in named
                  if kind == 'muon' and name.startswith('blocks.') and parameter.ndim == 2}
    groups = {'pointer': [], 'backbone': [], 'scalar': []}
    for name, parameter in named:
        if name not in muon_names:
            key = 'scalar' if parameter.ndim == 0 else ('pointer' if name.startswith('pointer_') else 'backbone')
            groups[key].append(parameter)
    rates = dict(pointer=head_lr, backbone=backbone_lr, scalar=scalar_lr)
    decay = dict(pointer=.01, backbone=.1, scalar=0.)
    optimizers = [torch.optim.AdamW([
        dict(params=values, lr=rates[key], base_lr=rates[key], weight_decay=decay[key])
        for key, values in groups.items() if values], betas=(.9,.95), fused=True)]
    if kind == 'muon':
        from push148_muon import BatchedMuon
        optimizers.append(BatchedMuon([parameter for name, parameter in named if name in muon_names],
                                     lr=muon_lr, weight_decay=.1, momentum=.95, ns_steps=5))
    parameter_ids = [id(parameter) for optimizer in optimizers for group in optimizer.param_groups for parameter in group['params']]
    assert len(parameter_ids) == len(set(parameter_ids)) == len(named)
    return optimizers, sorted(muon_names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--steps', default=8000, type=int)
    parser.add_argument('--eval-every', default=2000, type=int)
    parser.add_argument('--seed', default=157, type=int)
    parser.add_argument('--views', choices=[1,2], default=1, type=int)
    parser.add_argument('--rdrop-coefficient', default=0., type=float)
    parser.add_argument('--optimizer', choices=['adamw','muon'], default='adamw')
    parser.add_argument('--backbone-lr', default=.00003, type=float)
    parser.add_argument('--head-lr', default=.0003, type=float)
    parser.add_argument('--scalar-lr', default=.001, type=float)
    parser.add_argument('--muon-lr', default=.002, type=float)
    args = parser.parse_args()
    if args.rdrop_coefficient < 0 or (args.views == 1 and args.rdrop_coefficient != 0):
        raise ValueError('A nonzero R-Drop coefficient needs two views.')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    device, precision = setup('cuda','bf16',4)
    torch.manual_seed(args.seed)
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    assert parent['implementation'] == 'push147_signed_cache'
    parent_sha = sha(args.parent)
    model, implementation_sha = make_model(parent['implementation'], train_config(parent['config']), device)
    model.load_state_dict({key: value for key, value in parent['model'].items() if not key.startswith('ngram.')}, strict=True)
    assert not any(key.startswith('ngram.') for key in model.state_dict())
    averaged = copy.deepcopy(model).eval().requires_grad_(False)
    eval_config = copy.deepcopy(parent['config'])
    eval_config.update(temperature=1.25,mkn_weight=.15,dropout=.20)
    evaluator, _ = make_model(parent['implementation'], eval_config, device)
    evaluator.eval()
    optimizers, muon_names = build_optimizers(model,args.optimizer,args.backbone_lr,args.head_lr,args.scalar_lr,args.muon_lr)
    batch_size = 32//args.views
    assert batch_size*args.views*256 == 8192
    tokenizer = Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    data = {}
    for split in ('train','validation'):
        path = ROOT/'data'/f'wikitext_{split}.txt'
        assert sha(path) == manifest['sha256'][path.name]
        raw = path.read_bytes()
        data[split] = (torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids),len(raw))
    tokens = data['train'][0].to(device)
    sources = ['push148_train.py','push148_losses.py','push147_train.py']
    if args.optimizer == 'muon':
        sources.append('push148_muon.py')
    log = dict(status='running', parent_sha256=parent_sha, parent_train_targets=parent['train_tokens'],
        training_implementation=parent['implementation'],inference_implementation=parent['implementation'],
        inference_architecture_unchanged=True,training_config=model.config,validation_config=eval_config,
        seed=args.seed,steps=args.steps,eval_every=args.eval_every,batch_size=batch_size,
        dropout_views_per_sample=args.views,processed_view_targets_per_step=8192,
        sampled_label_targets_per_step=batch_size*256,
        optimizer=args.optimizer,rdrop_coefficient=args.rdrop_coefficient,
        backbone_lr=args.backbone_lr,head_lr=args.head_lr,scalar_lr=args.scalar_lr,muon_lr=args.muon_lr,
        muon_parameter_names=muon_names,muon_momentum=.95,muon_ns_steps=5,
        ema_decay=.99,training_precision=precision,parameters=sum(p.numel() for p in model.parameters()),
        no_statistical_tables_in_training=True,source_splits=['train','validation'],test_used=False,
        prior_test_exposure_disclosed=True,implementation_sha256=implementation_sha,
        source_sha256={name:sha(ROOT/name) for name in sources},
        ancestry=parent.get('ancestry',{}),history=[],validation_history=[])
    started = time.perf_counter()
    best = float('inf')
    completed_step = 0

    def save_log():
        log.update(completed_steps=completed_step,
            incremental_search_targets=completed_step*8192,
            incremental_sampled_label_targets=completed_step*batch_size*256,
            seconds=time.perf_counter()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        (args.run_dir/'metrics.json').write_text(json.dumps(log,indent=2))

    def validate(step):
        nonlocal best
        choices = [('raw',model)] if step == 0 else [('raw',model),('ema',averaged)]
        for label,candidate in choices:
            checkpoint = snapshot(candidate,parent,eval_config,parent_sha,args.seed,step,label)
            checkpoint['training_recipe'] = dict(optimizer=args.optimizer,views=args.views,
                batch_size=batch_size,rdrop_coefficient=args.rdrop_coefficient,
                processed_view_targets=step*8192,sampled_label_targets=step*batch_size*256)
            evaluator.load_state_dict(checkpoint['model'],strict=True)
            measured = score(evaluator,*data['validation'],device,'fp32')
            measured.pop('window_nll_nats')
            row = dict(step=step,weights=label,
                scalars={name:float(value.detach()) for name,value in candidate.named_parameters() if value.ndim==0},**measured)
            log['validation_history'].append(row)
            print(json.dumps(dict(validation=row)),flush=True)
            if measured['bpb'] < best:
                best = measured['bpb']
                torch.save(checkpoint,args.run_dir/'checkpoint.pt')
                log['best'] = dict(row,checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'),
                    selected_lineage_targets=checkpoint['train_tokens'],
                    new_selected_sampled_label_targets=step*batch_size*256)
        save_log()

    try:
        validate(0)
        rng = torch.Generator().manual_seed(args.seed)
        for step in range(1,args.steps+1):
            model.train()
            starts = torch.randint(len(tokens)-256,(batch_size,),generator=rng).to(device)
            batch = tokens[starts[:,None]+torch.arange(257,device=device)]
            schedule = min(1.,step/100)*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/args.steps)))
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group['lr'] = group['base_lr']*schedule
                optimizer.zero_grad(set_to_none=True)
            with autocast(device,precision):
                if args.views == 2:
                    loss,detail = dual_view_loss(model,batch[:,:-1],batch[:,1:],args.rdrop_coefficient)
                else:
                    loss = training_loss(model,batch[:,:-1],batch[:,1:],0.)
                    detail = dict(hard=loss.detach(),kl=loss.detach()*0)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at attempted step {step}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            for optimizer in optimizers:
                optimizer.step()
            model.project_cache_parameters()
            with torch.no_grad():
                for target,current in zip(averaged.parameters(),model.parameters()):
                    target.lerp_(current,.01)
            completed_step = step
            if step%400 == 0:
                torch.cuda.synchronize()
                row = dict(step=step,loss=loss.item(),detail={key:float(value) for key,value in detail.items()},
                    gradient_norm=float(norm),seconds=time.perf_counter()-started)
                log['history'].append(row)
                print(json.dumps(row),flush=True)
            if step%args.eval_every == 0 or step == args.steps:
                validate(step)
        log['status'] = 'completed'
        save_log()
    except Exception as error:
        log.update(status='failed',failure_type=type(error).__name__,failure_message=str(error),
            attempted_step=locals().get('step',0),
            failed_attempt_cost_note='Completed update counts exclude any work in the failed attempt; no failure is silently replaced.')
        save_log()
        raise
    print(json.dumps(dict(best=log['best'],seconds=log['seconds'])),flush=True)


if __name__ == '__main__':
    main()
