"""Bounded train-only trials with one frozen full-hybrid validation recipe."""
import argparse
import copy
import json
import math
import time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from common import ROOT, PROTOCOL, setup, make_model, sha, autocast
from evaluate import score


def train_config(config):
    result = {k:copy.deepcopy(v) for k,v in config.items()
              if not k.startswith(('ngram_','mkn_')) and k!='discount_scale'}
    result.update(temperature=1.15,dropout=.20)
    return result


def train_implementation(implementation):
    # The old inference class overrides neural_and_cache with no_grad.
    return 'push145_cache' if implementation=='push145_cache_hybrid' else implementation


def training_loss(model, ids, targets, auxiliary_weight):
    """A shared feature pass; no statistical table call or gradient is allowed."""
    features=model.features(ids)
    logits=model.head(features).float()/model.config.get('temperature',1.15)
    probabilities=logits.softmax(-1)
    attention,gate=model.cache_attention_and_gate(features)
    values=torch.cat([ids[:,1:],ids.new_zeros(ids.shape[0],1)],dim=1)
    memory=torch.zeros_like(probabilities)
    memory.scatter_add_(-1,values[:,None,:].expand(-1,ids.shape[1],-1),attention)
    mixture=probabilities*(1-gate)+memory*gate
    target=targets.unsqueeze(-1)
    mixed_nll=-mixture.gather(-1,target).clamp_min(1e-30).log().mean()
    if auxiliary_weight:
        neural_nll=-probabilities.gather(-1,target).clamp_min(1e-30).log().mean()
        return (1-auxiliary_weight)*mixed_nll+auxiliary_weight*neural_nll
    return mixed_nll


def snapshot(model,parent,eval_config,parent_sha,seed,step,weights):
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    state['head.weight']=state['token.weight']
    state.update({k:v for k,v in parent['model'].items() if k.startswith('ngram.')})
    return dict(protocol=PROTOCOL,implementation=parent['implementation'],
        config=copy.deepcopy(eval_config),model=state,seed=seed,
        train_tokens=parent['train_tokens']+step*32*256,parent_sha256=parent_sha,
        selected_step=step,weights=weights,training_config=copy.deepcopy(model.config),
        ancestry=copy.deepcopy(parent.get('ancestry',{})),
        selection='full validation only; frozen train-derived statistics')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True,type=Path)
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--steps',type=int,default=8000)
    p.add_argument('--eval-every',type=int,default=2000)
    p.add_argument('--seed',type=int,default=149)
    p.add_argument('--backbone-lr',type=float,default=.00003)
    p.add_argument('--head-lr',type=float,default=.0003)
    p.add_argument('--scalar-lr',type=float,default=.001)
    p.add_argument('--auxiliary-weight',type=float,default=0.)
    args=p.parse_args()
    if not 0<=args.auxiliary_weight<=1:
        raise ValueError('Auxiliary weight must be a convex mixing coefficient.')
    args.run_dir.mkdir(parents=True,exist_ok=False)
    device,precision=setup('cuda','bf16',4)
    torch.manual_seed(args.seed)
    parent=torch.load(args.parent,map_location='cpu',weights_only=True)
    parent_sha=sha(args.parent)
    implementation=train_implementation(parent['implementation'])
    model,_=make_model(implementation,train_config(parent['config']),device)
    state={k:v for k,v in parent['model'].items() if not k.startswith('ngram.')}
    model.load_state_dict(state,strict=True)
    averaged=copy.deepcopy(model).eval()
    for parameter in averaged.parameters():
        parameter.requires_grad_(False)
    eval_config=copy.deepcopy(parent['config'])
    eval_config.update(temperature=1.25,mkn_weight=.15,dropout=.20)
    evaluator,_=make_model(parent['implementation'],eval_config,device)
    evaluator.eval()
    groups={'pointer':[],'backbone':[],'scalar':[]}
    for name,parameter in model.named_parameters():
        group='scalar' if parameter.ndim==0 else ('pointer' if name.startswith('pointer_') else 'backbone')
        groups[group].append(parameter)
    rates={'pointer':args.head_lr,'backbone':args.backbone_lr,'scalar':args.scalar_lr}
    decay={'pointer':.01,'backbone':.1,'scalar':0.}
    optimizer=torch.optim.AdamW([
        dict(params=values,lr=rates[key],base_lr=rates[key],weight_decay=decay[key])
        for key,values in groups.items() if values],betas=(.9,.95),fused=True)
    tokenizer=Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    data={}
    for split in ('train','validation'):
        path=ROOT/'data'/f'wikitext_{split}.txt'
        assert sha(path)==manifest['sha256'][path.name]
        raw=path.read_bytes()
        data[split]=(torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids),len(raw))
    tokens=data['train'][0].to(device)
    log=dict(parent_sha256=parent_sha,parent_train_targets=parent['train_tokens'],
        training_implementation=implementation,inference_implementation=parent['implementation'],
        training_config=model.config,validation_config=eval_config,seed=args.seed,
        steps=args.steps,eval_every=args.eval_every,batch_size=32,context=256,
        backbone_lr=args.backbone_lr,head_lr=args.head_lr,scalar_lr=args.scalar_lr,
        auxiliary_neural_weight=args.auxiliary_weight,ema_decay=.99,
        training_precision=precision,parameters=sum(p.numel() for p in model.parameters()),
        no_statistical_tables_in_training=True,source_splits=['train','validation'],
        test_used=False,prior_test_exposure_disclosed=True,
        script_sha256=sha(__file__),implementation_sha256=sha(ROOT/(implementation+'.py')),
        ancestry=parent.get('ancestry',{}),history=[],validation_history=[])
    assert not any(k.startswith('ngram.') for k in model.state_dict())
    started=time.perf_counter()
    best=float('inf')

    def validate(step):
        nonlocal best
        for label,candidate in ([('raw',model)] if step==0 else [('raw',model),('ema',averaged)]):
            checkpoint=snapshot(candidate,parent,eval_config,parent_sha,args.seed,step,label)
            evaluator.load_state_dict(checkpoint['model'],strict=True)
            measured=score(evaluator,*data['validation'],device,'fp32')
            measured.pop('window_nll_nats')
            scalars={name:float(value.detach()) for name,value in candidate.named_parameters() if value.ndim==0}
            row=dict(step=step,weights=label,scalars=scalars,**measured)
            log['validation_history'].append(row)
            print(json.dumps(dict(validation=row)),flush=True)
            if measured['bpb']<best:
                best=measured['bpb']
                torch.save(checkpoint,args.run_dir/'checkpoint.pt')
                log['best']=dict(row,checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'),
                    selected_lineage_targets=checkpoint['train_tokens'])
            del checkpoint
        log.update(incremental_search_targets=step*32*256,seconds=time.perf_counter()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        (args.run_dir/'metrics.json').write_text(json.dumps(log,indent=2))

    validate(0)
    rng=torch.Generator().manual_seed(args.seed)
    for step in range(1,args.steps+1):
        model.train()
        starts=torch.randint(len(tokens)-256,(32,),generator=rng).to(device)
        batch=tokens[starts[:,None]+torch.arange(257,device=device)]
        schedule=min(1.,step/100)*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/args.steps)))
        for group in optimizer.param_groups:
            group['lr']=group['base_lr']*schedule
        optimizer.zero_grad(set_to_none=True)
        with autocast(device,precision):
            loss=training_loss(model,batch[:,:-1],batch[:,1:],args.auxiliary_weight)
        assert loss.requires_grad
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        project=getattr(model,'project_cache_parameters',None)
        if callable(project):
            project()
        with torch.no_grad():
            for target,current in zip(averaged.parameters(),model.parameters()):
                target.lerp_(current,.01)
        if step%400==0:
            torch.cuda.synchronize()
            row=dict(step=step,loss=loss.item(),gradient_norm=float(norm),seconds=time.perf_counter()-started)
            log['history'].append(row)
            print(json.dumps(row),flush=True)
        if step%args.eval_every==0 or step==args.steps:
            validate(step)
    print(json.dumps(dict(best=log['best'],seconds=log['seconds'])),flush=True)


if __name__=='__main__':
    main()
