"""Bounded train-only learned-cache experiment; no test data is opened."""
import argparse
import copy
import json
import math
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from common import ROOT, PROTOCOL, setup, make_model, sha, autocast
from evaluate import score
from push145_cache import LearnedCacheGPT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', required=True, type=Path)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--mode', choices=['head','joint','control'], required=True)
    p.add_argument('--steps', default=2000, type=int)
    p.add_argument('--eval-every', default=500, type=int)
    p.add_argument('--lr', default=.001, type=float)
    p.add_argument('--backbone-lr', default=.00003, type=float)
    p.add_argument('--seed', default=103, type=int)
    args = p.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=False)
    device, precision = setup('cuda','bf16',4)
    torch.manual_seed(args.seed)
    parent = torch.load(args.parent,map_location='cpu',weights_only=True)
    config = dict(parent['config'])
    # Frozen table interpolation is added only after learning/selecting the pointer.
    config.pop('ngram_shapes',None)
    config.update(pointer_rank=32,pointer_max_weight=.5)
    if args.mode == 'control':
        model, _ = make_model('lowrank',config,device)
        implementation = 'lowrank'
    else:
        model = LearnedCacheGPT(config).to(device)
        implementation = 'push145_cache'
    state = {k:v for k,v in parent['model'].items() if not k.startswith('ngram.')}
    result = model.load_state_dict(state,strict=False)
    assert not result.unexpected_keys
    assert all(k.startswith('pointer_') for k in result.missing_keys)
    pointer, backbone = [], []
    for name,param in model.named_parameters():
        if name.startswith('pointer_'):
            pointer.append(param)
        elif args.mode == 'head':
            param.requires_grad_(False)
        else:
            backbone.append(param)
    groups = []
    if pointer:
        groups.append(dict(params=pointer,lr=args.lr,base_lr=args.lr,weight_decay=.01))
    if backbone:
        groups.append(dict(params=backbone,lr=args.backbone_lr,base_lr=args.backbone_lr,weight_decay=.1))
    optimizer = torch.optim.AdamW(groups,betas=(.9,.95),fused=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    averaged = copy.deepcopy(model).eval()
    for param in averaged.parameters():
        param.requires_grad_(False)
    tokenizer = Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    data = {}
    for split in ('train','validation'):
        path = ROOT/'data'/f'wikitext_{split}.txt'
        assert sha(path) == manifest['sha256'][path.name]
        raw = path.read_bytes()
        data[split] = (torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids),len(raw))
    tokens = data['train'][0].to(device)
    log = dict(implementation=implementation,config=config,mode=args.mode,seed=args.seed,
        parent_sha256=sha(args.parent),parent_train_targets=parent['train_tokens'],
        max_steps=args.steps,batch_size=32,context=256,head_lr=args.lr,backbone_lr=args.backbone_lr,
        new_trainable_parameters=sum(p.numel() for p in trainable),
        parameters=sum(p.numel() for p in model.parameters()),training_precision=precision,
        source_splits=['train','validation'],test_used=False,
        script_sha256=sha(__file__),implementation_sha256=sha(ROOT/(implementation+'.py')),
        selection='validation only; existing test results known but not used for this search',
        history=[],validation_history=[])
    started = time.perf_counter()
    best = float('inf')

    def validate(step):
        nonlocal best
        for name,candidate in ([('raw',model)] if step == 0 else [('raw',model),('ema',averaged)]):
            measured = score(candidate,*data['validation'],device,'fp32')
            measured.pop('window_nll_nats')
            row = dict(step=step,weights=name,**measured)
            log['validation_history'].append(row)
            print(json.dumps(dict(validation=row)),flush=True)
            if row['bpb'] < best:
                best = row['bpb']
                weights = {k:v.detach().cpu() for k,v in candidate.state_dict().items()}
                weights['head.weight'] = weights['token.weight']
                checkpoint = dict(protocol=PROTOCOL,implementation=implementation,config=config,
                    model=weights,seed=args.seed,train_tokens=parent['train_tokens']+step*32*256,
                    parent_sha256=sha(args.parent),weights=name,selection='validation-only')
                torch.save(checkpoint,args.run_dir/'checkpoint.pt')
                log['best'] = dict(row,checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'))
        log.update(incremental_search_targets=step*32*256,seconds=time.perf_counter()-started,
            total_search_targets=parent['train_tokens']+step*32*256,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        (args.run_dir/'metrics.json').write_text(json.dumps(log,indent=2))

    validate(0)
    rng = torch.Generator().manual_seed(args.seed)
    for step in range(1,args.steps+1):
        # Head-only experiments retain the frozen backbone's evaluation features.
        model.train(args.mode != 'head')
        starts = torch.randint(len(tokens)-257,(32,),generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        schedule = min(1.,step/100) * (.1+.9*.5*(1+math.cos(math.pi*(step-1)/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr']*schedule
        optimizer.zero_grad(set_to_none=True)
        with autocast(device,precision):
            if args.mode == 'control':
                loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
            else:
                logp = model.training_log_probs(batch[:,:-1])
                loss = -logp.gather(-1,batch[:,1:,None]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable,1.)
        optimizer.step()
        with torch.no_grad():
            for target,current in zip(averaged.parameters(),model.parameters()):
                target.lerp_(current,.01)
        if step % 100 == 0:
            torch.cuda.synchronize()
            row = dict(step=step,loss=loss.item(),seconds=time.perf_counter()-started)
            log['history'].append(row)
            print(json.dumps(row),flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            validate(step)
    print(json.dumps(dict(best=log['best'],seconds=log['seconds'])),flush=True)


if __name__ == '__main__':
    main()
