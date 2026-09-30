"""Train only from supplied training text, select checkpoints only on validation."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--steps', default=8000, type=int)
    parser.add_argument('--eval-every', default=500, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--seed', default=31, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--implementation', default='sparse_ngram', choices=['efficient','model','sparse_ngram','lowrank'])
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    device, precision = setup('cuda', 'bf16', 4)
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    parent_targets, parent_hash = 0, None
    if args.parent:
        parent = torch.load(args.parent, map_location='cpu', weights_only=True)
        model.load_state_dict(parent['model'])
        parent_targets, parent_hash = parent['train_tokens'], sha(args.parent)
    averaged = copy.deepcopy(model).eval()
    for p in averaged.parameters():
        p.requires_grad_(False)
    tokenizer = Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    data = {}
    for split in ('train', 'validation'):
        path = ROOT/'data'/f'wikitext_{split}.txt'
        assert sha(path) == manifest['sha256'][path.name]
        raw = path.read_bytes()
        data[split] = (torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids), len(raw))
    tokens = data['train'][0].to(device)
    parameters = list(model.parameters())
    groups = [dict(params=[p for p in parameters if p.ndim >= 2], weight_decay=0.1),
              dict(params=[p for p in parameters if p.ndim < 2], weight_decay=0.)]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), fused=True)
    rng = torch.Generator().manual_seed(args.seed)
    best, stale = float('inf'), 0
    log = dict(config=config, implementation=args.implementation, seed=args.seed, parameters=sum(p.numel() for p in parameters),
               recipe=dict(max_steps=args.steps, lr=args.lr, batch_size=32, warmup=200,
                           weight_decay=0.1, betas=[0.9,0.95], ema_decay=0.995,
                           eval_every=args.eval_every, patience=args.patience),
               training_precision=precision, torch_version=str(torch.__version__),
               device=torch.cuda.get_device_name(), script_sha256=sha(__file__),
               implementation_sha256=implementation_sha, history=[], validation_history=[],
               parent_train_targets=parent_targets, parent_sha256=parent_hash,
               optimizer_initialization='fresh AdamW')
    started = time.perf_counter()
    for step in range(1,args.steps+1):
        model.train()
        starts = torch.randint(len(tokens)-257,(32,),generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        lr = args.lr * min(1.,step/200) * (0.1+0.9*0.5*(1+math.cos(math.pi*(step-1)/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast(device,precision):
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters,1.)
        optimizer.step()
        with torch.no_grad():
            for avg, current in zip(averaged.parameters(),parameters):
                avg.lerp_(current,0.005)
        if step%100==0:
            torch.cuda.synchronize()
            row = dict(step=step,loss=loss.item(),seconds=time.perf_counter()-started)
            log['history'].append(row)
            print(json.dumps(row),flush=True)
        if step%args.eval_every==0 or step==args.steps:
            improved = False
            for name,candidate in [('raw',model),('ema',averaged)]:
                result = score(candidate,*data['validation'],device,'fp32')
                result.pop('window_nll_nats')
                row = dict(step=step,weights=name,**result)
                log['validation_history'].append(row)
                print(json.dumps({'validation':row}),flush=True)
                if result['bpb'] < best:
                    improved=True
                    best=result['bpb']
                    state = {k:v.detach().cpu() for k,v in candidate.state_dict().items()}
                    # Preserve tied embedding storage in serialized inference assets.
                    state['head.weight'] = state['token.weight']
                    checkpoint = dict(protocol=PROTOCOL,implementation=args.implementation,config=config,
                                      model=state,seed=args.seed,train_tokens=parent_targets+step*32*256,
                                      parent_sha256=parent_hash,
                                      weights=name,selection='validation-only')
                    torch.save(checkpoint,args.run_dir/'checkpoint.pt')
                    log['best'] = dict(row,checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'))
            stale = 0 if improved else stale+1
            log.update(total_search_targets=parent_targets+step*32*256,
                       incremental_search_targets=step*32*256, seconds=time.perf_counter()-started,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            (args.run_dir/'metrics.json').write_text(json.dumps(log,indent=2))
            if stale >= args.patience:
                print(json.dumps({'early_stop':step,'best':log['best']}),flush=True)
                break
    print(json.dumps({'best':log['best'],'total_search_targets':log['total_search_targets']}),flush=True)


if __name__ == '__main__':
    main()
