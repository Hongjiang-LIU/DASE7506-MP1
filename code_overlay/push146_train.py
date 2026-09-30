"""Matched continuation runs, train-only loss and full-hybrid validation.

The inference implementation is unchanged. Never instantiate the inference-only
hybrid as the training network: its no_grad mixture would suppress gradients.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from common import ROOT, PROTOCOL, setup, sha, autocast
from evaluate import score
from push145_cache import LearnedCacheGPT
from push145_cache_hybrid import LearnedCacheHybrid


def training_config(parent_config, dropout, temperature=1.15):
    config = {k: copy.deepcopy(v) for k, v in parent_config.items()
              if not k.startswith(('mkn_', 'ngram_')) and k != 'discount_scale'}
    config.update(dropout=dropout, temperature=temperature)
    return config


def neural_state(state):
    return {k: v for k, v in state.items() if not k.startswith('ngram.')}


def hybrid_state(model, parent):
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    state['head.weight'] = state['token.weight']
    state.update({k: v for k, v in parent['model'].items() if k.startswith('ngram.')})
    return state


def make_checkpoint(model, parent, config, parent_sha, seed, step, weights):
    return dict(protocol=PROTOCOL, implementation='push145_cache_hybrid',
        config=copy.deepcopy(config), model=hybrid_state(model, parent), seed=seed,
        train_tokens=parent['train_tokens'] + step*32*256,
        parent_sha256=parent_sha, selected_step=step, weights=weights,
        selection='full validation only; same frozen training-derived statistics',
        training_config=copy.deepcopy(model.config))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', required=True, type=Path)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--dropout', type=float, required=True)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=131)
    p.add_argument('--head-lr', type=float, default=.0002)
    p.add_argument('--backbone-lr', type=float, default=.00002)
    args = p.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    device, precision = setup('cuda', 'bf16', 4)
    torch.manual_seed(args.seed)
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    assert parent['implementation'] == 'push145_cache_hybrid'
    parent_sha = sha(args.parent)
    config = training_config(parent['config'], args.dropout)
    model = LearnedCacheGPT(config).to(device)
    model.load_state_dict(neural_state(parent['model']), strict=True)
    averaged = copy.deepcopy(model).eval()
    for parameter in averaged.parameters():
        parameter.requires_grad_(False)
    eval_config = copy.deepcopy(parent['config'])
    eval_config.update(temperature=1.20, mkn_weight=.15, dropout=args.dropout)
    evaluator = LearnedCacheHybrid(eval_config).to(device).eval()
    pointer, backbone = [], []
    for name, parameter in model.named_parameters():
        (pointer if name.startswith('pointer_') else backbone).append(parameter)
    optimizer = torch.optim.AdamW([
        dict(params=pointer, lr=args.head_lr, base_lr=args.head_lr, weight_decay=.01),
        dict(params=backbone, lr=args.backbone_lr, base_lr=args.backbone_lr, weight_decay=.1)
    ], betas=(.9,.95), fused=True)
    tokenizer = Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    data = {}
    for split in ('train', 'validation'):
        path = ROOT/'data'/f'wikitext_{split}.txt'
        assert sha(path) == manifest['sha256'][path.name]
        raw = path.read_bytes()
        data[split] = (torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids),len(raw))
    tokens = data['train'][0].to(device)
    log = dict(parent_sha256=parent_sha, parent_train_targets=parent['train_tokens'],
        implementation='push145_cache_hybrid', inference_implementation_unchanged=True,
        training_config=config, validation_config=eval_config, seed=args.seed,
        max_steps=args.steps, batch_size=32, context=256, head_lr=args.head_lr,
        backbone_lr=args.backbone_lr, ema_decay=.99, training_precision=precision,
        loss='train-only neural+pointer, no statistical tables in loss',
        parameters=sum(p.numel() for p in model.parameters()),
        selection='full hybrid validation, raw/EMA and step0 parent eligible',
        source_splits=['train','validation'], test_used=False,
        prior_test_exposure_disclosed=True, script_sha256=sha(__file__),
        history=[], validation_history=[])
    started = time.perf_counter()
    best = float('inf')

    def validate(step):
        nonlocal best
        candidates = [('raw',model)] if step == 0 else [('raw',model),('ema',averaged)]
        for label, candidate in candidates:
            checkpoint = make_checkpoint(candidate,parent,eval_config,parent_sha,args.seed,step,label)
            evaluator.load_state_dict(checkpoint['model'], strict=True)
            measured = score(evaluator,*data['validation'],device,'fp32')
            measured.pop('window_nll_nats')
            row = dict(step=step, weights=label, **measured)
            log['validation_history'].append(row)
            print(json.dumps(dict(validation=row)),flush=True)
            if measured['bpb'] < best:
                best = measured['bpb']
                torch.save(checkpoint,args.run_dir/'checkpoint.pt')
                log['best'] = dict(row,checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'),
                    selected_lineage_targets=checkpoint['train_tokens'])
            del checkpoint
        log.update(incremental_search_targets=step*32*256,
            seconds=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        (args.run_dir/'metrics.json').write_text(json.dumps(log,indent=2))

    validate(0)
    rng = torch.Generator().manual_seed(args.seed)
    for step in range(1,args.steps+1):
        model.train()
        starts = torch.randint(len(tokens)-256,(32,),generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        schedule = min(1.,step/100)*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr']*schedule
        optimizer.zero_grad(set_to_none=True)
        with autocast(device,precision):
            logp = model.training_log_probs(batch[:,:-1])
            loss = -logp.gather(-1,batch[:,1:,None]).mean()
        assert loss.requires_grad
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for target,current in zip(averaged.parameters(),model.parameters()):
                target.lerp_(current,.01)
        if step % 200 == 0:
            torch.cuda.synchronize()
            row = dict(step=step,loss=loss.item(),gradient_norm=float(norm),seconds=time.perf_counter()-started)
            log['history'].append(row)
            print(json.dumps(row),flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            validate(step)
    print(json.dumps(dict(best=log['best'],seconds=log['seconds'])),flush=True)


if __name__ == '__main__':
    main()
