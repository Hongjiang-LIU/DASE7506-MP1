"""Validation-only continuation/KD experiment; never reads or scores test text."""
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


def distillation_loss(student_logits, teacher_logits, targets, alpha, temperature):
    hard = F.cross_entropy(student_logits.flatten(0, 1).float(), targets.flatten())
    if alpha == 0:
        return hard
    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_p = F.softmax(teacher_logits.detach().float() / temperature, dim=-1)
    soft = -(teacher_p * student_logp).sum(-1).mean() * temperature**2
    return (1-alpha)*hard + alpha*soft


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
    parser.add_argument('--implementation', default='efficient', choices=['efficient','model'])
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--alpha', type=float, default=0.)
    parser.add_argument('--temperature', type=float, default=1.)
    args = parser.parse_args()
    assert 0 <= args.alpha <= 1 and args.temperature > 0
    assert args.parent is not None
    assert args.alpha == 0 or args.teacher is not None
    args.run_dir.mkdir(parents=True, exist_ok=False)
    device, precision = setup('cuda', 'bf16', 4)
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    parent_targets, parent_hash = 0, None
    if args.parent:
        parent = torch.load(args.parent, map_location='cpu', weights_only=True)
        assert parent['protocol'] == PROTOCOL
        model.load_state_dict(parent['model'])
        parent_targets, parent_hash = parent['train_tokens'], sha(args.parent)
    averaged = copy.deepcopy(model).eval()
    for p in averaged.parameters():
        p.requires_grad_(False)
    teacher = None
    teacher_hash, teacher_targets = None, 0
    if args.teacher:
        teacher_checkpoint = torch.load(args.teacher, map_location='cpu', weights_only=True)
        assert teacher_checkpoint['protocol'] == PROTOCOL
        assert teacher_checkpoint['implementation'] == 'efficient'
        teacher, _ = make_model('efficient', teacher_checkpoint['config'], device)
        teacher.load_state_dict(teacher_checkpoint['model'])
        teacher.eval().requires_grad_(False)
        teacher_hash, teacher_targets = sha(args.teacher), teacher_checkpoint['train_tokens']
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
               teacher_sha256=teacher_hash, teacher_train_targets=teacher_targets,
               distillation=dict(alpha=args.alpha, temperature=args.temperature),
               optimizer_initialization='fresh AdamW')
    started = time.perf_counter()
    initial = score(model, *data['validation'], device, 'fp32')
    initial.pop('window_nll_nats')
    best = initial['bpb']
    initial_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    initial_state['head.weight'] = initial_state['token.weight']
    torch.save(dict(protocol=PROTOCOL, implementation=args.implementation, config=config,
                    model=initial_state, seed=args.seed, train_tokens=parent_targets,
                    parent_sha256=parent_hash, weights='parent', selection='validation-only',
                    teacher_sha256=teacher_hash, teacher_train_targets=teacher_targets),
               args.run_dir/'checkpoint.pt')
    log['best'] = dict(step=0,weights='parent',**initial,
                       checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'))
    log['initial_validation'] = initial
    print(json.dumps({'initial_validation':initial}), flush=True)
    for step in range(1,args.steps+1):
        model.train()
        starts = torch.randint(len(tokens)-257,(32,),generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        lr = args.lr * min(1.,step/200) * (0.1+0.9*0.5*(1+math.cos(math.pi*(step-1)/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast(device,precision):
            teacher_logits = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(batch[:,:-1])
            student_logits = model(batch[:,:-1])
            loss = distillation_loss(student_logits, teacher_logits, batch[:,1:],
                                     args.alpha, args.temperature)
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
                                      teacher_sha256=teacher_hash, teacher_train_targets=teacher_targets,
                                      distillation=dict(alpha=args.alpha,temperature=args.temperature),
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
