"""Validation-only continuation search; does not load or score the test split."""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from common import ROOT, PROTOCOL, setup, make_model, sha
from evaluate import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--eval-every', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--seed', type=int, default=23)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    device, _ = setup('cpu', 'fp32', 4)
    torch.manual_seed(args.seed)
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    model, implementation_sha = make_model(parent['implementation'], parent['config'], device)
    model.load_state_dict(parent['model'])
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/tokenizer.json'))
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    data = {}
    for split in ('train', 'validation'):
        path = ROOT / 'data' / f'wikitext_{split}.txt'
        assert sha(path) == manifest['sha256'][path.name]
        raw = path.read_bytes()
        data[split] = (torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids), len(raw))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    rng = torch.Generator().manual_seed(args.seed)
    parent_targets = parent['train_tokens']
    log = dict(parent_sha256=sha(args.parent), parent_train_targets=parent_targets,
               seed=args.seed, max_steps=args.steps, batch_size=32, lr=args.lr,
               optimizer='fresh AdamW; parent optimizer not available',
               ema_decay=0.995, implementation_sha256=implementation_sha,
               training_script_sha256=sha(__file__), validation_history=[], history=[])
    best = float('inf')
    started = time.perf_counter()
    train_tokens = data['train'][0]
    for step in range(1, args.steps + 1):
        model.train()
        starts = torch.randint(len(train_tokens)-257, (32,), generator=rng)
        batch = train_tokens[starts[:, None] + torch.arange(257)]
        lr = args.lr * min(1., step / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step-1) / args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch[:, :-1]).flatten(0, 1), batch[:, 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        with torch.no_grad():
            for averaged, current in zip(ema.parameters(), model.parameters()):
                averaged.lerp_(current, 0.005)
        if step % 100 == 0:
            row = dict(step=step, loss=loss.item(), seconds=time.perf_counter()-started)
            log['history'].append(row)
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            for name, candidate in [('raw', model), ('ema', ema)]:
                result = score(candidate, *data['validation'], device, 'fp32')
                result.pop('window_nll_nats')
                row = dict(step=step, weights=name, **result)
                log['validation_history'].append(row)
                print(json.dumps({'validation': row}), flush=True)
                if result['bpb'] < best:
                    best = result['bpb']
                    checkpoint = dict(protocol=PROTOCOL, implementation=parent['implementation'],
                                      config=parent['config'], model=candidate.state_dict(),
                                      seed=args.seed, train_tokens=parent_targets + step * 32 * 256,
                                      parent_sha256=log['parent_sha256'], selection='validation',
                                      weights=name, continuation_step=step)
                    torch.save(checkpoint, args.run_dir / 'checkpoint.pt')
                    log['best'] = dict(row, checkpoint_sha256=sha(args.run_dir / 'checkpoint.pt'))
            log['total_processed_targets_including_parent'] = parent_targets + step * 32 * 256
            log['continuation_seconds'] = time.perf_counter()-started
            (args.run_dir / 'metrics.json').write_text(json.dumps(log, indent=2))
    print(json.dumps({'best': log['best'], 'total_search_targets': log['total_processed_targets_including_parent']}), flush=True)


if __name__ == '__main__':
    main()
