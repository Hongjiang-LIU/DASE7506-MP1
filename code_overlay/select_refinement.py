"""Fit temperature/cache hyperparameters only on validation, never test."""
import argparse
import itertools
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from common import ROOT, setup, make_model, windows, sha
from refined import cache_probs


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--fine', action='store_true')
    parser.add_argument('--output-implementation',default='refined')
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    device, _ = setup(args.device, 'fp32', 4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    tokenizer = Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    raw = (ROOT/'data/wikitext_validation.txt').read_bytes()
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    assert sha(ROOT/'data/wikitext_validation.txt') == manifest['sha256']['wikitext_validation.txt']
    tokens = torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids)
    settings = [(t, 0., 20.) for t in (0.85, 0.9, 0.95, 1., 1.05, 1.1)]
    settings += list(itertools.product((0.9, 0.95, 1., 1.05), (0.05, 0.1, 0.15), (0., 2., 5., 10., 20.)))
    if args.fine:
        settings = [(t,0.,20.) for t in (0.95,1.,1.05,1.1,1.15,1.2,1.3)]
        settings += list(itertools.product((1.,1.05,1.1,1.15,1.2), (0.025,0.05,0.075,0.1), (5.,10.,15.,20.,30.)))
    temperatures = sorted(set(t for t,w,h in settings))
    thetas = sorted(set(h for t,w,h in settings))
    totals = [0.] * len(settings)
    started = time.perf_counter()
    for x, y in windows(tokens):
        x, y = x.to(device), y.to(device)
        features = model.features(x)
        logits = model.head(features).float()
        target = y.clamp_min(0).unsqueeze(-1)
        valid = y != -100
        neural = {t: F.log_softmax(logits/t, dim=-1).gather(-1, target).squeeze(-1).double().exp()
                  for t in temperatures}
        memories = {theta: cache_probs(features, x, theta).gather(-1, target).squeeze(-1).double()
                    for theta in thetas}
        for i, (temperature, weight, theta) in enumerate(settings):
            mix = torch.full_like(y, weight, dtype=torch.float64)
            mix[:, 0] = 0
            p = (1-mix)*neural[temperature] + mix*memories[theta]
            totals[i] += -p.clamp_min(1e-300).log()[valid].sum().item()
    results = [dict(temperature=t, cache_weight=w, cache_theta=theta,
                    validation_bpb=total / math.log(2) / len(raw))
               for (t,w,theta), total in zip(settings,totals)]
    results.sort(key=lambda row:row['validation_bpb'])
    best = results[0]
    checkpoint['implementation'] = args.output_implementation
    checkpoint['config'].update({k:best[k] for k in ('temperature','cache_weight','cache_theta')})
    checkpoint['selection'] = 'validation-only temperature and causal cache'
    checkpoint['parent_sha256'] = sha(args.checkpoint)
    torch.save(checkpoint,args.run_dir/'checkpoint.pt')
    report = dict(best=best, trials=results, script_sha256=sha(__file__), device=str(device), seconds=time.perf_counter()-started,
                  parent_sha256=sha(args.checkpoint), checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'))
    (args.run_dir/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
