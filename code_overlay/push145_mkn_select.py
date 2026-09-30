"""A fixed 12-setting VALIDATION-ONLY screen; scoring shortcut unit-tested.

Inference is target-free. Targets are used here only for scoring distributions.
One neural forward traversal is shared by all choices, with no training updates.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from common import ROOT, make_model, setup, sha, windows
from ngram_hybrid import prefix_codes
from push145_mkn import MKNTables


def target_probabilities(arrays, x, y, maximum=5):
    target = y.ravel().clip(min=0)
    result = arrays['unigram'][target].copy()
    for order in range(2, maximum+1):
        contexts = arrays[f'contexts_{order}']
        if not len(contexts):
            continue
        codes, valid = prefix_codes(x, order-1)
        pos = np.searchsorted(contexts, codes)
        safe = np.minimum(pos, len(contexts)-1)
        valid = valid & (pos < len(contexts)) & (contexts[safe] == codes)
        offsets = arrays[f'offsets_{order}']
        # Reconstituted exact keys are temporary scoring indices, not assets.
        keys = (np.repeat(contexts, np.diff(offsets)) << 11) | arrays[f'words_{order}']
        queries = (codes << 11) | target
        keypos = np.searchsorted(keys, queries)
        keysafe = np.minimum(keypos, len(keys)-1)
        found = (keypos < len(keys)) & (keys[keysafe] == queries)
        mass = np.where(found, arrays[f'masses_{order}'][keysafe], 0.)
        result = np.where(valid, result*arrays[f'backoffs_{order}'][safe]+mass, result).astype(np.float32)
    return result.reshape(y.shape)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tables-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device, _ = setup(args.device, 'fp32', 2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert checkpoint['implementation'] == 'lowrank', 'Require frozen neural-only refined lowrank parent.'
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    path = ROOT/'data/wikitext_validation.txt'
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    assert sha(path) == manifest['sha256'][path.name]
    raw = path.read_bytes()
    tokenizer = Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    tokens = torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids)
    xs, ys, neural = [], [], []
    for x, y in windows(tokens):
        n = model.predict_log_probs(x.to(device)).gather(-1, y.clamp_min(0).unsqueeze(-1).to(device)).squeeze(-1).exp().cpu()
        xs.append(x.numpy()); ys.append(y.numpy()); neural.append(n.numpy())
    x, y, n = np.concatenate(xs), np.concatenate(ys), np.concatenate(neural)
    valid = y != -100
    # Prespecified before validation: three scales and four mixture weights.
    scales, weights = (.7, 1., 1.3), (.10, .15, .20, .25)
    results = []
    for scale in scales:
        buffers = torch.load(args.tables_dir/f'tables-scale{scale:g}.pt', map_location='cpu', weights_only=True)
        arrays = {k: v.numpy() for k, v in buffers.items()}
        counts = target_probabilities(arrays, x, y)
        # Real-data full-distribution check covers 4 complete validation windows.
        tables = MKNTables({k: list(v.shape) for k, v in buffers.items()})
        tables.load_state_dict(buffers)
        direct = tables.probabilities(torch.from_numpy(x[:4])).gather(-1, torch.from_numpy(y[:4].clip(min=0)).unsqueeze(-1)).squeeze(-1).numpy()
        np.testing.assert_allclose(counts[:4], direct, atol=1e-7, rtol=1e-5)
        for weight in weights:
            mix = (1-weight)*n+weight*counts
            bpb = -np.log(mix[valid].astype(np.float64).clip(1e-300)).sum()/math.log(2)/len(raw)
            results.append(dict(discount_scale=scale, mkn_order=5, mkn_weight=weight, validation_bpb=float(bpb)))
    results.sort(key=lambda row: row['validation_bpb'])
    best = results[0]
    tables_path = args.tables_dir/f"tables-scale{best['discount_scale']:g}.pt"
    buffers = torch.load(tables_path, map_location='cpu', weights_only=True)
    checkpoint['implementation'] = 'push145_mkn'
    checkpoint['config'].update({k: v for k, v in best.items() if k != 'validation_bpb'})
    checkpoint['config']['mkn_shapes'] = {k: list(v.shape) for k, v in buffers.items()}
    checkpoint['model'].update({'ngram.'+k: v for k, v in buffers.items()})
    checkpoint['neural_parent_sha256'] = sha(args.checkpoint)
    checkpoint['mkn_train_tables_sha256'] = sha(tables_path)
    checkpoint['selection'] = '12 prespecified validation-only choices; train-derived MKN-style counts'
    torch.save(checkpoint, args.run_dir/'checkpoint.pt')
    report = dict(best=best, trials=results, seconds=time.perf_counter()-started,
                  checkpoint_bytes=(args.run_dir/'checkpoint.pt').stat().st_size,
                  checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'),
                  neural_parent_sha256=sha(args.checkpoint), tables_sha256=sha(tables_path),
                  validation_target_occurrences=int(valid.sum()), validation_bytes=len(raw),
                  script_sha256=sha(__file__), test_used=False, prior_test_exposure_disclosed=True,
                  neural_training_target_occurrences=0, shortcut_checked_against_full_distribution=True)
    (args.run_dir/'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'trials'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
