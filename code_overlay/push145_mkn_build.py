"""Build compact modified-Kneser-Ney-style tables, using TRAIN text only.

Counts below maximum order count distinct predecessor types. Each order's three
discounts are estimated on its unpruned count-of-counts. Removed entries back off
their entire mass: denominators are NOT renormalized to surviving counts.
This explicitly described pruned model is not claimed identical to any toolkit.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from common import ROOT, sha


def discounts(counts):
    ns = np.asarray([(counts == i).sum() for i in (1, 2, 3, 4)], dtype=np.float64)
    if ns[0] == 0 or ns[1] == 0:
        return np.asarray([.5, .75, 1.], dtype=np.float64), ns
    y = ns[0] / (ns[0] + 2*ns[1])
    values = [1-2*y*ns[1]/ns[0],
              2-3*y*ns[2]/ns[1] if ns[2] else .75,
              3-4*y*ns[3]/ns[2] if ns[2] and ns[3] else 1.]
    return np.clip(values, .01, [0.99, 1.99, 2.99]), ns


def effective_counts(ids, maximum=5):
    if not 2 <= maximum <= 5:
        raise ValueError('Lossless 64-bit encodings support maximum order 2..5.')
    keys, counts = {}, {}
    for order in range(1, maximum+1):
        packed = np.zeros(len(ids)-order+1, dtype=np.int64)
        for offset in range(order):
            packed = (packed << 11) | ids[offset:offset+len(packed)]
        keys[order], counts[order] = np.unique(packed, return_counts=True)
    effective = {}
    for order in range(1, maximum):
        effective[order] = np.unique(keys[order+1] & ((1 << (11*order))-1), return_counts=True)
    effective[maximum] = keys[maximum], counts[maximum]
    return effective


def make_tables(effective, scale=1., minimum=2, trigram_minimum=1):
    unigram = np.full(2048, .01, dtype=np.float32)
    unigram[effective[1][0]] += effective[1][1].astype(np.float32)
    unigram /= unigram.sum()
    arrays, stats = {'unigram': torch.from_numpy(unigram)}, []
    for order in range(2, max(effective)+1):
        keys, counts = effective[order]
        base, count_of_counts = discounts(counts)
        chosen = np.clip(base*scale, .01, [0.99, 1.99, 2.99])
        contexts_all, starts_all = np.unique(keys >> 11, return_index=True)
        totals_all = np.add.reduceat(counts.astype(np.float64), starts_all)
        threshold = 1 if order == 2 else trigram_minimum if order == 3 else minimum
        keep = counts >= threshold
        selected_keys, selected_counts = keys[keep], counts[keep]
        contexts, starts = np.unique(selected_keys >> 11, return_index=True)
        offsets = np.r_[starts, len(selected_keys)].astype(np.int32)
        lengths = np.diff(offsets)
        totals = totals_all[np.searchsorted(contexts_all, contexts)]
        per_item_discount = chosen[np.minimum(selected_counts, 3)-1]
        masses = ((selected_counts-per_item_discount)/np.repeat(totals, lengths)).astype(np.float32)
        # Summation happens in FP64 at build time; stored/inference values FP32.
        backoffs = (1-np.add.reduceat(masses.astype(np.float64), starts)).astype(np.float32) if len(starts) else np.zeros(0, np.float32)
        if np.any(backoffs < -1e-6) or np.any(backoffs > 1+1e-6):
            raise AssertionError('Invalid backoff mass')
        backoffs = np.clip(backoffs, 0, 1)
        for name, values in [('contexts', contexts), ('offsets', offsets),
                             ('words', (selected_keys & 2047).astype(np.int16)),
                             ('masses', masses), ('backoffs', backoffs)]:
            arrays[f'{name}_{order}'] = torch.from_numpy(values)
        stats.append(dict(order=order, original_entries=len(keys), entries=len(selected_keys),
                          contexts=len(contexts), minimum=threshold,
                          count_of_counts=count_of_counts.tolist(), base_discounts=base.tolist(),
                          applied_discounts=chosen.tolist(), retained_raw_mass=float(selected_counts.sum()/counts.sum())))
    return arrays, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--scales', default='0.7,1.0,1.3')
    args = p.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.perf_counter()
    path = ROOT/'data/wikitext_train.txt'
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    assert sha(path) == manifest['sha256'][path.name]
    tokenizer = Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    ids = np.asarray(tokenizer.encode(path.read_bytes().decode('utf-8')).ids, dtype=np.int64)
    effective = effective_counts(ids)
    variants = []
    for scale in map(float, args.scales.split(',')):
        arrays, stats = make_tables(effective, scale)
        destination = args.run_dir/f'tables-scale{scale:g}.pt'
        torch.save(arrays, destination)
        variants.append(dict(scale=scale, path=str(destination), bytes=destination.stat().st_size,
                             sha256=sha(destination), stats=stats))
    report = dict(source='supplied training text ONLY', train_tokens=len(ids),
                  train_text_sha256=sha(path), tokenizer_sha256=sha(ROOT/'data/tokenizer.json'),
                  neural_training_target_occurrences=0, count_scan_token_occurrences=len(ids),
                  variants=variants, seconds=time.perf_counter()-started,
                  test_used=False, prior_test_exposure_disclosed=True, script_sha256=sha(__file__))
    (args.run_dir/'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
