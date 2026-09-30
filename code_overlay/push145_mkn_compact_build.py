"""Exact integer repacking and precomputation, with no parameter search.

The old normalized absolute-discount D=.5 and mixture .15 remain fixed. Only
integer indices are made narrower; every stored probability remains FP32.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from common import ROOT, sha
from ngram_hybrid import NgramTables
from push145_mkn import MKNTables


def convert(buffers, discount=.5):
    arrays = {k: v.numpy() for k, v in buffers.items()}
    result = {'unigram': buffers['unigram'].clone()}
    for order in range(2, 6):
        offsets = arrays[f'offsets_{order}']
        lengths = np.diff(offsets)
        totals = arrays[f'totals_{order}']
        masses = (arrays[f'counts_{order}']-np.float32(discount))/np.repeat(totals, lengths)
        backoffs = np.float32(discount)*lengths.astype(np.float32)/totals
        for name, values in [('contexts', arrays[f'contexts_{order}'].copy()),
                             ('offsets', offsets.astype(np.int32)),
                             ('words', (arrays[f'keys_{order}'] & 2047).astype(np.int16)),
                             ('masses', masses), ('backoffs', backoffs)]:
            result[f'{name}_{order}'] = torch.from_numpy(values)
    return result


def compare_distributions(original, compact, ids):
    old = NgramTables({k: list(v.shape) for k, v in original.items()})
    old.load_state_dict(original)
    new = MKNTables({k: list(v.shape) for k, v in compact.items()})
    new.load_state_dict(compact)
    p, q = old.probabilities(ids, .5, 5), new.probabilities(ids, 5)
    torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)
    return dict(maximum_absolute_difference=float((p-q).abs().max()),
                bitwise_equal=bool(torch.equal(p, q)),
                normalization_max_error=float((q.sum(-1)-1).abs().max()),
                positions=ids.numel(), vocabulary=2048)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tables', type=Path, default=Path('runs/ngrams-full3/tables.pt'))
    p.add_argument('--lowrank', type=Path, default=Path('runs/refine-lowrank128-s71/checkpoint.pt'))
    p.add_argument('--dense', type=Path, default=Path('runs/refine-large-fine/checkpoint.pt'))
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.perf_counter()
    original = torch.load(args.tables, map_location='cpu', weights_only=True)
    compact = convert(original)
    torch.save(compact, args.run_dir/'tables.pt')
    # Train snippets (not held-out scores) test all 2048 predicted probabilities.
    path = ROOT/'data/wikitext_train.txt'
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    assert sha(path) == manifest['sha256'][path.name]
    tokenizer = Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    train_ids = np.asarray(tokenizer.encode(path.read_bytes().decode('utf-8')).ids)
    starts = np.random.default_rng(145).integers(0, len(train_ids)-256, size=16)
    x = torch.from_numpy(np.stack([train_ids[start:start+256] for start in starts])).long()
    comparison = compare_distributions(original, compact, x)
    candidates = []
    for parent, implementation, name in [(args.lowrank, 'push145_mkn_compact', 'lowrank'),
                                          (args.dense, 'push145_mkn_dense', 'dense')]:
        checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
        assert checkpoint['implementation'] in ('lowrank', 'refined')
        checkpoint['implementation'] = implementation
        checkpoint['config'].update(mkn_weight=.15, mkn_order=5,
                                     mkn_shapes={k: list(v.shape) for k, v in compact.items()})
        checkpoint['model'].update({'ngram.'+k: v for k, v in compact.items()})
        checkpoint['neural_parent_sha256'] = sha(parent)
        checkpoint['ngram_train_tables_sha256'] = sha(args.tables)
        checkpoint['selection'] = 'No new grid: fixed old D=.5 and mixture .15; compact exact integer encodings'
        destination = args.run_dir/f'{name}-checkpoint.pt'
        torch.save(checkpoint, destination)
        candidates.append(dict(name=name, implementation=implementation,
                               checkpoint_path=str(destination), checkpoint_sha256=sha(destination),
                               checkpoint_bytes=destination.stat().st_size, neural_parent_sha256=sha(parent),
                               newly_scored=False))
    report = dict(algorithm='Old normalized absolute discount .5, mixture .15: exact integer repacking and FP32 precomputation',
                  new_grid_trials=0, train_only_distribution_comparison=comparison,
                  neural_training_target_occurrences=0, count_reused_from=str(args.tables),
                  source_tables_sha256=sha(args.tables), tables_sha256=sha(args.run_dir/'tables.pt'),
                  tables_bytes=(args.run_dir/'tables.pt').stat().st_size, candidates=candidates,
                  seconds=time.perf_counter()-started, test_used=False, prior_test_exposure_disclosed=True,
                  new_hypothesis='Dense parent may now fit asset limit; timing and scores NOT YET VERIFIED',
                  script_sha256=sha(__file__))
    (args.run_dir/'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
