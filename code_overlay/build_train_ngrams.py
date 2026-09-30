"""Fit exact-key pruned continuation-count tables ONLY on supplied train text."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from common import ROOT, sha


def build_tables(ids, minimum=2, maximum=5, trigram_minimum=None):
    # Five 11-bit token IDs fit losslessly in int64 (55 bits).
    keys, counts = {}, {}
    for order in range(1, maximum+1):
        packed = np.zeros(len(ids)-order+1,dtype=np.int64)
        for offset in range(order):
            packed = (packed << 11) | ids[offset:offset+len(packed)]
        keys[order], counts[order] = np.unique(packed,return_counts=True)
    # Lower orders use distinct predecessor counts; highest order raw counts.
    effective = {}
    for order in range(1, maximum):
        effective[order] = np.unique(keys[order+1] & ((1 << (11*order))-1),return_counts=True)
    effective[maximum] = keys[maximum],counts[maximum]
    unigram = np.full(2048,.01,dtype=np.float32)
    unigram[effective[1][0]] += effective[1][1].astype(np.float32)
    unigram /= unigram.sum()
    arrays = {'unigram':torch.from_numpy(unigram)}
    stats = []
    for order in range(2,maximum+1):
        nkeys,ncounts = effective[order]
        threshold=1 if order==2 else (trigram_minimum if order==3 and trigram_minimum is not None else minimum)
        keep = ncounts >= threshold
        nkeys,ncounts = nkeys[keep],ncounts[keep].astype(np.float32)
        contexts,starts = np.unique(nkeys >> 11,return_index=True)
        offsets = np.concatenate([starts,np.array([len(nkeys)],dtype=np.int64)])
        totals = np.add.reduceat(ncounts,starts)
        for name,value in [('keys',nkeys),('counts',ncounts),('contexts',contexts),('offsets',offsets),('totals',totals)]:
            arrays[f'{name}_{order}'] = torch.from_numpy(value)
        stats.append(dict(order=order,entries=len(nkeys),contexts=len(contexts),minimum=threshold))
    return arrays, stats


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--minimum',default=2,type=int)
    p.add_argument('--maximum',default=5,type=int)
    p.add_argument('--long-minimum',default=3,type=int)
    p.add_argument('--trigram-minimum',default=None,type=int)
    args=p.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    path=ROOT/'data/wikitext_train.txt'
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    assert sha(path)==manifest['sha256'][path.name]
    raw=path.read_bytes()
    tokenizer=Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    ids=np.asarray(tokenizer.encode(raw.decode('utf-8')).ids,dtype=np.int64)
    arrays,stats=build_tables(ids,args.minimum,trigram_minimum=args.trigram_minimum)
    for order in range(6,args.maximum+1):
        size=len(ids)-order+1
        contexts=np.zeros(size,dtype=np.int64)
        for offset in range(order-1):
            contexts=(contexts*1000003+ids[offset:offset+size]+1)&((1<<52)-1)
        packed=(contexts<<11)|ids[order-1:order-1+size]
        nkeys,ncounts=np.unique(packed,return_counts=True)
        keep=ncounts>=args.long_minimum
        nkeys,ncounts=nkeys[keep],ncounts[keep].astype(np.float32)
        contexts,starts=np.unique(nkeys>>11,return_index=True)
        offsets=np.concatenate([starts,np.array([len(nkeys)],dtype=np.int64)])
        totals=np.add.reduceat(ncounts,starts)
        for name,value in [('keys',nkeys),('counts',ncounts),('contexts',contexts),('offsets',offsets),('totals',totals)]:
            arrays[f'{name}_{order}']=torch.from_numpy(value)
        stats.append(dict(order=order,entries=len(nkeys),contexts=len(contexts),minimum=args.long_minimum))
    torch.save(arrays,args.run_dir/'tables.pt')
    info=dict(source='supplied train text ONLY',train_text_sha256=sha(path),train_tokens=len(ids),
              training_target_occurrences=len(ids)-1,minimum=args.minimum,maximum=args.maximum,long_minimum=args.long_minimum,trigram_minimum=args.trigram_minimum,stats=stats,
              algorithm='Continuation counts, highest order raw counts; count pruning followed by recomputed normalized absolute-discount interpolation',
              bytes=(args.run_dir/'tables.pt').stat().st_size,seconds=time.perf_counter()-started,
              script_sha256=sha(__file__),tables_sha256=sha(args.run_dir/'tables.pt'))
    (args.run_dir/'metrics.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':
    main()
