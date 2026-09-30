"""Validation-only calibration; target lookups here are scoring, not inference."""
import argparse
import itertools
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from common import ROOT,make_model,setup,sha,windows
from ngram_hybrid import prefix_codes,reliability_gate


def target_probabilities(arrays, x, y, discount, maximum):
    # A scoring shortcut checked against full target-free inference in tests.
    target=y.ravel().clip(min=0)
    result=arrays['unigram'][target].copy()
    for order in range(2,maximum+1):
        codes,valid=prefix_codes(x,order-1)
        contexts=arrays[f'contexts_{order}']
        pos=np.searchsorted(contexts,codes)
        safe=np.minimum(pos,len(contexts)-1)
        valid &= (pos<len(contexts)) & (contexts[safe]==codes)
        offsets=arrays[f'offsets_{order}']
        totals=arrays[f'totals_{order}'][safe]
        lengths=offsets[safe+1]-offsets[safe]
        lam=np.float32(discount)*lengths.astype(np.float32)/totals
        query=(codes << 11)|target
        keys=arrays[f'keys_{order}']
        keypos=np.searchsorted(keys,query)
        keysafe=np.minimum(keypos,len(keys)-1)
        found=(keypos<len(keys)) & (keys[keysafe]==query)
        masses=np.where(found,(arrays[f'counts_{order}'][keysafe]-np.float32(discount))/totals,0.)
        result=np.where(valid,result*lam+masses,result).astype(np.float32)
    return result.reshape(y.shape)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--tables',required=True,type=Path)
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--orders',default='3,4,5')
    p.add_argument('--output-implementation',default='ngram_hybrid')
    p.add_argument('--gated',action='store_true')
    args=p.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    device,_=setup('cuda','fp32',4)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model,_=make_model(checkpoint['implementation'],checkpoint['config'],device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    buffers=torch.load(args.tables,map_location='cpu',weights_only=True)
    arrays={k:v.numpy() for k,v in buffers.items()}
    raw=(ROOT/'data/wikitext_validation.txt').read_bytes()
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    assert sha(ROOT/'data/wikitext_validation.txt')==manifest['sha256']['wikitext_validation.txt']
    tokenizer=Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    tokens=torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids)
    xs,ys,neural=[],[],[]
    for x,y in windows(tokens):
        n=model.predict_log_probs(x.to(device)).gather(-1,y.clamp_min(0).unsqueeze(-1).to(device)).squeeze(-1).exp().cpu()
        xs.append(x.numpy());ys.append(y.numpy());neural.append(n.numpy())
    x,y,n=np.concatenate(xs),np.concatenate(ys),np.concatenate(neural)
    valid=y!=-100
    results=[]
    for order,discount in itertools.product(tuple(map(int,args.orders.split(','))),(.25,.5,.75)):
        counts=target_probabilities(arrays,x,y,discount,order)
        for weight in (0.,.05,.1,.15,.2,.3,.4,.6,1.):
            mix=(1-weight)*n+weight*counts
            bpb=-np.log(mix[valid].astype(np.float64).clip(1e-300)).sum()/math.log(2)/len(raw)
            results.append(dict(ngram_order=order,ngram_discount=discount,ngram_weight=weight,validation_bpb=bpb))
        if args.gated:
            for minimum,scale in itertools.product((3,4,5),(0.,5.,20.)):
                if minimum>order:
                    continue
                gate=reliability_gate(arrays,x,order,minimum,scale)
                for weight in (.1,.2,.3,.4,.6):
                    mix=(1-weight*gate)*n+weight*gate*counts
                    bpb=-np.log(mix[valid].astype(np.float64).clip(1e-300)).sum()/math.log(2)/len(raw)
                    results.append(dict(ngram_order=order,ngram_discount=discount,ngram_weight=weight,
                                        ngram_gate_min_order=minimum,ngram_gate_scale=scale,validation_bpb=bpb))
    results.sort(key=lambda row:row['validation_bpb'])
    best=results[0]
    checkpoint['implementation']=args.output_implementation
    checkpoint['config'].update({k:v for k,v in best.items() if k!='validation_bpb'})
    checkpoint['config']['ngram_shapes']={k:list(v.shape) for k,v in buffers.items()}
    checkpoint['model'].update({'ngram.'+k:v for k,v in buffers.items()})
    checkpoint['neural_parent_sha256']=sha(args.checkpoint)
    checkpoint['ngram_train_tables_sha256']=sha(args.tables)
    checkpoint['selection']='validation-only train-derived ngram mixture'
    torch.save(checkpoint,args.run_dir/'checkpoint.pt')
    report=dict(best=best,trials=results,seconds=time.perf_counter()-started,
                checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'),
                checkpoint_bytes=(args.run_dir/'checkpoint.pt').stat().st_size,
                neural_parent_sha256=sha(args.checkpoint),tables_sha256=sha(args.tables),
                script_sha256=sha(__file__),test_used=False)
    (args.run_dir/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2),flush=True)


if __name__=='__main__':
    main()
