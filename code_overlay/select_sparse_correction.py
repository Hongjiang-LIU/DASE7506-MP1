"""Validation-only sparse-count correction selection; inference gets no targets."""
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
from sparse_correction import corrected_targets


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--tables',required=True,type=Path)
    p.add_argument('--run-dir',required=True,type=Path)
    args=p.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=False)
    device,_=setup('cuda','fp32',4);started=time.perf_counter()
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model,_=make_model(cp['implementation'],cp['config'],device)
    model.load_state_dict(cp['model']);model.eval()
    buffers=torch.load(args.tables,map_location='cpu',weights_only=True)
    arrays={k:v.numpy() for k,v in buffers.items()}
    raw=(ROOT/'data/wikitext_validation.txt').read_bytes()
    assert sha(ROOT/'data/wikitext_validation.txt')==json.loads((ROOT/'data/manifest.json').read_text())['sha256']['wikitext_validation.txt']
    tokenizer=Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    tokens=torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids)
    xs,ys,ns=[],[],[]
    for x,y in windows(tokens):
        prob=model.predict_log_probs(x.to(device)).gather(-1,y.clamp_min(0).unsqueeze(-1).to(device)).squeeze(-1).exp().cpu().numpy()
        xs.append(x.numpy());ys.append(y.numpy());ns.append(prob)
    x,y,n=map(np.concatenate,(xs,ys,ns));valid=y!=-100;rows=[]
    for minimum,strength,scale in itertools.product((2,3,4),(.025,.05,.1,.15,.2,.3),(1.,5.,20.,100.)):
        probabilities=corrected_targets(arrays,x,y,n,minimum,5,strength,scale)
        bpb=-np.log(probabilities[valid].astype(np.float64).clip(1e-300)).sum()/math.log(2)/len(raw)
        rows.append(dict(correction_min=minimum,correction_max=5,correction_strength=strength,correction_scale=scale,validation_bpb=bpb))
    rows.sort(key=lambda r:r['validation_bpb']);best=rows[0]
    cp['implementation']='sparse_correction'
    cp['config'].update({k:v for k,v in best.items() if k!='validation_bpb'})
    cp['config']['ngram_shapes']={k:list(v.shape) for k,v in buffers.items()}
    cp['model'].update({'ngram.'+k:v for k,v in buffers.items()})
    cp['neural_parent_sha256']=sha(args.checkpoint);cp['ngram_train_tables_sha256']=sha(args.tables)
    torch.save(cp,args.run_dir/'checkpoint.pt')
    report=dict(best=best,trials=rows,seconds=time.perf_counter()-started,
                checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'),checkpoint_bytes=(args.run_dir/'checkpoint.pt').stat().st_size,
                neural_parent_sha256=sha(args.checkpoint),tables_sha256=sha(args.tables),test_used=False)
    (args.run_dir/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2),flush=True)


if __name__=='__main__':
    main()
