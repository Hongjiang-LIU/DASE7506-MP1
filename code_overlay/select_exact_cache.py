"""Validation-only selection of a window-local exact-prefix cache."""
import argparse
import itertools
import json
import math
from pathlib import Path
import time
import torch
from tokenizers import Tokenizer
from common import ROOT,setup,make_model,sha,windows
from exact_refined import exact_cache


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--run-dir',required=True,type=Path)
    args=p.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=False)
    device,_=setup('cuda','fp32',4)
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model,_=make_model(cp['implementation'],cp['config'],device)
    model.load_state_dict(cp['model']);model.eval()
    raw=(ROOT/'data/wikitext_validation.txt').read_bytes()
    assert sha(ROOT/'data/wikitext_validation.txt')==json.loads((ROOT/'data/manifest.json').read_text())['sha256']['wikitext_validation.txt']
    tokenizer=Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    tokens=torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids)
    grid=list(itertools.product((1.,1.05,1.1,1.15,1.2),(1,2,3,4,5),(0.,1.,3.),(.05,.1,.15,.2,.3,.4)))
    totals=[0.]*len(grid);started=time.perf_counter()
    for x,y in windows(tokens):
        x,y=x.to(device),y.to(device)
        features=model.features(x);logits=model.head(features).float()
        target=y.clamp_min(0).unsqueeze(-1);valid=y!=-100
        neural={t:(logits/t).softmax(-1).gather(-1,target).squeeze(-1) for t in (1.,1.05,1.1,1.15,1.2)}
        memories={}
        for h in (1,2,3,4,5):
            memory,evidence=exact_cache(x,h)
            memories[h]=(memory.gather(-1,target).squeeze(-1),evidence)
        for i,(t,h,s,w) in enumerate(grid):
            memory,evidence=memories[h]
            gate=(evidence>0).float() if s==0 else evidence/(evidence+s)
            mix=w*gate
            prob=(1-mix)*neural[t]+mix*memory
            totals[i]+=-prob[valid].double().clamp_min(1e-300).log().sum().item()
    rows=[dict(temperature=t,exact_history=h,exact_scale=s,cache_weight=w,validation_bpb=total/math.log(2)/len(raw))
          for (t,h,s,w),total in zip(grid,totals)]
    rows.sort(key=lambda x:x['validation_bpb']);best=rows[0]
    cp['implementation']='exact_refined'
    cp['config'].update({k:v for k,v in best.items() if k!='validation_bpb'})
    cp['config']['memory_kind']='exact'
    cp['exact_parent_sha256']=sha(args.checkpoint)
    torch.save(cp,args.run_dir/'checkpoint.pt')
    info=dict(best=best,trials=rows,seconds=time.perf_counter()-started,parent_sha256=sha(args.checkpoint),
              checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'),test_used=False)
    (args.run_dir/'metrics.json').write_text(json.dumps(info,indent=2))
    print(json.dumps({k:v for k,v in info.items() if k!='trials'},indent=2),flush=True)


if __name__=='__main__':
    main()
