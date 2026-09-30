"""Nine prespecified validation settings; target-free full-scorer confirmation."""
import argparse
import itertools
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from common import ROOT,setup,make_model,sha,windows
from evaluate import score
from push145_mkn_select import target_probabilities


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--run-dir',required=True,type=Path)
    args=p.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    device,_=setup('cuda','fp32',4)
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model,_=make_model(cp['implementation'],cp['config'],device)
    model.load_state_dict(cp['model']);model.eval()
    path=ROOT/'data/wikitext_validation.txt'
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    assert sha(path)==manifest['sha256'][path.name]
    raw=path.read_bytes()
    tokens=torch.tensor(Tokenizer.from_file(str(ROOT/'data/tokenizer.json')).encode(raw.decode('utf-8')).ids)
    temperatures=(1.10,1.15,1.20)
    weights=(.10,.15,.20)
    xs,ys=[],[]
    values={temperature:[] for temperature in temperatures}
    for x,y in windows(tokens):
        ids=x.to(device)
        features=model.features(ids)
        logits=model.head(features).float()
        attention,gate=model.cache_attention_and_gate(features)
        word_ids=torch.cat([ids[:,1:],ids.new_zeros(ids.shape[0],1)],1)
        memory=torch.zeros_like(logits)
        memory.scatter_add_(-1,word_ids[:,None,:].expand(-1,ids.shape[1],-1),attention)
        target=y.clamp_min(0).to(device)[...,None]
        m=memory.gather(-1,target).squeeze(-1)
        g=gate.squeeze(-1)
        for temperature in temperatures:
            n=(logits/temperature).softmax(-1).gather(-1,target).squeeze(-1)
            values[temperature].append((n*(1-g)+m*g).cpu().numpy())
        xs.append(x.numpy());ys.append(y.numpy())
    x,y=np.concatenate(xs),np.concatenate(ys)
    valid=y!=-100
    arrays={k.removeprefix('ngram.'):v.cpu().numpy() for k,v in cp['model'].items() if k.startswith('ngram.')}
    counts=target_probabilities(arrays,x,y,maximum=5)
    trials=[]
    for temperature,weight in itertools.product(temperatures,weights):
        neural=np.concatenate(values[temperature])
        mix=(1-weight)*neural+weight*counts
        bpb=-np.log(mix[valid].astype(np.float64).clip(1e-300)).sum()/math.log(2)/len(raw)
        trials.append(dict(temperature=temperature,mkn_weight=weight,validation_bpb=float(bpb)))
    trials.sort(key=lambda row:row['validation_bpb'])
    best=trials[0]
    cp['config'].update(temperature=best['temperature'],mkn_weight=best['mkn_weight'])
    cp['calibration_parent_sha256']=sha(args.checkpoint)
    cp['selection']='Train-only learned pointer; nine prespecified validation-only temperature/mixture settings'
    torch.save(cp,args.run_dir/'checkpoint.pt')
    # Confirm selection with the actual normalized full-vocabulary predictor.
    model.config.update(temperature=best['temperature'],mkn_weight=best['mkn_weight'])
    measured=score(model,tokens,len(raw),device,'fp32');measured.pop('window_nll_nats')
    assert abs(measured['bpb']-best['validation_bpb'])<1e-6
    result=dict(best=best,trials=trials,full_validation=measured,device=str(device),precision='fp32',
        checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'),parent_sha256=sha(args.checkpoint),
        new_training_targets=0,test_used=False,prior_test_exposure_disclosed=True,
        script_sha256=sha(__file__),evaluator_sha256=sha(ROOT/'evaluate.py'),
        seconds_total=time.perf_counter()-started)
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
