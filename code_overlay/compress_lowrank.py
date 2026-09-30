"""Compress only train-derived model weights with truncated SVD; no data read."""
import argparse
import json
import time
from pathlib import Path
import torch
from common import sha
from lowrank import LowRankGPT,LowRankLinear


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--rank',default=96,type=int)
    args=p.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    started=time.perf_counter()
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    config=dict(checkpoint['config'],lowrank_rank=args.rank)
    model=LowRankGPT(config)
    original=checkpoint['model']
    state=model.state_dict()
    for name in state:
        if name in original:
            state[name]=original[name]
    errors=[]
    for name,module in model.named_modules():
        if isinstance(module,LowRankLinear):
            weight=original[name+'.weight']
            u,s,vh=torch.linalg.svd(weight,full_matrices=False)
            root=s[:args.rank].sqrt()
            state[name+'.left.weight']=vh[:args.rank]*root[:,None]
            state[name+'.right.weight']=u[:,:args.rank]*root[None,:]
            errors.append(dict(layer=name,discarded_energy_fraction=(s[args.rank:].square().sum()/s.square().sum()).item()))
    model.load_state_dict(state)
    state['head.weight']=state['token.weight']
    checkpoint.update(config=config,model=state,implementation='lowrank',svd_parent_sha256=sha(args.checkpoint))
    torch.save(checkpoint,args.run_dir/'checkpoint.pt')
    (args.run_dir/'config.json').write_text(json.dumps(config,indent=2))
    info=dict(parent_sha256=sha(args.checkpoint),checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'),
              parent_training_targets=checkpoint['train_tokens'],rank=args.rank,errors=errors,
              parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-started)
    (args.run_dir/'compression.json').write_text(json.dumps(info,indent=2))
    print(json.dumps({k:v for k,v in info.items() if k!='errors'},indent=2),flush=True)


if __name__=='__main__':
    main()
