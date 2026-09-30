"""Attach fixed previously selected train tables; evaluate validation only."""
import argparse
import json
import time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from common import ROOT,make_model,setup,sha
from evaluate import score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--neural',type=Path,required=True)
    p.add_argument('--statistics',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    neural=torch.load(args.neural,map_location='cpu',weights_only=True)
    statistic=torch.load(args.statistics,map_location='cpu',weights_only=True)
    config=dict(neural['config'])
    config.update({k:v for k,v in statistic['config'].items() if k.startswith(('ngram_','mkn_')) or k=='discount_scale'})
    for key in ['ngram_shapes','mkn_shapes']:
        if key not in statistic['config']:
            config.pop(key,None)
    if neural['implementation']=='push145_cache':
        implementation='push145_cache_hybrid'
    elif neural['implementation']=='lowrank':
        implementation='push145_mkn' if config.get('mkn_shapes') else 'lowrank_hybrid'
    else:
        raise ValueError('Unsupported neural parent; explicitly review new architecture.')
    neural['implementation']=implementation
    neural['config']=config
    neural['model'].update({k:v for k,v in statistic['model'].items() if k.startswith('ngram.')})
    neural['neural_parent_sha256']=sha(args.neural)
    neural['statistics_parent_sha256']=sha(args.statistics)
    neural['selection']='Frozen train-derived statistical recipe plus validation-selected neural parent; no test'
    torch.save(neural,args.run_dir/'checkpoint.pt')
    device,_=setup(args.device,'fp32',4)
    model,implementation_sha=make_model(implementation,config,device)
    model.load_state_dict(neural['model']);model.eval()
    path=ROOT/'data/wikitext_validation.txt'
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    assert sha(path)==manifest['sha256'][path.name]
    raw=path.read_bytes()
    tokenizer=Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    tokens=torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids)
    measured=score(model,tokens,len(raw),device,'fp32')
    measured.pop('window_nll_nats')
    result=dict(measurement=measured,split='validation',device=str(device),precision='fp32',
        checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'),implementation_sha256=implementation_sha,
        checkpoint_bytes=(args.run_dir/'checkpoint.pt').stat().st_size,
        neural_parent_sha256=sha(args.neural),statistics_parent_sha256=sha(args.statistics),
        train_targets=neural['train_tokens'],new_training_targets=0,test_used=False,
        script_sha256=sha(__file__),evaluator_sha256=sha(ROOT/'evaluate.py'),
        seconds_total=time.perf_counter()-started)
    (args.run_dir/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
