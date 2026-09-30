"""Initialize two cache scalars at exact old-predictor behavior; no data files."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from common import make_model,sha
from push147_cache import RecencyLearnedCache


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True,type=Path)
    p.add_argument('--run-dir',required=True,type=Path)
    args=p.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(149)
    parent=torch.load(args.parent,map_location='cpu',weights_only=True)
    original,_=make_model(parent['implementation'],parent['config'],torch.device('cpu'))
    original.load_state_dict(parent['model']);original.eval()
    candidate=RecencyLearnedCache(copy.deepcopy(parent['config'])).eval()
    candidate.load_parent_state_dict(parent['model'])
    ids=torch.randint(2048,(2,19))
    reference=original.predict_log_probs(ids)
    actual=candidate.predict_log_probs(ids)
    torch.testing.assert_close(reference,actual,atol=0,rtol=0)
    state={k:v.detach().cpu() for k,v in candidate.state_dict().items()}
    state['head.weight']=state['token.weight']
    result=copy.deepcopy({k:v for k,v in parent.items() if k!='model'})
    result.update(implementation='push147_cache',model=state,config=copy.deepcopy(candidate.config),
        parent_sha256=sha(args.parent),selection='Equivalent two-scalar initialization; no data or held-out scoring')
    torch.save(result,args.run_dir/'checkpoint.pt')
    theta,alpha=candidate.cache_scales()
    report=dict(parent_sha256=sha(args.parent),checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'),
        train_targets=parent['train_tokens'],new_training_targets=0,
        parameter_count=sum(p.numel() for p in candidate.parameters()),
        scalar_theta=float(theta),scalar_recency=float(alpha),synthetic_prediction_max_difference=float((actual-reference).abs().max()),
        data_text_loaded=False,test_used=False,
        all_parent_weights_equal=all(torch.equal(v,state[k]) for k,v in parent['model'].items()),
        script_sha256=sha(__file__),seconds=time.perf_counter()-started)
    assert report['all_parent_weights_equal']
    (args.run_dir/'initialization.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
