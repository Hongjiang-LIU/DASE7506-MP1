"""TRAIN-prefix timing probe, no fitting and no validation/test selection."""
import argparse
import ctypes
from ctypes import wintypes
import gc
import json
import statistics
import time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from common import ROOT,setup,make_model,sha
from push145_cpu_fast import FastLearnedCache


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--run-dir',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier probe')
    args.run_dir.mkdir(parents=True,exist_ok=False)
    process=ctypes.windll.kernel32.GetCurrentProcess;process.restype=wintypes.HANDLE
    setter=ctypes.windll.kernel32.SetProcessAffinityMask;setter.argtypes=[wintypes.HANDLE,ctypes.c_size_t]
    if not setter(process(),15):
        raise ctypes.WinError()
    device,_=setup('cpu','fp32',4)
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    original,_=make_model(cp['implementation'],cp['config'],device)
    original.load_state_dict(cp['model']);original.eval()
    tokenizer=Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    path=ROOT/'data/wikitext_train.txt'
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    assert sha(path)==manifest['sha256'][path.name]
    # Prefix timing inputs only; no losses or target words are used.
    tokens=tokenizer.encode(path.read_bytes().decode('utf-8')[:50000]).ids
    ids=torch.tensor(tokens[:8192]).reshape(32,256)
    expected=original.predict_log_probs(ids)
    rows=[]
    for size in (32,16,8):
        config=dict(cp['config'],cpu_feature_microbatch=size)
        fast=FastLearnedCache(config).eval()
        fast.load_state_dict(cp['model'])
        actual=fast.predict_log_probs(ids)
        max_logp_difference=(actual-expected).abs().max().item()
        torch.testing.assert_close(actual,expected,atol=3e-5,rtol=2e-6)
        assert actual.dtype==torch.float32
        assert actual.logsumexp(-1).abs().max().item()<1e-3
        timings=[]
        for repeat in range(3):
            pair={}
            for name,model in ([('original',original),('optimized',fast)] if repeat%2==0 else [('optimized',fast),('original',original)]):
                started=time.perf_counter()
                for _ in range(2):
                    model.predict_log_probs(ids)
                pair[name]=(time.perf_counter()-started)/2
            timings.append(dict(repeat=repeat,**pair))
        row=dict(feature_microbatch=size,max_logp_difference=max_logp_difference,timings=timings,
            median_optimized_seconds=statistics.median(r['optimized'] for r in timings),
            median_original_seconds=statistics.median(r['original'] for r in timings))
        rows.append(row)
        print(json.dumps(row),flush=True)
        del fast,actual
        gc.collect()
    winner=min(rows,key=lambda row:row['median_optimized_seconds']/row['median_original_seconds'])
    cp['implementation']='push145_cpu_fast'
    cp['config']['cpu_feature_microbatch']=winner['feature_microbatch']
    cp['runtime_parent_sha256']=sha(args.checkpoint)
    cp['selection']='Same weights; runtime variant selected on train-prefix paired speed only, not held-out quality'
    torch.save(cp,args.run_dir/'checkpoint.pt')
    report=dict(rows=rows,winner=winner,checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'),
        source_checkpoint_sha256=sha(args.checkpoint),test_used=False,validation_used=False,
        training_updates=0,scope='one 32x256 training-prefix batch, three paired repeats; not full evaluation evidence',
        implementation_sha256=sha(ROOT/'push145_cpu_fast.py'),script_sha256=sha(__file__))
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(winner=winner)),flush=True)


if __name__=='__main__':
    main()
