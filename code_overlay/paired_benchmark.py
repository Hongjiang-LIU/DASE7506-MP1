"""Alternate baseline and candidate in one process to control CPU scheduling drift."""
import argparse
import ctypes
from ctypes import wintypes
import json
import statistics
from pathlib import Path
import torch
from tokenizers import Tokenizer
from common import ROOT,setup,make_model,sha
from evaluate import score
from measure_validation import peak_memory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True,type=Path)
    p.add_argument('--baseline',default=Path('runs/baseline-s17/checkpoint.pt'),type=Path)
    p.add_argument('--repeats',default=2,type=int)
    p.add_argument('--split',choices=['validation','test'],default='validation')
    args=p.parse_args()
    get_process=ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype=wintypes.HANDLE
    setter=ctypes.windll.kernel32.SetProcessAffinityMask
    setter.argtypes=[wintypes.HANDLE,ctypes.c_size_t]
    if not setter(get_process(),15):
        raise ctypes.WinError()
    device,_=setup('cpu','fp32',4)
    models={}
    source_hashes={}
    for name,path in [('baseline',args.baseline),('candidate',args.candidate)]:
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        model,implementation_sha=make_model(checkpoint['implementation'],checkpoint['config'],device)
        source_hashes[name]=implementation_sha
        model.load_state_dict(checkpoint['model'])
        models[name]=model.eval()
    raw=(ROOT/'data'/f'wikitext_{args.split}.txt').read_bytes()
    tokenizer=Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    tokens=torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids)
    measurements=[]
    for repeat in range(args.repeats):
        for name in (['baseline','candidate'] if repeat%2==0 else ['candidate','baseline']):
            measured=score(models[name],tokens,len(raw),device,'fp32')
            measured.pop('window_nll_nats')
            row=dict(model=name,repeat=repeat,**measured)
            measurements.append(row)
            print(json.dumps(row),flush=True)
    medians={name:statistics.median(m['seconds'] for m in measurements if m['model']==name)
             for name in models}
    result=dict(split=args.split,cpu_affinity_mask=15,threads=4,
                measurements=measurements,median_seconds=medians,
                cpu_time_ratio=medians['candidate']/medians['baseline'],
                conservative_time_ratio=max(m['seconds'] for m in measurements if m['model']=='candidate') /
                                        min(m['seconds'] for m in measurements if m['model']=='baseline'),
                peak_process_working_set_bytes=peak_memory(),
                candidate_sha256=sha(args.candidate),baseline_sha256=sha(args.baseline),
                scorer_sha256=sha(ROOT/'evaluate.py'), implementation_hashes=source_hashes)
    (args.candidate.parent/f'paired_{args.split}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
