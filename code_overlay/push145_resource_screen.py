"""Baseline-bracketed full validation CPU FP32 screen; never loads test text."""
import argparse
import ctypes
from ctypes import wintypes
import gc
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from common import ROOT,setup,make_model,sha
from evaluate import score
from measure_validation import peak_memory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not overwrite earlier timing evidence')
    get_process=ctypes.windll.kernel32.GetCurrentProcess;get_process.restype=wintypes.HANDLE
    setter=ctypes.windll.kernel32.SetProcessAffinityMask;setter.argtypes=[wintypes.HANDLE,ctypes.c_size_t]
    if not setter(get_process(),15):
        raise ctypes.WinError()
    device,_=setup('cpu','fp32',4)
    def load(path):
        cp=torch.load(path,map_location='cpu',weights_only=True)
        model,_=make_model(cp['implementation'],cp['config'],device)
        model.load_state_dict(cp['model'])
        return model.eval()
    baseline=load(ROOT/'runs/baseline-s17/checkpoint.pt')
    path=ROOT/'data/wikitext_validation.txt'
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    assert sha(path)==manifest['sha256'][path.name]
    raw=path.read_bytes()
    tokens=torch.tensor(Tokenizer.from_file(str(ROOT/'data/tokenizer.json')).encode(raw.decode('utf-8')).ids)
    report=dict(scope='full validation only',test_used=False,device='cpu',precision='fp32',
        threads=4,affinity_mask=15,measurements=[],ratios=[],
        evaluator_sha256=sha(ROOT/'evaluate.py'),script_sha256=sha(__file__),
        baseline_sha256=sha(ROOT/'runs/baseline-s17/checkpoint.pt'),
        checkpoint_hashes={str(path):sha(path) for path in args.candidates})
    def measure(name,model):
        result=score(model,tokens,len(raw),device,'fp32');result.pop('window_nll_nats')
        row=dict(name=name,**result)
        report['measurements'].append(row)
        report['peak_process_working_set_bytes']=peak_memory()
        args.output.write_text(json.dumps(report,indent=2))
        print(json.dumps(row),flush=True)
        return row
    before=measure('baseline',baseline)
    for candidate_path in args.candidates:
        candidate=load(candidate_path)
        label=candidate_path.parent.name
        if candidate_path.name!='checkpoint.pt':
            label+=':'+candidate_path.stem
        result=measure(label,candidate)
        del candidate
        gc.collect()
        after=measure('baseline',baseline)
        report['ratios'].append(dict(name=result['name'],validation_bpb=result['bpb'],
            mean_bracket_ratio=result['seconds']/((before['seconds']+after['seconds'])/2),
            conservative_bracket_ratio=result['seconds']/min(before['seconds'],after['seconds'])))
        args.output.write_text(json.dumps(report,indent=2))
        before=after
    print(json.dumps(report['ratios'],indent=2),flush=True)


if __name__=='__main__':
    main()
