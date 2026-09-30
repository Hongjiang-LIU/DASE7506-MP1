"""Audit actual candidate weights and full validation, excluding test entirely."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys
import torch
from tokenizers import Tokenizer
from common import ROOT,setup,make_model,sha
from evaluate import score
from measure_validation import peak_memory


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('Retain previous audit output')
    process=ctypes.windll.kernel32.GetCurrentProcess;process.restype=wintypes.HANDLE
    setter=ctypes.windll.kernel32.SetProcessAffinityMask;setter.argtypes=[wintypes.HANDLE,ctypes.c_size_t]
    if not setter(process(),15):
        raise ctypes.WinError()
    device,_=setup('cpu','fp32',4)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model,_=make_model(checkpoint['implementation'],checkpoint['config'],device)
    model.load_state_dict(checkpoint['model']);model.eval()
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
    fixed=['common.py','evaluate.py','model.py','tests/test_contract.py','data/manifest.json',
           'data/tokenizer.json','data/wikitext_train.txt','data/wikitext_validation.txt']
    checks={name:sha(ROOT/name)==manifest['code/'+name] for name in fixed}
    assert all(checks.values())
    tokenizer=Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    data={}
    for split in ('train','validation'):
        raw=(ROOT/'data'/f'wikitext_{split}.txt').read_bytes()
        ids=tokenizer.encode(raw.decode('utf-8')).ids
        data[split]=(torch.tensor(ids),len(raw))
    torch.manual_seed(127)
    inputs=[torch.randint(2048,(2,256)),data['train'][0][:512].reshape(2,256)]
    for x in inputs:
        original=model.predict_log_probs(x)
        assert torch.isfinite(original).all()
        assert torch.logsumexp(original,dim=-1).abs().max().item()<1e-3
        changed=x.clone();changed[:,128:]=torch.randint(2048,(2,128))
        torch.testing.assert_close(original[:,:128],model.predict_log_probs(changed)[:,:128],atol=4e-6,rtol=1e-6)
        other=x.clone();other[1]=0
        torch.testing.assert_close(original[0],model.predict_log_probs(other)[0],atol=4e-6,rtol=1e-6)
        model.predict_log_probs(torch.randint(2048,(2,256)))
        torch.testing.assert_close(original,model.predict_log_probs(x),atol=0,rtol=0)
        short=model.predict_log_probs(x[:,:19])
        torch.testing.assert_close(original[:,:19],short,atol=5e-6,rtol=1e-6)
    del original,changed,other,short,inputs,x
    measured=score(model,*data['validation'],device,'fp32');measured.pop('window_nll_nats')
    # Include all locally imported implementation dependencies, even audit helpers.
    local_files=set()
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.parent==ROOT and path.suffix=='.py' and path.is_file():
                local_files.add(path)
    local_files.update([ROOT/'data/tokenizer.json',ROOT/'data/manifest.json'])
    assets={str(path.relative_to(ROOT)):dict(bytes=path.stat().st_size,sha256=sha(path))
            for path in sorted(local_files)}
    assets['checkpoint.pt']=dict(bytes=args.checkpoint.stat().st_size,sha256=sha(args.checkpoint))
    total=sum(a['bytes'] for a in assets.values())
    result=dict(status='Validation-only candidate audit; test not loaded',measurement=measured,
        contract_checks=dict(future_invariance=True,example_independence=True,window_reset=True,
                             short_prefix_equivalence=True,normalized=True,finite=True),
        checks_on=['random tokens','supplied train text prefixes'],fixed_file_checks=checks,
        assets=assets,inference_assets_bytes=total,inference_assets_mib=total/2**20,
        peak_process_working_set_bytes=peak_memory(),peak_ram_gib=peak_memory()/2**30,
        memory_scope='Actual CPU validation scorer after train+validation loading, including audit; test not loaded',
        train_targets_on_checkpoint=checkpoint['train_tokens'],parameters=sum(p.numel() for p in model.parameters()),
        test_used=False,evaluation_network_access=False,
        asset_limit_pass=total<=64*2**20,ram_limit_pass=peak_memory()<=4*2**30)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    assert result['asset_limit_pass'] and result['ram_limit_pass']


if __name__=='__main__':
    main()
