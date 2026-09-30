"""Audit selected validation-only candidate, including train+validation loader RAM."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from common import ROOT,setup,make_model,sha
from evaluate import score
from measure_validation import peak_memory


@torch.no_grad()
def main():
    process=ctypes.windll.kernel32.GetCurrentProcess;process.restype=wintypes.HANDLE
    setter=ctypes.windll.kernel32.SetProcessAffinityMask;setter.argtypes=[wintypes.HANDLE,ctypes.c_size_t]
    if not setter(process(),15):
        raise ctypes.WinError()
    device,_=setup('cpu','fp32',4)
    folder=ROOT/'runs/hybrid-lowrank128-full3'
    cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    model,_=make_model(cp['implementation'],cp['config'],device)
    model.load_state_dict(cp['model']);model.eval()
    torch.manual_seed(97)
    x=torch.randint(2048,(2,256))
    original=model.predict_log_probs(x)
    altered=x.clone();altered[:,128:]=torch.randint(2048,(2,128))
    future=model.predict_log_probs(altered)
    torch.testing.assert_close(original[:,:128],future[:,:128],atol=3e-6,rtol=1e-6)
    independent=x.clone();independent[1]=0
    other=model.predict_log_probs(independent)
    torch.testing.assert_close(original[0],other[0],atol=3e-6,rtol=1e-6)
    model.predict_log_probs(torch.randint(2048,(2,256)))
    again=model.predict_log_probs(x)
    torch.testing.assert_close(original,again,atol=0,rtol=0)
    assert torch.logsumexp(original,dim=-1).abs().max().item()<1e-3
    del original,future,other,again,altered,independent,x
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
    measured=score(model,*data['validation'],device,'fp32');measured.pop('window_nll_nats')
    sources=['lowrank_hybrid.py','lowrank.py','efficient.py','refined.py','student.py','ngram_hybrid.py',
             'exact_refined.py','fast_refined.py','common.py','evaluate.py','data/tokenizer.json','data/manifest.json']
    assets={name:dict(bytes=(ROOT/name).stat().st_size,sha256=sha(ROOT/name)) for name in sources}
    assets['checkpoint.pt']=dict(bytes=(folder/'checkpoint.pt').stat().st_size,sha256=sha(folder/'checkpoint.pt'))
    size=sum(item['bytes'] for item in assets.values())
    output=dict(status='Validation-only candidate; no test evaluation',measurement=measured,
                contract_checks=dict(future_invariance=True,example_independence=True,window_reset=True,normalized=True),
                fixed_file_checks=checks,assets=assets,inference_assets_bytes=size,inference_assets_mib=size/2**20,
                peak_process_working_set_bytes=peak_memory(),peak_ram_gib=peak_memory()/2**30,
                memory_scope='Actual CPU scorer after loading train and validation. Test text was not loaded; this is not a full-test resource claim.',
                train_targets_on_checkpoint=cp['train_tokens'],parameters=sum(p.numel() for p in model.parameters()),
                evaluation_network_access=False,test_used=False)
    assert size<=64*2**20
    assert output['peak_ram_gib']<=4
    (folder/'audit.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':
    main()
