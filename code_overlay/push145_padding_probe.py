"""One additional train-prefix engineering hypothesis; not score tuning."""
import ctypes
from ctypes import wintypes
import json
import statistics
import time
import torch
from tokenizers import Tokenizer
from common import ROOT,setup,make_model,sha
from push145_cpu_padded import PaddedAttentionCache


@torch.no_grad()
def main():
    output=ROOT/'runs/push145_padding_probe.json'
    target=ROOT/'runs/push145-cache-padded'
    if output.exists():
        raise FileExistsError('Do not overwrite probe')
    target.mkdir(exist_ok=False)
    process=ctypes.windll.kernel32.GetCurrentProcess;process.restype=wintypes.HANDLE
    setter=ctypes.windll.kernel32.SetProcessAffinityMask;setter.argtypes=[wintypes.HANDLE,ctypes.c_size_t]
    if not setter(process(),15):
        raise ctypes.WinError()
    device,_=setup('cpu','fp32',4)
    path=ROOT/'runs/push145-cache-fast/checkpoint.pt'
    cp=torch.load(path,map_location='cpu',weights_only=True)
    original,_=make_model(cp['implementation'],cp['config'],device)
    original.load_state_dict(cp['model']);original.eval()
    padded=PaddedAttentionCache(dict(cp['config'],cpu_attention_pad_to=64)).eval()
    padded.load_state_dict(cp['model'])
    tokenizer=Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    train=ROOT/'data/wikitext_train.txt'
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    assert sha(train)==manifest['sha256'][train.name]
    ids=torch.tensor(tokenizer.encode(train.read_bytes().decode('utf-8')[:50000]).ids[:8192]).reshape(32,256)
    expected=original.predict_log_probs(ids)
    actual=padded.predict_log_probs(ids)
    torch.testing.assert_close(actual,expected,atol=3e-5,rtol=2e-6)
    rows=[]
    for repeat in range(3):
        row=dict(repeat=repeat)
        for name,model in ([('fast',original),('padded',padded)] if repeat%2==0 else [('padded',padded),('fast',original)]):
            started=time.perf_counter()
            for _ in range(2):
                model.predict_log_probs(ids)
            row[name]=(time.perf_counter()-started)/2
        rows.append(row)
    cp['implementation']='push145_cpu_padded'
    cp['config']['cpu_attention_pad_to']=64
    cp['runtime_parent_sha256']=sha(path)
    cp['selection']='Identical weights; one train-prefix attention-padding runtime probe, no held-out fitting'
    torch.save(cp,target/'checkpoint.pt')
    ratio=statistics.median(r['padded'] for r in rows)/statistics.median(r['fast'] for r in rows)
    report=dict(rows=rows,median_ratio_to_fast=ratio,max_logp_difference=(actual-expected).abs().max().item(),
        validation_used=False,test_used=False,training_updates=0,checkpoint_sha256=sha(target/'checkpoint.pt'),
        source_sha256=sha(path),script_sha256=sha(__file__),implementation_sha256=sha(ROOT/'push145_cpu_padded.py'))
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
