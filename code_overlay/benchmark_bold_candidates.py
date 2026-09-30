"""Full validation fixed-scorer screen, baseline bracketing every candidate."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from common import ROOT,setup,make_model,sha
from evaluate import score
from measure_validation import peak_memory


def main():
    get_process=ctypes.windll.kernel32.GetCurrentProcess;get_process.restype=wintypes.HANDLE
    setter=ctypes.windll.kernel32.SetProcessAffinityMask
    setter.argtypes=[wintypes.HANDLE,ctypes.c_size_t]
    if not setter(get_process(),15):
        raise ctypes.WinError()
    device,_=setup('cpu','fp32',4)
    names=['micro-hybrid-compact-min2','hybrid-compact-exact','correction-compact-exact',
           'correction-medium-continue','hybrid-lowrank128']
    paths={name:ROOT/'runs'/name/'checkpoint.pt' for name in ['baseline-s17']+names}
    models={}
    for name,path in paths.items():
        cp=torch.load(path,map_location='cpu',weights_only=True)
        model,_=make_model(cp['implementation'],cp['config'],device)
        model.load_state_dict(cp['model']);models[name]=model.eval()
    raw=(ROOT/'data/wikitext_validation.txt').read_bytes()
    tokens=torch.tensor(Tokenizer.from_file(str(ROOT/'data/tokenizer.json')).encode(raw.decode('utf-8')).ids)
    rows=[]
    info=dict(scope='Full validation CPU FP32 single-pass screen; not test evaluation',
              threads=4,affinity_mask=15,checkpoint_hashes={n:sha(p) for n,p in paths.items()},
              source_hashes={p.name:sha(p) for p in ROOT.glob('*.py')},measurements=rows)
    order=['baseline-s17']
    for name in names:
        order.extend([name,'baseline-s17'])
    for name in order:
        result=score(models[name],tokens,len(raw),device,'fp32');result.pop('window_nll_nats')
        row=dict(name=name,**result);rows.append(row)
        info['peak_process_working_set_bytes']=peak_memory()
        (ROOT/'runs/bold_validation_resource_screen.json').write_text(json.dumps(info,indent=2))
        print(json.dumps(row),flush=True)
    ratios=[]
    for i in range(1,len(rows),2):
        before,after=rows[i-1]['seconds'],rows[i+1]['seconds']
        ratios.append(dict(name=rows[i]['name'],validation_bpb=rows[i]['bpb'],
                           mean_bracket_ratio=rows[i]['seconds']/((before+after)/2),
                           conservative_bracket_ratio=rows[i]['seconds']/min(before,after)))
    info['ratios']=ratios
    (ROOT/'runs/bold_validation_resource_screen.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(ratios,indent=2),flush=True)


if __name__=='__main__':
    main()
