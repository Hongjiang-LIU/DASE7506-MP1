"""Synthetic engineering screen, NOT the official resource acceptance test."""
import json
import time
import statistics
from pathlib import Path
import torch
from common import make_model


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(73)
    x=torch.randint(2048,(32,256))
    rows=[]
    for label,path,module in [
        ('compact-original','runs/refine-extension-control-s41/checkpoint.pt','refined'),
        ('compact-fast','runs/refine-extension-control-s41/checkpoint.pt','fast_refined'),
        ('compact-micro','runs/refine-extension-control-s41/checkpoint.pt','micro_refined'),
        ('hybrid-fast','runs/hybrid-compact-min2/checkpoint.pt','fast_ngram_hybrid'),
        ('hybrid-micro','runs/hybrid-compact-min2/checkpoint.pt','micro_hybrid'),
        ('large-original','runs/refine-large-fine/checkpoint.pt','refined'),
        ('large-micro','runs/refine-large-fine/checkpoint.pt','micro_refined')]:
        cp=torch.load(path,map_location='cpu',weights_only=True)
        model,_=make_model(module,cp['config'],torch.device('cpu'))
        model.load_state_dict(cp['model']);model.eval()
        times=[]
        for _ in range(4):
            start=time.perf_counter(); result=model.predict_log_probs(x); times.append(time.perf_counter()-start)
        row=dict(label=label,seconds=times,median=statistics.median(times[1:]))
        rows.append(row); print(json.dumps(row),flush=True)
        del model,cp,result
    Path('runs/synthetic_inference_screen.json').write_text(json.dumps(dict(
        scope='Synthetic screen; concurrent GPU training possible, NOT acceptance evidence',rows=rows),indent=2))


if __name__=='__main__':
    main()
