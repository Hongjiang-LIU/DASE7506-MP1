"""Synthetic causal-window CPU scoring benchmark for architecture screening."""
import gc
import json
import statistics
from pathlib import Path
import torch
from common import setup, make_model
from evaluate import score


def main():
    device, _ = setup('cpu','fp32',4)
    torch.manual_seed(17)
    tokens = torch.randint(0,2048,(65537,))
    results=[]
    for filename,module in [('baseline.json','model'),('medium_gpu.json','efficient'),('large_gpu.json','efficient')]:
        config=json.loads((Path('configs')/filename).read_text())
        model,_=make_model(module,config,device)
        row=dict(config=filename,parameters=sum(p.numel() for p in model.parameters()))
        times=[score(model,tokens,65537,device,'fp32')['seconds'] for _ in range(3)]
        row.update(seconds=times,median=statistics.median(times))
        results.append(row)
        print(json.dumps(row),flush=True)
        del model
        gc.collect()
    for row in results:
        row['baseline_ratio']=row['median']/results[0]['median']
    Path('runs/capacity_probe.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)


if __name__ == '__main__':
    main()
