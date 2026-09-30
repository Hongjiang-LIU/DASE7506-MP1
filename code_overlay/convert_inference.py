"""Repack unchanged tensors with a new inference module; no data is loaded."""
import argparse
import json
from pathlib import Path
import torch
from common import sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--implementation',required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=False)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    checkpoint['implementation']=args.implementation
    checkpoint['inference_parent_sha256']=sha(args.checkpoint)
    torch.save(checkpoint,args.run_dir/'checkpoint.pt')
    info=dict(checkpoint_sha256=sha(args.run_dir/'checkpoint.pt'),parent_sha256=sha(args.checkpoint),
              implementation=args.implementation,change='Inference algebra only; identical parameter tensors',test_used=False)
    (args.run_dir/'conversion.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info),flush=True)


if __name__=='__main__':
    main()
