"""Unchanged official score(), loading VALIDATION ONLY, no search or test I/O.

Repeat --checkpoint and --output together to evaluate fixed candidates serially
in one process. CPU results use two threads and are not the 4-thread paired
resource benchmark; GPU results are screening measurements, not ranked scores.
"""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from common import PROTOCOL, ROOT, make_model, setup, sha, device_metrics
from evaluate import score


def load_validation():
    path = ROOT/'data/wikitext_validation.txt'
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    if sha(path) != manifest['sha256'][path.name]:
        raise ValueError('Changed validation file')
    tokenizer_path = ROOT/'data/tokenizer.json'
    if sha(tokenizer_path) != manifest['sha256'][tokenizer_path.name]:
        raise ValueError('Changed tokenizer')
    raw = path.read_bytes()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids, dtype=torch.long), len(raw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, action='append', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    if len(args.checkpoint) != len(args.output):
        p.error('Supply one --output for each --checkpoint')
    tokens, byte_count = load_validation()
    for checkpoint_path, output in zip(args.checkpoint, args.output):
        if output.exists() or output.with_suffix('.window-nll.npy').exists():
            raise FileExistsError(f'Refusing to overwrite existing measurement: {output}')
        device, precision = setup(args.device, 'fp32', 2)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint['protocol'] != PROTOCOL:
            raise ValueError('Checkpoint belongs to a different protocol')
        model, implementation_sha = make_model(checkpoint['implementation'], checkpoint['config'], device)
        model.load_state_dict(checkpoint['model'])
        result = score(model, tokens, byte_count, device, precision)
        losses = result.pop('window_nll_nats')
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output.with_suffix('.window-nll.npy'), np.asarray(losses, dtype=np.float64))
        result.update(protocol=PROTOCOL, split='validation', precision='fp32', threads=2,
                      test_used=False, prior_test_exposure_disclosed=True,
                      checkpoint_sha256=sha(checkpoint_path), checkpoint_bytes=checkpoint_path.stat().st_size,
                      implementation=checkpoint['implementation'], implementation_sha256=implementation_sha,
                      evaluator_sha256=sha(ROOT/'evaluate.py'), tokenizer_sha256=sha(ROOT/'data/tokenizer.json'),
                      validation_sha256=sha(ROOT/'data/wikitext_validation.txt'),
                      wrapper_sha256=sha(__file__), **device_metrics(device))
        output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2), flush=True)
        del model, checkpoint
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
