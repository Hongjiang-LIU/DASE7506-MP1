"""Run the unchanged scorer on validation, with OS process peak memory."""
import argparse
import ctypes
from ctypes import wintypes
import json
import statistics
from pathlib import Path

import torch
from tokenizers import Tokenizer
from common import ROOT, make_model, setup, sha
from evaluate import score


def peak_memory():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not query(get_process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return counters.PeakWorkingSetSize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--split', choices=['validation','test'], default='validation')
    parser.add_argument('--affinity', type=int, default=15,
                        help='Windows CPU mask; 15 selects the same four logical CPUs for both models.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    set_affinity = ctypes.windll.kernel32.SetProcessAffinityMask
    set_affinity.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    if not set_affinity(get_process(), args.affinity):
        raise ctypes.WinError()
    device, _ = setup('cpu', 'fp32', 4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, implementation_sha = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    raw = (ROOT/'data'/f'wikitext_{args.split}.txt').read_bytes()
    tokenizer = Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    tokens = torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids)
    measurements = []
    for _ in range(args.repeats):
        measured = score(model, tokens, len(raw), device, 'fp32')
        measured.pop('window_nll_nats')
        measurements.append(measured)
    result = dict(split=args.split, repeats=measurements, cpu_affinity_mask=args.affinity,
                  median_scoring_seconds=statistics.median(m['seconds'] for m in measurements),
                  peak_process_working_set_bytes=peak_memory(),
                  checkpoint_bytes=args.checkpoint.stat().st_size,
                  checkpoint_sha256=sha(args.checkpoint), implementation_sha256=implementation_sha,
                  evaluator_sha256=sha(ROOT/'evaluate.py'), torch_version=str(torch.__version__), threads=4)
    output = args.output or args.checkpoint.parent/f'resource_{args.split}_affinity.json'
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
