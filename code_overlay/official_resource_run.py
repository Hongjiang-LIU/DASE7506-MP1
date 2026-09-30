"""Wrap the unmodified evaluator CLI and record Windows peak process memory."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import runpy
import sys
from measure_validation import peak_memory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--split',default='test',choices=['test','validation'])
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    get_process=ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype=wintypes.HANDLE
    setter=ctypes.windll.kernel32.SetProcessAffinityMask
    setter.argtypes=[wintypes.HANDLE,ctypes.c_size_t]
    if not setter(get_process(),15):
        raise ctypes.WinError()
    evaluator=Path(__file__).resolve().parent/'evaluate.py'
    sys.argv=[str(evaluator),'--checkpoint',str(args.checkpoint),'--device','cpu',
              '--precision','fp32','--threads','4','--split',args.split,'--output',str(args.output)]
    runpy.run_path(str(evaluator),run_name='__main__')
    result=dict(peak_process_working_set_bytes=peak_memory(), cpu_affinity_mask=15,
                includes='entire evaluator process including all fixed dataset loading',
                evaluator_output=str(args.output))
    args.output.with_suffix('.resources.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
