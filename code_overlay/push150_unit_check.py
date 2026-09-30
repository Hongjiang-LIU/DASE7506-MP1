"""Run dataset-free CPU unit tests and preserve a machine-readable result."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import unittest

from common import ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous test evidence')
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    report = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  tests_run=result.testsRun, successful=result.wasSuccessful(),
                  failures=[(str(test), detail) for test,detail in result.failures],
                  errors=[(str(test), detail) for test,detail in result.errors],
                  skipped=[(str(test), reason) for test,reason in result.skipped],
                  seconds=time.perf_counter()-started, test_text_loaded=False,
                  source_sha256={path.name:sha(path) for path in sorted(ROOT.glob('push150*.py'))},
                  unit_source_sha256={path.name:sha(path) for path in sorted((ROOT/'tests').glob('test_push150*.py'))})
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key:report[key] for key in ('tests_run','successful','seconds')}), flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
