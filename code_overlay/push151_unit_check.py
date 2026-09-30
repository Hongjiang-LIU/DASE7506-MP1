"""Dataset-free complete unit suite; immutable push151 source/test evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import unittest
from common import ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier unit-test evidence')
    source = {path.name: sha(path) for path in sorted(ROOT.glob('push151*.py'))}
    tests = {path.name: sha(path) for path in sorted((ROOT/'tests').glob('test_push151*.py'))}
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if source != {path.name: sha(path) for path in sorted(ROOT.glob('push151*.py'))} or tests != {
            path.name: sha(path) for path in sorted((ROOT/'tests').glob('test_push151*.py'))}:
        raise RuntimeError('Sources changed while testing')
    record = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(), tests_run=result.testsRun,
        successful=result.wasSuccessful(), failures=[(str(test),detail) for test,detail in result.failures],
        errors=[(str(test),detail) for test,detail in result.errors], skipped=[(str(test),why) for test,why in result.skipped],
        source_sha256=source, unit_source_sha256=tests, seconds=time.perf_counter()-started, test_text_loaded=False)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key:record[key] for key in ('tests_run','successful','seconds')}), flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
