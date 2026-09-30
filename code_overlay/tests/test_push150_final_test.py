"""Dataset-free tests of post-freeze orchestration validation; no scorer calls."""
import math
from pathlib import Path
import tempfile
import unittest

from push150_final_test import check_measurement, child, read, sha, write_new
from push150_final_test import SCORER_SHA, TOKENIZER_SHA


class FinalTestGuardTests(unittest.TestCase):
    def fixture(self):
        bpb = 1.43
        measured = dict(protocol='7506-mp1-wt2-v2', split='test', precision='fp32',
            checkpoint_sha256='a'*64, implementation_sha256='b'*64,
            evaluator_sha256=SCORER_SHA, tokenizer_sha256=TOKENIZER_SHA,
            device='cpu', device_name='CPU', targets=428405, utf8_bytes=1292013,
            bpb=bpb, nll_nats=bpb*math.log(2)*1292013, seconds=34.)
        resource = dict(cpu_affinity_mask=15, peak_process_working_set_bytes=2**30)
        return measured, resource

    def test_valid_official_result(self):
        measured, resource = self.fixture()
        check_measurement(measured, resource, 'a'*64, 'b'*64)

    def test_changed_protocol_fields_refused(self):
        for key, value in [('split', 'validation'), ('device', 'cuda:0'), ('precision', 'bf16'),
                           ('targets', 428404), ('utf8_bytes', 1292012),
                           ('checkpoint_sha256', 'c'*64), ('evaluator_sha256', 'c'*64)]:
            with self.subTest(key=key):
                measured, resource = self.fixture()
                measured[key] = value
                with self.assertRaises(ValueError):
                    check_measurement(measured, resource, 'a'*64, 'b'*64)

    def test_inconsistent_bpb_refused(self):
        measured, resource = self.fixture()
        measured['bpb'] += .001
        with self.assertRaises(ValueError):
            check_measurement(measured, resource, 'a'*64, 'b'*64)

    def test_invalid_times_refused(self):
        for value in (0., -1., float('nan'), float('inf')):
            measured, resource = self.fixture()
            measured['seconds'] = value
            with self.assertRaises(ValueError):
                check_measurement(measured, resource, 'a'*64, 'b'*64)

    def test_missing_resources_refused(self):
        measured, resource = self.fixture()
        for key in resource:
            changed = dict(resource)
            del changed[key]
            with self.assertRaises(ValueError):
                check_measurement(measured, changed, 'a'*64, 'b'*64)

    def test_safe_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(child(directory, 'runs/example.json'),
                             (Path(directory)/'runs/example.json').resolve())
            for name in ('../test', '/test', 'C:/test', 'runs/./test', 'runs//test', ''):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    child(directory, name)

    def test_corpus_io_blocked_before_open(self):
        for operation in (sha, read):
            with self.assertRaises(ValueError):
                operation(Path('does-not-exist/wikitext_test.txt'))

    def test_exclusive_ledger_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'launch_plan.json'
            write_new(path, {'candidate_pass_cap': 1})
            digest = sha(path)
            with self.assertRaises(FileExistsError):
                write_new(path, {'candidate_pass_cap': 2})
            self.assertEqual(read(path), {'candidate_pass_cap': 1})
            self.assertEqual(sha(path), digest)


if __name__ == '__main__':
    unittest.main()
