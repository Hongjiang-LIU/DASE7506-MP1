"""Pure-metadata final timing gate fixtures: no corpus or scoring."""
import copy
import unittest

from push150_freeze import check_timing, BASELINE_SHA, SCORER_SHA


class FreezeTimingTests(unittest.TestCase):
    def fixture(self):
        winner = {'checkpoint_sha256': 'a'*64}
        order = [(0,'baseline'),(0,'candidate'),(1,'candidate'),(1,'baseline'),(2,'baseline'),(2,'candidate')]
        timing = dict(test_used=False, split='validation', device='cpu', precision='fp32', threads=4,
            affinity_mask=15, candidate_sha256='a'*64, baseline_sha256=BASELINE_SHA,
            evaluator_sha256=SCORER_SHA, cpu_time_ratio=4.5,
            measurements=[dict(repeat=repeat,name=name,seconds=7. if name=='baseline' else 31.5,
                              targets=376599,utf8_bytes=1148007,peak_process_working_set_bytes=2**30)
                          for repeat,name in order])
        return timing,winner

    def test_valid_three_pairs(self):
        timing,winner=self.fixture()
        self.assertEqual(check_timing(timing,winner)['ratio_of_medians'],4.5)

    def test_five_is_allowed_no_extra_margin(self):
        timing,winner=self.fixture()
        for row in timing['measurements']:
            if row['name']=='candidate': row['seconds']=35.
        timing['cpu_time_ratio']=5.
        self.assertEqual(check_timing(timing,winner)['ratio_of_medians'],5.)

    def test_worst_case_diagnostic_not_extra_gate(self):
        timing,winner=self.fixture()
        timing['measurements'][1]['seconds']=38.
        result=check_timing(timing,winner)
        self.assertGreater(result['max_candidate_divided_by_min_baseline'],5.)
        self.assertFalse(result['extra_conservative_gate_required'])

    def test_median_failure_refused(self):
        timing,winner=self.fixture()
        for row in timing['measurements']:
            if row['name']=='candidate': row['seconds']=36.
        timing['cpu_time_ratio']=36./7
        with self.assertRaises(ValueError):check_timing(timing,winner)

    def test_missing_reordered_changed_identity_refused(self):
        timing,winner=self.fixture()
        variants=[]
        value=copy.deepcopy(timing);value['measurements'].pop();variants.append(value)
        value=copy.deepcopy(timing);value['measurements'].reverse();variants.append(value)
        value=copy.deepcopy(timing);value['baseline_sha256']='b'*64;variants.append(value)
        value=copy.deepcopy(timing);value['test_used']=True;variants.append(value)
        value=copy.deepcopy(timing);value['cpu_time_ratio']=1.;variants.append(value)
        for value in variants:
            with self.assertRaises(ValueError):check_timing(value,winner)

    def test_ram_limit_and_invalid_duration(self):
        timing,winner=self.fixture()
        timing['measurements'][1]['peak_process_working_set_bytes']=4*2**30+1
        with self.assertRaises(ValueError):check_timing(timing,winner)
        timing,winner=self.fixture();timing['measurements'][0]['seconds']=float('nan')
        with self.assertRaises(ValueError):check_timing(timing,winner)


if __name__=='__main__':unittest.main()
