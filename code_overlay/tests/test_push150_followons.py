import unittest
from pathlib import Path
from push150_followons import declared_commands


class FollowonCommandsTests(unittest.TestCase):
    def setUp(self):
        self.rows=declared_commands(Path('fake-parent.pt'),'a'*64)
        self.commands=dict(self.rows)

    def test_order_and_no_test_command(self):
        self.assertEqual(len(self.rows),14)
        self.assertEqual([name for name,_ in self.rows[:5]],
                         ['gate-build','oof-prepare','gate-extract','gate-train','gate-score'])
        self.assertFalse(any('evaluate.py' in argument or argument=='test'
                             for _,command in self.rows for argument in command))

    def test_suffix_control_explicit_phase_c_recipe(self):
        command=self.commands['suffix-control']
        for key,value in {'--steps':'4000','--eval-every':'1000','--seed':'191',
                          '--warmup':'100','--decay-steps':'1000','--schedule':'cosine'}.items():
            self.assertEqual(command[command.index(key)+1],value)

    def test_distinct_suffix_calibrator(self):
        self.assertIn('push150_suffix_calibrate.py',self.commands['suffix-joint-calibrate'])
        self.assertIn('push146_calibrate.py',self.commands['suffix-control-calibrate'])

    def test_two_oof_objectives_same_parent_cache(self):
        for objective in ('control','joint'):
            command=self.commands['oof-'+objective]
            self.assertEqual(command[command.index('--objective')+1],objective)
            self.assertEqual(command[command.index('--parent')+1],'fake-parent.pt')
            self.assertEqual(command[command.index('--prepared-dir')+1],'runs/push150-oof-data')


if __name__=='__main__':unittest.main()
