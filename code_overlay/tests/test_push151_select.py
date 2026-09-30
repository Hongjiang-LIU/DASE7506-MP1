"""Synthetic states/metadata only; never open real checkpoints or corpora."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import push151_select as subject


NAMES = {'A': ['control', 'tail-ema-average', 'last-raw-ema-average'],
         'B': ['control', 'information-equal', 'information-plus8mib']}


def checkpoint_fixture():
    tied = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    average = dict(implementation='push150_expand', protocol='synthetic', train_tokens=638976000,
        config=dict(subject.FIXED_CONFIG, width=2, mkn_shapes={'unigram': [3], 'masses_3': [1]}),
        model={'token.weight': tied, 'head.weight': tied, 'buffer': torch.tensor([3]),
               'ngram.unigram': torch.tensor([.2, .3, .5]), 'ngram.masses_3': torch.tensor([.1])})
    statistics = copy.deepcopy(average)
    statistics['model']['token.weight'].fill_(99.)
    statistics['model']['buffer'].fill_(19)
    statistics['model']['ngram.masses_3'] = torch.tensor([.2, .4])
    statistics['config']['mkn_shapes']['masses_3'] = [2]
    return average, statistics


def measurement(path, digest, bpb):
    return dict(checkpoint=path, checkpoint_sha256=digest, bpb=bpb, targets=376599,
        utf8_bytes=1148007, precision='fp32', split='validation', device='cuda:0',
        test_used=False, seconds=.25, evaluator_sha256='e'*64, implementation_sha256='i'*64)


def metadata_fixture(a=(1.44, 1.439, 1.4397), b=(1.44, 1.4395, 1.4398)):
    root = subject.ROOT
    plan = dict(parent=dict(validation_bpb_gpu_fp32=1.44, sha256=subject.PARENT_SHA,
        selected_lineage_targets=638976000),
        phase_a=dict(candidates=[{'name': name} for name in NAMES['A'][1:]]),
        phase_b=dict(names=NAMES['B'][1:]), selection=dict(minimum_improvement=.0005))
    documents, hashes, records = {}, {}, {}
    source = 'synthetic-source.py'
    hashes[root/source] = 's'*64
    for phase, scores in [('A', a), ('B', b)]:
        directory = root/('runs/push151-average' if phase == 'A' else 'runs/push151-statistics')
        built, measured, rows = [], [], []
        for index, (name, bpb) in enumerate(zip(NAMES[phase], scores)):
            path = 'runs/synthetic-parent/checkpoint.pt' if index == 0 else f'runs/synthetic-{phase}-{index}/checkpoint.pt'
            digest = subject.PARENT_SHA if index == 0 else (phase.lower()+str(index))*32
            hashes[root/path] = digest
            built.append(dict(name=name, checkpoint=path, checkpoint_sha256=digest))
            item = dict(name=name, **measurement(path, digest, bpb))
            measured.append(item)
            rows.append(dict(phase=phase, name=name, checkpoint=path, checkpoint_sha256=digest,
                validation_bpb_gpu_fp32=bpb, measured=item))
        build = dict(status='completed', plan_sha256=subject.PLAN_SHA, parent_sha256=subject.PARENT_SHA,
            source_sha256={source:'s'*64}, candidates=built, test_used=False, seconds_total=1.,
            count_source_token_occurrences=100, new_gradient_targets=0,
            inherited_shared_lineage_targets=638976000)
        build_hash, score_hash = phase.lower()*64, str(1 if phase == 'A' else 2)*64
        scored = dict(status='completed', plan_sha256=subject.PLAN_SHA, build_sha256=build_hash,
            source_sha256={source:'s'*64}, candidates=measured, test_used=False, seconds_total=2.)
        documents[directory/'build.json'], documents[directory/'score.json'] = build, scored
        hashes[directory/'build.json'], hashes[directory/'score.json'] = build_hash, score_hash
        records[phase] = dict(rows=rows, build=build, score=scored, directory=directory)
    hashes[subject.PLAN_PATH] = subject.PLAN_SHA
    hashes[Path(subject.__file__)] = 'c'*64
    return plan, records, documents, hashes


def combo_fixture(records, hashes, active=True):
    _, a, b = subject.combination_spec(records)
    result = dict(status='completed' if active else 'skipped', plan_sha256=subject.PLAN_SHA,
        script_sha256=hashes[Path(subject.__file__)], test_used=False,
        phase_a_score_sha256=hashes[records['A']['directory']/'score.json'],
        phase_b_score_sha256=hashes[records['B']['directory']/'score.json'],
        new_gradient_targets=0, shared_ancestor_targets=638976000, seconds_total=3.,
        source_a=copy.deepcopy(a), source_b=copy.deepcopy(b), best_a=copy.deepcopy(a), best_b=copy.deepcopy(b))
    if active:
        path = 'runs/push151-combination/checkpoint.pt'
        hashes[subject.ROOT/path] = 'f'*64
        result['measurement'] = measurement(path, 'f'*64, 1.438)
    hashes[subject.COMBO/'result.json'] = 'd'*64
    return result


class SelectorTests(unittest.TestCase):
    def test_combination_requires_both_strictly_better(self):
        for a, b, expected in [((1.44,1.439,1.4397),(1.44,1.4395,1.4398),True),
                               ((1.44,1.4401,1.4402),(1.44,1.4395,1.4398),False),
                               ((1.44,1.439,1.4397),(1.44,1.44,1.441),False)]:
            _, records, _, _ = metadata_fixture(a,b)
            self.assertEqual(subject.combination_spec(records)[0], expected)

    def test_combination_menu_is_not_final_improvement_gate(self):
        _, records, _, _ = metadata_fixture((1.44,1.4399,1.441),(1.44,1.4399,1.441))
        active,a,b = subject.combination_spec(records)
        self.assertTrue(active)
        self.assertEqual(a['name'],'tail-ema-average')
        self.assertEqual(b['name'],'information-equal')

    def test_tie_retains_parent_and_skips(self):
        _,records,_,_ = metadata_fixture((1.44,1.44,1.44),(1.44,1.439,1.439))
        active,a,_ = subject.combination_spec(records)
        self.assertFalse(active)
        self.assertEqual(a['name'],'control')

    def test_combination_changes_only_statistics_and_shapes(self):
        average,statistics = checkpoint_fixture()
        original = copy.deepcopy(average)
        result = subject.combine_states(average,statistics)
        for name,tensor in result['model'].items():
            expected = statistics['model'][name] if name.startswith('ngram.') else average['model'][name]
            self.assertTrue(torch.equal(tensor,expected),name)
        self.assertIs(result['model']['token.weight'],result['model']['head.weight'])
        self.assertEqual(result['config']['mkn_shapes'],statistics['config']['mkn_shapes'])
        for key,value in subject.FIXED_CONFIG.items():self.assertEqual(result['config'][key],value)
        result['model']['ngram.masses_3'][0] = 0.
        self.assertEqual(statistics['model']['ngram.masses_3'][0].item(),torch.tensor(.2).item())
        for name,value in original['model'].items():self.assertTrue(torch.equal(average['model'][name],value))

    def test_different_ancestry_or_implementation_rejected(self):
        for field,value in [('train_tokens',638976001),('implementation','other')]:
            average,statistics=checkpoint_fixture();statistics[field]=value
            with self.assertRaisesRegex(ValueError,'identity/ancestry'):subject.combine_states(average,statistics)

    def test_even_equal_but_wrong_ancestry_rejected(self):
        average,statistics=checkpoint_fixture()
        average['train_tokens']=statistics['train_tokens']=1277952000
        with self.assertRaises(ValueError):subject.combine_states(average,statistics)

    def test_fixed_calibration_and_architecture_rejected_if_changed(self):
        for key,value in [('temperature',1.25),('mkn_weight',.2),('mkn_order',4),('discount_scale',.55),('width',3)]:
            average,statistics=checkpoint_fixture();statistics['config'][key]=value
            with self.assertRaisesRegex(ValueError,'calibration or architecture'):subject.combine_states(average,statistics)
        average,statistics=checkpoint_fixture()
        average['config']['temperature']=statistics['config']['temperature']=1.25
        with self.assertRaises(ValueError):subject.combine_states(average,statistics)

    def test_mismatched_state_keys_rejected(self):
        average,statistics=checkpoint_fixture();statistics['model'].pop('ngram.masses_3')
        with self.assertRaisesRegex(ValueError,'state keys'):subject.combine_states(average,statistics)

    def phase_check(self, documents, hashes, plan):
        with mock.patch.object(subject,'load_plan',return_value=plan), \
             mock.patch.object(subject,'read',side_effect=lambda path:copy.deepcopy(documents[Path(path)])), \
             mock.patch.object(subject,'sha',side_effect=lambda path:hashes[Path(path)]):
            return subject.phase_records()

    def test_complete_bound_A_B_metadata_accepted(self):
        plan,_,documents,hashes=metadata_fixture()
        actual_plan,records=self.phase_check(documents,hashes,plan)
        self.assertEqual(actual_plan,plan)
        self.assertEqual([row['name'] for row in records['A']['rows']],NAMES['A'])
        self.assertEqual([row['name'] for row in records['B']['rows']],NAMES['B'])

    def test_phase_checkpoint_and_build_hashes_must_match(self):
        for alteration in ('checkpoint','build'):
            plan,records,documents,hashes=metadata_fixture()
            scored=documents[records['A']['directory']/'score.json']
            if alteration=='checkpoint':scored['candidates'][1]['checkpoint_sha256']='x'*64
            else:scored['build_sha256']='x'*64
            with self.assertRaises(ValueError):self.phase_check(documents,hashes,plan)

    def test_phase_source_drift_rejected(self):
        plan,_,documents,hashes=metadata_fixture();hashes[subject.ROOT/'synthetic-source.py']='x'*64
        with self.assertRaisesRegex(ValueError,'source changed'):self.phase_check(documents,hashes,plan)

    def test_phase_incomplete_or_test_scoring_rejected(self):
        for target,key,value in [('score','status','running'),('score','test_used',True),
                                 ('measurement','split','test'),('measurement','test_used',True),
                                 ('measurement','targets',376598),('measurement','utf8_bytes',1),
                                 ('measurement','precision','bf16'),('measurement','bpb',float('nan'))]:
            plan,records,documents,hashes=metadata_fixture()
            scored=documents[records['A']['directory']/'score.json']
            (scored if target=='score' else scored['candidates'][1])[key]=value
            with self.subTest(target=target,key=key),self.assertRaises(ValueError):
                self.phase_check(documents,hashes,plan)

    def test_parent_controls_must_agree(self):
        plan,records,documents,hashes=metadata_fixture()
        documents[records['B']['directory']/'score.json']['candidates'][0]['bpb']+=1e-7
        with self.assertRaisesRegex(ValueError,'parent scores disagree'):self.phase_check(documents,hashes,plan)

    def test_phase_duplicate_or_undeclared_candidates_rejected(self):
        for invalid in ('duplicate','undeclared'):
            plan,records,documents,hashes=metadata_fixture()
            for kind in ('build','score'):
                rows=documents[records['A']['directory']/(kind+'.json')]['candidates']
                rows[2]=copy.deepcopy(rows[1])
                if invalid=='undeclared':rows[2]['name']='another-average'
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                self.phase_check(documents,hashes,plan)

    def call_report(self, plan, records, combo, hashes):
        with mock.patch.object(subject,'phase_records',return_value=(plan,records)), \
             mock.patch.object(subject,'read',side_effect=lambda path:copy.deepcopy(
                 combo if Path(path).name=='result.json' else combo['measurement'])), \
             mock.patch.object(subject,'sha',side_effect=lambda path:hashes[Path(path)]), \
             mock.patch.object(subject,'write_new') as write:
            result=subject.report()
            self.assertEqual(write.call_count,1)
            return result

    def test_report_counts_shared_ancestor_once_and_remains_validation_only(self):
        plan,records,_,hashes=metadata_fixture();combo=combo_fixture(records,hashes)
        report=self.call_report(plan,records,combo,hashes)
        self.assertEqual(report['selected_shared_ancestor_targets'],638976000)
        self.assertEqual(report['new_neural_gradient_targets'],0)
        self.assertEqual(report['statistical_train_tokens_scanned'],100)
        self.assertEqual(report['construction_seconds'],{'A':1.,'B':1.})
        self.assertEqual(report['development_score_seconds'],{'A':2.,'B':2.})
        self.assertEqual(report['combination_seconds_including_build_and_score'],3.)
        self.assertEqual(report['full_validation_passes'],7)
        self.assertEqual(len(report['candidates']),6)
        self.assertEqual(report['validation_qualified_in_rank_order'][0]['name'],'combined')
        self.assertEqual(report['selection_split'],'validation')
        self.assertFalse(report['test_used'])
        self.assertTrue(report['final_cpu_audit_and_freeze_required'])
        self.assertTrue(all(row['cpu_resource_status']=='not_yet_audited'
            for row in report['validation_qualified_in_rank_order']))

    def test_report_no_qualified_candidate_does_not_authorize_test(self):
        plan,records,_,hashes=metadata_fixture((1.44,1.4401,1.4402),(1.44,1.4401,1.4402))
        combo=combo_fixture(records,hashes,False)
        report=self.call_report(plan,records,combo,hashes)
        self.assertEqual(report['full_validation_passes'],6)
        self.assertEqual(report['validation_qualified_in_rank_order'],[])
        self.assertEqual(report['status'],'no_new_candidate_meets_validation_gate')
        self.assertFalse(report['final_cpu_audit_and_freeze_required'])

    def test_report_combo_measurement_must_be_full_validation(self):
        for key,value in [('targets',376598),('utf8_bytes',1),('precision','bf16'),('split','test'),
                          ('test_used',True),('bpb',float('nan')),('bpb',0.),('bpb',-1.),('bpb',float('inf'))]:
            plan,records,_,hashes=metadata_fixture();combo=combo_fixture(records,hashes)
            combo['measurement'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.call_report(plan,records,combo,hashes)

    def test_report_combo_must_bind_declared_winners(self):
        for key in ('source_a','source_b'):
            plan,records,_,hashes=metadata_fixture();combo=combo_fixture(records,hashes)
            combo[key]['checkpoint_sha256']='x'*64
            with self.subTest(key=key),self.assertRaises(ValueError):self.call_report(plan,records,combo,hashes)

    def test_report_unaccounted_or_rebound_combo_rejected(self):
        for key,value in [('status','failed'),('plan_sha256','x'*64),('script_sha256','x'*64),
                          ('phase_a_score_sha256','x'*64),('phase_b_score_sha256','x'*64),('test_used',True)]:
            plan,records,_,hashes=metadata_fixture();combo=combo_fixture(records,hashes);combo[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.call_report(plan,records,combo,hashes)

    def test_skip_never_loads_checkpoint_corpus_or_scorer_and_is_exclusive(self):
        plan,records,_,_=metadata_fixture((1.44,1.441,1.442),(1.44,1.439,1.4395))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            destination=Path(directory)/'synthetic-combination'
            stack.enter_context(mock.patch.object(subject,'COMBO',destination))
            stack.enter_context(mock.patch.object(subject,'phase_records',return_value=(plan,records)))
            stack.enter_context(mock.patch.object(subject,'sha',return_value='x'*64))
            stack.enter_context(mock.patch.object(subject,'write_new'))
            for owner,name in [(subject.torch,'load'),(subject,'load_one_split'),(subject,'measure_checkpoint'),(subject,'setup')]:
                stack.enter_context(mock.patch.object(owner,name,side_effect=AssertionError('must not execute')))
            result=subject.run_combination()
            self.assertEqual(result['status'],'skipped')
            self.assertEqual(result['new_gradient_targets'],0)
            with self.assertRaises(FileExistsError):subject.run_combination()

    def test_active_combination_scores_only_once_and_only_validation(self):
        plan,records,_,_=metadata_fixture();average,statistics=checkpoint_fixture()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(mock.patch.object(subject,'COMBO',Path(directory)/'synthetic-combination'))
            stack.enter_context(mock.patch.object(subject,'phase_records',return_value=(plan,records)))
            stack.enter_context(mock.patch.object(subject,'sha',return_value='x'*64))
            stack.enter_context(mock.patch.object(subject,'write_new'))
            saved=stack.enter_context(mock.patch.object(subject,'save_new'))
            loaded=stack.enter_context(mock.patch.object(subject.torch,'load',side_effect=[average,statistics]))
            setup=stack.enter_context(mock.patch.object(subject,'setup',return_value=(torch.device('cpu'),'fp32')))
            split=stack.enter_context(mock.patch.object(subject,'load_one_split',return_value=(np.array([1,2]),{'utf8_bytes':1148007})))
            score=stack.enter_context(mock.patch.object(subject,'measure_checkpoint',return_value=measurement('runs/fake.pt','x'*64,1.438)))
            result=subject.run_combination()
            self.assertEqual(result['status'],'completed')
            self.assertEqual(loaded.call_count,2)
            self.assertEqual(saved.call_count,1)
            self.assertEqual(score.call_count,1)
            setup.assert_called_once_with('cuda','fp32',2)
            split.assert_called_once_with('validation')
            self.assertEqual(saved.call_args.args[1]['train_tokens'],638976000)
            self.assertTrue(saved.call_args.args[1]['push151_combination']['duplicate_ancestors_not_summed'])
            with self.assertRaises(FileExistsError):subject.run_combination()


if __name__=='__main__':
    unittest.main()
