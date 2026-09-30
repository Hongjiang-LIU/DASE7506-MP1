"""Tiny synthetic math/metadata tests; no actual model, corpus or screen runs."""
import ast
import copy
import inspect
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import push153_screen as screen
from push153_mixture import CausalResponsibilityMixture


def experts(ids):
    vocabulary=torch.arange(11,device=ids.device,dtype=torch.float32)
    a=-(vocabulary[None,None,:]-ids[...,None]).abs()*.2
    b=-(vocabulary[None,None,:]-(ids[...,None]+3)%11).abs()*.4
    return a.softmax(-1),b.softmax(-1)


class SyntheticPredictor(CausalResponsibilityMixture):
    def __init__(self,kappa):
        nn.Module.__init__(self)
        self.config=dict(push153_kappa=kappa,mkn_weight=.15,mkn_order=5,mkn_shapes={'synthetic':True})
        self.ngram=SimpleNamespace(probabilities=lambda ids,order:experts(ids)[1])

    def neural_and_cache(self,ids):return experts(ids)[0]


def fixture():
    parent=dict(checkpoint='runs/parent/checkpoint.pt',checkpoint_sha256='p'*64,validation_bpb_gpu_fp32=1.44,
        selected_lineage_targets=802816000,parameters=6279315)
    lock=dict(parent=parent,source_sha256={'synthetic.py':'s'*64})
    rows=[]
    for index,(name,kappa) in enumerate(screen.MENU):
        measured=screen.score_metadata((1.44-index*.001)*math.log(2)*screen.BYTES,screen.TARGETS,1.)
        gain=parent['validation_bpb_gpu_fp32']-measured['bpb']
        rows.append(dict(name=name,kappa=kappa,validation_bpb_gpu_fp32=measured['bpb'],screen_measurement=measured,
            improvement_vs_parent=gain,passes_validation_improvement_gate=name!='control' and gain>=.001,
            selected_lineage_targets=802816000,parameters=6279315))
    ranking=screen.ranked_trials(rows,parent)
    selection=dict(status='validation_screen_complete',plan_sha256=screen.PLAN_SHA,test_used=False,selection_split='validation',
        source_sha256=lock['source_sha256'],parent=parent,full_validation_scoring_choices=4,actual_shared_expert_passes=1,
        heldout_probability_or_label_arrays_saved=False,trials=rows,validation_qualified_in_rank_order=ranking,
        resource_audit_candidates=ranking[:3])
    return lock,selection


class ShortcutTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_full_production_gather_matches_shortcut_every_menu_and_shape(self):
        generator=torch.Generator().manual_seed(773)
        for batch,length in ((1,1),(3,2),(2,19),(2,63),(1,256)):
            ids=torch.randint(11,(batch,length),generator=generator)
            target=torch.cat((ids[:,1:],torch.randint(11,(batch,1),generator=generator)),1)
            neural,counts=experts(ids)
            for _,kappa in screen.MENU:
                loss,weights=screen.target_losses(neural,counts,target,kappa)
                actual=-SyntheticPredictor(kappa).predict_log_probs(ids).gather(-1,target[...,None]).squeeze(-1)
                torch.testing.assert_close(loss,actual,atol=2e-7,rtol=1e-7)
                self.assertEqual(weights.shape,(batch,length))

    def test_last_target_and_current_target_never_enter_current_weight(self):
        ids=torch.tensor([[1,2,3,4,5]])
        targets=torch.tensor([[2,3,4,5,6]])
        neural,counts=experts(ids)
        for _,kappa in screen.MENU:
            _,before=screen.target_losses(neural,counts,targets,kappa)
            for position in range(5):
                changed=targets.clone();changed[:,position]=9
                _,after=screen.target_losses(neural,counts,changed,kappa)
                torch.testing.assert_close(before[:,:position+1],after[:,:position+1],atol=0,rtol=0)

    def test_windows_are_independent_and_reset(self):
        ids=torch.tensor([[1,2,3,4],[7,8,9,10]])
        targets=torch.cat((ids[:,1:],ids[:,:1]),1);n,m=experts(ids)
        first=screen.target_losses(n,m,targets,8)
        for row in range(2):
            separate=screen.target_losses(n[row:row+1],m[row:row+1],targets[row:row+1],8)
            for a,b in zip(first,separate):torch.testing.assert_close(a[row:row+1],b,atol=0,rtol=0)
        again=screen.target_losses(n,m,targets,8)
        for a,b in zip(first,again):torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_padding_masks_only_nll_and_does_not_change_prior_weights(self):
        ids=torch.tensor([[1,2,3,4]]);targets=torch.tensor([[2,3,4,-100]])
        loss,_=screen.target_losses(*experts(ids),targets,32)
        self.assertEqual(loss[0,-1].item(),0.)
        valid,_=screen.target_losses(*experts(ids),torch.tensor([[2,3,4,5]]),32)
        torch.testing.assert_close(loss[:,:3],valid[:,:3],atol=0,rtol=0)

    def test_mixed_batch_multiple_trailing_pads_cannot_affect_valid_positions(self):
        ids=torch.tensor([[1,2,3,4,5],[2,7,0,0,0]])
        targets=torch.tensor([[2,3,4,5,6],[7,8,-100,-100,-100]])
        for _,kappa in screen.MENU:
            loss,_=screen.target_losses(*experts(ids),targets,kappa)
            full=-SyntheticPredictor(kappa).predict_log_probs(ids).gather(-1,targets.clamp_min(0)[...,None]).squeeze(-1)
            full.masked_fill_(targets==-100,0)
            torch.testing.assert_close(loss,full,atol=2e-7,rtol=1e-7)

    def test_disabled_control_and_first_position_use_exact_parent_arithmetic(self):
        ids=torch.tensor([[1,3,7]]);targets=torch.tensor([[3,7,8]]);n,m=experts(ids)
        fixed=n.clone().mul_(.85).add_(m,alpha=.15).clamp_min_(1e-30).log_()
        control,_=screen.target_losses(n,m,targets,None)
        torch.testing.assert_close(control,-fixed.gather(-1,targets[...,None]).squeeze(-1),atol=0,rtol=0)
        for kappa in (8,32,128):
            dynamic,_=screen.target_losses(n,m,targets,kappa)
            torch.testing.assert_close(dynamic[:,0],control[:,0],atol=0,rtol=0)

    def test_input_tensors_are_not_mutated(self):
        ids=torch.tensor([[1,2]]);n,m=experts(ids);a,b=n.clone(),m.clone()
        screen.target_losses(n,m,torch.tensor([[2,3]]),8)
        self.assertTrue(torch.equal(n,a));self.assertTrue(torch.equal(m,b))

    def test_invalid_precision_shape_or_denominator_refused(self):
        n,m=experts(torch.tensor([[1,2]]));targets=torch.tensor([[2,3]])
        for first,second in ((n.double(),m.double()),(n[:,:1],m),(n*float('nan'),m)):
            with self.assertRaises((ValueError,TypeError)):screen.target_losses(first,second,targets,8)


class MetadataTests(unittest.TestCase):
    def test_fixed_four_menu_rank_and_control_anchor(self):
        lock,selection=fixture();screen.check_selection(selection,lock)
        self.assertEqual(selection['resource_audit_candidates'][0]['name'],'k128')
        selection['trials'][0]['validation_bpb_gpu_fp32']+=.001
        with self.assertRaises(ValueError):screen.check_selection(selection,lock)

    def test_extra_menu_test_based_selection_or_changed_gain_refused(self):
        for fault in ('menu','test','gain'):
            lock,selection=fixture()
            if fault=='menu':selection['trials'].append(copy.deepcopy(selection['trials'][1]))
            elif fault=='test':selection['test_used']=True
            else:selection['trials'][1]['passes_validation_improvement_gate']=True
            if fault=='gain':selection['trials'][1]['improvement_vs_parent']=.01
            with self.assertRaises(ValueError):screen.check_selection(selection,lock)

    def test_full_validation_denominator_and_no_short_split(self):
        with self.assertRaises(ValueError):screen.score_metadata(10,20,1.)
        lock,selection=fixture();selection['trials'][2]['screen_measurement']['nll_nats']=1.
        with self.assertRaises(ValueError):screen.check_selection(selection,lock)

    def test_real_confirmation_failure_only_after_provenance(self):
        lock,selection=fixture();row=selection['resource_audit_candidates'][0]
        record=dict(status='failed',plan_sha256=screen.PLAN_SHA,parent_lock_sha256='l'*64,selection_sha256='s'*64,
            resource_rank=0,name=row['name'],kappa=row['kappa'],candidate_confirmation_started=True,test_used=False,
            no_automatic_retry=True,failure_type='AssertionError',failure_message='')
        screen.verify_confirmation_failure(record,'l'*64,'s'*64,0,row)
        record['candidate_confirmation_started']=False
        with self.assertRaises(ValueError):screen.verify_confirmation_failure(record,'l'*64,'s'*64,0,row)

    def test_confirmation_binds_full_scorer_and_saved_checkpoint(self):
        lock,selection=fixture();row=selection['resource_audit_candidates'][0]
        lock['source_sha256']=dict(**{'push153_screen.py':'s'*64,'push153_mixture.py':'m'*64,'evaluate.py':'e'*64})
        measured=dict(row['screen_measurement'],checkpoint_sha256='c'*64,
            implementation_sha256='m'*64,evaluator_sha256='e'*64)
        record=dict(status='completed',plan_sha256=screen.PLAN_SHA,resource_rank=0,name=row['name'],kappa=row['kappa'],
            implementation='push153_mixture',selected_lineage_targets=802816000,all_parent_tensors_bitwise_identical=True,
            statistics_buffers=21,parameters=6279315,new_training_targets=0,test_used=False,
            source_sha256=lock['source_sha256'],script_sha256='s'*64,implementation_sha256='m'*64,
            checkpoint_sha256='c'*64,measurement=measured,shortcut_full_bpb_absolute_difference=0.)
        screen.check_confirmation(record,lock,selection,0)
        measured['checkpoint_sha256']='z'*64
        with self.assertRaises(ValueError):screen.check_confirmation(record,lock,selection,0)

    def test_candidate_changes_no_tensor_or_ancestry(self):
        state={'token.weight':torch.arange(6.).reshape(2,3),'head.weight':torch.arange(6.).reshape(2,3)}
        state.update({'ngram.buffer'+str(i):torch.tensor([i],dtype=torch.int64) for i in range(21)})
        parent=dict(protocol=screen.PROTOCOL,implementation=screen.PARENT_IMPLEMENTATION,train_tokens=123,
            config=dict(temperature=1.3),model=state)
        candidate=screen.candidate_checkpoint(parent,8,{'parent':{'checkpoint_sha256':'a'*64}},torch)
        self.assertEqual(candidate['train_tokens'],123)
        self.assertEqual(candidate['implementation'],'push153_mixture')
        self.assertEqual(candidate['config'],dict(temperature=1.3,push153_kappa=8))
        for name,value in state.items():self.assertTrue(torch.equal(value,candidate['model'][name]))
        self.assertNotIn('push153_kappa',parent['config'])

    def test_canonical_configuration_hash_order_independent(self):
        self.assertEqual(screen.canonical_sha(dict(a=1,b=2)),screen.canonical_sha(dict(b=2,a=1)))
        self.assertNotEqual(screen.canonical_sha(dict(a=1)),screen.canonical_sha(dict(a=2)))

    def test_metadata_mode_imports_no_tensor_runtime_at_module_level(self):
        tree=ast.parse(inspect.getsource(screen))
        names=[entry.name for node in tree.body if isinstance(node,ast.Import) for entry in node.names]
        self.assertNotIn('torch',names);self.assertNotIn('numpy',names)
        calls=[ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)]
        self.assertNotIn('load_data',calls);self.assertNotIn('common.load_data',calls)

    def test_paths_and_exclusive_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):screen.child(directory,'../bad')
            with self.assertRaises(ValueError):screen.sha(Path(directory)/'wikitext_test.txt')
            path=Path(directory)/'fixture.json';screen.write_new(path,{})
            with self.assertRaises(FileExistsError):screen.write_new(path,{})


if __name__=='__main__':unittest.main()
