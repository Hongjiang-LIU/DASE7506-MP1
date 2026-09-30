"""Tiny synthetic CPU tests; no real corpus/checkpoint, scoring or GPU use."""
import ast
import copy
import inspect
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common import PROTOCOL, ROOT, make_model, sha
from push147_train import training_loss as old_loss
from push150_crossfit import contiguous_folds
from push150_oof_prepare import valid_start_ranges, gather_window_probabilities
import push150_oof_train as trainer


class OOFTrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        random.seed(211); np.random.seed(211); torch.manual_seed(211)

    def models(self):
        config=dict(vocab=2048,width=16,heads=2,depth=1,context=256,
            ffn_hidden=32,rope=True,lowrank_rank=4,pointer_rank=4,
            dropout=.2,temperature=1.25,cache_weight=.05,cache_theta=15.,
            push150_ranks=dict(qkv=6,proj=0,gate=4,value=4,out=4))
        model,_=make_model('push150_expand',trainer.training_config(config,.2),torch.device('cpu'))
        averaged=copy.deepcopy(model).eval().requires_grad_(False)
        parent=dict(protocol=PROTOCOL,implementation='push150_expand',config=config,
            model=trainer.model_state(model),train_tokens=540672000,ancestry={'shared':'phase A'})
        return model,averaged,parent

    def metadata(self):
        tokens=7680
        parent=dict(implementation='push150_expand',train_tokens=540672000)
        parent_sha='a'*64; selection_sha='b'*64
        selection=dict(status='phase_a_choice_complete',test_used=False,evidence_errors=[],
            winner=dict(checkpoint_sha256=parent_sha,selected_lineage_targets=parent['train_tokens'],
                        implementation=parent['implementation']))
        train=dict(split='train',tokens=tokens,text_sha256='c'*64,utf8_bytes=10000,
                   tokenizer_sha256='d'*64,manifest_sha256='e'*64)
        folds=[dict(fold=index,start=start,end=end,valid_random_starts=end-start-256)
               for index,(start,end) in enumerate(contiguous_folds(tokens))]
        metadata=dict(status='completed',test_used=False,plan_sha256=trainer.PLAN_SHA256,
            parent_sha256=parent_sha,parent_lineage_targets=parent['train_tokens'],
            phase_a_selection_sha256=selection_sha,train=train,folds=folds,
            array=dict(path='train_oof_probabilities.npy',dtype='float32',shape=[tokens,4],
                       maximum_orders=[2,3,4,5]),parity={'passes':True},costs={'target_queries':tokens*4},
            source_sha256={'common.py':sha(ROOT/'common.py')},gate_identity_sha256='f'*64,
            gate_build_sha256='0'*64)
        return metadata,parent,parent_sha,selection,selection_sha,train

    def equal(self,first,second):
        if torch.is_tensor(first):
            torch.testing.assert_close(first,second,atol=0,rtol=0)
        elif isinstance(first,dict):
            self.assertEqual(set(first),set(second))
            for key in first:
                self.equal(first[key],second[key])
        elif isinstance(first,(list,tuple)):
            self.assertEqual(type(first),type(second)); self.assertEqual(len(first),len(second))
            for left,right in zip(first,second):
                self.equal(left,right)
        else:
            self.assertEqual(first,second)

    def test_uniform_sampler_is_bijection_over_all_legal_starts(self):
        ranges=torch.tensor([[0,2],[10,14]],dtype=torch.long)
        with patch.object(torch,'randint',side_effect=[torch.arange(6),torch.tensor(1)]):
            starts,keep=trainer.sample_starts_and_coin(torch.Generator(),ranges,6)
        self.equal(starts,torch.tensor([0,1,10,11,12,13]))
        self.assertTrue(keep)

    def test_sampler_keeps_entire_257_token_windows_inside_one_fold(self):
        metadata,*_=self.metadata()
        ranges=torch.from_numpy(valid_start_ranges(metadata['folds']))
        sampler=torch.Generator().manual_seed(211)
        for _ in range(6):
            starts,keep=trainer.sample_starts_and_coin(sampler,ranges)
            self.assertIsInstance(keep,bool)
            for start in starts.tolist():
                containing=[fold for fold in metadata['folds']
                            if fold['start']<=start and start+256<fold['end']]
                self.assertEqual(len(containing),1)

    def test_control_joint_share_starts_and_coin_without_global_rng_consumption(self):
        ranges=torch.tensor([[2,30],[45,66]],dtype=torch.long)
        first=torch.Generator().manual_seed(211); second=torch.Generator().manual_seed(211)
        global_state=torch.get_rng_state().clone()
        coins=[]
        for _ in range(20):
            self.equal(trainer.sample_starts_and_coin(first,ranges,3),
                       trainer.sample_starts_and_coin(second,ranges,3))
            coins.append(trainer.sample_starts_and_coin(first,ranges,1)[1])
            trainer.sample_starts_and_coin(second,ranges,1)
        self.assertIn(True,coins); self.assertIn(False,coins)
        self.equal(torch.get_rng_state(),global_state)
        self.equal(first.get_state(),second.get_state())

    def test_window_probability_columns_reset_at_every_arbitrary_start(self):
        array=np.stack([np.full(80,.01*(column+1),dtype=np.float32) for column in range(4)],axis=1)
        gathered=gather_window_probabilities(array,np.array([2,17,35],dtype=np.int64),context=8)
        expected=np.array([[.01,.02,.03,.04,.04,.04,.04,.04]]*3,dtype=np.float32)
        np.testing.assert_array_equal(gathered,expected)

    def test_control_and_dropped_joint_equal_original_probability_nll(self):
        neural=torch.tensor([[.2,.7,.01],[.3,.1,.6]],requires_grad=True)
        counts=torch.full_like(neural,.8,requires_grad=True)
        expected=-neural.log().mean()
        for objective,keep in (('control',False),('control',True),('joint',False)):
            self.equal(trainer.mixture_target_loss(neural,counts,objective,keep),expected)
        actual=trainer.mixture_target_loss(neural,counts,'joint',True)
        self.equal(actual,-(.85*neural+.15*counts.detach()).log().mean())

    def test_mixture_is_normalized_for_kept_or_dropped_whole_count_block(self):
        neural=torch.tensor([[[.1,.2,.7]]]); counts=torch.tensor([[[.4,.2,.4]]])
        for keep in (True,False):
            probability=[]
            for target in range(3):
                loss=trainer.mixture_target_loss(neural[...,target],counts[...,target],'joint',keep)
                probability.append(loss.exp().reciprocal())
            torch.testing.assert_close(torch.stack(probability).sum(),torch.tensor(1.),atol=1e-7,rtol=0)

    def test_count_targets_have_no_gradient_and_neural_gradients_are_finite(self):
        logits=torch.randn(2,8,requires_grad=True)
        counts=torch.full((2,8),.03,requires_grad=True)
        loss=trainer.mixture_target_loss(logits.sigmoid(),counts,'joint',True)
        loss.backward()
        self.assertIsNone(counts.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(),0)

    def test_control_model_loss_and_inherited_gradients_match_original(self):
        model,_,_=self.models(); other=copy.deepcopy(model)
        ids=torch.tensor([[1,2,3,4,1,2,3,4],[5,6,5,6,5,6,5,6]])
        targets=ids.roll(-1,dims=1); counts=torch.full((2,8),.04)
        random_state=torch.get_rng_state().clone()
        expected=old_loss(model,ids,targets,0.)
        torch.set_rng_state(random_state)
        actual=trainer.training_loss(other,ids,targets,counts,'control',True)
        self.equal(actual,expected)
        expected.backward(); actual.backward()
        for first,second in zip(model.parameters(),other.parameters()):
            self.equal(first.grad,second.grad)

    def test_coin_accounting_retains_discarded_and_failed_attempts(self):
        log=dict(count_block_coins=trainer.empty_coin_counts(),
                 discarded_count_block_coins=trainer.empty_coin_counts())
        trainer.add_coin(log,True); trainer.add_coin(log,False)
        disk=copy.deepcopy(log)
        trainer.add_coin(disk,True); trainer.add_coin(disk,True)
        extra=trainer.retain_discarded_coins(log,disk)
        self.assertEqual(extra,dict(drawn=2,kept=2,dropped=0))
        self.assertEqual(log['discarded_count_block_coins'],extra)
        self.assertEqual(log['count_block_coins'],dict(drawn=2,kept=1,dropped=1))

    def test_complete_next_stochastic_update_resume_preserves_coin_and_two_optimizers(self):
        model,averaged,_=self.models()
        optimizers=trainer.build_optimizers(model,'muon',7.5e-6,7.5e-5,.00025,.0005)[0]
        sampler=torch.Generator().manual_seed(211)
        identity=dict(recipe={'steps':4,'objective':'joint'},parent_sha256='synthetic',
                      oof_preparation={'array_sha256':'synthetic-fixed-array'})
        tokens=torch.arange(80)%13+1
        ranges=torch.tensor([[0,22],[31,72]],dtype=torch.long)
        array=np.full((80,4),.03,dtype=np.float32)
        def advance(network,average,opts,rng):
            network.train()
            starts,keep=trainer.sample_starts_and_coin(rng,ranges,2)
            batch=tokens[starts[:,None]+torch.arange(9)]
            counts=torch.from_numpy(gather_window_probabilities(array,starts.numpy(),context=8))
            noise=(random.random(),float(np.random.random()))
            for optimizer in opts:
                optimizer.zero_grad(set_to_none=True)
            loss=trainer.training_loss(network,batch[:,:-1],batch[:,1:],counts,'joint',keep)
            loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(),1.)
            for optimizer in opts:
                optimizer.step()
            network.project_cache_parameters()
            with torch.no_grad():
                for target,current in zip(average.parameters(),network.parameters()):
                    target.lerp_(current,.01)
            return starts,keep,loss.detach(),noise
        advance(model,averaged,optimizers,sampler)
        state=trainer.pack_train_state(model,averaged,optimizers,sampler,identity,1,{'completed_steps':1})
        stream=io.BytesIO(); torch.save(state,stream); stream.seek(0)
        saved=torch.load(stream,weights_only=True)
        expected_result=advance(model,averaged,optimizers,sampler)
        expected=trainer.pack_train_state(model,averaged,optimizers,sampler,identity,2,{'completed_steps':2})
        resumed,ema,_=self.models()
        new_optimizers=trainer.build_optimizers(resumed,'muon',7.5e-6,7.5e-5,.00025,.0005)[0]
        new_sampler=torch.Generator().manual_seed(3)
        trainer.restore_train_state(saved,resumed,ema,new_optimizers,new_sampler,identity)
        actual_result=advance(resumed,ema,new_optimizers,new_sampler)
        actual=trainer.pack_train_state(resumed,ema,new_optimizers,new_sampler,identity,2,{'completed_steps':2})
        self.equal(actual_result,expected_result); self.equal(actual,expected)

    def test_changed_cache_or_objective_prevents_resume_before_mutation(self):
        model,averaged,_=self.models()
        optimizers=[torch.optim.SGD(model.parameters(),lr=.01)]
        sampler=torch.Generator().manual_seed(211)
        identity=dict(recipe={'steps':12000,'objective':'joint'},oof_preparation={'array_sha256':'original'})
        state=trainer.pack_train_state(model,averaged,optimizers,sampler,identity,0,{'completed_steps':0})
        original=trainer.model_state(model)
        for changed in (dict(identity,oof_preparation={'array_sha256':'other'}),
                        dict(identity,recipe={'steps':12000,'objective':'control'})):
            with self.assertRaises(ValueError):
                trainer.restore_train_state(state,model,averaged,optimizers,sampler,changed)
            self.equal(trainer.model_state(model),original)

    def test_metadata_requires_completed_same_parent_same_train_and_probability_parity(self):
        values=self.metadata()
        self.assertEqual(len(trainer.validate_prepared_metadata(*values)),10)
        for key,value in (('status','failed'),('test_used',True),('parent_sha256','wrong'),
                          ('parent_lineage_targets',1),('plan_sha256','wrong'),('parity',{'passes':False})):
            altered=copy.deepcopy(values); altered[0][key]=value
            with self.assertRaises(ValueError):
                trainer.validate_prepared_metadata(*altered)
        altered=copy.deepcopy(values); altered[0]['folds'][0]['end']+=1
        with self.assertRaises(ValueError):
            trainer.validate_prepared_metadata(*altered)
        altered=copy.deepcopy(values); altered[0]['array']['maximum_orders']=[1,2,3,4]
        with self.assertRaises(ValueError):
            trainer.validate_prepared_metadata(*altered)

    def test_training_only_array_loader_checks_file_hash_and_shape(self):
        metadata,parent,parent_sha,selection,_,train=self.metadata()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            plan=path/'plan.json'; plan.write_text('{}',encoding='utf-8')
            selected=path/'selection.json'; selected.write_text(json.dumps(selection),encoding='utf-8')
            identity=path/'identity.json'; identity.write_text('{}',encoding='utf-8')
            array_path=path/'train_oof_probabilities.npy'
            np.save(array_path,np.full((train['tokens'],4),.03,dtype=np.float32))
            metadata.update(plan_sha256=sha(plan),phase_a_selection_sha256=sha(selected),identity_sha256=sha(identity))
            metadata['array'].update(sha256=sha(array_path),bytes=array_path.stat().st_size)
            (path/'metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
            with patch.object(trainer,'PLAN_SHA256',sha(plan)):
                array,ranges,provenance,pins=trainer.load_prepared_cache(path,parent,parent_sha,selected,train,plan)
                self.assertEqual(array.shape,(train['tokens'],4)); self.assertEqual(ranges.shape,(10,2))
                self.assertTrue(provenance['training_only_array_excluded_from_inference_assets'])
                self.assertEqual(provenance['reused_B_count_scans_counted_again_in_D'],0)
                self.assertEqual(pins[str(array_path)],sha(array_path))
                del array
                metadata['array']['sha256']='0'*64
                (path/'metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'OOF array'):
                    trainer.load_prepared_cache(path,parent,parent_sha,selected,train,plan)

    def test_snapshot_keeps_architecture_and_lineage_without_training_arrays(self):
        model,_,parent=self.models()
        saved=trainer.snapshot(model,parent,parent['config'],'a'*64,{'seed':211},12000,'ema')
        self.assertEqual(saved['implementation'],'push150_expand')
        self.assertEqual(saved['train_tokens'],638976000)
        self.assertEqual(set(saved['model']),set(parent['model']))
        self.assertIs(saved['model']['head.weight'],saved['model']['token.weight'])
        self.assertFalse(any('oof' in name for name in saved['model']))

    def test_source_pins_preparer_and_driver_and_main_has_only_train_validation(self):
        sources=trainer.source_digests('push150_expand')
        self.assertTrue({'push150_oof_train.py','push150_oof_prepare.py','push150_expand.py',
                         'push150_crossfit.py','push148_muon.py','evaluate.py'}<=set(sources))
        source=inspect.getsource(trainer.run); tree=ast.parse(source)
        corpus_loop=next(node for node in ast.walk(tree) if isinstance(node,ast.For)
                         and isinstance(node.target,ast.Name) and node.target.id=='split')
        self.assertEqual(ast.literal_eval(corpus_loop.iter),('train','validation'))
        self.assertIn('sample_starts_and_coin(sampler,legal_ranges)',source)
        self.assertIn('count_targets,args.objective,keep_count',source)
        self.assertIn('declared_raw_ema_validation_choices=13',source)
        self.assertIn('except BaseException',source)

    def test_preflight_failure_record_preserved_only_for_confirmed_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            args=SimpleNamespace(run_dir=path,parent=Path('synthetic.pt'),objective='joint',
                                 _run_directory_confirmed=True)
            for _ in range(2):
                with patch.object(trainer,'parse_args',return_value=args), patch.object(
                        trainer,'run',side_effect=ValueError('synthetic cache mismatch')):
                    with self.assertRaisesRegex(ValueError,'synthetic cache'):
                        trainer.main()
            self.assertEqual(len(list(path.glob('preflight-failure-*.json'))),2)
            report=json.loads((path/'preflight-failure-001.json').read_text())
            self.assertEqual(report['new_completed_neural_updates'],0)
            self.assertFalse(report['test_used'])
            args._run_directory_confirmed=False
            with patch.object(trainer,'parse_args',return_value=args), patch.object(
                    trainer,'run',side_effect=FileExistsError('unrelated existing directory')):
                with self.assertRaises(FileExistsError):
                    trainer.main()
            self.assertEqual(len(list(path.glob('preflight-failure-*.json'))),2)


if __name__=='__main__':
    unittest.main()
