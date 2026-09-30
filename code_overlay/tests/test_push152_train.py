"""Small CPU/synthetic Phase-B tests: no corpus, checkpoint, scoring or GPU."""
import copy
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import push152_train as t
from push147_train import training_loss
from push150_train import capture_rng, restore_rng


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=dict(vocab=11,width=4,context=256,temperature=1.15,dropout=.2)
        self.token=nn.Embedding(11,4)
        self.head=nn.Linear(4,11,bias=False); self.head.weight=self.token.weight
        self.blocks=nn.ModuleList([nn.Linear(4,4,bias=False)])
        self.pointer_gate=nn.Linear(4,1)
        self.cache_scalar=nn.Parameter(torch.tensor(0.))
        self.feature_calls=0

    def features(self,ids):
        self.feature_calls += 1
        return self.blocks[0](F.dropout(self.token(ids),p=.2,training=self.training))

    def cache_attention_and_gate(self,features):
        length=features.shape[1]
        legal=torch.arange(length)[None,:] < torch.arange(length)[:,None]
        logits=features@features.transpose(-2,-1)
        attention=logits.masked_fill(~legal,-1e4).softmax(-1)
        gate=(self.pointer_gate(features)+self.cache_scalar).sigmoid()*.2
        gate=gate*(torch.arange(length)>0)[None,:,None]
        return attention,gate


def plan_fixture():
    rates=dict(backbone=7.5e-6,pointer=7.5e-5,scalar=2.5e-4,muon=5e-4)
    phase={key:t.COMMON[key] for key in ('steps','eval_every','seed','schedule','warmup','decay_steps','lr_floor','ema_decay')}
    phase.update(learning_rates=rates,new_targets_per_completed_branch=163840000,variants={
        name:dict(run_dir='runs/push152-'+name,parent='runs/push152-expanded-init/checkpoint.pt' if name=='expanded' else t.PARENT_PATH,
                  auxiliary_weight=.1 if name=='auxiliary' else 0.) for name in t.BRANCHES})
    shared={key:t.COMMON[key] for key in ('optimizer','batch_size','context','dropout','training_temperature',
                                         'validation_temperature','gradient_clip')}
    shared.update(mkn_weight=.15,targets_per_step=8192,training_split='train',selection_split='validation',
        training_precision='bf16',validation_precision='fp32',validation_device='cuda',no_calibration_grid=True)
    return dict(phase_b=phase,shared=shared,parent=dict(path=t.PARENT_PATH,sha256=t.PARENT_SHA,
        selected_lineage_targets=t.LINEAGE),external_training_data_or_weights=False,evaluation_network_access=False,
        expansion=dict(authorization='runs/push152-expanded-preflight/authorization.json'))


def parent_fixture(model):
    state={name:value.detach().clone() for name,value in model.state_dict().items()}
    state.update({'ngram.synthetic_'+str(i):torch.tensor([i],dtype=torch.float32) for i in range(21)})
    return dict(protocol=t.PROTOCOL,implementation=t.IMPLEMENTATION,train_tokens=t.LINEAGE,
                model=state,config=copy.deepcopy(model.config),ancestry={'shared_parent_once':True})


class PhaseBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads(); torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(31)

    def test_plan_exact_three_variants_and_alpha(self):
        plan=plan_fixture()
        for name in t.BRANCHES:
            result=t.branch_spec(plan,name)
            self.assertEqual(result['run_dir'],'runs/push152-'+name)
            self.assertEqual(result['auxiliary_weight'],.1 if name=='auxiliary' else 0.)
        self.assertNotIn('name',plan['phase_b']['variants']['control'])

    def test_changed_plan_budget_seed_or_menu_refused(self):
        for key,value in [('steps',24000),('seed',194),('eval_every',2000)]:
            plan=plan_fixture(); plan['phase_b'][key]=value
            with self.assertRaises(ValueError): t.branch_spec(plan,'control')
        plan=plan_fixture(); plan['phase_b']['variants']['auxiliary']['auxiliary_weight']=.2
        with self.assertRaises(ValueError): t.branch_spec(plan,'auxiliary')
        plan=plan_fixture(); plan['phase_b']['variants']['fourth']={}
        with self.assertRaises(ValueError): t.branch_spec(plan,'expanded')

    def test_training_config_strips_statistics_eval_keeps_current_tables(self):
        config=dict(vocab=11,temperature=1.3,dropout=.2,mkn_weight=.15,mkn_order=5,
            mkn_shapes={'unigram':[11]},discount_scale=.7,push150_ranks={'gate':176})
        before=copy.deepcopy(config)
        recipe=t.recipe_for(config,'auxiliary',.1)
        self.assertEqual(recipe['training_config']['temperature'],1.15)
        self.assertFalse(any(name.startswith(('ngram_','mkn_')) for name in recipe['training_config']))
        self.assertNotIn('discount_scale',recipe['training_config'])
        self.assertEqual(recipe['validation_config']['temperature'],1.3)
        self.assertEqual(recipe['validation_config']['mkn_shapes'],config['mkn_shapes'])
        self.assertEqual(config,before)

    def test_alpha_zero_exact_existing_loss_and_gradient(self):
        model=Toy(); other=copy.deepcopy(model)
        ids=torch.randint(11,(2,7)); targets=torch.randint(11,(2,7))
        torch.manual_seed(17); actual=t.branch_loss(model,ids,targets,0.); actual.backward()
        torch.manual_seed(17); expected=training_loss(other,ids,targets,0.); expected.backward()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for a,b in zip(model.parameters(),other.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)

    def test_auxiliary_point_one_matches_declared_mixture_one_features(self):
        model=Toy().eval(); ids=torch.randint(11,(2,7)); targets=torch.randint(11,(2,7))
        actual=t.branch_loss(model,ids,targets,.1)
        self.assertEqual(model.feature_calls,1)
        mixed=training_loss(model,ids,targets,0.)
        neural=training_loss(model,ids,targets,1.)
        torch.testing.assert_close(actual,.9*mixed+.1*neural,rtol=0,atol=0)
        actual.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_statistics_are_rejected_before_any_feature_call(self):
        model=Toy(); model.ngram=nn.Linear(1,1)
        with self.assertRaises(ValueError): t.branch_loss(model,torch.zeros(1,3,dtype=torch.long),torch.zeros(1,3,dtype=torch.long),.1)
        self.assertEqual(model.feature_calls,0)

    def test_unplanned_loss_coefficient_refused(self):
        for alpha in (-.1,.2,float('nan'),1.):
            with self.assertRaises(ValueError): t.branch_loss(Toy(),torch.zeros(1,3,dtype=torch.long),torch.zeros(1,3,dtype=torch.long),alpha)

    def test_two_loss_terms_do_not_double_unique_targets(self):
        control=t.target_accounting(20000,0,0.); aux=t.target_accounting(20000,0,.1)
        self.assertEqual(control['incremental_search_targets'],163840000)
        self.assertEqual(aux['incremental_search_targets'],163840000)
        self.assertEqual(aux['mixed_nll_target_exposures'],aux['auxiliary_neural_nll_target_exposures'])
        self.assertEqual(control['auxiliary_neural_nll_target_exposures'],0)

    def test_replayed_targets_count_cost_not_model_lineage(self):
        value=t.target_accounting(4000,399,0.)
        self.assertEqual(value['incremental_search_targets'],4399*8192)
        self.assertEqual(value['lineage_new_targets'],4000*8192)

    def test_sample_final_legal_start_is_reachable(self):
        sampler=torch.Generator().manual_seed(193)
        self.assertTrue(torch.equal(t.sample_starts(257,sampler),torch.zeros(32,dtype=torch.long)))
        values=t.sample_starts(258,sampler,1024)
        self.assertEqual(set(values.tolist()),{0,1})
        with self.assertRaises(ValueError): t.sample_starts(256,sampler)

    def test_sampler_independent_of_global_dropout_draws(self):
        first=torch.Generator().manual_seed(193); second=torch.Generator().manual_seed(193)
        a=t.sample_starts(400,first); torch.rand(93); b=t.sample_starts(400,second)
        self.assertTrue(torch.equal(a,b))

    def test_schedule_starts_warm_and_ends_exact_floor(self):
        self.assertAlmostEqual(t.schedule_factor(1,20000,'cosine',200,4000),.005)
        self.assertEqual(t.schedule_factor(20000,20000,'cosine',200,4000),.1)
        self.assertGreater(t.schedule_factor(200,20000,'cosine',200,4000),.99)

    def test_snapshot_preserves_tie_statistics_ancestry_and_temperature(self):
        model=Toy(); parent=parent_fixture(model); recipe=t.recipe_for(parent['config'],'auxiliary',.1)
        result=t.make_snapshot(model,parent,recipe['validation_config'],'a'*64,recipe,20000,'ema',t.PLAN_SHA)
        self.assertIs(result['model']['head.weight'],result['model']['token.weight'])
        self.assertEqual(result['train_tokens'],t.LINEAGE+163840000)
        self.assertEqual(result['config']['temperature'],1.3)
        self.assertEqual(result['training_config']['temperature'],1.15)
        self.assertEqual(result['auxiliary_neural_weight'],.1)
        self.assertEqual(result['push152_plan_sha256'],t.PLAN_SHA)
        for name,value in parent['model'].items():
            if name.startswith('ngram.'):
                self.assertTrue(torch.equal(value,result['model'][name])); self.assertNotEqual(value.data_ptr(),result['model'][name].data_ptr())

    def test_snapshot_archive_is_exclusive_and_reloadable(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'synthetic.pt'
            value={'weight':torch.tensor([1.]),'plan_sha256':t.PLAN_SHA}
            t.save_snapshot_new(path,value); original=t.sha(path)
            reloaded=torch.load(path,map_location='cpu',weights_only=True)
            self.assertEqual(reloaded['plan_sha256'],t.PLAN_SHA)
            self.assertTrue(torch.equal(reloaded['weight'],value['weight']))
            with self.assertRaises(FileExistsError): t.save_snapshot_new(path,{'weight':torch.tensor([2.])})
            self.assertEqual(t.sha(path),original)

    def test_only_eleven_declared_snapshot_choices(self):
        model=Toy(); parent=parent_fixture(model); recipe=t.recipe_for(parent['config'],'control',0.)
        choices=[(0,'raw')]+[(step,label) for step in range(4000,20001,4000) for label in ('raw','ema')]
        self.assertEqual(len(choices),11)
        for step,label in choices: t.make_snapshot(model,parent,recipe['validation_config'],'a'*64,recipe,step,label,t.PLAN_SHA)
        for step,label in [(0,'ema'),(1000,'raw'),(24000,'raw')]:
            with self.assertRaises(ValueError): t.make_snapshot(model,parent,recipe['validation_config'],'a'*64,recipe,step,label,t.PLAN_SHA)

    def test_statistics_dtype_value_shape_all_immutable(self):
        parent=parent_fixture(Toy()); fixed=copy.deepcopy(parent); t.validate_statistics(parent,fixed)
        for operation in ('value','dtype','remove'):
            changed=copy.deepcopy(parent); name='ngram.synthetic_0'
            if operation=='value': changed['model'][name]+=1
            elif operation=='dtype': changed['model'][name]=changed['model'][name].double()
            else: del changed['model'][name]
            with self.assertRaises(ValueError): t.validate_statistics(changed,fixed)

    def test_optimizer_partition_unique_tied_embedding_not_muon(self):
        model=Toy(); optimizers,names=t.build_optimizers(model,'muon',7.5e-6,7.5e-5,2.5e-4,5e-4)
        self.assertEqual(names,['blocks.0.weight'])
        ids=[id(p) for optimizer in optimizers for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids))); self.assertEqual(set(ids),{id(p) for p in model.parameters()})
        self.assertEqual(len(optimizers),2)

    def test_full_state_restores_exact_next_step_two_optimizers_and_rng(self):
        model=Toy(); averaged=copy.deepcopy(model).eval().requires_grad_(False)
        optimizers,_=t.build_optimizers(model,'muon',7.5e-6,7.5e-5,2.5e-4,5e-4)
        for group in optimizers[0].param_groups: group['fused']=False
        sampler=torch.Generator().manual_seed(193)
        identity=dict(recipe={'steps':20000},plan_sha256=t.PLAN_SHA,parent_sha256='a'*64,source_sha256={'fixture':'b'*64})
        def update():
            model.train(); ids=torch.randint(11,(2,7),generator=sampler); targets=(ids+1)%11
            for optimizer in optimizers: optimizer.zero_grad(set_to_none=True)
            loss=t.branch_loss(model,ids,targets,.1); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            for optimizer in optimizers: optimizer.step()
            with torch.no_grad():
                for a,b in zip(averaged.parameters(),model.parameters()): a.lerp_(b,.01)
            return loss.detach().clone()
        update()
        state=t.pack_train_state(model,averaged,optimizers,sampler,identity,1,{'completed_steps':1},cuda=False)
        expected=update(); expected_raw=t.cpu_clone(model.state_dict()); expected_ema=t.cpu_clone(averaged.state_dict())
        restored,log=t.restore_train_state(state,model,averaged,optimizers,sampler,identity,cuda=False)
        self.assertEqual(restored,1); actual=update()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for name,value in model.state_dict().items(): torch.testing.assert_close(value,expected_raw[name],rtol=0,atol=0)
        for name,value in averaged.state_dict().items(): torch.testing.assert_close(value,expected_ema[name],rtol=0,atol=0)
        self.assertTrue(state['optimizers'][0]['state']); self.assertTrue(state['optimizers'][1]['state'])

    def test_resume_changed_recipe_parent_plan_source_refused_before_mutation(self):
        model=Toy(); averaged=copy.deepcopy(model); optimizers,_=t.build_optimizers(model,'muon',7.5e-6,7.5e-5,2.5e-4,5e-4)
        sampler=torch.Generator().manual_seed(193)
        identity=dict(recipe={'steps':20000},plan_sha256='a',parent_sha256='b',source_sha256={'x':'y'})
        state=t.pack_train_state(model,averaged,optimizers,sampler,identity,0,{'completed_steps':0},cuda=False)
        before=t.cpu_clone(model.state_dict()); rng=sampler.get_state().clone()
        for key in identity:
            changed=copy.deepcopy(identity); changed[key]='different'
            with self.assertRaises(ValueError): t.restore_train_state(state,model,averaged,optimizers,sampler,changed,cuda=False)
        for name,value in model.state_dict().items(): self.assertTrue(torch.equal(value,before[name]))
        self.assertTrue(torch.equal(rng,sampler.get_state()))

    def test_python_numpy_torch_sampler_rng_all_restore(self):
        random.seed(12); np.random.seed(12); torch.manual_seed(12); sampler=torch.Generator().manual_seed(12)
        state=capture_rng(sampler,cuda=False)
        expected=(random.random(),np.random.rand(),torch.rand(3),torch.rand(3,generator=sampler))
        restore_rng(state,sampler,cuda=False)
        actual=(random.random(),np.random.rand(),torch.rand(3),torch.rand(3,generator=sampler))
        self.assertEqual(expected[:2],actual[:2])
        self.assertTrue(torch.equal(expected[2],actual[2])); self.assertTrue(torch.equal(expected[3],actual[3]))

    def test_source_graph_pins_new_and_reused_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            sources={'push152_train.py':'from helper import thing\n','helper.py':'import nested\n',
                     'nested.py':'value=1\n','push150_expand.py':'pass\n','push148_muon.py':'pass\n'}
            for name,text in sources.items(): (root/name).write_text(text,encoding='utf-8')
            result=t.source_digests(root=root)
            self.assertEqual(set(result),set(sources))
            before=result['nested.py']; (root/'nested.py').write_text('value=2\n',encoding='utf-8')
            self.assertNotEqual(t.source_digests(root=root)['nested.py'],before)

    def test_allowed_loader_menu_has_no_test_path(self):
        self.assertEqual(set(t.DATA_SHA),{'data/tokenizer.json','data/manifest.json','data/wikitext_train.txt','data/wikitext_validation.txt'})


class ResourceTests(unittest.TestCase):
    def timing(self):
        from push150_final_test import BASELINE_SHA,SCORER_SHA
        order=[(0,'baseline'),(0,'candidate'),(1,'candidate'),(1,'baseline'),(2,'baseline'),(2,'candidate')]
        return dict(test_used=False,split='validation',device='cpu',precision='fp32',threads=4,affinity_mask=15,
            candidate_sha256='a'*64,baseline_sha256=BASELINE_SHA,evaluator_sha256=SCORER_SHA,cpu_time_ratio=5.,
            measurements=[dict(repeat=repeat,name=name,seconds=35. if name=='candidate' else 7.,
                targets=376599,utf8_bytes=1148007,peak_process_working_set_bytes=2**30) for repeat,name in order])

    def test_five_limit_exactly_passes_no_extra_margin(self):
        value=self.timing(); self.assertEqual(t.check_expanded_timing(value,'a'*64),5.)
        value['measurements'][1]['seconds']=38.
        self.assertEqual(t.check_expanded_timing(value,'a'*64),5.)

    def test_missing_overbudget_nonfinite_or_wrong_cp_refused(self):
        variants=[]
        v=self.timing(); v['measurements'].pop(); variants.append(v)
        v=self.timing(); v['candidate_sha256']='b'*64; variants.append(v)
        v=self.timing(); v['measurements'][1]['seconds']=float('nan'); variants.append(v)
        v=self.timing(); v['measurements'][1]['peak_process_working_set_bytes']=4*2**30+1; variants.append(v)
        v=self.timing()
        for row in v['measurements']:
            if row['name']=='candidate': row['seconds']=36.
        v['cpu_time_ratio']=36/7; variants.append(v)
        for value in variants:
            with self.assertRaises(ValueError): t.check_expanded_timing(value,'a'*64)

    def test_control_requires_completed_preflight_but_not_expansion_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/t.AUTHORIZATION_PATH).parent.mkdir(parents=True)
            source=root/'push152_preflight.py'; source.write_text('# synthetic preflight',encoding='utf-8')
            t.write_new(root/t.AUTHORIZATION_PATH,dict(status='completed',training_authorized=False,all_limits_pass=False,
                plan_sha256=t.PLAN_SHA,test_used=False,no_layout_fallback=True,script_sha256=t.sha(source)))
            pins,parent=t.verify_authorization(root,{'name':'control','parent_sha256':t.PARENT_SHA},t.PLAN_SHA)
            self.assertIn(t.AUTHORIZATION_PATH,pins); self.assertEqual(parent,t.PARENT_SHA)

    def test_no_branch_can_start_before_preflight_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in t.BRANCHES:
                with self.assertRaises(FileNotFoundError):
                    t.verify_authorization(Path(directory),{'name':name,'parent_sha256':t.PARENT_SHA},t.PLAN_SHA)

    def test_expanded_without_root_authorization_fails_before_model_io(self):
        spec=dict(name='expanded',resource_authorization=t.AUTHORIZATION_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/t.AUTHORIZATION_PATH).parent.mkdir(parents=True)
            t.write_new(root/t.AUTHORIZATION_PATH,dict(training_authorized=False))
            with self.assertRaises(ValueError): t.verify_authorization(root,spec,t.PLAN_SHA)


if __name__ == '__main__':
    unittest.main()
