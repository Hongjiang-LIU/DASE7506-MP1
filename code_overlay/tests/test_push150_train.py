"""Small CPU synthetic tests only; no course corpus, checkpoints or GPU."""
import ast
import copy
import inspect
import io
import random
import unittest

import numpy as np
import torch

from common import PROTOCOL, make_model
import push150_train as trainer


class Push150TrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        random.seed(173); np.random.seed(173); torch.manual_seed(173)
        self.ids=torch.tensor([[1,2,3,1,2,3,4,1],[5,6,5,6,5,6,7,5]])

    def models(self,implementation='push147_signed_cache'):
        tables=dict(unigram=torch.full((2048,),1/2048),
            contexts_2=torch.tensor([1],dtype=torch.int64),
            offsets_2=torch.tensor([0,1],dtype=torch.int32),
            words_2=torch.tensor([7],dtype=torch.int16),
            masses_2=torch.tensor([.5]),backoffs_2=torch.tensor([.5]))
        config=dict(vocab=2048,width=16,heads=2,depth=1,context=256,
            ffn_hidden=32,rope=True,lowrank_rank=8,pointer_rank=4,
            dropout=.2,temperature=1.25,cache_weight=.05,cache_theta=15.,
            mkn_order=2,mkn_weight=.15,mkn_shapes={k:list(v.shape) for k,v in tables.items()})
        if implementation=='push150_expand':
            config['push150_ranks']=dict(qkv=12,proj=0,gate=8,value=8,out=8)
        full,_=make_model(implementation,config,torch.device('cpu'))
        full.ngram.load_state_dict(tables)
        full.eval()
        parent=dict(protocol=PROTOCOL,implementation=implementation,config=copy.deepcopy(config),
            model=trainer.model_state(full),train_tokens=376832000,ancestry={'synthetic':{'targets':98304000}})
        model,_=make_model(implementation,trainer.training_config(config,.2),torch.device('cpu'))
        model.load_state_dict({k:v for k,v in parent['model'].items() if not k.startswith('ngram.')},strict=True)
        averaged=copy.deepcopy(model).eval().requires_grad_(False)
        return model,averaged,parent,full

    def optimizers(self,model):
        return trainer.build_optimizers(model,'muon',.0000075,.000075,.00025,.0005)[0]

    def assert_nested_equal(self,first,second):
        if torch.is_tensor(first):
            torch.testing.assert_close(first,second,atol=0.,rtol=0.)
        elif isinstance(first,dict):
            self.assertEqual(set(first),set(second))
            for key in first:
                self.assert_nested_equal(first[key],second[key])
        elif isinstance(first,(list,tuple)):
            self.assertEqual(type(first),type(second))
            self.assertEqual(len(first),len(second))
            for a,b in zip(first,second):
                self.assert_nested_equal(a,b)
        else:
            self.assertEqual(first,second)

    def test_schedule_boundaries_monotonic_cooldown_and_paired_warmup(self):
        factors=[trainer.schedule_factor(step,20000,'wsd',200,4000) for step in
                 (1,100,200,15999,16000,16001,18000,20000)]
        self.assertEqual(factors[:5],[.005,.5,1.,1.,1.])
        self.assertLess(factors[5],1.)
        self.assertAlmostEqual(factors[6],.55)
        self.assertEqual(factors[-1],.1)
        cosine=[trainer.schedule_factor(step,20000,'cosine',200,4000)
                for step in (1,200,10000,18000,20000)]
        self.assertEqual(cosine[0],.005)
        self.assertEqual(cosine[-1],.1)
        self.assertTrue(all(a>b for a,b in zip(cosine[1:],cosine[2:])))
        for arguments in ((0,20000,'wsd',200,4000),(20001,20000,'cosine',200,4000),
                          (1,20,'wsd',19,5),(1,20,'other',2,5),(1,20,'wsd',2,0)):
            with self.assertRaises(ValueError):
                trainer.schedule_factor(*arguments)

    def test_training_config_strips_statistics_without_modifying_parent(self):
        config=dict(vocab=2048,dropout=.3,temperature=1.25,mkn_order=5,mkn_weight=.15,
                    mkn_shapes={'x':[2]},ngram_shapes={'z':[3]},ngram_weight=.1,
                    discount_scale=.7,push150_ranks={'qkv':12})
        original=copy.deepcopy(config)
        actual=trainer.training_config(config,.1)
        self.assertFalse(any(k.startswith(('mkn_','ngram_')) for k in actual))
        self.assertNotIn('discount_scale',actual)
        self.assertEqual(actual['temperature'],1.15)
        self.assertEqual(actual['dropout'],.1)
        actual['push150_ranks']['qkv']=99
        self.assertEqual(config,original)
        for dropout in (-.1,1.,float('nan')):
            with self.assertRaises(ValueError):
                trainer.training_config({},dropout)

    def test_generic_snapshot_roundtrip_keeps_parent_implementation_cost_and_tables(self):
        for implementation in ('push147_signed_cache','push150_expand'):
            with self.subTest(implementation=implementation):
                model,_,parent,full=self.models(implementation)
                before=copy.deepcopy(parent)
                checkpoint=trainer.snapshot(model,parent,parent['config'],'synthetic-parent',
                    {'seed':173,'steps':20000,'schedule':'wsd'},20000,'ema')
                self.assertEqual(checkpoint['implementation'],implementation)
                self.assertEqual(checkpoint['train_tokens'],540672000)
                self.assertEqual(checkpoint['training_config']['temperature'],1.15)
                self.assertEqual(checkpoint['config']['temperature'],1.25)
                self.assertEqual(checkpoint['model']['token.weight'].data_ptr(),checkpoint['model']['head.weight'].data_ptr())
                for key in parent['model']:
                    if key.startswith('ngram.'):
                        self.assertNotEqual(checkpoint['model'][key].data_ptr(),parent['model'][key].data_ptr())
                clone,_=make_model(implementation,checkpoint['config'],torch.device('cpu'))
                clone.load_state_dict(checkpoint['model'],strict=True)
                clone.eval()
                torch.testing.assert_close(clone.predict_log_probs(self.ids),full.predict_log_probs(self.ids),atol=0.,rtol=0.)
                with torch.no_grad():
                    model.token.weight.add_(1.)
                    checkpoint['model']['ngram.unigram'].fill_(0.)
                self.assert_nested_equal(parent,before)

    def test_all_cpu_rng_streams_roundtrip_without_serializing_numpy_objects(self):
        sampler=torch.Generator().manual_seed(999)
        np.random.normal()  # Exercise NumPy's cached Gaussian component too.
        state=trainer.capture_rng(sampler,cuda=False)
        stream=io.BytesIO(); torch.save(state,stream); stream.seek(0)
        restored=torch.load(stream,map_location='cpu',weights_only=True)
        def draw():
            return random.random(),np.random.normal(),torch.rand(4),torch.randint(100,(4,),generator=sampler)
        expected=draw()
        trainer.restore_rng(restored,sampler,cuda=False)
        self.assert_nested_equal(expected,draw())
        with self.assertRaises(ValueError):
            trainer.restore_rng(restored,sampler,cuda=True)

    def test_serialized_raw_ema_two_optimizer_resume_matches_next_stochastic_update(self):
        model,averaged,_,_=self.models()
        optimizers=self.optimizers(model)
        sampler=torch.Generator().manual_seed(173)
        identity=dict(parent_sha256='synthetic',source_sha256={'source.py':'sha'},
                      recipe={'steps':8,'schedule':'wsd'},data_sha256={'train':'allowed'})
        def advance(network,average,opts,rng,step):
            network.train()
            batch=torch.randint(1,14,(2,9),generator=rng)
            extra_random=(random.random(),float(np.random.random()))
            factor=trainer.schedule_factor(step,8,'wsd',2,2)
            for optimizer in opts:
                for group in optimizer.param_groups:
                    group['lr']=group['base_lr']*factor
                optimizer.zero_grad(set_to_none=True)
            loss=trainer.training_loss(network,batch[:,:-1],batch[:,1:],0.)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True)
            for optimizer in opts:
                optimizer.step()
            network.project_cache_parameters()
            with torch.no_grad():
                for target,current in zip(average.parameters(),network.parameters()):
                    target.lerp_(current,.01)
            return loss.detach(),extra_random
        advance(model,averaged,optimizers,sampler,1)
        state=trainer.pack_train_state(model,averaged,optimizers,sampler,identity,1,
                                      {'completed_steps':1,'history':[]},cuda=False)
        stream=io.BytesIO(); torch.save(state,stream); stream.seek(0)
        saved=torch.load(stream,map_location='cpu',weights_only=True)
        expected_result=advance(model,averaged,optimizers,sampler,2)
        expected=trainer.pack_train_state(model,averaged,optimizers,sampler,identity,2,
                                         {'completed_steps':2},cuda=False)
        resumed,ema,_,_=self.models()
        resumed_optimizers=self.optimizers(resumed)
        resumed_sampler=torch.Generator().manual_seed(3)
        step,log=trainer.restore_train_state(saved,resumed,ema,resumed_optimizers,resumed_sampler,identity,cuda=False)
        self.assertEqual(step,1)
        self.assertEqual(log['completed_steps'],1)
        actual_result=advance(resumed,ema,resumed_optimizers,resumed_sampler,2)
        actual=trainer.pack_train_state(resumed,ema,resumed_optimizers,resumed_sampler,identity,2,
                                       {'completed_steps':2},cuda=False)
        self.assert_nested_equal(actual_result,expected_result)
        self.assert_nested_equal(actual,expected)
        self.assertTrue(all(optimizer.state for optimizer in resumed_optimizers))

    def test_resume_identity_change_rejected_before_model_mutation(self):
        model,averaged,_,_=self.models()
        optimizers=self.optimizers(model)
        sampler=torch.Generator().manual_seed(173)
        identity=dict(recipe={'steps':20000,'seed':173},parent_sha256='parent',
                      source_sha256={'file.py':'hash'},data_sha256={'train':'hash'})
        state=trainer.pack_train_state(model,averaged,optimizers,sampler,identity,0,{'completed_steps':0})
        before=trainer.model_state(model)
        for key,value in (('parent_sha256','other'),('source_sha256',{'file.py':'changed'}),
                          ('recipe',{'steps':40000,'seed':173}),('data_sha256',{'train':'changed'})):
            altered=copy.deepcopy(identity); altered[key]=value
            with self.assertRaises(ValueError):
                trainer.restore_train_state(state,model,averaged,optimizers,sampler,altered)
            self.assert_nested_equal(before,trainer.model_state(model))
        broken=copy.deepcopy(state); broken['step']=20001
        with self.assertRaises(ValueError):
            trainer.restore_train_state(broken,model,averaged,optimizers,sampler,identity)

    def test_source_graph_pins_dynamic_model_and_all_local_inference_dependencies(self):
        sources=trainer.source_digests('push150_expand')
        required={'push150_train.py','push150_expand.py','push148_train.py','push148_muon.py',
                  'push147_train.py','push147_signed_cache.py','push147_cache.py',
                  'push145_cache.py','push145_cache_hybrid.py','push145_mkn.py',
                  'common.py','evaluate.py','efficient.py','student.py','lowrank.py'}
        self.assertTrue(required.issubset(sources),required-set(sources))
        self.assertTrue(all(name.endswith('.py') and len(digest)==64 for name,digest in sources.items()))

    def test_main_ast_prohibits_test_loader_and_keeps_training_update_order(self):
        source=inspect.getsource(trainer.main)
        tree=ast.parse(source)
        corpus_loop=next(node for node in ast.walk(tree) if isinstance(node,ast.For)
                         and isinstance(node.target,ast.Name) and node.target.id=='split')
        self.assertEqual(ast.literal_eval(corpus_loop.iter),('train','validation'))
        calls=[ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)]
        self.assertNotIn('load_data',calls)
        self.assertIn("manifest['sha256']['tokenizer.json']",source)
        self.assertTrue(any(isinstance(node,ast.ExceptHandler) and ast.unparse(node.type)=='BaseException'
                            for node in ast.walk(tree)))
        loop=next(node for node in ast.walk(tree) if isinstance(node,ast.For)
                  and isinstance(node.target,ast.Name) and node.target.id=='step')
        body=[ast.unparse(node) for node in loop.body]
        updates=next(i for i,line in enumerate(body) if 'optimizer.step()' in line)
        projection=next(i for i,line in enumerate(body) if line.startswith('project ='))
        ema=next(i for i,line in enumerate(body) if 'target.lerp_' in line)
        completed=next(i for i,line in enumerate(body) if line=='completed_step = step')
        self.assertLess(updates,projection); self.assertLess(projection,ema); self.assertLess(ema,completed)
        starts=next(line for line in body if line.startswith('starts ='))
        self.assertIn('len(tokens) - CONTEXT',starts)


if __name__=='__main__':
    unittest.main()
