"""Synthetic CPU checks of push148's training-only driver and cost semantics."""
import ast
import copy
import inspect
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import push148_train as trainer
from push147_signed_cache import SignedDistanceCache
from push148_losses import dual_view_loss


class Push148TrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(157)
        self.ids = torch.tensor([[1,2,3,1,2,3,4,1], [5,6,5,6,5,6,7,5]])
        self.targets = torch.tensor([[2,3,1,2,3,4,1,2], [6,5,6,5,6,7,5,6]])

    def models(self):
        tables = dict(unigram=torch.full((2048,), 1/2048),
            contexts_2=torch.tensor([1], dtype=torch.int64),
            offsets_2=torch.tensor([0,1], dtype=torch.int32),
            words_2=torch.tensor([7], dtype=torch.int16),
            masses_2=torch.tensor([.5]), backoffs_2=torch.tensor([.5]))
        config = dict(vocab=2048, width=16, heads=2, depth=1, context=256,
            ffn_hidden=32, rope=True, lowrank_rank=8, pointer_rank=4,
            dropout=.2, temperature=1.25, cache_weight=.05, cache_theta=15.,
            pointer_max_weight=.5, mkn_order=2, mkn_weight=.15,
            mkn_shapes={k:list(v.shape) for k,v in tables.items()})
        evaluator = SignedDistanceCache(config).eval()
        evaluator.ngram.load_state_dict(tables)
        with torch.no_grad():
            evaluator.cache_recency_strength.fill_(-.3)
            evaluator.pointer_key.weight.normal_(0., .05)
        parent = dict(implementation='push147_signed_cache',config=copy.deepcopy(config),
            model={k:v.detach().clone() for k,v in evaluator.state_dict().items()},
            train_tokens=278528000,ancestry={'original': {'targets':98304000}})
        model = SignedDistanceCache(trainer.train_config(parent['config']))
        model.load_state_dict({k:v for k,v in parent['model'].items()
                               if not k.startswith('ngram.')}, strict=True)
        return model, parent, evaluator

    def optimizers(self, model, kind='adamw'):
        return trainer.build_optimizers(model,kind,.00003,.0003,.001,.002)

    def test_optimizer_groups_cover_every_tied_parameter_once(self):
        for kind in ('adamw','muon'):
            with self.subTest(kind=kind):
                model,_,_ = self.models()
                optimizers,muon_names = self.optimizers(model,kind)
                named = dict(model.named_parameters())
                group_by_id = {id(p):(i,g) for i,o in enumerate(optimizers)
                               for g in o.param_groups for p in g['params']}
                all_ids = [id(p) for o in optimizers for g in o.param_groups for p in g['params']]
                self.assertEqual(len(all_ids),len(set(all_ids)))
                self.assertEqual(set(all_ids),{id(p) for p in named.values()})
                self.assertIs(model.token.weight,model.head.weight)
                selected = sorted(k for k,p in named.items()
                                  if kind=='muon' and k.startswith('blocks.') and p.ndim==2)
                self.assertEqual(muon_names,selected)
                for name,parameter in named.items():
                    owner,group=group_by_id[id(parameter)]
                    if name in selected:
                        self.assertEqual(owner,1)
                        rate,decay=.002,.1
                    elif parameter.ndim==0:
                        rate,decay=.001,0.
                    elif name.startswith('pointer_'):
                        rate,decay=.0003,.01
                    else:
                        rate,decay=.00003,.1
                    self.assertEqual(group['lr'],rate,name)
                    self.assertEqual(group['base_lr'],rate,name)
                    self.assertEqual(group['weight_decay'],decay,name)
                self.assertEqual(optimizers[0].defaults['betas'],(.9,.95))

    def test_single_view_one_step_matches_direct_signed_training_nll(self):
        model,_,_ = self.models()
        reference = copy.deepcopy(model)
        actual_optimizers,_ = self.optimizers(model)
        reference_optimizers,_ = self.optimizers(reference)
        torch.manual_seed(771)
        actual = trainer.training_loss(model,self.ids,self.targets,0.)
        torch.manual_seed(771)
        expected = -reference.training_log_probs(self.ids).gather(-1,self.targets[...,None]).mean()
        torch.testing.assert_close(actual,expected,atol=0.,rtol=0.)
        for loss,network,optimizers in ((actual,model,actual_optimizers),
                                        (expected,reference,reference_optimizers)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True)
            for optimizer in optimizers:
                optimizer.step()
            network.project_cache_parameters()
        for (name,a),(_,b) in zip(model.named_parameters(),reference.named_parameters()):
            torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-6,msg=name)

    def test_actual_driver_loss_dispatch_uses_two_views_even_at_zero_coefficient(self):
        tree = ast.parse(inspect.getsource(trainer.main))
        branch = next(node for node in ast.walk(tree) if isinstance(node,ast.If)
                      and ast.unparse(node.test)=='args.views == 2')
        snippet = compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),
                          '<actual-push148-loss-dispatch>','exec')
        model,_,_ = self.models()
        batch = torch.cat([self.ids,self.targets[:,-1:]],dim=1)
        for views in (1,2):
            with self.subTest(views=views):
                scope = dict(model=model,batch=batch,args=SimpleNamespace(views=views,rdrop_coefficient=0.),
                             dual_view_loss=dual_view_loss,training_loss=trainer.training_loss)
                with patch.object(model,'training_log_probs',wraps=model.training_log_probs) as forward:
                    exec(snippet,scope)
                self.assertEqual(forward.call_count,2 if views==2 else 0)
                self.assertTrue(scope['loss'].requires_grad)
                self.assertEqual(set(scope['detail']),{'hard','kl'})
                self.assertTrue(all(not value.requires_grad for value in scope['detail'].values()))

    def test_training_clone_has_no_statistical_tables_or_parent_alias(self):
        model,parent,evaluator = self.models()
        before = copy.deepcopy(parent)
        self.assertFalse(hasattr(model,'ngram'))
        self.assertFalse(any(key.startswith('ngram.') for key in model.state_dict()))
        self.assertFalse(any(key.startswith(('ngram_','mkn_')) for key in model.config))
        self.assertEqual(model.config['temperature'],1.15)
        self.assertEqual(evaluator.config['temperature'],1.25)
        for key,value in model.state_dict().items():
            self.assertNotEqual(value.data_ptr(),parent['model'][key].data_ptr())
        with patch.object(evaluator.ngram,'probabilities',side_effect=AssertionError('Train table lookup')):
            loss,_ = dual_view_loss(model,self.ids,self.targets,.5)
            loss.backward()
        with torch.no_grad():
            model.token.weight.add_(.1)
            evaluator.ngram.unigram.fill_(float('nan'))
        for key,value in before['model'].items():
            torch.testing.assert_close(parent['model'][key],value,atol=0.,rtol=0.)

    def test_snapshot_roundtrip_and_actual_dual_recipe_disclose_both_cost_counts(self):
        model,parent,evaluator = self.models()
        tree = ast.parse(inspect.getsource(trainer.main))
        recipe = next(node for node in ast.walk(tree) if isinstance(node,ast.Assign)
                      and isinstance(node.targets[0],ast.Subscript)
                      and ast.unparse(node.targets[0])=="checkpoint['training_recipe']")
        snippet = compile(ast.fix_missing_locations(ast.Module(body=[recipe],type_ignores=[])),
                          '<actual-push148-recipe>','exec')
        for views in (1,2):
            with self.subTest(views=views):
                checkpoint = trainer.snapshot(model,parent,parent['config'],'synthetic-parent',157,8000,'ema')
                scope=dict(checkpoint=checkpoint,step=8000,batch_size=32//views,
                    args=SimpleNamespace(optimizer='adamw',views=views,rdrop_coefficient=.1 if views==2 else 0.))
                exec(snippet,scope)
                self.assertEqual(checkpoint['train_tokens'],344064000)
                self.assertEqual(checkpoint['training_recipe']['processed_view_targets'],65536000)
                self.assertEqual(checkpoint['training_recipe']['sampled_label_targets'],65536000//views)
                self.assertEqual(checkpoint['training_config']['temperature'],1.15)
                self.assertEqual(checkpoint['config']['temperature'],1.25)
                stream=io.BytesIO()
                torch.save(checkpoint,stream); stream.seek(0)
                restored=torch.load(stream,map_location='cpu',weights_only=True)
                clone=SignedDistanceCache(restored['config']).eval()
                clone.load_state_dict(restored['model'],strict=True)
                self.assertIs(clone.token.weight,clone.head.weight)
                torch.testing.assert_close(clone.predict_log_probs(self.ids),
                                           evaluator.predict_log_probs(self.ids),atol=0.,rtol=0.)

    def test_actual_driver_projects_after_all_optimizers_before_ema_and_uses_safe_sampler(self):
        tree = ast.parse(inspect.getsource(trainer.main))
        loop = next(node for node in ast.walk(tree) if isinstance(node,ast.For)
                    and isinstance(node.target,ast.Name) and node.target.id=='step')
        project_index=next(i for i,node in enumerate(loop.body)
                           if ast.unparse(node)=='model.project_cache_parameters()')
        self.assertIn('for optimizer in optimizers:',ast.unparse(loop.body[project_index-1]))
        self.assertIn('optimizer.step()',ast.unparse(loop.body[project_index-1]))
        self.assertIn('target.lerp_(current, 0.01)',ast.unparse(loop.body[project_index+1]))
        starts=next(node for node in loop.body if isinstance(node,ast.Assign)
                    and ast.unparse(node.targets[0])=='starts')
        # randint's upper bound is exclusive: maximal start N-257 gives a
        # complete 257-token input/target slice ending at N-1, no skipped window.
        self.assertEqual(ast.unparse(starts.value.func.value.args[0]),'len(tokens) - 256')
        for views in (1,2):
            self.assertEqual((32//views)*views*256,8192)


if __name__=='__main__':
    unittest.main()
