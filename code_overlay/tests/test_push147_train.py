"""Synthetic CPU-only checks of the generic trainer; no corpus or GPU access."""
import ast
import copy
import inspect
import io
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import push147_train as trainer
from common import make_model


class Push147TrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(149)
        self.ids = torch.tensor([[1,2,3,1,2,3,4,1], [5,6,5,6,5,6,7,5]])
        self.targets = torch.tensor([[2,3,1,2,3,4,1,2], [6,5,6,5,6,7,5,6]])
        self.tables = dict(unigram=torch.full((2048,), 1/2048),
            contexts_2=torch.tensor([1], dtype=torch.int64),
            offsets_2=torch.tensor([0,1], dtype=torch.int32),
            words_2=torch.tensor([7], dtype=torch.int16),
            masses_2=torch.tensor([.5]), backoffs_2=torch.tensor([.5]))
        self.config = dict(vocab=2048, width=32, heads=4, depth=2, context=256,
            ffn_hidden=64, rope=True, lowrank_rank=16, pointer_rank=8,
            dropout=.2, temperature=1.25, cache_weight=.05, cache_theta=15.,
            pointer_max_weight=.5, mkn_weight=.15, mkn_order=2,
            mkn_shapes={k:list(v.shape) for k,v in self.tables.items()})

    def models(self, implementation='push145_cache_hybrid'):
        config = copy.deepcopy(self.config)
        if implementation == 'push147_rank':
            config['rank_layout'] = dict(qkv=12, proj=0, gate=8, value=8, out=8)
        original, _ = make_model(implementation, config, torch.device('cpu'))
        original.ngram.load_state_dict(self.tables)
        original.eval()
        parent = dict(implementation=implementation, config=copy.deepcopy(config),
            model={k:v.detach().clone() for k,v in original.state_dict().items()},
            train_tokens=212992000,
            ancestry={'dense': {'sha256':'synthetic-dense','targets':98304000},
                      'shared_ancestry_targets':98304000})
        model, _ = make_model(trainer.train_implementation(implementation),
                              trainer.train_config(config), torch.device('cpu'))
        model.load_state_dict({k:v for k,v in parent['model'].items()
                               if not k.startswith('ngram.')}, strict=True)
        model.eval()  # Deterministic features; autograd remains enabled.
        return model, parent, original

    def assert_gradient_sets_close(self, expected, model):
        for name, parameter in model.named_parameters():
            actual = parameter.grad
            if expected[name] is None:
                self.assertTrue(actual is None or torch.count_nonzero(actual).item() == 0, name)
            else:
                self.assertIsNotNone(actual, name)
                torch.testing.assert_close(actual, expected[name], atol=2e-7, rtol=2e-5)

    def test_alpha_zero_matches_existing_loss_and_gradients_for_all_architectures(self):
        for implementation in ('push145_cache_hybrid','push147_cache','push147_rank'):
            with self.subTest(implementation=implementation):
                model, _, _ = self.models(implementation)
                expected = -model.training_log_probs(self.ids).gather(
                    -1, self.targets[...,None]).mean()
                expected.backward()
                gradients = {k:None if p.grad is None else p.grad.detach().clone()
                             for k,p in model.named_parameters()}
                model.zero_grad(set_to_none=True)
                actual = trainer.training_loss(model,self.ids,self.targets,0.)
                self.assertTrue(actual.requires_grad)
                torch.testing.assert_close(actual,expected.detach(),atol=0.,rtol=0.)
                actual.backward()
                self.assert_gradient_sets_close(gradients,model)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))

    def test_alpha_one_matches_temperature_scaled_neural_cross_entropy(self):
        model, _, _ = self.models('push147_cache')
        logits = model.head(model.features(self.ids)).float()/model.config['temperature']
        expected = F.cross_entropy(logits.flatten(0,1),self.targets.flatten())
        expected.backward()
        gradients = {k:None if p.grad is None else p.grad.detach().clone()
                     for k,p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        actual = trainer.training_loss(model,self.ids,self.targets,1.)
        torch.testing.assert_close(actual,expected.detach(),atol=1e-6,rtol=1e-6)
        actual.backward()
        self.assert_gradient_sets_close(gradients,model)

    def test_auxiliary_loss_shares_one_feature_and_output_projection_pass(self):
        model, _, _ = self.models()
        with patch.object(model,'features',wraps=model.features) as features, \
             patch.object(model.head,'forward',wraps=model.head.forward) as head:
            mixed = trainer.training_loss(model,self.ids,self.targets,.1)
        self.assertEqual(features.call_count,1)
        self.assertEqual(head.call_count,1)
        pure_mixed = trainer.training_loss(model,self.ids,self.targets,0.)
        pure_neural = trainer.training_loss(model,self.ids,self.targets,1.)
        torch.testing.assert_close(mixed,.9*pure_mixed+.1*pure_neural,atol=0.,rtol=0.)

    def test_poisoned_statistical_tables_cannot_affect_training_loss(self):
        for implementation in ('push145_cache_hybrid','push147_cache','push147_rank'):
            with self.subTest(implementation=implementation):
                _, _, full_model = self.models(implementation)
                before = trainer.training_loss(full_model,self.ids,self.targets,.1)
                with torch.no_grad():
                    for value in full_model.ngram.buffers():
                        if value.is_floating_point():
                            value.fill_(float('nan'))
                with patch.object(full_model.ngram,'probabilities',
                                  side_effect=AssertionError('Training called statistics')):
                    after = trainer.training_loss(full_model,self.ids,self.targets,.1)
                    self.assertTrue(after.requires_grad)
                    after.backward()
                torch.testing.assert_close(before,after,atol=0.,rtol=0.)
                self.assertTrue(torch.isfinite(full_model.token.weight.grad).all())

    def test_snapshot_strict_roundtrip_preserves_architecture_cost_and_parent(self):
        for implementation in ('push145_cache_hybrid','push147_cache','push147_rank'):
            with self.subTest(implementation=implementation):
                model,parent,original = self.models(implementation)
                parent_before = copy.deepcopy(parent)
                eval_config = copy.deepcopy(parent['config'])
                checkpoint = trainer.snapshot(model,parent,eval_config,'synthetic-parent',149,8000,'ema')
                self.assertEqual(checkpoint['implementation'],implementation)
                self.assertEqual(checkpoint['train_tokens'],278528000)
                self.assertEqual(checkpoint['selected_step'],8000)
                self.assertEqual(checkpoint['weights'],'ema')
                self.assertEqual(checkpoint['training_config']['temperature'],1.15)
                self.assertEqual(checkpoint['config']['temperature'],1.25)
                self.assertEqual(checkpoint['model']['head.weight'].data_ptr(),
                                 checkpoint['model']['token.weight'].data_ptr())
                stream = io.BytesIO()
                torch.save(checkpoint,stream)
                stream.seek(0)
                restored = torch.load(stream,map_location='cpu',weights_only=True)
                evaluator,_ = make_model(restored['implementation'],restored['config'],torch.device('cpu'))
                evaluator.load_state_dict(restored['model'],strict=True)
                evaluator.eval()
                self.assertIs(evaluator.token.weight,evaluator.head.weight)
                torch.testing.assert_close(evaluator.predict_log_probs(self.ids),
                                           original.predict_log_probs(self.ids),atol=0.,rtol=0.)
                checkpoint['config']['temperature']=9.
                checkpoint['ancestry']['dense']['targets']=0
                with torch.no_grad():
                    model.token.weight.add_(.5)
                for key,value in parent_before['model'].items():
                    torch.testing.assert_close(parent['model'][key],value,atol=0.,rtol=0.)
                self.assertEqual(parent['config'],parent_before['config'])
                self.assertEqual(parent['ancestry'],parent_before['ancestry'])
                torch.testing.assert_close(checkpoint['model']['token.weight'],
                                           parent_before['model']['token.weight'],atol=0.,rtol=0.)

    def test_actual_projection_hook_runs_after_optimizer_and_before_ema(self):
        tree = ast.parse(inspect.getsource(trainer.main))
        loop = next(node for node in ast.walk(tree) if isinstance(node,ast.For)
                    and isinstance(node.target,ast.Name) and node.target.id=='step')
        project_index = next(i for i,node in enumerate(loop.body)
                             if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
                             and node.targets[0].id=='project')
        self.assertIn('optimizer.step()',ast.unparse(loop.body[project_index-1]))
        self.assertIn('target.lerp_',ast.unparse(loop.body[project_index+2]))
        snippet = ast.fix_missing_locations(ast.Module(
            body=loop.body[project_index:project_index+2],type_ignores=[]))
        model,parent,_ = self.models('push147_cache')
        with torch.no_grad():
            model.cache_log_theta_delta.fill_(100.)
            model.cache_recency_strength.fill_(-10.)
        with patch.object(model,'project_cache_parameters',wraps=model.project_cache_parameters) as project:
            exec(compile(snippet,'<actual-trainer-projection>','exec'),dict(model=model))
        self.assertEqual(project.call_count,1)
        theta,alpha = model.cache_scales()
        self.assertLessEqual(theta.item(),40.)
        self.assertEqual(alpha.item(),0.)
        checkpoint=trainer.snapshot(model,parent,parent['config'],'synthetic-parent',149,1,'raw')
        self.assertLessEqual(checkpoint['model']['cache_log_theta_delta'].item(),model._theta_log_upper+1e-7)
        self.assertEqual(checkpoint['model']['cache_recency_strength'].item(),0.)


if __name__=='__main__':
    unittest.main()
