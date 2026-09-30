"""Small synthetic CPU checks; never reads dataset or checkpoint files."""
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from push147_signed_cache import SignedDistanceCache
from push148_losses import dual_view_loss


class TwoLogitViews(nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = nn.Parameter(torch.tensor(first, dtype=torch.float32))
        self.second = nn.Parameter(torch.tensor(second, dtype=torch.float32))
        self.calls = 0
        self.outputs = []

    def training_log_probs(self, ids):
        logits = self.first if self.calls == 0 else self.second
        self.calls += 1
        result = F.log_softmax(logits, dim=-1).expand(*ids.shape, -1)
        self.outputs.append(result)
        return result


class DualViewLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(148)
        self.ids = torch.tensor([[1,2,3,1,2,3,4,1], [5,6,5,6,5,6,7,5]])
        self.targets = torch.tensor([[2,3,1,2,3,4,1,2], [6,5,6,5,6,7,5,6]])

    def cache_model(self, dropout=.2):
        tables = dict(unigram=torch.full((2048,), 1/2048),
            contexts_2=torch.tensor([1], dtype=torch.int64),
            offsets_2=torch.tensor([0,1], dtype=torch.int32),
            words_2=torch.tensor([7], dtype=torch.int16),
            masses_2=torch.tensor([.5]), backoffs_2=torch.tensor([.5]))
        config = dict(vocab=2048, width=16, heads=2, depth=1, context=256,
            ffn_hidden=32, rope=True, lowrank_rank=8, pointer_rank=4,
            dropout=dropout, temperature=1.15, cache_weight=.05, cache_theta=15.,
            mkn_order=2, mkn_weight=.15,
            mkn_shapes={k:list(v.shape) for k,v in tables.items()})
        model = SignedDistanceCache(config)
        model.ngram.load_state_dict(tables)
        with torch.no_grad():
            model.cache_recency_strength.fill_(-.2)
            model.pointer_key.weight.normal_(0., .05)
            model.pointer_gate[-1].weight.normal_(0., .03)
        return model

    def test_zero_coefficient_matches_same_two_stochastic_passes_and_rng(self):
        model = self.cache_model().train()
        torch.manual_seed(512)
        first = model.training_log_probs(self.ids)
        second = model.training_log_probs(self.ids)
        expected = .5 * (-first.gather(-1,self.targets[...,None]).mean()
                         -second.gather(-1,self.targets[...,None]).mean())
        next_random = torch.rand(4)
        torch.manual_seed(512)
        with patch.object(model,'training_log_probs',wraps=model.training_log_probs) as forward:
            actual,metrics = dual_view_loss(model,self.ids,self.targets,0.)
        self.assertEqual(forward.call_count,2)
        torch.testing.assert_close(actual,expected,atol=0.,rtol=0.)
        torch.testing.assert_close(torch.rand(4),next_random,atol=0.,rtol=0.)
        self.assertGreater(metrics['kl'].item(),0.)
        self.assertFalse(metrics['hard'].requires_grad)
        self.assertFalse(metrics['kl'].requires_grad)
        actual.backward()
        self.assertTrue(torch.isfinite(model.token.weight.grad).all())

    def test_matches_analytic_symmetric_kl_with_vocab_sum_token_mean(self):
        p, q = [.7,.2,.1], [.2,.3,.5]
        model = TwoLogitViews([math.log(v) for v in p],[math.log(v) for v in q])
        ids = torch.zeros((2,3),dtype=torch.long)
        targets = torch.tensor([[0,1,2],[2,1,0]])
        coefficient=.4
        loss,metrics = dual_view_loss(model,ids,targets,coefficient)
        expected_kl = .5*sum((a-b)*(math.log(a)-math.log(b)) for a,b in zip(p,q))
        expected_hard = -.5*sum(math.log(p[i])+math.log(q[i]) for i in range(3))/3
        self.assertAlmostEqual(metrics['kl'].item(),expected_kl,places=6)
        self.assertAlmostEqual(metrics['hard'].item(),expected_hard,places=6)
        self.assertAlmostEqual(loss.item(),expected_hard+coefficient*expected_kl,places=6)

    def test_both_kl_sides_keep_their_gradients(self):
        model = TwoLogitViews([1.3,-.2,.4],[-.3,.9,.1])
        ids = torch.zeros((1,2),dtype=torch.long)
        targets = torch.tensor([[0,2]])
        loss,_ = dual_view_loss(model,ids,targets,.7)
        first,second = model.outputs
        hard = .5*(-first.gather(-1,targets[...,None]).mean()
                    -second.gather(-1,targets[...,None]).mean())
        parameters = (model.first,model.second)
        actual = torch.autograd.grad((loss-hard)/.7,parameters,retain_graph=True)
        p,q = first.exp(),second.exp()
        reference = .5*((p*(first-second)).sum(-1)+(q*(second-first)).sum(-1)).mean()
        expected = torch.autograd.grad(reference,parameters)
        for a,b in zip(actual,expected):
            self.assertGreater(a.abs().sum().item(),0.)
            self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-6)

    def test_identical_distributions_have_exactly_zero_consistency(self):
        model = TwoLogitViews([.3,-.7,1.1],[.3,-.7,1.1])
        ids = torch.zeros((2,3),dtype=torch.long)
        loss,metrics = dual_view_loss(model,ids,ids,1.)
        self.assertEqual(metrics['kl'].item(),0.)
        torch.testing.assert_close(loss,metrics['hard'],atol=0.,rtol=0.)
        model = self.cache_model(dropout=0.).train()
        _,metrics = dual_view_loss(model,self.ids,self.targets,1.)
        self.assertEqual(metrics['kl'].item(),0.)

    def test_underflowed_probabilities_still_produce_finite_loss_and_gradients(self):
        model = TwoLogitViews([1000.,-1000.,0.],[-1000.,1000.,0.])
        ids = torch.zeros((1,2),dtype=torch.long)
        targets = torch.tensor([[0,2]])
        loss,metrics = dual_view_loss(model,ids,targets,.5)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(metrics['kl']))
        self.assertGreater(metrics['kl'].item(),0.)
        loss.backward()
        self.assertTrue(torch.isfinite(model.first.grad).all())
        self.assertTrue(torch.isfinite(model.second.grad).all())

    def test_statistical_table_poison_does_not_enter_either_training_view(self):
        model = self.cache_model().train()
        torch.manual_seed(611)
        before,before_metrics = dual_view_loss(model,self.ids,self.targets,.5)
        with torch.no_grad():
            for value in model.ngram.buffers():
                if value.is_floating_point():
                    value.fill_(float('nan'))
        torch.manual_seed(611)
        with patch.object(model.ngram,'probabilities',side_effect=AssertionError('Count lookup in loss')):
            after,metrics = dual_view_loss(model,self.ids,self.targets,.5)
            after.backward()
        torch.testing.assert_close(before,after,atol=0.,rtol=0.)
        torch.testing.assert_close(before_metrics['kl'],metrics['kl'],atol=0.,rtol=0.)
        self.assertTrue(torch.isfinite(model.token.weight.grad).all())

    def test_loss_does_not_mutate_inference_parameters_mode_or_window_contract(self):
        model = self.cache_model().eval()
        before = model.predict_log_probs(self.ids)
        state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        loss,_ = dual_view_loss(model,self.ids,self.targets,.5)
        loss.backward()
        self.assertFalse(model.training)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,state[key],atol=0.,rtol=0.)
        changed = self.ids.clone(); changed[:,5:] = 13
        torch.testing.assert_close(before[:,:5],model.predict_log_probs(changed)[:,:5],atol=2e-6,rtol=1e-6)
        changed[1] = 21
        torch.testing.assert_close(before[0,:5],model.predict_log_probs(changed)[0,:5],atol=2e-6,rtol=1e-6)
        model.predict_log_probs(changed)
        torch.testing.assert_close(before,model.predict_log_probs(self.ids),atol=0.,rtol=0.)

    def test_invalid_coefficient_rejected_before_any_forward(self):
        for value in (-.1,float('nan'),float('inf')):
            model = TwoLogitViews([0.,1.],[1.,0.])
            ids = torch.zeros((1,2),dtype=torch.long)
            with self.assertRaises(ValueError):
                dual_view_loss(model,ids,ids,value)
            self.assertEqual(model.calls,0)


if __name__=='__main__':
    unittest.main()
