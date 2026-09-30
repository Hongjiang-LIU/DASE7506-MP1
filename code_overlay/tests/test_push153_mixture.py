"""Synthetic probabilities and tiny random models only; no corpus/checkpoint."""
import copy
import unittest
from unittest.mock import patch

import torch

from push150_expand import ExpandedSignedCache
from push153_mixture import (BASE_WEIGHT, KAPPAS, CausalResponsibilityMixture,
                             build_model, validate_kappa,
                             weights_from_observed_targets)


def synthetic_tables():
    tables = dict(unigram=torch.full((2048,), 1/2048),
        contexts_2=torch.tensor([1, 2], dtype=torch.int64),
        offsets_2=torch.tensor([0, 2, 3], dtype=torch.int32),
        words_2=torch.tensor([2, 5, 7], dtype=torch.int16),
        masses_2=torch.tensor([.4, .2, .5]),
        backoffs_2=torch.tensor([.4, .5]))
    for order in (3, 4, 5):
        tables.update({f'contexts_{order}':torch.empty(0, dtype=torch.int64),
            f'offsets_{order}':torch.zeros(1, dtype=torch.int32),
            f'words_{order}':torch.empty(0, dtype=torch.int16),
            f'masses_{order}':torch.empty(0), f'backoffs_{order}':torch.empty(0)})
    return tables


def synthetic_config():
    tables = synthetic_tables()
    return dict(vocab=2048, width=8, heads=2, depth=1, context=256,
        ffn_hidden=16, rope=True, dropout=0., lowrank_rank=2, temperature=1.3,
        cache_weight=.05, cache_theta=15., pointer_rank=2, pointer_max_weight=.5,
        push150_ranks=dict(qkv=4, proj=0, gate=4, value=4, out=4),
        mkn_shapes={name:list(value.shape) for name,value in tables.items()},
        mkn_order=5, mkn_weight=.15, discount_scale=.7)


class ResponsibilityMathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_exact_fixed_menu(self):
        self.assertEqual(KAPPAS, (8, 32, 128))
        self.assertEqual(BASE_WEIGHT, .15)
        for kappa in (None, *KAPPAS):
            self.assertEqual(validate_kappa(kappa), kappa)
        for kappa in (0, -8, 1, 16, 8., True, False, '8', float('inf')):
            with self.subTest(kappa=kappa), self.assertRaises(ValueError):
                validate_kappa(kappa)

    def test_fp64_matches_independent_scalar_formula(self):
        n = torch.tensor([[.7, .2, .02, .5], [.3, .9, .4, .2]], dtype=torch.float64)
        m = torch.tensor([[.1, .4, .8, .05], [.8, .05, .5, .7]], dtype=torch.float64)
        for kappa in KAPPAS:
            expected = []
            for row_n, row_m in zip(n.tolist(), m.tolist()):
                total = 0.
                row = [.15]
                for count, (pn, pm) in enumerate(zip(row_n, row_m), 1):
                    total += .15*pm/(.85*pn+.15*pm)
                    row.append((kappa*.15+total)/(kappa+count))
                expected.append(row)
            torch.testing.assert_close(weights_from_observed_targets(n,m,kappa),
                torch.tensor(expected, dtype=torch.float64), atol=2e-16, rtol=2e-16)

    def test_zero_history_is_exact_prior_for_every_menu(self):
        for dtype in (torch.float32, torch.float64):
            empty = torch.empty(3, 0, dtype=dtype)
            for kappa in (None, *KAPPAS):
                torch.testing.assert_close(weights_from_observed_targets(empty,empty,kappa),
                    torch.full((3,1), .15, dtype=dtype), atol=0, rtol=0)

    def test_disabled_does_not_adapt_to_evidence(self):
        n = torch.tensor([[.01, .9, .5], [.8, .05, .4]])
        m = torch.tensor([[.9, .01, .5], [.01, .8, .4]])
        torch.testing.assert_close(weights_from_observed_targets(n,m,None),
            torch.full((2,4), .15), atol=0, rtol=0)

    def test_identical_experts_preserve_prior_up_to_fp_rounding(self):
        n = torch.linspace(.001, .9, 18).reshape(2,9)
        for kappa in KAPPAS:
            torch.testing.assert_close(weights_from_observed_targets(n,n,kappa),
                torch.full((2,10), .15), atol=3e-8, rtol=0)

    def test_zero_neural_probability_needs_no_epsilon(self):
        n = torch.zeros(2,6)
        m = torch.full_like(n, .2)
        t = torch.arange(7, dtype=torch.float32)
        for kappa in KAPPAS:
            expected = ((kappa*.15+t)/(kappa+t))[None].expand(2,-1)
            torch.testing.assert_close(weights_from_observed_targets(n,m,kappa),
                expected, atol=0, rtol=0)

    def test_zero_mkn_probability_with_positive_neural_is_mathematically_valid(self):
        n = torch.ones(1,4)
        m = torch.zeros_like(n)
        for kappa in KAPPAS:
            # An explicit FP32 numerator avoids Python scalar/tensor reverse
            # division's different reciprocal rounding in this exact check.
            expected = torch.full((5,), kappa*.15)/(kappa+torch.arange(5, dtype=torch.float32))
            torch.testing.assert_close(weights_from_observed_targets(n,m,kappa)[0],
                expected, atol=0, rtol=0)

    def test_invalid_denominator_is_explicit_failure_not_fallback(self):
        for n,m in ((0.,0.), (-1.,0.), (float('nan'),.2), (.2,float('inf'))):
            for kappa in KAPPAS:
                with self.subTest(n=n,m=m,kappa=kappa), self.assertRaisesRegex(ValueError,'denominators'):
                    weights_from_observed_targets(torch.tensor([[n]]), torch.tensor([[m]]), kappa)

    def test_shape_dtype_and_argument_checks(self):
        n = torch.ones(2,3)
        cases = [(n[0], n[0]), (n,n[:,:2]), (n.long(),n.long()), (n,n.double())]
        for left,right in cases:
            with self.assertRaises((TypeError,ValueError)):
                weights_from_observed_targets(left,right,8)
        with self.assertRaises(TypeError):
            weights_from_observed_targets([[.1]], torch.tensor([[.2]]), 8)

    def test_inputs_are_not_mutated(self):
        n = torch.tensor([[.2,.5,.8]])
        m = torch.tensor([[.8,.4,.1]])
        n_before, m_before = n.clone(), m.clone()
        for kappa in (None,*KAPPAS):
            weights_from_observed_targets(n,m,kappa)
        torch.testing.assert_close(n,n_before,atol=0,rtol=0)
        torch.testing.assert_close(m,m_before,atol=0,rtol=0)

    def test_current_target_never_affects_its_weight_and_last_target_is_unused(self):
        # These are evaluator target probabilities, not production inputs.
        n = torch.full((2,6), .3)
        m = torch.full((2,6), .6)
        for kappa in KAPPAS:
            original = weights_from_observed_targets(n[:,:-1],m[:,:-1],kappa)
            for current in range(n.shape[1]):
                altered = m.clone(); altered[:,current] = .01
                result = weights_from_observed_targets(n[:,:-1],altered[:,:-1],kappa)
                torch.testing.assert_close(result[:,:current+1],original[:,:current+1],atol=0,rtol=0)
                if current < n.shape[1]-1:
                    self.assertFalse(torch.equal(result[:,current+1],original[:,current+1]))

    def test_history_prefix_and_row_independence(self):
        n = torch.tensor([[.2,.8,.5,.1],[.7,.05,.6,.3]])
        m = torch.tensor([[.5,.2,.4,.9],[.3,.8,.1,.6]])
        for kappa in KAPPAS:
            full = weights_from_observed_targets(n,m,kappa)
            for length in range(n.shape[1]+1):
                torch.testing.assert_close(weights_from_observed_targets(n[:,:length],m[:,:length],kappa),
                    full[:,:length+1],atol=0,rtol=0)
            torch.testing.assert_close(weights_from_observed_targets(n.flip(0),m.flip(0),kappa).flip(0),
                full,atol=0,rtol=0)


class ProductionMixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(193)
        cls.parent = ExpandedSignedCache(synthetic_config()).eval()
        cls.parent.ngram.load_state_dict(synthetic_tables(), strict=True)
        with torch.no_grad():
            cls.parent.cache_recency_strength.fill_(-.4)
            cls.parent.cache_log_theta_delta.fill_(.03)
            cls.parent.pointer_key.weight.normal_(0,.02)
            cls.parent.pointer_gate[-1].weight.normal_(0,.02)
        cls.parent_state = {key:value.clone() for key,value in cls.parent.state_dict().items()}
        cls.ids = torch.tensor([[1,2,7,1,5,2,7,3,4],[3,1,2,7,1,5,2,7,9]])

    def model(self, kappa):
        model = build_model(dict(self.parent.config, push153_kappa=kappa)).eval()
        model.load_state_dict(self.parent_state, strict=True)
        return model

    def test_disabled_delegates_to_parent_and_is_bitwise_equal(self):
        model = self.model(None)
        with patch.object(ExpandedSignedCache,'predict_log_probs',autospec=True,
                          return_value=torch.tensor([123.])) as inherited:
            self.assertEqual(model.predict_log_probs(self.ids).item(),123.)
            inherited.assert_called_once_with(model,self.ids)
        torch.testing.assert_close(model.predict_log_probs(self.ids),
            self.parent.predict_log_probs(self.ids),atol=0,rtol=0)

    def test_one_token_delegates_for_every_kappa(self):
        ids = self.ids[:,:1]
        expected = self.parent.predict_log_probs(ids)
        for kappa in KAPPAS:
            model = self.model(kappa)
            torch.testing.assert_close(model.predict_log_probs(ids),expected,atol=0,rtol=0)
            with patch.object(model,'neural_and_cache',wraps=model.neural_and_cache) as neural:
                model.predict_log_probs(ids)
                self.assertEqual(neural.call_count,1)

    def test_all_first_positions_are_exact_parent_mixture(self):
        expected = self.parent.predict_log_probs(self.ids)[:,0]
        for kappa in KAPPAS:
            torch.testing.assert_close(self.model(kappa).predict_log_probs(self.ids)[:,0],
                expected,atol=0,rtol=0)

    def test_no_added_parameters_buffers_or_changed_tied_weights(self):
        for kappa in (None,*KAPPAS):
            model = self.model(kappa)
            self.assertEqual(set(model.state_dict()),set(self.parent_state))
            self.assertEqual(dict(model.named_parameters()).keys(),dict(self.parent.named_parameters()).keys())
            self.assertEqual(dict(model.named_buffers()).keys(),dict(self.parent.named_buffers()).keys())
            self.assertEqual(model.token.weight.data_ptr(),model.head.weight.data_ptr())
            for name,value in model.state_dict().items():
                torch.testing.assert_close(value,self.parent_state[name],atol=0,rtol=0)
            self.assertEqual(len(list(model.ngram.named_buffers())),21)

    def test_bad_config_rejected_and_callers_config_unchanged(self):
        for changed in (dict(push153_kappa=4),dict(mkn_weight=.2),dict(mkn_shapes={})):
            with self.assertRaises(ValueError):
                build_model(dict(self.parent.config,**changed))
        config = copy.deepcopy(self.parent.config)
        before = copy.deepcopy(config)
        model = build_model(config)
        self.assertEqual(config,before)
        self.assertIsNone(model.config['push153_kappa'])

    def test_exactly_one_neural_forward_and_one_statistics_query(self):
        model = self.model(32)
        with patch.object(model,'neural_and_cache',wraps=model.neural_and_cache) as neural:
            with patch.object(model.ngram,'probabilities',wraps=model.ngram.probabilities) as counts:
                model.predict_log_probs(self.ids)
                self.assertEqual(neural.call_count,1)
                self.assertEqual(counts.call_count,1)

    def test_independent_explicit_formula_matches_full_distribution(self):
        for kappa in KAPPAS:
            model = self.model(kappa)
            with torch.no_grad():
                n = model.neural_and_cache(self.ids)
                m = model.ngram.probabilities(self.ids,5)
                expected = torch.empty_like(n)
                for row in range(self.ids.shape[0]):
                    total = 0.
                    for t in range(self.ids.shape[1]):
                        if t:
                            token = self.ids[row,t]
                            pn, pm = float(n[row,t-1,token]), float(m[row,t-1,token])
                            total += .15*pm/(.85*pn+.15*pm)
                        weight = (kappa*.15+total)/(kappa+t)
                        expected[row,t] = (1-weight)*n[row,t]+weight*m[row,t]
            torch.testing.assert_close(model.predict_log_probs(self.ids).exp(),expected,atol=5e-8,rtol=2e-6)

    def test_shortcut_all_menus_variable_batch_and_length(self):
        for batch,length in ((1,1),(2,2),(1,5),(2,9)):
            ids = self.ids[:batch,:length]
            # Every target except the final one is literally the next input.
            targets = torch.cat((ids[:,1:],torch.full((batch,1),11,dtype=torch.long)),dim=1)
            with torch.no_grad():
                n = self.parent.neural_and_cache(ids)
                m = self.parent.ngram.probabilities(ids,5)
                n_targets = n.gather(-1,targets[...,None]).squeeze(-1)
                m_targets = m.gather(-1,targets[...,None]).squeeze(-1)
            for kappa in (None,*KAPPAS):
                weights = weights_from_observed_targets(n_targets[:,:-1],m_targets[:,:-1],kappa)
                shortcut = (1-weights)*n_targets+weights*m_targets
                # Exact parent operation order at first output (all if disabled).
                fixed = n_targets.clone().mul_(.85).add_(m_targets,alpha=.15)
                if kappa is None:
                    shortcut = fixed
                else:
                    shortcut[:,0] = fixed[:,0]
                output = self.model(kappa).predict_log_probs(ids)
                actual = output.gather(-1,targets[...,None]).squeeze(-1)
                torch.testing.assert_close(actual,shortcut.clamp_min(1e-30).log(),atol=2e-6,rtol=1e-6)
                self.assertAlmostEqual(float(-actual.double().sum()),
                    float(-shortcut.clamp_min(1e-30).log().double().sum()),places=4)

    def test_finite_normalized_fp32_and_no_grad(self):
        for kappa in (None,*KAPPAS):
            output = self.model(kappa).predict_log_probs(self.ids)
            self.assertEqual(output.dtype,torch.float32)
            self.assertFalse(output.requires_grad)
            self.assertTrue(torch.isfinite(output).all())
            torch.testing.assert_close(output.logsumexp(-1),torch.zeros_like(output[...,0]),atol=1e-6,rtol=0)

    def test_future_tail_cannot_change_earlier_outputs(self):
        for kappa in KAPPAS:
            model = self.model(kappa)
            expected = model.predict_log_probs(self.ids)
            for prefix in (1,3,7):
                changed = self.ids.clone(); changed[:,prefix:] += 31
                torch.testing.assert_close(model.predict_log_probs(changed)[:,:prefix],
                    expected[:,:prefix],atol=4e-6,rtol=1e-6)

    def test_short_prefix_contract_at_declared_lengths(self):
        ids = (torch.arange(512).reshape(2,256)%17).long()
        model = self.model(32)
        expected = model.predict_log_probs(ids)
        for length in (1,2,3,19,63,128,255):
            torch.testing.assert_close(model.predict_log_probs(ids[:,:length]),
                expected[:,:length],atol=1e-5,rtol=1e-5)

    def test_padding_cannot_change_valid_prefix(self):
        model = self.model(8)
        ids = self.ids[:,:5]
        padded = torch.cat((ids,torch.zeros((2,4),dtype=torch.long)),dim=1)
        torch.testing.assert_close(model.predict_log_probs(padded)[:,:5],
            model.predict_log_probs(ids),atol=1e-5,rtol=1e-5)

    def test_per_row_independence_permutation_and_exact_reset(self):
        model = self.model(128)
        original = model.predict_log_probs(self.ids)
        changed = self.ids.clone(); changed[1] = 17
        torch.testing.assert_close(model.predict_log_probs(changed)[0],original[0],atol=0,rtol=0)
        torch.testing.assert_close(model.predict_log_probs(self.ids.flip(0)).flip(0),original,atol=4e-6,rtol=1e-6)
        torch.testing.assert_close(model.predict_log_probs(self.ids[:1]),original[:1],atol=1e-5,rtol=1e-5)
        model.predict_log_probs(torch.tensor([[4,3],[1,2],[6,7]]))
        torch.testing.assert_close(model.predict_log_probs(self.ids),original,atol=0,rtol=0)

    def test_forward_does_not_update_any_parameter_table_or_config(self):
        model = self.model(32)
        before = {key:value.clone() for key,value in model.state_dict().items()}
        config = copy.deepcopy(model.config)
        rng = torch.random.get_rng_state().clone()
        model.predict_log_probs(self.ids)
        model.predict_log_probs(self.ids.flip(0))
        self.assertEqual(model.config,config)
        torch.testing.assert_close(torch.random.get_rng_state(),rng,atol=0,rtol=0)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,before[key],atol=0,rtol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
