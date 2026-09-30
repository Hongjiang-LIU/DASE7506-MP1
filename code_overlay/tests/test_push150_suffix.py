"""Synthetic CPU-only suffix checks: no corpus, checkpoints, GPU or timing."""
import copy
import io
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from common import PROTOCOL
from push147_train import training_loss as original_training_loss
from push148_train import build_optimizers
from push150_expand import ExpandedSignedCache
from push150_suffix import (SuffixSignedCache, SUFFIX_ORDERS, SUFFIX_PARAMETER_NAMES,
                            suffix_match_masks, training_loss)
from push150_suffix_init import initialize_from_parent, initialized_checkpoint
from push150_suffix_calibrate import (TEMPERATURES, MIXTURE_WEIGHTS, neural_cache_targets,
                                      score_grid)
from push150_suffix_train import (source_digests, training_loss as driver_loss,
                                  pack_train_state, restore_train_state, snapshot)


def config(**extra):
    result = dict(vocab=2048, width=16, heads=4, depth=2, context=256, ffn_hidden=32,
        rope=True, dropout=0., lowrank_rank=4, temperature=1.25, cache_weight=.05,
        cache_theta=15., pointer_rank=4, pointer_max_weight=.5,
        push150_ranks=dict(qkv=6, proj=0, gate=6, value=6, out=6))
    result.update(extra)
    return result


def count_arrays():
    return dict(unigram=torch.full((2048,), 1/2048),
        contexts_2=torch.tensor([1], dtype=torch.int64), offsets_2=torch.tensor([0,1], dtype=torch.int32),
        words_2=torch.tensor([7], dtype=torch.int16), masses_2=torch.tensor([.5], dtype=torch.float32),
        backoffs_2=torch.tensor([.5], dtype=torch.float32))


def reference_masks(ids):
    result = {}
    for order in SUFFIX_ORDERS:
        if order > ids.shape[1]:
            break
        mask = torch.zeros(ids.shape[0], ids.shape[1], ids.shape[1], dtype=torch.bool)
        for row in range(ids.shape[0]):
            for position in range(order-1, ids.shape[1]):
                for source in range(order-1, position):
                    mask[row, position, source] = torch.equal(
                        ids[row, position-order+1:position+1], ids[row, source-order+1:source+1])
        result[order] = mask
    return result


class SuffixTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(191)
        self.source = ExpandedSignedCache(config()).eval()
        with torch.no_grad():
            self.source.cache_recency_strength.fill_(-.6)
            self.source.cache_log_theta_delta.fill_(.1)
            self.source.pointer_key.weight.normal_(0,.02)
            self.source.pointer_gate[-1].weight.normal_(0,.02)
        self.parent = dict(protocol=PROTOCOL, implementation='push150_expand',
            config=copy.deepcopy(self.source.config), model=self.source.state_dict(),
            train_tokens=500000000, ancestry={'synthetic_phase_a': 'one inherited chain'})
        self.model, self.report = initialize_from_parent(self.parent)
        self.model.eval()
        self.ids = torch.tensor([[1,2,3,4,10,1,2,3,4,11,1,2,3,4,12],
                                 [5,6,7,8,20,5,6,7,8,21,5,6,7,8,22]])

    def nonzero(self):
        with torch.no_grad():
            for name, value in zip(SUFFIX_PARAMETER_NAMES, (.8,-.3,.5,.9)):
                getattr(self.model, name).fill_(value)

    def test_masks_equal_naive_suffix_reference_and_no_short_prefix_matches(self):
        actual = dict(suffix_match_masks(self.ids))
        expected = reference_masks(self.ids)
        for order in SUFFIX_ORDERS:
            torch.testing.assert_close(actual[order], expected[order], atol=0, rtol=0)
            self.assertFalse(actual[order][:, :order-1].any())
            self.assertFalse(actual[order][:, :, :order-1].any())
            self.assertFalse(actual[order].triu().any())
        self.assertTrue(actual[4][0,8,3])
        self.assertTrue(actual[4][0,13,8])
        self.assertFalse(actual[4][0,3,3])

    def test_exact_encoding_supports_max_token_zero_padding_and_int32(self):
        ids = torch.tensor([[2047,0,2047,0,7,2047,0,2047,0,8]], dtype=torch.int32)
        for order, mask in suffix_match_masks(ids):
            torch.testing.assert_close(mask, reference_masks(ids)[order], atol=0, rtol=0)
        for length in (1,2,3,4):
            self.assertEqual(len(dict(suffix_match_masks(ids[:,:length]))), length)

    def test_exactly_four_added_fp32_scalar_parameters_and_no_weight_changes(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters())-
                         sum(p.numel() for p in self.source.parameters()), 4)
        self.assertEqual(set(self.model.state_dict())-set(self.source.state_dict()), set(SUFFIX_PARAMETER_NAMES))
        for name in SUFFIX_PARAMETER_NAMES:
            value = getattr(self.model, name)
            self.assertEqual(value.ndim, 0)
            self.assertEqual(value.dtype, torch.float32)
            self.assertEqual(value.item(), 0)
        for name, value in self.source.state_dict().items():
            torch.testing.assert_close(value, self.model.state_dict()[name], atol=0, rtol=0)

    def test_zero_initialization_preserves_features_attention_gate_and_predictor(self):
        with torch.no_grad():
            original = self.source.features(self.ids)
            current = self.model.features(self.ids)
            torch.testing.assert_close(current, original, atol=0, rtol=0)
            for first, second in zip(self.source.cache_attention_and_gate(original),
                                     self.model.cache_attention_and_gate(current, self.ids)):
                torch.testing.assert_close(first, second, atol=0, rtol=0)
            torch.testing.assert_close(self.model.neural_and_cache(self.ids),
                                       self.source.neural_and_cache(self.ids), atol=0, rtol=0)
        torch.testing.assert_close(self.model.predict_log_probs(self.ids),
                                   self.source.predict_log_probs(self.ids), atol=0, rtol=0)

    def test_zero_initialization_preserves_training_objective_and_inherited_gradients(self):
        targets = self.ids.roll(-1, dims=1)
        self.source.train(); self.model.train()
        original = original_training_loss(self.source, self.ids, targets, .1)
        current = training_loss(self.model, self.ids, targets, .1)
        torch.testing.assert_close(original, current, atol=0, rtol=0)
        original.backward(); current.backward()
        other = dict(self.model.named_parameters())
        for name, parameter in self.source.named_parameters():
            torch.testing.assert_close(parameter.grad, other[name].grad, atol=2e-7, rtol=2e-5)

    def test_nonzero_attention_is_manual_semantic_plus_nested_suffix_formula(self):
        self.nonzero()
        features = torch.randn(2,15,16)
        normalized = F.normalize(features, dim=-1)
        theta, alpha = self.model.cache_scales()
        query, key = self.model.pointer_query(normalized), self.model.pointer_key(normalized)
        scores = theta*(normalized@normalized.transpose(-1,-2))
        scores = scores+(query@key.transpose(-1,-2))/math.sqrt(query.shape[-1])
        scores = scores-alpha*self.model._cache_log_distance[:15,:15]
        for order, mask in reference_masks(self.ids).items():
            scores = scores+getattr(self.model, SUFFIX_PARAMETER_NAMES[order-1])*mask
        expected = scores.masked_fill(~self.model._cache_allowed[:15,:15], -1e4).softmax(-1)
        actual, _ = self.model.cache_attention_and_gate(features, self.ids)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_cache_successor_values_never_come_from_future_positions(self):
        self.nonzero()
        ids = torch.tensor([[1,2,1,2,1,99]])
        with torch.no_grad():
            attention, gate = self.model.cache_attention_and_gate(self.model.features(ids), ids)
            values = torch.cat([ids[:,1:], ids.new_zeros(1,1)], dim=1)
            contribution = torch.zeros(1,6,2048)
            contribution.scatter_add_(-1, values[:,None,:].expand(-1,6,-1), attention*gate)
        self.assertEqual(contribution[0,:5,99].count_nonzero().item(), 0)
        self.assertEqual(contribution[0,0].count_nonzero().item(), 0)
        self.assertGreater(contribution[0,2,2].item(), 0)

    def test_future_edits_do_not_change_earlier_predictions(self):
        self.nonzero()
        original = self.model.predict_log_probs(self.ids)
        changed = self.ids.clone(); changed[:,8:] = torch.randint(2048, (2,7))
        torch.testing.assert_close(original[:,:8], self.model.predict_log_probs(changed)[:,:8], atol=4e-6, rtol=2e-6)

    def test_short_prefixes_and_first_position_are_consistent(self):
        self.nonzero()
        original = self.model.predict_log_probs(self.ids)
        for length in (1,2,3,4,9):
            torch.testing.assert_close(original[:,:length], self.model.predict_log_probs(self.ids[:,:length]), atol=4e-6, rtol=2e-6)
        with torch.no_grad():
            neural = (self.model.head(self.model.features(self.ids)).float()/self.model.config['temperature']).softmax(-1)
        torch.testing.assert_close(original[:,0].exp(), neural[:,0], atol=1e-7, rtol=1e-5)

    def test_batch_permutation_independence_reset_and_no_input_state(self):
        self.nonzero()
        attributes = set(vars(self.model))
        state = {name:value.clone() for name,value in self.model.state_dict().items()}
        original = self.model.predict_log_probs(self.ids)
        torch.testing.assert_close(self.model.predict_log_probs(self.ids.flip(0)).flip(0), original, atol=4e-6, rtol=2e-6)
        changed = self.ids.clone(); changed[1] = 17
        torch.testing.assert_close(self.model.predict_log_probs(changed)[0], original[0], atol=4e-6, rtol=2e-6)
        self.model.predict_log_probs(torch.randint(2048,(3,7)))
        torch.testing.assert_close(self.model.predict_log_probs(self.ids), original, atol=0, rtol=0)
        self.assertEqual(set(vars(self.model)), attributes)
        for name,value in state.items():
            torch.testing.assert_close(value, self.model.state_dict()[name], atol=0, rtol=0)

    def test_biases_have_finite_nonzero_gradients_and_scalar_optimizer_group(self):
        self.model.train()
        with torch.no_grad():
            self.model.cache_log_theta_delta.fill_(self.model._theta_log_lower)
            self.model.pointer_gate[-1].bias.zero_()
        loss = training_loss(self.model, self.ids, self.ids.roll(-1,dims=1), 0.)
        loss.backward()
        for name in SUFFIX_PARAMETER_NAMES:
            gradient = getattr(self.model,name).grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().item(), 1e-10, name)
        optimizers, _ = build_optimizers(self.model, 'adamw', 7.5e-6, 7.5e-5, .00025, .0005)
        scalar_group = next(group for group in optimizers[0].param_groups if group['base_lr'] == .00025)
        ids = {id(value) for value in scalar_group['params']}
        self.assertTrue(all(id(getattr(self.model,name)) in ids for name in SUFFIX_PARAMETER_NAMES))
        self.assertEqual(scalar_group['weight_decay'], 0.)

    def test_parameter_projection_preserves_declared_and_inherited_bounds(self):
        with torch.no_grad():
            for name,value in zip(SUFFIX_PARAMETER_NAMES, (-100,100,-5,6)):
                getattr(self.model,name).fill_(value)
            self.model.cache_recency_strength.fill_(-99)
            self.model.cache_log_theta_delta.fill_(99)
        self.model.project_cache_parameters()
        self.assertEqual([getattr(self.model,name).item() for name in SUFFIX_PARAMETER_NAMES], [-4,4,-4,4])
        theta, alpha = self.model.cache_scales()
        self.assertEqual(alpha.item(), -4)
        self.assertLessEqual(theta.item(), 40)

    def test_inference_and_differentiable_training_probability_paths_match(self):
        self.nonzero()
        differentiable = self.model.training_log_probs(self.ids)
        predicted = self.model.predict_log_probs(self.ids)
        torch.testing.assert_close(differentiable, predicted, atol=2e-6, rtol=2e-6)
        self.assertTrue(differentiable.requires_grad)
        self.assertFalse(predicted.requires_grad)
        self.assertTrue(torch.isfinite(predicted).all())
        torch.testing.assert_close(predicted.logsumexp(-1), torch.zeros(2,15), atol=1e-6, rtol=0)
        self.assertIs(driver_loss, training_loss)

    def test_explicit_ids_api_cannot_silently_drop_suffix_training(self):
        features = self.model.features(self.ids)
        with self.assertRaises(TypeError):
            self.model.cache_attention_and_gate(features)
        with self.assertRaises(ValueError):
            self.model.cache_attention_and_gate(features, self.ids[:,:3])
        with patch.object(self.model, 'cache_attention_and_gate', wraps=self.model.cache_attention_and_gate) as observed:
            training_loss(self.model, self.ids, self.ids.roll(-1,dims=1), 0.)
            self.assertIs(observed.call_args.args[1], self.ids)

    def test_mkn_tables_metadata_and_zero_bias_full_predictor_preserved(self):
        arrays = count_arrays()
        cfg = config(mkn_shapes={name:list(value.shape) for name,value in arrays.items()},
            mkn_order=2, mkn_weight=.15, discount_scale=.7, statistics_recipe={'synthetic':True})
        original = ExpandedSignedCache(cfg).eval(); original.ngram.load_state_dict(arrays)
        parent = dict(self.parent, config=cfg, model=original.state_dict())
        model, _ = initialize_from_parent(parent); model.eval()
        ids = torch.tensor([[1,4,1,7,1,2]])
        torch.testing.assert_close(model.predict_log_probs(ids), original.predict_log_probs(ids), atol=0, rtol=0)
        for name,value in arrays.items():
            torch.testing.assert_close(model.ngram.state_dict()[name], value, atol=0, rtol=0)
        self.assertEqual(model.config['discount_scale'], .7)
        self.assertEqual(model.config['statistics_recipe'], cfg['statistics_recipe'])
        self.assertIsNot(model.config['statistics_recipe'], cfg['statistics_recipe'])
        with patch.object(model.ngram, 'probabilities', side_effect=AssertionError('Training used statistics')):
            training_loss(model, ids, ids.roll(-1,dims=1), 0.).backward()
            model.training_log_probs(ids).sum().backward()

    def test_calibration_explicit_cache_gathers_match_full_predictor_all_15_choices(self):
        self.nonzero()
        arrays = count_arrays()
        cfg = dict(self.model.config, mkn_shapes={name:list(value.shape) for name,value in arrays.items()},
                   mkn_order=2, mkn_weight=.15)
        model = SuffixSignedCache(cfg).eval()
        state = dict(self.model.state_dict(), **{'ngram.'+name:value for name,value in arrays.items()})
        model.load_state_dict(state)
        ids = self.ids[:1,:7]; targets = ids.roll(-1,dims=1)
        gathered = neural_cache_targets(model, ids, targets)
        with torch.no_grad():
            counts = model.ngram.probabilities(ids, 2).gather(-1, targets[...,None]).squeeze(-1)
        for temperature in TEMPERATURES:
            for weight in MIXTURE_WEIGHTS:
                model.config.update(temperature=temperature,mkn_weight=weight)
                actual = model.predict_log_probs(ids).exp().gather(-1,targets[...,None]).squeeze(-1)
                expected = (1-weight)*gathered[temperature]+weight*counts
                torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
        trials, best = score_grid({key:value.numpy() for key,value in gathered.items()},
                                  counts.numpy(), targets.numpy(), 100)
        self.assertEqual(len(trials), 15)
        self.assertIn(best, trials)

    def test_initialization_preserves_parent_and_rng_and_serialized_tying(self):
        before = {name:value.clone() for name,value in self.parent['model'].items()}
        rng = torch.get_rng_state().clone()
        checkpoint, report = initialized_checkpoint(self.parent, 'a'*64)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        self.assertEqual(checkpoint['train_tokens'], self.parent['train_tokens'])
        self.assertEqual(checkpoint['ancestry']['shared_inherited_targets_counted_once'], self.parent['train_tokens'])
        self.assertEqual(checkpoint['ancestry']['prior_ancestry'], self.parent['ancestry'])
        self.assertIs(checkpoint['ancestry'], checkpoint['initialization_ancestry'])
        self.assertEqual(report['new_training_targets'], 0)
        self.assertEqual(report['added_parameters'], 4)
        for name,value in before.items():
            torch.testing.assert_close(value, self.parent['model'][name], atol=0, rtol=0)
        buffer = io.BytesIO(); torch.save(checkpoint, buffer); buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)
        self.assertEqual(restored['model']['head.weight'].data_ptr(), restored['model']['token.weight'].data_ptr())
        model = SuffixSignedCache(restored['config']); model.load_state_dict(restored['model'])
        self.assertIs(model.head.weight, model.token.weight)

    def test_new_scalars_survive_raw_ema_optimizer_and_rng_resume(self):
        averaged = copy.deepcopy(self.model).eval().requires_grad_(False)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.01, momentum=.9)
        for parameter in self.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        with torch.no_grad():
            averaged.cache_suffix_bias_1.fill_(.3)
        sampler = torch.Generator().manual_seed(191)
        identity = {'recipe':{'steps':4000}, 'parent':'synthetic'}
        state = pack_train_state(self.model, averaged, [optimizer], sampler, identity, 1, {'completed_steps':1})
        expected = torch.randint(100,(4,),generator=sampler)
        with torch.no_grad():
            self.model.cache_suffix_bias_1.fill_(2)
            averaged.cache_suffix_bias_1.fill_(3)
        optimizer.state.clear()
        step, log = restore_train_state(state,self.model,averaged,[optimizer],sampler,identity)
        self.assertEqual(step, 1); self.assertEqual(log['completed_steps'], 1)
        self.assertAlmostEqual(self.model.cache_suffix_bias_1.item(), -.01)
        self.assertAlmostEqual(averaged.cache_suffix_bias_1.item(), .3)
        self.assertTrue(optimizer.state)
        torch.testing.assert_close(torch.randint(100,(4,),generator=sampler), expected, atol=0, rtol=0)

    def test_snapshot_lineage_and_recursive_sources_include_new_loss_and_calibrator(self):
        parent, _ = initialized_checkpoint(self.parent,'a'*64)
        recipe = {'seed':191}
        saved = snapshot(self.model,parent,self.model.config,'b'*64,recipe,1000,'ema')
        self.assertEqual(saved['implementation'], 'push150_suffix')
        self.assertEqual(saved['train_tokens'], self.parent['train_tokens']+1000*8192)
        self.assertEqual(saved['ancestry'], parent['ancestry'])
        self.assertTrue(set(SUFFIX_PARAMETER_NAMES) <= set(saved['model']))
        sources = source_digests('push150_suffix')
        self.assertTrue({'push150_suffix.py','push150_suffix_init.py','push150_suffix_train.py',
                         'push150_suffix_calibrate.py','evaluate.py','push145_mkn_select.py'} <= set(sources))
        self.assertTrue(all(len(value) == 64 for value in sources.values()))

    def test_invalid_parent_configuration_and_input_rejected(self):
        with self.assertRaises(ValueError):
            SuffixSignedCache(config(push150_suffix_orders=[1,2]))
        with self.assertRaises(ValueError):
            SuffixSignedCache(config(push150_suffix_bounds=[0,4]))
        with self.assertRaises(ValueError):
            initialize_from_parent(dict(self.parent, implementation='push150_fast'))
        extra = copy.deepcopy(self.parent); extra['model']['undeclared'] = torch.zeros(1)
        with self.assertRaises(ValueError):
            initialize_from_parent(extra)
        with self.assertRaises(ValueError):
            list(suffix_match_masks(torch.ones(2,3)))
        with self.assertRaises(ValueError):
            list(suffix_match_masks(torch.empty(2,0,dtype=torch.long)))
        with self.assertRaises(ValueError):
            list(suffix_match_masks(self.ids, 1<<20))


if __name__ == '__main__':
    unittest.main()
