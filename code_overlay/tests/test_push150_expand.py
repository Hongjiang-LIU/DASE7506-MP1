"""Tiny synthetic CPU checks; no course text, real checkpoint or benchmark."""
import copy
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn

from common import PROTOCOL
from efficient import EfficientGPT
from lowrank import LowRankLinear
from push147_signed_cache import SignedDistanceCache
from push150_expand import ExpandedSignedCache, LAYOUTS, projection_modules
from push150_init import initialize_from_parent, initialized_checkpoint


TINY_EXPANDED = dict(qkv=6, proj=0, gate=6, value=6, out=6)
TINY_CONTROL = dict.fromkeys(TINY_EXPANDED, 4)


def small_config(**extra):
    config = dict(vocab=2048, width=16, heads=4, depth=2, context=256,
        ffn_hidden=32, rope=True, dropout=0., lowrank_rank=4, temperature=1.25,
        cache_weight=.05, cache_theta=15., pointer_rank=4, pointer_max_weight=.5)
    config.update(extra)
    return config


def count_arrays():
    return dict(unigram=torch.full((2048,), 1/2048),
        contexts_2=torch.tensor([1], dtype=torch.int64), offsets_2=torch.tensor([0,1], dtype=torch.int32),
        words_2=torch.tensor([7], dtype=torch.int16), masses_2=torch.tensor([.5], dtype=torch.float32),
        backoffs_2=torch.tensor([.5], dtype=torch.float32))


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(173)
        self.source = SignedDistanceCache(small_config()).eval()
        with torch.no_grad():
            self.source.cache_recency_strength.fill_(-.7)
            self.source.cache_log_theta_delta.fill_(.12)
            self.source.pointer_key.weight.normal_(0,.02)
            self.source.pointer_gate[-1].weight.normal_(0,.02)
        self.parent = dict(protocol=PROTOCOL, implementation='push147_signed_cache',
            config=copy.deepcopy(self.source.config), model=self.source.state_dict(),
            train_tokens=376832000, ancestry={'retained': 'synthetic ancestor'})
        self.ids = torch.randint(2048, (2,13))

    def init(self, ranks=None, **kwargs):
        model, report = initialize_from_parent(self.parent,
            rank_layout=TINY_EXPANDED if ranks is None else ranks, **kwargs)
        return model.eval(), report

    def test_layout_sizes_and_exact_existing_features_implementation(self):
        model, _ = self.init()
        self.assertIs(ExpandedSignedCache.features, EfficientGPT.features)
        for block in model.blocks:
            self.assertIsInstance(block.proj, nn.Linear)
            self.assertEqual(block.qkv.left.weight.shape, (6,16))
            self.assertEqual(block.qkv.right.weight.shape, (48,6))
            self.assertEqual(block.mlp.out.left.weight.shape, (6,32))
            self.assertEqual(block.mlp.out.right.weight.shape, (16,6))

    def test_all_old_factors_preserved_and_extra_outputs_zero(self):
        model, _ = self.init()
        for prefix, _, module in projection_modules(model):
            if isinstance(module, LowRankLinear):
                torch.testing.assert_close(module.left.weight[:4], self.parent['model'][prefix+'.left.weight'], atol=0, rtol=0)
                torch.testing.assert_close(module.right.weight[:,:4], self.parent['model'][prefix+'.right.weight'], atol=0, rtol=0)
                self.assertEqual(module.right.weight[:,4:].count_nonzero().item(), 0)
                self.assertGreater(module.left.weight[4:].abs().sum().item(), 0)
            else:
                expected = self.parent['model'][prefix+'.right.weight']@self.parent['model'][prefix+'.left.weight']
                torch.testing.assert_close(module.weight, expected, atol=0, rtol=0)

    def test_expansion_preserves_features_and_signed_probabilities(self):
        model, _ = self.init()
        with torch.no_grad():
            torch.testing.assert_close(model.features(self.ids), self.source.features(self.ids), atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(model.neural_and_cache(self.ids), self.source.neural_and_cache(self.ids), atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(model.predict_log_probs(self.ids), self.source.predict_log_probs(self.ids), atol=4e-6, rtol=2e-6)
        self.assertLess(model.cache_recency_strength.item(), 0)

    def test_control_preserves_all_weights_and_predictions(self):
        model, _ = self.init(TINY_CONTROL, layout='control')
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, self.parent['model'][name], atol=0, rtol=0)
        torch.testing.assert_close(model.predict_log_probs(self.ids), self.source.predict_log_probs(self.ids), atol=4e-6, rtol=2e-6)

    def test_dense_fold_preserves_signed_probabilities(self):
        model, _ = self.init(LAYOUTS['dense'], layout='dense')
        self.assertTrue(all(isinstance(module, nn.Linear) for _,_,module in projection_modules(model)))
        torch.testing.assert_close(model.predict_log_probs(self.ids), self.source.predict_log_probs(self.ids), atol=5e-6, rtol=2e-6)

    def test_new_rank_receives_gradient_and_then_new_left_learns(self):
        model, _ = self.init()
        model.train()
        target = self.ids.roll(-1, dims=1)
        loss = -model.training_log_probs(self.ids).gather(-1, target[...,None]).mean()
        self.assertTrue(loss.requires_grad)
        loss.backward()
        for prefix,_,module in projection_modules(model):
            if isinstance(module, LowRankLinear):
                self.assertGreater(module.right.weight.grad[:,4:].abs().sum().item(), 0, prefix)
                torch.testing.assert_close(module.left.weight.grad[4:], torch.zeros_like(module.left.weight.grad[4:]), atol=0, rtol=0)
        with torch.no_grad():
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                parameter.add_(parameter.grad, alpha=-.1)
        model.zero_grad(set_to_none=True)
        (-model.training_log_probs(self.ids).gather(-1, target[...,None]).mean()).backward()
        for prefix,_,module in projection_modules(model):
            if isinstance(module, LowRankLinear):
                self.assertGreater(module.left.weight.grad[4:].abs().sum().item(), 0, prefix)

    def test_future_tokens_cannot_change_earlier_predictions(self):
        model, _ = self.init()
        original = model.predict_log_probs(self.ids)
        changed = self.ids.clone(); changed[:,7:] = torch.randint(2048, (2,6))
        torch.testing.assert_close(original[:,:7], model.predict_log_probs(changed)[:,:7], atol=4e-6, rtol=2e-6)

    def test_short_prefix_normalization_and_first_position(self):
        model, _ = self.init()
        original = model.predict_log_probs(self.ids)
        for length in (1,4,9):
            torch.testing.assert_close(original[:,:length], model.predict_log_probs(self.ids[:,:length]), atol=4e-6, rtol=2e-6)
        torch.testing.assert_close(original.logsumexp(-1), torch.zeros(2,13), atol=1e-6, rtol=0)
        with torch.no_grad():
            neural = (model.head(model.features(self.ids)).float()/model.config['temperature']).softmax(-1)
            torch.testing.assert_close(original[:,0].exp(), neural[:,0], atol=1e-7, rtol=1e-5)

    def test_permutation_independence_and_no_cross_window_state(self):
        model, _ = self.init()
        original = model.predict_log_probs(self.ids)
        # CPU FP32 batched kernels need not be bitwise permutation invariant.
        # This is tighter than the fixed contract's prediction tolerance.
        torch.testing.assert_close(model.predict_log_probs(self.ids.flip(0)).flip(0), original, atol=4e-6, rtol=2e-6)
        changed = self.ids.clone(); changed[1] = 7
        torch.testing.assert_close(model.predict_log_probs(changed)[0], original[0], atol=4e-6, rtol=2e-6)
        model.predict_log_probs(torch.randint(2048, (3,5)))
        torch.testing.assert_close(model.predict_log_probs(self.ids), original, atol=0, rtol=0)

    def test_mkn_tensors_metadata_and_full_mixture_preserved(self):
        arrays = count_arrays()
        config = small_config(mkn_shapes={name:list(value.shape) for name,value in arrays.items()},
            mkn_order=2, mkn_weight=.15, discount_scale=.7, statistics_recipe={'source':'synthetic'})
        source = SignedDistanceCache(config).eval()
        source.ngram.load_state_dict(arrays)
        parent = dict(self.parent, config=config, model=source.state_dict())
        model, _ = initialize_from_parent(parent, rank_layout=TINY_EXPANDED)
        model.eval()
        for name,value in arrays.items():
            torch.testing.assert_close(model.ngram.state_dict()[name], value, atol=0, rtol=0)
        self.assertEqual(model.config['statistics_recipe'], config['statistics_recipe'])
        self.assertIsNot(model.config['statistics_recipe'], config['statistics_recipe'])
        self.assertEqual(model.config['discount_scale'], .7)
        ids = torch.tensor([[1,4,1,7,2]])
        torch.testing.assert_close(model.predict_log_probs(ids), source.predict_log_probs(ids), atol=4e-6, rtol=2e-6)
        with patch.object(model.ngram, 'probabilities', side_effect=AssertionError('Training read counts')):
            (-model.training_log_probs(ids).gather(-1, ids[...,None]).mean()).backward()

    def test_initialization_does_not_mutate_parent_or_global_rng(self):
        before = {name:value.clone() for name,value in self.parent['model'].items()}
        config = copy.deepcopy(self.parent['config'])
        random_state = torch.random.get_rng_state().clone()
        self.init()
        torch.testing.assert_close(torch.random.get_rng_state(), random_state, atol=0, rtol=0)
        self.assertEqual(self.parent['config'], config)
        for name,value in before.items():
            torch.testing.assert_close(self.parent['model'][name], value, atol=0, rtol=0)

    def test_seed_reproducible_new_capacity(self):
        first,_ = self.init(seed=173); second,_ = self.init(seed=173); third,_ = self.init(seed=174)
        torch.testing.assert_close(first.blocks[0].qkv.left.weight, second.blocks[0].qkv.left.weight, atol=0, rtol=0)
        self.assertFalse(torch.equal(first.blocks[0].qkv.left.weight[4:], third.blocks[0].qkv.left.weight[4:]))

    def test_tied_embedding_and_single_parent_cost_survive_serialization(self):
        checkpoint, report = initialized_checkpoint(self.parent, 'a'*64, rank_layout=TINY_EXPANDED)
        self.assertEqual(checkpoint['model']['token.weight'].data_ptr(), checkpoint['model']['head.weight'].data_ptr())
        self.assertEqual(checkpoint['train_tokens'], 376832000)
        self.assertEqual(report['new_training_targets'], 0)
        self.assertEqual(checkpoint['ancestry']['shared_inherited_targets_counted_once'], 376832000)
        self.assertEqual(checkpoint['ancestry']['prior_ancestry'], self.parent['ancestry'])
        self.assertIs(checkpoint['ancestry'], checkpoint['initialization_ancestry'])
        buffer = io.BytesIO(); torch.save(checkpoint, buffer); buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)
        self.assertEqual(restored['model']['token.weight'].data_ptr(), restored['model']['head.weight'].data_ptr())
        model = ExpandedSignedCache(restored['config']); model.load_state_dict(restored['model'])
        self.assertEqual(model.token.weight.data_ptr(), model.head.weight.data_ptr())

    def test_invalid_layouts_and_rank_reduction_rejected(self):
        for layout in ({'qkv':6}, dict(TINY_EXPANDED,qkv=True), dict(TINY_EXPANDED,out=17), dict(TINY_EXPANDED,gate=-1)):
            with self.assertRaises(ValueError):
                self.init(layout)
        with self.assertRaisesRegex(ValueError, 'Cannot shrink'):
            self.init(dict(TINY_EXPANDED,out=3))

    def test_unknown_source_tensor_or_untied_parent_rejected(self):
        parent = copy.deepcopy(self.parent); parent['model']['unexpected'] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, 'Unaccounted'):
            initialize_from_parent(parent, rank_layout=TINY_EXPANDED)
        parent = copy.deepcopy(self.parent); parent['model']['head.weight'] = parent['model']['head.weight']+1
        with self.assertRaisesRegex(ValueError, 'tied embedding'):
            initialize_from_parent(parent, rank_layout=TINY_EXPANDED)

    def test_signed_scalar_projection_bounds_remain_inherited(self):
        model,_ = self.init()
        with torch.no_grad():
            model.cache_recency_strength.fill_(-100)
            model.cache_log_theta_delta.fill_(100)
        model.project_cache_parameters()
        theta, alpha = model.cache_scales()
        self.assertEqual(alpha.item(), -4)
        self.assertLessEqual(theta.item(), 40)


if __name__ == '__main__':
    unittest.main()
