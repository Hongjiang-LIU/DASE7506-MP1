"""Tiny CPU mathematical/contract checks only; no real data or timing probes."""
import unittest

import torch
from torch import nn

from push150_expand import ExpandedSignedCache
from push150_fast import FastExpandedSignedCache, pair_rotary, rms_norm_no_grad


class MockMKN(nn.Module):
    """Normalized current-token-only counts, never a hidden future-token source."""
    def __init__(self):
        super().__init__()
        self.register_buffer('unigram', torch.full((2048,), 1/2048))

    def probabilities(self, ids, order):
        result = self.unigram.expand(*ids.shape, -1).clone()*.5
        return result.scatter_add_(-1, ids[...,None], torch.full_like(ids[...,None], .5, dtype=torch.float32))


def config(**updates):
    value = dict(vocab=2048, width=32, heads=4, depth=2, context=256, ffn_hidden=64,
        rope=True, dropout=.2, lowrank_rank=8, temperature=1.25, cache_weight=.05,
        cache_theta=15., pointer_rank=8, pointer_max_weight=.5,
        push150_ranks=dict(qkv=0,proj=0,gate=0,value=0,out=0),
        mkn_shapes={'unigram':[2048]}, mkn_order=2, mkn_weight=.15, cpu_feature_microbatch=32)
    value.update(updates)
    return value


class FastProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(179)
        self.reference, self.fast = self.models(config())
        self.ids = torch.randint(0,31,(3,17))

    def models(self, recipe):
        reference = ExpandedSignedCache(recipe).eval()
        reference.ngram = MockMKN()
        with torch.no_grad():
            reference.cache_recency_strength.fill_(-.6)
            reference.cache_log_theta_delta.fill_(.1)
            reference.pointer_key.weight.normal_(0,.3)
            reference.pointer_gate[-1].weight.normal_(0,.2)
        fast = FastExpandedSignedCache(recipe).eval()
        fast.ngram = MockMKN()
        fast.load_state_dict(reference.state_dict())
        return reference, fast

    def same(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=6e-6, rtol=1e-6)

    def test_full_signed_cache_mkn_prediction_and_fp32(self):
        actual = self.fast.predict_log_probs(self.ids)
        self.same(actual, self.reference.predict_log_probs(self.ids))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.logsumexp(-1), torch.zeros(3,17), atol=1e-6, rtol=0)
        self.assertEqual(len(self.fast._fused_dense_weights), 2)

    def test_paired_rope_and_nonmutating_rms_match_original(self):
        rope = self.reference.blocks[0].rotary
        tensor = torch.randn(3,4,17,8)
        self.same(pair_rotary(tensor, rope), rope(tensor))
        x = torch.randn(3,17,32); original = x.clone()
        with torch.no_grad():
            torch.testing.assert_close(rms_norm_no_grad(x,self.reference.norm), self.reference.norm(x), atol=0,rtol=0)
        torch.testing.assert_close(x, original, atol=0,rtol=0)

    def test_internal_microbatches_all_three_declared_sizes(self):
        ids = torch.randint(0,31,(17,11))
        expected = self.reference.predict_log_probs(ids)
        for size in (8,16,32):
            with self.subTest(size=size):
                model = FastExpandedSignedCache(config(cpu_feature_microbatch=size)).eval()
                model.ngram = MockMKN(); model.load_state_dict(self.reference.state_dict())
                self.same(model.predict_log_probs(ids), expected)

    def test_future_prefix_permutation_and_cross_window_reset(self):
        expected = self.fast.predict_log_probs(self.ids)
        changed = self.ids.clone(); changed[:,8:] = torch.randint(0,2048,(3,9))
        self.same(self.fast.predict_log_probs(changed)[:,:8], expected[:,:8])
        self.same(self.fast.predict_log_probs(self.ids[:,:8]), expected[:,:8])
        self.same(self.fast.predict_log_probs(self.ids[:,:1]), expected[:,:1])
        self.same(self.fast.predict_log_probs(self.ids.flip(0)).flip(0), expected)
        self.same(self.fast.predict_log_probs(self.ids[:1]), expected[:1])
        self.fast.predict_log_probs(torch.randint(0,2048,(2,23)))
        torch.testing.assert_close(self.fast.predict_log_probs(self.ids), expected, atol=0,rtol=0)

    def test_low_rank_and_mixed_layouts_keep_correct_fallback(self):
        for layout in (dict(qkv=8,proj=8,gate=8,value=8,out=8),
                       dict(qkv=12,proj=0,gate=8,value=0,out=12)):
            with self.subTest(layout=layout):
                reference, fast = self.models(config(push150_ranks=layout))
                self.same(fast.predict_log_probs(self.ids), reference.predict_log_probs(self.ids))
                self.assertFalse(fast._fused_dense_weights)

    def test_learned_positions_and_width_configuration(self):
        reference, fast = self.models(config(width=16,heads=4,ffn_hidden=48,depth=1,rope=False))
        self.same(fast.predict_log_probs(self.ids), reference.predict_log_probs(self.ids))

    def test_optional_optimizations_disabled_are_still_same_predictor(self):
        recipe = config(push150_fast_fuse_dense_mlp=False, push150_fast_paired_rope=False,
                        push150_fast_inplace_rms=False)
        reference, fast = self.models(recipe)
        self.same(fast.predict_log_probs(self.ids), reference.predict_log_probs(self.ids))
        self.assertFalse(fast._fused_dense_weights)

    def test_no_new_persistent_weights_or_parameters(self):
        before = {name: value.clone() for name,value in self.fast.state_dict().items()}
        self.fast.predict_log_probs(self.ids)
        self.assertEqual(set(self.fast.state_dict()), set(self.reference.state_dict()))
        self.assertEqual(sum(p.numel() for p in self.fast.parameters()), sum(p.numel() for p in self.reference.parameters()))
        for name,value in self.fast.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0,rtol=0)
        for signature,weight in self.fast._fused_dense_weights.values():
            self.assertEqual(weight.shape, (128,32))
            self.assertEqual(weight.dtype, torch.float32)

    def test_load_state_dict_invalidates_derived_weights(self):
        self.fast.predict_log_probs(self.ids)
        with torch.no_grad():
            self.reference.blocks[0].mlp.gate.weight.add_(.03)
        self.fast.load_state_dict(self.reference.state_dict())
        self.assertFalse(self.fast._fused_dense_weights)
        self.same(self.fast.predict_log_probs(self.ids), self.reference.predict_log_probs(self.ids))

    def test_apply_and_train_invalidate_derived_weights(self):
        self.fast.predict_log_probs(self.ids)
        self.fast._apply(lambda tensor: tensor.clone())
        self.assertFalse(self.fast._fused_dense_weights)
        self.same(self.fast.predict_log_probs(self.ids), self.reference.predict_log_probs(self.ids))
        self.fast.train()
        self.assertFalse(self.fast._fused_dense_weights)

    def test_in_place_parameter_update_invalidates_by_version(self):
        self.fast.predict_log_probs(self.ids)
        old_version = self.fast._fused_dense_weights[0][0][0][1]
        with torch.no_grad():
            self.reference.blocks[0].mlp.gate.weight.add_(.02)
            self.fast.blocks[0].mlp.gate.weight.add_(.02)
        self.same(self.fast.predict_log_probs(self.ids), self.reference.predict_log_probs(self.ids))
        self.assertGreater(self.fast._fused_dense_weights[0][0][0][1], old_version)

    def test_training_loss_and_all_gradients_use_original_path(self):
        self.reference.train(); self.fast.train()
        target = self.ids.roll(-1,dims=1)
        torch.manual_seed(181)
        expected = -self.reference.training_log_probs(self.ids).gather(-1,target[...,None]).mean()
        expected.backward()
        torch.manual_seed(181)
        actual = -self.fast.training_log_probs(self.ids).gather(-1,target[...,None]).mean()
        actual.backward()
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        for (name,parameter),(other_name,other) in zip(self.fast.named_parameters(),self.reference.named_parameters()):
            self.assertEqual(name,other_name)
            self.assertIsNotNone(parameter.grad,name)
            torch.testing.assert_close(parameter.grad,other.grad,atol=0,rtol=0)
        self.assertFalse(self.fast._fused_dense_weights)

    def test_eval_with_gradients_does_not_detach_features(self):
        features = self.fast.features(self.ids)
        self.assertTrue(features.requires_grad)
        torch.testing.assert_close(features,self.reference.features(self.ids),atol=0,rtol=0)
        self.assertFalse(self.fast._fused_dense_weights)

    def test_invalid_microbatch_and_flags_rejected(self):
        for value in (0,1,7,True,8.0,64):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FastExpandedSignedCache(config(cpu_feature_microbatch=value))
        with self.assertRaises(ValueError):
            FastExpandedSignedCache(config(push150_fast_fuse_dense_mlp=1))


if __name__ == '__main__':
    unittest.main()
