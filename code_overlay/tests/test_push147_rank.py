import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from efficient import EfficientGPT
from lowrank import LowRankLinear
from push145_cache import LearnedCacheGPT
from push147_rank import RankLayoutCache
from common import PROTOCOL
from push147_rank_init import initialize_from_dense, lineage_targets, main, verified_shared_evidence


def small_config(**updates):
    config = dict(vocab=2048, width=32, heads=4, depth=2, context=256, ffn_hidden=64,
                  rope=True, dropout=0., lowrank_rank=16, temperature=1.15,
                  cache_weight=.05, cache_theta=15., pointer_rank=8,
                  rank_layout=dict(qkv=12, proj=0, gate=8, value=8, out=8))
    config.update(updates)
    return config


def count_arrays():
    return dict(unigram=torch.full((2048,), 1 / 2048),
                contexts_2=torch.tensor([1], dtype=torch.int64),
                offsets_2=torch.tensor([0, 1], dtype=torch.int32),
                words_2=torch.tensor([7], dtype=torch.int16),
                masses_2=torch.tensor([.5], dtype=torch.float32),
                backoffs_2=torch.tensor([.5], dtype=torch.float32))


class RankLayoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(147)
        self.model = RankLayoutCache(small_config()).eval()

    def test_layout_and_parameter_count(self):
        for block in self.model.blocks:
            self.assertIsInstance(block.proj, nn.Linear)
            self.assertIsInstance(block.qkv, LowRankLinear)
            self.assertEqual(block.qkv.left.out_features, 12)
            self.assertEqual(block.mlp.gate.left.out_features, 8)
        # No real model weights/data are loaded by this architecture-only check.
        model = RankLayoutCache(dict(vocab=2048, width=320, heads=8, depth=6,
            context=256, ffn_hidden=864, dropout=.2, lowrank_rank=128,
            pointer_rank=32, cache_weight=.05))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4915345)

    def test_invalid_layout_is_rejected(self):
        for layout in ({"qkv": 12}, dict(qkv=-1, proj=0, gate=8, value=8, out=8),
                       dict(qkv=33, proj=0, gate=8, value=8, out=8),
                       dict(qkv=True, proj=0, gate=8, value=8, out=8)):
            with self.assertRaises(ValueError):
                RankLayoutCache(small_config(rank_layout=layout))

    def test_training_remains_differentiable(self):
        self.model.train()
        ids = torch.tensor([[1, 2, 3, 1, 2, 3, 4, 1, 2]])
        target = torch.tensor([[2, 3, 1, 2, 3, 4, 1, 2, 3]])
        logp = self.model.training_log_probs(ids)
        self.assertTrue(logp.requires_grad)
        (-logp.gather(-1, target[..., None]).mean()).backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(self.model.pointer_key.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(self.model.blocks[0].proj.weight.grad.abs().sum().item(), 0.)

    def test_causality_batch_independence_reset_and_normalization(self):
        ids = torch.randint(2048, (2, 23))
        original = self.model.predict_log_probs(ids)
        changed = ids.clone()
        changed[:, 11:] = torch.randint(2048, (2, 12))
        torch.testing.assert_close(original[:, :11], self.model.predict_log_probs(changed)[:, :11],
                                   atol=3e-6, rtol=1e-6)
        changed = ids.clone()
        changed[1] = 9
        torch.testing.assert_close(original[0], self.model.predict_log_probs(changed)[0], atol=0, rtol=0)
        self.model.predict_log_probs(torch.randint(2048, (3, 17)))
        torch.testing.assert_close(original, self.model.predict_log_probs(ids), atol=0, rtol=0)
        torch.testing.assert_close(original.logsumexp(-1), torch.zeros(2, 23), atol=1e-6, rtol=0)
        torch.testing.assert_close(original[:, :1], self.model.predict_log_probs(ids[:, :1]),
                                   atol=3e-6, rtol=1e-6)

    def test_mkn_prediction_and_no_table_training(self):
        arrays = count_arrays()
        model = RankLayoutCache(small_config(mkn_shapes={k: list(v.shape) for k, v in arrays.items()},
                                             mkn_order=2, mkn_weight=.2)).eval()
        model.ngram.load_state_dict(arrays)
        ids = torch.tensor([[1, 4, 1, 7, 2]])
        with torch.no_grad():
            expected = model.neural_and_cache(ids) * .8 + model.ngram.probabilities(ids, 2) * .2
        torch.testing.assert_close(model.predict_log_probs(ids).exp(), expected, atol=1e-7, rtol=1e-5)
        with patch.object(model.ngram, "probabilities", side_effect=AssertionError("Training read count tables")):
            loss = -model.training_log_probs(ids).gather(-1, ids[..., None]).mean()
            loss.backward()
        self.assertIsNotNone(model.pointer_gate[-1].weight.grad)


class RankInitializationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(147)
        self.config = small_config()
        self.dense = EfficientGPT(self.config).eval()
        self.dense_checkpoint = dict(config=self.config, model=self.dense.state_dict())

    def test_dense_copy_and_truncated_svd(self):
        model, errors = initialize_from_dense(self.dense_checkpoint, rank_layout=self.config["rank_layout"])
        torch.testing.assert_close(model.token.weight, self.dense.token.weight, atol=0, rtol=0)
        torch.testing.assert_close(model.blocks[0].proj.weight, self.dense.blocks[0].proj.weight, atol=0, rtol=0)
        source = self.dense.blocks[0].qkv.weight
        u, singular, vh = torch.linalg.svd(source, full_matrices=False)
        expected = (u[:, :12] * singular[:12]) @ vh[:12]
        actual = model.blocks[0].qkv.right.weight @ model.blocks[0].qkv.left.weight
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
        self.assertEqual(len(errors), self.config["depth"] * 5)
        self.assertTrue(all(0 <= row["discarded_energy_fraction"] <= 1 for row in errors))
        self.assertEqual(model.head.weight.data_ptr(), model.token.weight.data_ptr())

    def test_pointer_and_statistics_are_copied_explicitly(self):
        arrays = count_arrays()
        pointer = RankLayoutCache(small_config(mkn_shapes={k: list(v.shape) for k, v in arrays.items()},
                                               mkn_order=2, mkn_weight=.2, discount_scale=.7,
                                               statistics_recipe={"source": "synthetic", "minimum": 2})).eval()
        pointer.ngram.load_state_dict(arrays)
        with torch.no_grad():
            pointer.pointer_key.weight.fill_(.25)
        parent = dict(config=pointer.config, model=pointer.state_dict())
        model, _ = initialize_from_dense(self.dense_checkpoint, parent,
            rank_layout=self.config["rank_layout"], include_statistics=True)
        for name, value in model.state_dict().items():
            if name.startswith(("pointer_", "ngram.")):
                torch.testing.assert_close(value, parent["model"][name], atol=0, rtol=0)
        self.assertEqual(model.config["mkn_weight"], .2)
        self.assertEqual(model.config["discount_scale"], .7)
        self.assertEqual(model.config["statistics_recipe"], {"source": "synthetic", "minimum": 2})
        self.assertIsNot(model.config["statistics_recipe"], parent["config"]["statistics_recipe"])

    def test_full_rank_initialization_preserves_dense_features(self):
        layout = dict(qkv=32, proj=0, gate=32, value=32, out=32)
        model, errors = initialize_from_dense(self.dense_checkpoint, rank_layout=layout)
        model.eval()
        ids = torch.tensor([[1, 2, 3, 1, 2, 4]])
        with torch.no_grad():
            torch.testing.assert_close(model.features(ids), self.dense.features(ids),
                                       atol=3e-6, rtol=2e-5)
        self.assertTrue(all(row["discarded_energy_fraction"] == 0 for row in errors))

    def test_factorized_parent_is_rejected(self):
        lowrank = LearnedCacheGPT(self.config)
        with self.assertRaisesRegex(ValueError, "Dense source required"):
            initialize_from_dense(dict(config=self.config, model=lowrank.state_dict()),
                                  rank_layout=self.config["rank_layout"])

    def test_parent_costs_are_not_silently_dropped(self):
        self.assertEqual(lineage_targets(98304000, 212992000), 311296000)
        self.assertEqual(lineage_targets(98304000, 212992000, 98304000), 212992000)
        with self.assertRaises(ValueError):
            lineage_targets(10, 20, 11)

    def test_checkpoint_ancestry_alias_and_evidence_report(self):
        # Exercise the complete CLI only with tiny synthetic tensors and no data.
        with tempfile.TemporaryDirectory(prefix="push147-rank-test-") as directory:
            directory = Path(directory)
            source = directory / "dense.pt"
            output = directory / "initialized"
            torch.save(dict(self.dense_checkpoint, protocol=PROTOCOL, train_tokens=123), source)
            arguments = ["push147_rank_init.py", "--dense-parent", str(source), "--run-dir", str(output),
                         "--rank-layout", json.dumps(self.config["rank_layout"])]
            with patch("sys.argv", arguments), patch("builtins.print"):
                main()
            checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
            self.assertEqual(checkpoint["ancestry"], checkpoint["initialization_ancestry"])
            self.assertIs(checkpoint["ancestry"], checkpoint["initialization_ancestry"])
            info = json.loads((output / "initialization.json").read_text())
            evidence = json.loads((output / info["shared_ancestor_evidence"]["path"]).read_text())
            self.assertEqual(evidence["status"], "no_overlap_deducted")
            self.assertEqual(checkpoint["train_tokens"], 123)

    def test_shared_evidence_unknown_amount_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "known"):
            verified_shared_evidence("unrecognized", "unrecognized", 10)


if __name__ == "__main__":
    unittest.main()
