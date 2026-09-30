import copy
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from push148_muon import BatchedMuon, partition_named_parameters, zeropower_via_newtonschulz5


def reference_ns(matrix, steps=5):
    """Independent, unbatched CPU transcription of the cited iteration."""
    result = matrix.float().clone()
    was_tall = result.shape[0] > result.shape[1]
    if was_tall:
        result = result.T
    result = result / (torch.linalg.vector_norm(result) + 1e-7)
    for _ in range(steps):
        square = result @ result.T
        polynomial = -4.7750 * square + (2.0315 * square) @ square
        result = 3.4445 * result + polynomial @ result
    return result.T if was_tall else result


class MuonTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(148)
        torch.set_num_threads(2)

    def test_cpu_matrices_finite_fp32_and_input_unchanged(self):
        for shape in ((4, 9), (9, 4), (5, 5), (1, 7), (7, 1)):
            matrix = torch.randn(shape, dtype=torch.float64)
            original = matrix.clone()
            actual = zeropower_via_newtonschulz5(matrix)
            self.assertEqual(actual.shape, matrix.shape)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(matrix, original, rtol=0, atol=0)
            torch.testing.assert_close(actual, reference_ns(matrix), atol=2e-6, rtol=2e-5)

    def test_batched_ns_matches_per_matrix_and_transpose(self):
        for shape in ((3, 4, 9), (2, 9, 4), (4, 5, 5)):
            matrix = torch.randn(shape)
            actual = zeropower_via_newtonschulz5(matrix)
            expected = torch.stack([reference_ns(item) for item in matrix])
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(actual.mT, zeropower_via_newtonschulz5(matrix.mT),
                                       atol=3e-6, rtol=3e-5)

    def test_zero_gradient_and_none_gradient(self):
        active = nn.Parameter(torch.randn(3, 5))
        skipped = nn.Parameter(torch.randn(3, 5))
        before_active, before_skipped = active.detach().clone(), skipped.detach().clone()
        optimizer = BatchedMuon([active, skipped], lr=.1)
        active.grad = torch.zeros_like(active)
        optimizer.step()
        torch.testing.assert_close(active, before_active, atol=0, rtol=0)
        torch.testing.assert_close(skipped, before_skipped, atol=0, rtol=0)
        self.assertNotIn(skipped, optimizer.state)
        self.assertEqual(optimizer.state[active]["step"], 1)
        self.assertTrue(torch.isfinite(optimizer.state[active]["momentum_buffer"]).all())

    def test_decoupled_decay_and_skip_none(self):
        active, skipped = nn.Parameter(torch.ones(2, 3)), nn.Parameter(torch.ones(2, 3))
        active.grad = torch.zeros_like(active)
        optimizer = BatchedMuon([active, skipped], lr=.1, weight_decay=.2)
        optimizer.step()
        torch.testing.assert_close(active, torch.full_like(active, .98), atol=1e-7, rtol=0)
        torch.testing.assert_close(skipped, torch.ones_like(skipped), atol=0, rtol=0)

    def test_rectangular_update_scaling(self):
        for shape in ((8, 2), (2, 8), (4, 4)):
            parameter = nn.Parameter(torch.zeros(shape))
            gradient = torch.randn(shape)
            parameter.grad = gradient.clone()
            optimizer = BatchedMuon([parameter], lr=.25, momentum=0.)
            optimizer.step()
            expected = -.25 * reference_ns(gradient) * math.sqrt(max(1, shape[0] / shape[1]))
            torch.testing.assert_close(parameter, expected, atol=2e-6, rtol=2e-5)

    def test_momentum_nesterov_matches_reference_without_changing_gradients(self):
        for nesterov in (True, False):
            parameter = nn.Parameter(torch.randn(3, 7))
            expected = parameter.detach().clone()
            buffer = torch.zeros_like(parameter)
            optimizer = BatchedMuon([parameter], lr=.003, weight_decay=.1,
                                    momentum=.95, nesterov=nesterov)
            for _ in range(3):
                gradient = torch.randn_like(parameter)
                parameter.grad = gradient.clone()
                buffer.lerp_(gradient, .05)
                direction = gradient.lerp(buffer, .95) if nesterov else buffer
                expected.mul_(1 - .003 * .1).add_(reference_ns(direction), alpha=-.003)
                optimizer.step()
                torch.testing.assert_close(parameter.grad, gradient, atol=0, rtol=0)
                torch.testing.assert_close(parameter, expected, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], buffer,
                                           atol=1e-7, rtol=1e-6)

    def test_equal_shapes_share_one_ns_call(self):
        parameters = [nn.Parameter(torch.randn(shape)) for shape in ((3, 7), (3, 7), (7, 3), (3, 7))]
        optimizer = BatchedMuon(parameters)
        for parameter in parameters:
            parameter.grad = torch.randn_like(parameter)
        with patch("push148_muon.zeropower_via_newtonschulz5", wraps=zeropower_via_newtonschulz5) as spy:
            optimizer.step()
        self.assertEqual(spy.call_count, 2)
        self.assertEqual({tuple(call.args[0].shape) for call in spy.call_args_list}, {(3, 3, 7), (1, 7, 3)})

    def test_state_round_trip_and_group_schedule_metadata(self):
        parameters = [nn.Parameter(torch.randn(3, 7)) for _ in range(2)]
        optimizer = BatchedMuon([dict(params=parameters, lr=.004, base_lr=.008)], weight_decay=.02)
        for parameter in parameters:
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        restored_parameters = [nn.Parameter(parameter.detach().clone()) for parameter in parameters]
        restored = BatchedMuon(restored_parameters, lr=.9)
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        self.assertEqual(restored.param_groups[0]["lr"], .004)
        self.assertEqual(restored.param_groups[0]["base_lr"], .008)
        for parameter, other in zip(parameters, restored_parameters):
            gradient = torch.randn_like(parameter)
            parameter.grad, other.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        restored.step()
        for parameter, other in zip(parameters, restored_parameters):
            torch.testing.assert_close(parameter, other, atol=0, rtol=0)
            self.assertEqual(restored.state[other]["step"], 2)
            torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"],
                                       restored.state[other]["momentum_buffer"], atol=0, rtol=0)

    def test_default_base_lr_follows_each_group(self):
        p, q = nn.Parameter(torch.ones(2, 3)), nn.Parameter(torch.ones(3, 2))
        optimizer = BatchedMuon([dict(params=[p], lr=.003), dict(params=[q], lr=.007)], lr=.002)
        self.assertEqual([group["base_lr"] for group in optimizer.param_groups], [.003, .007])

    def test_partition_excludes_embedding_tied_head_norm_pointer_scalars(self):
        model = nn.Module()
        model.token = nn.Embedding(11, 4)
        model.blocks = nn.ModuleList([nn.Sequential(nn.Linear(4, 7, bias=False), nn.LayerNorm(7))])
        model.head = nn.Linear(4, 11, bias=False)
        model.head.weight = model.token.weight
        model.pointer_query = nn.Linear(4, 3, bias=False)
        model.scalar = nn.Parameter(torch.tensor(0.))
        hidden, auxiliary = partition_named_parameters(model)
        self.assertEqual([name for name, _ in hidden], ["blocks.0.0.weight"])
        self.assertEqual(len({id(parameter) for _, parameter in hidden + auxiliary}),
                         len(list(model.parameters())))
        self.assertIn("token.weight", [name for name, _ in auxiliary])
        self.assertIn("pointer_query.weight", [name for name, _ in auxiliary])
        self.assertIn("scalar", [name for name, _ in auxiliary])
        self.assertIn("blocks.0.1.weight", [name for name, _ in auxiliary])

    def test_validation_rejects_invalid_parameters_and_settings(self):
        with self.assertRaises(ValueError):
            BatchedMuon([nn.Parameter(torch.ones(3))])
        for settings in (dict(lr=-1), dict(momentum=1.), dict(ns_steps=0), dict(weight_decay=-.1)):
            with self.assertRaises(ValueError):
                BatchedMuon([nn.Parameter(torch.ones(2, 3))], **settings)
        with self.assertRaises(ValueError):
            zeropower_via_newtonschulz5(torch.ones(3))


if __name__ == "__main__":
    unittest.main()
