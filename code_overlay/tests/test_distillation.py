import unittest
import torch
from torch.nn import functional as F
from train_validation_extension import distillation_loss


class DistillationTests(unittest.TestCase):
    def test_zero_alpha_is_exact_hard_target_loss(self):
        logits = torch.randn(2, 5, 11, requires_grad=True)
        targets = torch.randint(11, (2, 5))
        actual = distillation_loss(logits, None, targets, 0., 1.)
        expected = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_teacher_does_not_receive_gradients(self):
        student = torch.randn(2, 5, 11, requires_grad=True)
        teacher = torch.randn(2, 5, 11, requires_grad=True)
        loss = distillation_loss(student, teacher, torch.randint(11, (2, 5)), .5, 2.)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(student.grad.abs().sum().item(), 0)

    def test_matching_teacher_is_stationary_for_soft_loss(self):
        teacher = torch.randn(2, 5, 11)
        student = teacher.clone().requires_grad_(True)
        loss = distillation_loss(student, teacher, torch.zeros(2, 5, dtype=torch.long), 1., 2.)
        loss.backward()
        self.assertLess(student.grad.abs().max().item(), 1e-6)


if __name__ == '__main__':
    unittest.main()
