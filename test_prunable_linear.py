"""
Sanity tests for PrunableLinear, run BEFORE any CIFAR-10 training.

These are deliberately plain assert-based tests (stdlib unittest, no extra
dependency) covering exactly the checks the design review called for:
shapes, gate range, forward correctness against a hand-computed example,
gradient flow to both `weight` and `gate_scores`, and gate behavior at the
extremes of gate_scores.

Run with:
    python test_prunable_linear.py
"""

import unittest

import torch

from self_pruning_nn import PrunableLinear, SelfPruningMLP, sparsity_loss


class TestShapes(unittest.TestCase):
    def test_parameter_and_output_shapes(self):
        layer = PrunableLinear(in_features=5, out_features=3)
        self.assertEqual(layer.weight.shape, (3, 5))
        self.assertEqual(layer.gate_scores.shape, layer.weight.shape)
        self.assertEqual(layer.bias.shape, (3,))

        x = torch.randn(4, 5)  # batch of 4
        out = layer(x)
        self.assertEqual(out.shape, (4, 3))

        gates = torch.sigmoid(layer.gate_scores)
        self.assertEqual(gates.shape, layer.weight.shape)


class TestGateRange(unittest.TestCase):
    def test_gates_strictly_between_zero_and_one(self):
        layer = PrunableLinear(in_features=8, out_features=6)
        # Spread of finite values, including large-magnitude ones, to probe
        # near-saturation behavior. NOTE: we deliberately stop at +-16, not
        # +-18 or beyond -- float32 sigmoid saturates to an EXACT 1.0 once
        # the score exceeds ~17-18 (mirroring the exact-0.0 underflow
        # discussed for very negative scores in report.md). That is expected
        # floating-point behavior, not a bug, but it means "strictly < 1"
        # only holds within this moderate range.
        with torch.no_grad():
            layer.gate_scores.copy_(
                torch.tensor([[-16.0, -5.0, -1.0, 0.0, 1.0, 5.0, 16.0, 16.0]] * 6)
            )
        gates = torch.sigmoid(layer.gate_scores)
        self.assertTrue(torch.all(gates > 0.0))
        self.assertTrue(torch.all(gates < 1.0))

    def test_extreme_scores_can_saturate_to_exact_float32_boundaries(self):
        # This documents (not "fixes") the floating-point reality discussed
        # in the design review: sigmoid never mathematically reaches 0 or 1,
        # but float32 rounding makes it reach those values EXACTLY once the
        # score is far enough out. This is why "pruned" is defined via a
        # threshold (gate < 1e-2), not via a literal-zero check.
        very_negative = torch.sigmoid(torch.tensor(-100.0))
        very_positive = torch.sigmoid(torch.tensor(100.0))
        self.assertEqual(very_negative.item(), 0.0)
        self.assertEqual(very_positive.item(), 1.0)


class TestForwardCorrectness(unittest.TestCase):
    def test_matches_hand_computed_example(self):
        layer = PrunableLinear(in_features=2, out_features=2)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            layer.bias.copy_(torch.tensor([0.5, -0.5]))
            layer.gate_scores.copy_(torch.tensor([[0.0, 0.0], [0.0, 0.0]]))  # sigmoid(0) = 0.5

        x = torch.tensor([[1.0, 1.0]])
        out = layer(x)

        # Expected: effective_weight = weight * 0.5 = [[0.5, 1.0], [1.5, 2.0]]
        # out = x @ effective_weight.T + bias
        #     = [1*0.5 + 1*1.0, 1*1.5 + 1*2.0] + [0.5, -0.5]
        #     = [1.5, 3.5] + [0.5, -0.5] = [2.0, 3.0]
        expected = torch.tensor([[2.0, 3.0]])
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_matches_manual_recomputation_with_random_values(self):
        torch.manual_seed(0)
        layer = PrunableLinear(in_features=6, out_features=4)
        x = torch.randn(3, 6)
        out = layer(x)

        gates = torch.sigmoid(layer.gate_scores)
        eff_weight = layer.weight * gates
        expected = x @ eff_weight.T + layer.bias
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))


class TestGradientFlow(unittest.TestCase):
    def test_gradients_exist_and_are_finite(self):
        layer = PrunableLinear(in_features=4, out_features=3)
        x = torch.randn(5, 4)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(layer.weight.grad)
        self.assertIsNotNone(layer.gate_scores.grad)
        self.assertIsNotNone(layer.bias.grad)

        self.assertTrue(torch.isfinite(layer.weight.grad).all())
        self.assertTrue(torch.isfinite(layer.gate_scores.grad).all())
        self.assertTrue(torch.isfinite(layer.bias.grad).all())

        # Gradients should not be trivially all-zero for a generic input/loss.
        self.assertGreater(layer.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(layer.gate_scores.grad.abs().sum().item(), 0.0)

    def test_full_model_and_sparsity_loss_gradient_flow(self):
        model = SelfPruningMLP(input_dim=12, hidden_dims=(8, 6), num_classes=3)
        x = torch.randn(5, 12)
        logits = model(x)
        labels = torch.randint(0, 3, (5,))
        ce = torch.nn.functional.cross_entropy(logits, labels)
        sparse = sparsity_loss(model)
        total = ce + 1e-3 * sparse
        total.backward()

        for layer in model.prunable_layers():
            self.assertIsNotNone(layer.weight.grad)
            self.assertIsNotNone(layer.gate_scores.grad)
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertTrue(torch.isfinite(layer.gate_scores.grad).all())


class TestGateBehaviorAtExtremes(unittest.TestCase):
    def test_strongly_positive_scores_open_the_gate(self):
        layer = PrunableLinear(in_features=3, out_features=3)
        with torch.no_grad():
            layer.gate_scores.fill_(10.0)
            layer.weight.fill_(2.0)
        gates = torch.sigmoid(layer.gate_scores)
        eff_weight = layer.weight * gates
        self.assertTrue(torch.all(gates > 0.999))
        self.assertTrue(torch.allclose(eff_weight, layer.weight, atol=1e-3))

    def test_strongly_negative_scores_close_the_gate(self):
        layer = PrunableLinear(in_features=3, out_features=3)
        with torch.no_grad():
            layer.gate_scores.fill_(-10.0)
            layer.weight.fill_(2.0)
        gates = torch.sigmoid(layer.gate_scores)
        eff_weight = layer.weight * gates
        self.assertTrue(torch.all(gates < 0.001))
        self.assertTrue(torch.all(eff_weight.abs() < 0.01))
        # But never mathematically exact zero for finite scores.
        self.assertTrue(torch.all(gates > 0.0))

    def test_gate_gradient_vanishes_at_extreme_saturation(self):
        # d(gate)/d(score) = gate * (1 - gate) -> ~0 as score -> +-infinity.
        layer = PrunableLinear(in_features=3, out_features=3)
        with torch.no_grad():
            layer.gate_scores.fill_(-50.0)
        x = torch.randn(2, 3)
        out = layer(x)
        out.sum().backward()
        # Gradient w.r.t. a deeply-saturated gate_score should be extremely small.
        self.assertLess(layer.gate_scores.grad.abs().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
