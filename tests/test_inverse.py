from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class InverseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from quantumpinn.inverse import (
            EvenPolynomialPotential,
            WavefunctionObservation,
            inverse_loss,
            relative_potential_l2,
        )
        from quantumpinn.pinn import BoundStatePINN

        cls.BoundStatePINN = BoundStatePINN
        cls.EvenPolynomialPotential = EvenPolynomialPotential
        cls.WavefunctionObservation = WavefunctionObservation
        cls.inverse_loss = staticmethod(inverse_loss)
        cls.relative_potential_l2 = staticmethod(relative_potential_l2)

    def test_polynomial_potential_and_gradients(self) -> None:
        potential = self.EvenPolynomialPotential(quadratic=0.5, quartic=0.1)
        x = torch.tensor([[-2.0], [0.0], [2.0]], dtype=torch.float64)
        values = potential(x)
        torch.testing.assert_close(
            values[:, 0],
            torch.tensor([3.6, 0.0, 3.6], dtype=torch.float64),
        )
        values.sum().backward()
        self.assertIsNotNone(potential.coefficients.grad)

    def test_inverse_loss_is_finite_and_differentiable(self) -> None:
        torch.manual_seed(0)
        model = self.BoundStatePINN(
            x_min=-5.0,
            x_max=5.0,
            initial_energy=0.6,
            hidden_layers=(16, 16),
            parity="even",
        )
        potential = self.EvenPolynomialPotential()
        x = torch.linspace(-5.0, 5.0, 64, dtype=torch.float64).reshape(-1, 1)
        x.requires_grad_(True)
        observed_x = torch.linspace(-2.0, 2.0, 9, dtype=torch.float64).reshape(-1, 1)
        observed = torch.exp(-0.5 * observed_x.square()) / torch.pi**0.25
        observation = self.WavefunctionObservation(
            state=0,
            x=observed_x,
            values=observed,
            energy=0.5,
        )
        loss, _ = self.inverse_loss((model,), potential, x, (observation,))
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertIsNotNone(potential.coefficients.grad)

    def test_relative_potential_error(self) -> None:
        x = torch.linspace(-2.0, 2.0, 20, dtype=torch.float64).reshape(-1, 1)
        reference = 0.5 * x.square()
        self.assertAlmostEqual(self.relative_potential_l2(reference, reference, x), 0.0)


if __name__ == "__main__":
    unittest.main()
