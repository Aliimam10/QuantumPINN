from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # The numerical baseline does not require the ML extra.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PINNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from quantumpinn.losses import (
            DeflationState,
            LossWeights,
            normalise_on_grid,
            pinn_loss,
            project_lower_states,
        )
        from quantumpinn.pinn import BoundStatePINN
        from quantumpinn.train import TrainingConfig, train_bound_state

        cls.BoundStatePINN = BoundStatePINN
        cls.DeflationState = DeflationState
        cls.LossWeights = LossWeights
        cls.normalise_on_grid = staticmethod(normalise_on_grid)
        cls.pinn_loss = staticmethod(pinn_loss)
        cls.project_lower_states = staticmethod(project_lower_states)
        cls.TrainingConfig = TrainingConfig
        cls.train_bound_state = staticmethod(train_bound_state)

    def make_grid(self):
        return torch.linspace(0.0, 1.0, 64, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)

    def test_trial_function_obeys_boundaries_exactly(self) -> None:
        model = self.BoundStatePINN(x_min=0.0, x_max=1.0, initial_energy=4.0)
        values = model(self.make_grid()).detach()
        self.assertEqual(float(values[0]), 0.0)
        self.assertEqual(float(values[-1]), 0.0)

    def test_even_and_odd_parity(self) -> None:
        x = self.make_grid()
        for parity, sign in (("even", 1.0), ("odd", -1.0)):
            model = self.BoundStatePINN(
                x_min=0.0,
                x_max=1.0,
                initial_energy=4.0,
                parity=parity,
            )
            values = model(x)
            torch.testing.assert_close(
                values,
                sign * torch.flip(values, dims=(0,)),
                atol=1e-12,
                rtol=1e-12,
            )

    def test_physics_loss_is_finite_and_differentiable(self) -> None:
        model = self.BoundStatePINN(x_min=0.0, x_max=1.0, initial_energy=4.0)
        loss = self.pinn_loss(model, self.make_grid(), lambda x: torch.zeros_like(x))
        self.assertTrue(bool(torch.isfinite(loss.total)))
        loss.total.backward()
        self.assertIsNotNone(model.energy.grad)
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_grid_normalisation(self) -> None:
        x = self.make_grid()
        wavefunction = x * (1.0 - x)
        normalised = self.normalise_on_grid(wavefunction, x)
        norm = torch.trapezoid(normalised[:, 0].square(), x[:, 0])
        torch.testing.assert_close(norm, torch.tensor(1.0, dtype=x.dtype))

    def test_orthogonality_penalty_detects_overlap(self) -> None:
        model = self.BoundStatePINN(x_min=0.0, x_max=1.0, initial_energy=4.0)
        x = self.make_grid()
        reference = model(x).detach()
        weights = self.LossWeights(residual=0.0, boundary=0.0, normalisation=0.0, orthogonality=1.0)
        loss = self.pinn_loss(
            model,
            x,
            lambda x: torch.zeros_like(x),
            weights=weights,
            reference_states=(reference,),
        )
        self.assertGreater(float(loss.orthogonality.detach()), 0.0)

    def test_deflation_projects_out_a_lower_state(self) -> None:
        x = self.make_grid()
        reference = torch.sin(torch.pi * x)
        reference_second = -(torch.pi**2) * reference
        candidate = reference + 0.25 * torch.sin(2.0 * torch.pi * x)
        candidate_second = reference_second - torch.pi**2 * torch.sin(2.0 * torch.pi * x)
        projected, _, _ = self.project_lower_states(
            candidate,
            candidate_second,
            x,
            (
                self.DeflationState(
                    wavefunction=reference,
                    second_derivative=reference_second,
                ),
            ),
        )
        overlap = torch.trapezoid(projected[:, 0] * reference[:, 0], x[:, 0])
        self.assertLess(abs(float(overlap.detach())), 1e-12)

    def test_short_training_recovers_square_well_ground_energy(self) -> None:
        torch.manual_seed(0)
        model = self.BoundStatePINN(
            x_min=0.0,
            x_max=1.0,
            initial_energy=3.0,
            hidden_layers=(16, 16),
            parity="even",
        )
        result = self.train_bound_state(
            model,
            lambda x: torch.zeros_like(x),
            config=self.TrainingConfig(
                num_collocation=64,
                adam_epochs=200,
                lbfgs_steps=50,
                seed=0,
                history_every=50,
            ),
        )
        self.assertAlmostEqual(result.energy, torch.pi**2 / 2.0, delta=5e-3)
        self.assertLess(result.final_loss["total"], 1e-3)


if __name__ == "__main__":
    unittest.main()
