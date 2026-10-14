from __future__ import annotations

import unittest
from functools import partial

import numpy as np

from quantumpinn.analytical import harmonic_oscillator_energy, infinite_well_energy
from quantumpinn.finite_difference import solve_bound_states
from quantumpinn.metrics import normalisation_error, sign_invariant_l2
from quantumpinn.potentials import harmonic_oscillator, infinite_well


class FiniteDifferenceTests(unittest.TestCase):
    def test_infinite_well_first_three_energies(self) -> None:
        solution = solve_bound_states(
            partial(infinite_well, left=0.0, right=1.0),
            x_min=0.0,
            x_max=1.0,
            num_points=500,
            num_states=3,
        )
        references = np.array([infinite_well_energy(state) for state in range(3)])
        np.testing.assert_allclose(solution.energies, references, rtol=5e-5)

    def test_harmonic_oscillator_first_three_energies(self) -> None:
        solution = solve_bound_states(
            harmonic_oscillator,
            x_min=-8.0,
            x_max=8.0,
            num_points=1000,
            num_states=3,
        )
        references = np.array([harmonic_oscillator_energy(state) for state in range(3)])
        np.testing.assert_allclose(solution.energies, references, rtol=2e-4)

    def test_wavefunctions_are_normalised_and_obey_boundaries(self) -> None:
        solution = solve_bound_states(
            harmonic_oscillator,
            x_min=-7.0,
            x_max=7.0,
            num_points=600,
            num_states=3,
        )
        for wavefunction in solution.wavefunctions:
            self.assertLess(normalisation_error(wavefunction, solution.x), 1e-12)
            self.assertEqual(wavefunction[0], 0.0)
            self.assertEqual(wavefunction[-1], 0.0)

    def test_l2_metric_ignores_global_sign(self) -> None:
        x = np.linspace(0.0, 1.0, 20)
        psi = np.sqrt(2.0) * np.sin(np.pi * x)
        self.assertLess(sign_invariant_l2(-psi, psi, x), 1e-14)

    def test_rejects_non_finite_interior_potential(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            solve_bound_states(
                lambda x: np.full_like(x, np.inf),
                x_min=0.0,
                x_max=1.0,
                num_points=20,
            )


if __name__ == "__main__":
    unittest.main()
