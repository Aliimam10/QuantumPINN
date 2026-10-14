from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from quantumpinn.evaluate import (
    aggregate_records,
    align_global_sign,
    evaluate_state,
    finite_difference_residual_rms,
    write_records_csv,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.linspace(0.0, 1.0, 500)
        self.psi = np.sqrt(2.0) * np.sin(np.pi * self.x)
        self.energy = np.pi**2 / 2.0

    def test_global_sign_alignment(self) -> None:
        aligned = align_global_sign(-self.psi, self.psi, self.x)
        np.testing.assert_allclose(aligned, self.psi)

    def test_discrete_residual_converges_for_square_well(self) -> None:
        residual = finite_difference_residual_rms(
            self.x,
            self.psi,
            self.energy,
            np.zeros_like(self.x),
        )
        self.assertLess(residual, 1e-4)

    def test_evaluation_and_aggregation(self) -> None:
        records = [
            evaluate_state(
                system="infinite_well",
                state=0,
                method="test",
                seed=seed,
                x=self.x,
                energy=self.energy * (1.0 + seed * 1e-4),
                wavefunction=(-1.0) ** seed * self.psi,
                reference_energy=self.energy,
                reference_wavefunction=self.psi,
                potential=np.zeros_like(self.x),
                runtime_seconds=0.1,
            )
            for seed in (0, 1)
        ]
        summary = aggregate_records(records)
        self.assertEqual(summary[0]["runs"], 2)
        self.assertAlmostEqual(summary[0]["relative_energy_error_mean"], 5e-5)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metrics.csv"
            write_records_csv(records, output)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
