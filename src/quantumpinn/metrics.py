"""Evaluation metrics that respect eigenfunction conventions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def relative_energy_error(predicted: float, reference: float) -> float:
    if reference == 0.0:
        return float(abs(predicted - reference))
    return float(abs(predicted - reference) / abs(reference))


def sign_invariant_l2(predicted: ArrayLike, reference: ArrayLike, x: ArrayLike) -> float:
    """Continuous relative L2 error minimised over the eigenfunction's global sign."""

    predicted_values = np.asarray(predicted, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    coordinates = np.asarray(x, dtype=float)
    if predicted_values.shape != reference_values.shape or predicted_values.shape != coordinates.shape:
        raise ValueError("predicted, reference, and x must have the same shape")

    reference_norm = np.sqrt(np.trapezoid(reference_values**2, x=coordinates))
    if reference_norm == 0.0:
        raise ValueError("reference wavefunction must have non-zero norm")
    errors = [
        np.sqrt(np.trapezoid((predicted_values - sign * reference_values) ** 2, x=coordinates))
        for sign in (1.0, -1.0)
    ]
    return float(min(errors) / reference_norm)


def normalisation_error(wavefunction: ArrayLike, x: ArrayLike) -> float:
    values = np.asarray(wavefunction, dtype=float)
    coordinates = np.asarray(x, dtype=float)
    return float(abs(np.trapezoid(np.abs(values) ** 2, x=coordinates) - 1.0))
