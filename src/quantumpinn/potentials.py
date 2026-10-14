"""Reusable one-dimensional potentials in dimensionless units."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

Potential = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def infinite_well(x: ArrayLike, *, left: float = 0.0, right: float = 1.0) -> NDArray[np.float64]:
    """Zero potential inside a well.

    Dirichlet boundary conditions provide the infinite walls; points outside the
    requested interval are assigned infinity for explicit validation/visualisation.
    """

    values = np.asarray(x, dtype=float)
    return np.where((values >= left) & (values <= right), 0.0, np.inf)


def harmonic_oscillator(
    x: ArrayLike, *, mass: float = 1.0, omega: float = 1.0, center: float = 0.0
) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=float)
    return 0.5 * mass * omega**2 * (values - center) ** 2


def quartic_oscillator(
    x: ArrayLike, *, quadratic: float = 0.5, quartic: float = 0.1
) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=float)
    return quadratic * values**2 + quartic * values**4


def finite_square_well(
    x: ArrayLike, *, half_width: float = 1.0, outside_height: float = 10.0
) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=float)
    return np.where(np.abs(values) <= half_width, 0.0, outside_height)


def square_barrier(
    x: ArrayLike, *, half_width: float = 0.5, height: float = 10.0
) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=float)
    return np.where(np.abs(values) <= half_width, height, 0.0)


def double_well(
    x: ArrayLike, *, scale: float = 1.0, separation: float = 1.5
) -> NDArray[np.float64]:
    """Smooth symmetric double well: ``scale * (x**2 - separation**2)**2``."""

    values = np.asarray(x, dtype=float)
    return scale * (values**2 - separation**2) ** 2
