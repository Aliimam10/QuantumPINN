"""Finite-difference reference solver for 1D bound-state eigenproblems."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh_tridiagonal


@dataclass(frozen=True)
class EigenSolution:
    """Low-lying eigenpairs sampled on a grid including Dirichlet endpoints."""

    x: NDArray[np.float64]
    energies: NDArray[np.float64]
    wavefunctions: NDArray[np.float64]
    potential: NDArray[np.float64]


def _normalise(psi: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = np.sqrt(np.trapezoid(np.abs(psi) ** 2, x=x))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("Cannot normalise a zero or non-finite wavefunction")
    return psi / norm


def solve_bound_states(
    potential_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    x_min: float,
    x_max: float,
    num_points: int = 1000,
    num_states: int = 3,
    hbar: float = 1.0,
    mass: float = 1.0,
) -> EigenSolution:
    """Solve the stationary Schrödinger equation with Dirichlet boundaries.

    The Hamiltonian uses a centred, second-order finite difference on the interior
    points. Returned wavefunctions include exact zeros at both domain boundaries and
    are normalised by trapezoidal quadrature.
    """

    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")
    if num_points < 4:
        raise ValueError("num_points must be at least 4")
    if not 1 <= num_states <= num_points - 2:
        raise ValueError("num_states must be between 1 and num_points - 2")
    if hbar <= 0.0 or mass <= 0.0:
        raise ValueError("hbar and mass must be positive")

    x = np.linspace(x_min, x_max, num_points, dtype=float)
    interior = x[1:-1]
    dx = x[1] - x[0]
    potential = np.asarray(potential_fn(x), dtype=float)
    if potential.shape != x.shape:
        raise ValueError("potential_fn must return one value per input coordinate")
    if not np.all(np.isfinite(potential[1:-1])):
        raise ValueError("potential must be finite at every interior grid point")

    kinetic_scale = hbar**2 / (2.0 * mass * dx**2)
    diagonal = 2.0 * kinetic_scale + potential[1:-1]
    off_diagonal = np.full(interior.size - 1, -kinetic_scale)

    energies, interior_states = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(0, num_states - 1),
        check_finite=True,
    )

    wavefunctions = np.zeros((num_states, num_points), dtype=float)
    for state in range(num_states):
        wavefunctions[state, 1:-1] = interior_states[:, state]
        wavefunctions[state] = _normalise(wavefunctions[state], x)

    return EigenSolution(
        x=x,
        energies=np.asarray(energies),
        wavefunctions=wavefunctions,
        potential=potential,
    )
