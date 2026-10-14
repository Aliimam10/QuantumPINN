"""Analytical reference solutions for selected one-dimensional systems."""

from __future__ import annotations

from math import factorial

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import eval_hermite


def infinite_well_energy(
    state: int, *, length: float = 1.0, hbar: float = 1.0, mass: float = 1.0
) -> float:
    """Energy for a zero-indexed state in a well on ``[0, length]``."""

    if state < 0 or length <= 0.0 or hbar <= 0.0 or mass <= 0.0:
        raise ValueError("state must be non-negative and physical parameters positive")
    n = state + 1
    return float(n**2 * np.pi**2 * hbar**2 / (2.0 * mass * length**2))


def infinite_well_wavefunction(
    x: ArrayLike, state: int, *, left: float = 0.0, right: float = 1.0
) -> NDArray[np.float64]:
    if state < 0 or right <= left:
        raise ValueError("state must be non-negative and right must exceed left")
    values = np.asarray(x, dtype=float)
    length = right - left
    n = state + 1
    inside = (values >= left) & (values <= right)
    psi = np.zeros_like(values)
    psi[inside] = np.sqrt(2.0 / length) * np.sin(n * np.pi * (values[inside] - left) / length)
    return psi


def harmonic_oscillator_energy(state: int, *, hbar: float = 1.0, omega: float = 1.0) -> float:
    if state < 0 or hbar <= 0.0 or omega <= 0.0:
        raise ValueError("state must be non-negative and physical parameters positive")
    return float(hbar * omega * (state + 0.5))


def harmonic_oscillator_wavefunction(
    x: ArrayLike,
    state: int,
    *,
    hbar: float = 1.0,
    mass: float = 1.0,
    omega: float = 1.0,
    center: float = 0.0,
) -> NDArray[np.float64]:
    if state < 0 or hbar <= 0.0 or mass <= 0.0 or omega <= 0.0:
        raise ValueError("state must be non-negative and physical parameters positive")
    values = np.asarray(x, dtype=float)
    alpha = mass * omega / hbar
    coordinate = np.sqrt(alpha) * (values - center)
    prefactor = (alpha / np.pi) ** 0.25 / np.sqrt(2.0**state * factorial(state))
    return prefactor * eval_hermite(state, coordinate) * np.exp(-0.5 * coordinate**2)
