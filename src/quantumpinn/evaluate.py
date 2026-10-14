"""Evaluation records and numerical diagnostics for quantum eigenstates."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .metrics import normalisation_error, relative_energy_error, sign_invariant_l2


@dataclass(frozen=True)
class EvaluationRecord:
    system: str
    state: int
    method: str
    seed: int | None
    energy: float
    reference_energy: float
    relative_energy_error: float
    wavefunction_l2_error: float
    boundary_error: float
    normalisation_error: float
    pde_residual_rms: float
    runtime_seconds: float


@dataclass(frozen=True)
class StatePrediction:
    system: str
    state: int
    method: str
    seed: int | None
    x: NDArray[np.float64]
    energy: float
    wavefunction: NDArray[np.float64]


def align_global_sign(predicted: ArrayLike, reference: ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    """Return a copy of ``predicted`` with its arbitrary global sign aligned."""

    predicted_values = np.asarray(predicted, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    coordinates = np.asarray(x, dtype=float)
    if predicted_values.shape != reference_values.shape or predicted_values.shape != coordinates.shape:
        raise ValueError("predicted, reference, and x must have the same shape")
    overlap = np.trapezoid(predicted_values * reference_values, x=coordinates)
    return predicted_values.copy() if overlap >= 0.0 else -predicted_values


def finite_difference_residual_rms(
    x: ArrayLike,
    wavefunction: ArrayLike,
    energy: float,
    potential: ArrayLike,
    *,
    hbar: float = 1.0,
    mass: float = 1.0,
) -> float:
    """Estimate the interior Schrödinger residual using a centred stencil."""

    coordinates = np.asarray(x, dtype=float)
    psi = np.asarray(wavefunction, dtype=float)
    potential_values = np.asarray(potential, dtype=float)
    if coordinates.ndim != 1 or coordinates.size < 3:
        raise ValueError("x must be a one-dimensional grid with at least three points")
    if psi.shape != coordinates.shape or potential_values.shape != coordinates.shape:
        raise ValueError("wavefunction and potential must have the same shape as x")
    spacing = np.diff(coordinates)
    if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=1e-12):
        raise ValueError("finite-difference residual requires a uniform grid")
    if hbar <= 0.0 or mass <= 0.0:
        raise ValueError("hbar and mass must be positive")

    second_derivative = (psi[:-2] - 2.0 * psi[1:-1] + psi[2:]) / spacing[0] ** 2
    residual = -(hbar**2 / (2.0 * mass)) * second_derivative
    residual = residual + potential_values[1:-1] * psi[1:-1] - energy * psi[1:-1]
    return float(np.sqrt(np.mean(residual**2)))


def evaluate_state(
    *,
    system: str,
    state: int,
    method: str,
    seed: int | None,
    x: ArrayLike,
    energy: float,
    wavefunction: ArrayLike,
    reference_energy: float,
    reference_wavefunction: ArrayLike,
    potential: ArrayLike,
    runtime_seconds: float,
    pde_residual_rms: float | None = None,
) -> EvaluationRecord:
    coordinates = np.asarray(x, dtype=float)
    psi = np.asarray(wavefunction, dtype=float)
    reference = np.asarray(reference_wavefunction, dtype=float)
    if pde_residual_rms is None:
        pde_residual_rms = finite_difference_residual_rms(
            coordinates,
            psi,
            energy,
            potential,
        )
    return EvaluationRecord(
        system=system,
        state=state,
        method=method,
        seed=seed,
        energy=float(energy),
        reference_energy=float(reference_energy),
        relative_energy_error=relative_energy_error(energy, reference_energy),
        wavefunction_l2_error=sign_invariant_l2(psi, reference, coordinates),
        boundary_error=float(max(abs(psi[0]), abs(psi[-1]))),
        normalisation_error=normalisation_error(psi, coordinates),
        pde_residual_rms=float(pde_residual_rms),
        runtime_seconds=float(runtime_seconds),
    )


def write_records_csv(records: list[EvaluationRecord], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EvaluationRecord.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def aggregate_records(records: list[EvaluationRecord]) -> list[dict[str, Any]]:
    """Aggregate repeated seeds by system, state, and method."""

    groups: dict[tuple[str, int, str], list[EvaluationRecord]] = {}
    for record in records:
        groups.setdefault((record.system, record.state, record.method), []).append(record)

    metrics = (
        "energy",
        "relative_energy_error",
        "wavefunction_l2_error",
        "boundary_error",
        "normalisation_error",
        "pde_residual_rms",
        "runtime_seconds",
    )
    summary: list[dict[str, Any]] = []
    for (system, state, method), group in sorted(groups.items()):
        row: dict[str, Any] = {
            "system": system,
            "state": state,
            "method": method,
            "runs": len(group),
            "reference_energy": group[0].reference_energy,
        }
        for metric in metrics:
            values = np.asarray([getattr(record, metric) for record in group])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        summary.append(row)
    return summary


def write_summary_csv(summary: list[dict[str, Any]], destination: str | Path) -> None:
    if not summary:
        raise ValueError("summary must not be empty")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def write_metadata(metadata: dict[str, Any], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
