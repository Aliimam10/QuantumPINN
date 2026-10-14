"""Physics-informed objectives for stationary Schrödinger eigenproblems."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .pinn import BoundStatePINN

TorchPotential = Callable[[Tensor], Tensor | float]


@dataclass(frozen=True)
class LossWeights:
    residual: float = 1.0
    boundary: float = 1.0
    normalisation: float = 10.0
    orthogonality: float = 10.0

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in vars(self).values()):
            raise ValueError("loss weights must be non-negative")


@dataclass(frozen=True)
class LossComponents:
    total: Tensor
    residual: Tensor
    boundary: Tensor
    normalisation: Tensor
    orthogonality: Tensor
    norm: Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "residual": float(self.residual.detach()),
            "boundary": float(self.boundary.detach()),
            "normalisation": float(self.normalisation.detach()),
            "orthogonality": float(self.orthogonality.detach()),
            "norm": float(self.norm.detach()),
        }


@dataclass(frozen=True)
class DeflationState:
    """A learned state and its second derivative on the collocation grid."""

    wavefunction: Tensor
    second_derivative: Tensor


def _coordinate_vector(x: Tensor) -> Tensor:
    if x.ndim != 2 or x.shape[1] != 1 or x.shape[0] < 3:
        raise ValueError("x must have shape (num_points, 1) with at least three points")
    coordinates = x[:, 0]
    if not bool(torch.all(coordinates[1:] > coordinates[:-1])):
        raise ValueError("collocation coordinates must be strictly increasing")
    return coordinates


def _potential_values(potential_fn: TorchPotential, x: Tensor) -> Tensor:
    values = potential_fn(x)
    if not isinstance(values, Tensor):
        values = torch.as_tensor(values, dtype=x.dtype, device=x.device)
    values = values.to(dtype=x.dtype, device=x.device)
    if values.ndim == 0:
        values = values.expand_as(x)
    if values.shape != x.shape:
        raise ValueError("potential_fn must return a scalar or a tensor with the shape of x")
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError("potential must be finite at all collocation points")
    return values


def normalise_on_grid(wavefunction: Tensor, x: Tensor) -> Tensor:
    coordinates = _coordinate_vector(x)
    if wavefunction.shape != x.shape:
        raise ValueError("wavefunction and x must have the same shape")
    norm = torch.trapezoid(wavefunction[:, 0].square(), coordinates)
    return wavefunction / torch.sqrt(torch.clamp_min(norm, torch.finfo(x.dtype).eps))


def wavefunction_and_second_derivative(
    model: BoundStatePINN, x: Tensor
) -> tuple[Tensor, Tensor]:
    wavefunction = model(x)
    first_derivative = torch.autograd.grad(
        wavefunction,
        x,
        grad_outputs=torch.ones_like(wavefunction),
        create_graph=True,
    )[0]
    second_derivative = torch.autograd.grad(
        first_derivative,
        x,
        grad_outputs=torch.ones_like(first_derivative),
        create_graph=True,
    )[0]
    return wavefunction, second_derivative


def project_lower_states(
    wavefunction: Tensor,
    second_derivative: Tensor,
    x: Tensor,
    deflation_states: Sequence[DeflationState],
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """Project known lower states from a wavefunction and its second derivative."""

    coordinates = _coordinate_vector(x)
    projected_wavefunction = wavefunction
    projected_second_derivative = second_derivative
    normalised_references: list[Tensor] = []
    for state in deflation_states:
        reference = state.wavefunction.to(dtype=x.dtype, device=x.device)
        reference_second = state.second_derivative.to(dtype=x.dtype, device=x.device)
        if reference.shape != x.shape or reference_second.shape != x.shape:
            raise ValueError("each deflation state must have the same shape as x")
        reference_norm = torch.trapezoid(reference[:, 0].square(), coordinates)
        reference_amplitude = torch.sqrt(
            torch.clamp_min(reference_norm, torch.finfo(x.dtype).eps)
        )
        reference = reference / reference_amplitude
        reference_second = reference_second / reference_amplitude
        overlap = torch.trapezoid(
            projected_wavefunction[:, 0] * reference[:, 0],
            coordinates,
        )
        projected_wavefunction = projected_wavefunction - overlap * reference
        projected_second_derivative = projected_second_derivative - overlap * reference_second
        normalised_references.append(reference)
    return projected_wavefunction, projected_second_derivative, normalised_references


def schrodinger_residual(
    model: BoundStatePINN,
    x: Tensor,
    potential_fn: TorchPotential,
    *,
    hbar: float = 1.0,
    mass: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Return the pointwise residual and the unnormalised model wavefunction."""

    if hbar <= 0.0 or mass <= 0.0:
        raise ValueError("hbar and mass must be positive")
    _coordinate_vector(x)
    if not x.requires_grad:
        raise ValueError("x must require gradients for automatic differentiation")

    wavefunction, second_derivative = wavefunction_and_second_derivative(model, x)
    potential = _potential_values(potential_fn, x)
    residual = -(hbar**2 / (2.0 * mass)) * second_derivative
    residual = residual + potential * wavefunction - model.energy * wavefunction
    return residual, wavefunction


def pinn_loss(
    model: BoundStatePINN,
    x: Tensor,
    potential_fn: TorchPotential,
    *,
    weights: LossWeights = LossWeights(),
    reference_states: Sequence[Tensor] = (),
    deflation_states: Sequence[DeflationState] = (),
    hbar: float = 1.0,
    mass: float = 1.0,
) -> LossComponents:
    """Evaluate the PDE, boundary, normalisation, and orthogonality losses.

    Previously learned ``deflation_states`` are projected out before evaluating the
    residual. Unlike a soft overlap penalty, this deflation cannot become stationary
    at a lower eigenstate. Tensor ``reference_states`` remain available as a soft
    constraint when derivative data for a reference state is not available.
    """

    coordinates = _coordinate_vector(x)
    if hbar <= 0.0 or mass <= 0.0:
        raise ValueError("hbar and mass must be positive")
    if not x.requires_grad:
        raise ValueError("x must require gradients for automatic differentiation")

    raw_wavefunction, raw_second_derivative = wavefunction_and_second_derivative(model, x)
    raw_wavefunction, raw_second_derivative, deflation_references = project_lower_states(
        raw_wavefunction,
        raw_second_derivative,
        x,
        deflation_states,
    )

    potential = _potential_values(potential_fn, x)
    raw_residual = -(hbar**2 / (2.0 * mass)) * raw_second_derivative
    raw_residual = (
        raw_residual
        + potential * raw_wavefunction
        - model.energy * raw_wavefunction
    )
    raw_norm = torch.trapezoid(raw_wavefunction[:, 0].square(), coordinates)
    amplitude = torch.sqrt(torch.clamp_min(raw_norm, torch.finfo(x.dtype).eps))
    wavefunction = raw_wavefunction / amplitude
    residual = raw_residual / amplitude
    residual_loss = residual.square().mean()
    boundary_loss = wavefunction[[0, -1]].square().mean()
    normalisation_loss = (raw_norm - 1.0).square()

    orthogonality_loss = torch.zeros((), dtype=x.dtype, device=x.device)
    for reference in deflation_references:
        overlap = torch.trapezoid(wavefunction[:, 0] * reference[:, 0], coordinates)
        orthogonality_loss = orthogonality_loss + overlap.square()
    for reference in reference_states:
        reference = reference.to(dtype=x.dtype, device=x.device)
        if reference.shape != wavefunction.shape:
            raise ValueError("each reference state must have the same shape as x")
        reference = normalise_on_grid(reference.detach(), x)
        overlap = torch.trapezoid(wavefunction[:, 0] * reference[:, 0], coordinates)
        orthogonality_loss = orthogonality_loss + overlap.square()

    total = (
        weights.residual * residual_loss
        + weights.boundary * boundary_loss
        + weights.normalisation * normalisation_loss
        + weights.orthogonality * orthogonality_loss
    )
    return LossComponents(
        total=total,
        residual=residual_loss,
        boundary=boundary_loss,
        normalisation=normalisation_loss,
        orthogonality=orthogonality_loss,
        norm=raw_norm,
    )

