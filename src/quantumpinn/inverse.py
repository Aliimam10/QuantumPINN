"""Parametric inverse-potential reconstruction with stationary PINNs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from .losses import LossWeights, normalise_on_grid, pinn_loss
from .pinn import BoundStatePINN

ObservationKind = Literal["wavefunction", "density"]


class EvenPolynomialPotential(nn.Module):
    """An even potential ``a*x**2 + b*x**4`` with a fixed zero-energy gauge."""

    def __init__(
        self,
        *,
        quadratic: float = 0.3,
        quartic: float = 0.02,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.coefficients = nn.Parameter(
            torch.tensor([float(quadratic), float(quartic)], dtype=dtype)
        )

    @property
    def quadratic(self) -> Tensor:
        return self.coefficients[0]

    @property
    def quartic(self) -> Tensor:
        return self.coefficients[1]

    def forward(self, x: Tensor) -> Tensor:
        return self.quadratic * x.square() + self.quartic * x.pow(4)

    def values(self) -> dict[str, float]:
        coefficients = self.coefficients.detach().cpu()
        return {"quadratic": float(coefficients[0]), "quartic": float(coefficients[1])}


@dataclass(frozen=True)
class WavefunctionObservation:
    state: int
    x: Tensor
    values: Tensor
    energy: float
    kind: ObservationKind = "wavefunction"

    def __post_init__(self) -> None:
        if self.state < 0:
            raise ValueError("state must be non-negative")
        if self.kind not in ("wavefunction", "density"):
            raise ValueError("kind must be 'wavefunction' or 'density'")
        if self.x.ndim != 2 or self.x.shape[1] != 1 or self.x.shape != self.values.shape:
            raise ValueError("observation coordinates and values must have shape (n, 1)")
        if self.x.shape[0] < 2:
            raise ValueError("at least two observations are required")


@dataclass(frozen=True)
class InverseLossWeights:
    physics: LossWeights = field(default_factory=LossWeights)
    data: float = 100.0
    coefficient_positivity: float = 10.0

    def __post_init__(self) -> None:
        if self.data < 0.0 or self.coefficient_positivity < 0.0:
            raise ValueError("inverse loss weights must be non-negative")


@dataclass(frozen=True)
class InverseTrainingConfig:
    num_collocation: int = 256
    adam_epochs: int = 3_000
    learning_rate: float = 1e-3
    lbfgs_steps: int = 300
    seed: int = 0
    history_every: int = 100
    loss_weights: InverseLossWeights = field(default_factory=InverseLossWeights)

    def __post_init__(self) -> None:
        if self.num_collocation < 3:
            raise ValueError("num_collocation must be at least 3")
        if self.adam_epochs < 0 or self.lbfgs_steps < 0:
            raise ValueError("optimizer step counts must be non-negative")
        if self.adam_epochs + self.lbfgs_steps == 0:
            raise ValueError("at least one optimizer must take a step")
        if self.learning_rate <= 0.0 or self.history_every < 1:
            raise ValueError("learning_rate and history_every must be positive")


@dataclass
class InverseTrainingResult:
    potential: EvenPolynomialPotential
    models: tuple[BoundStatePINN, ...]
    x: Tensor
    wavefunctions: tuple[Tensor, ...]
    history: list[dict[str, float | str]]
    final_loss: dict[str, float]


def inverse_loss(
    models: tuple[BoundStatePINN, ...],
    potential: EvenPolynomialPotential,
    x: Tensor,
    observations: tuple[WavefunctionObservation, ...],
    *,
    weights: InverseLossWeights = InverseLossWeights(),
) -> tuple[Tensor, dict[str, Tensor]]:
    if len(models) != len(observations) or not models:
        raise ValueError("provide one model for each observation set")

    physics_terms: list[Tensor] = []
    data_terms: list[Tensor] = []
    for model, observation in zip(models, observations):
        components = pinn_loss(model, x, potential, weights=weights.physics)
        physics_terms.append(components.total)
        predicted = model(observation.x)
        if observation.kind == "density":
            predicted = predicted.square()
        data_terms.append((predicted - observation.values).square().mean())

    physics_loss = torch.stack(physics_terms).mean()
    data_loss = torch.stack(data_terms).mean()
    positivity_loss = torch.relu(-potential.coefficients).square().sum()
    total = (
        physics_loss
        + weights.data * data_loss
        + weights.coefficient_positivity * positivity_loss
    )
    return total, {
        "total": total,
        "physics": physics_loss,
        "data": data_loss,
        "coefficient_positivity": positivity_loss,
    }


def train_inverse_potential(
    models: tuple[BoundStatePINN, ...],
    potential: EvenPolynomialPotential,
    observations: tuple[WavefunctionObservation, ...],
    *,
    x_min: float,
    x_max: float,
    config: InverseTrainingConfig = InverseTrainingConfig(),
    device: str | torch.device = "cpu",
) -> InverseTrainingResult:
    """Jointly fit wavefunctions and potential coefficients to sparse observations."""

    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(device)
    dtype = potential.coefficients.dtype
    potential = potential.to(device=device, dtype=dtype)
    models = tuple(model.to(device=device, dtype=dtype) for model in models)
    prepared_observations: list[WavefunctionObservation] = []
    for model, observation in zip(models, observations):
        model.energy.data.fill_(float(observation.energy))
        model.energy.requires_grad_(False)
        prepared_observations.append(
            WavefunctionObservation(
                state=observation.state,
                x=observation.x.detach().to(device=device, dtype=dtype),
                values=observation.values.detach().to(device=device, dtype=dtype),
                energy=observation.energy,
                kind=observation.kind,
            )
        )
    observations = tuple(prepared_observations)
    x = torch.linspace(
        x_min,
        x_max,
        config.num_collocation,
        dtype=dtype,
        device=device,
    ).reshape(-1, 1)
    x.requires_grad_(True)
    parameters = list(potential.parameters())
    for model in models:
        parameters.extend(parameter for parameter in model.parameters() if parameter.requires_grad)
    history: list[dict[str, float | str]] = []

    def objective() -> tuple[Tensor, dict[str, Tensor]]:
        return inverse_loss(
            models,
            potential,
            x,
            observations,
            weights=config.loss_weights,
        )

    def record(step: int, stage: str, components: dict[str, Tensor]) -> None:
        history.append(
            {
                "step": float(step),
                "stage": stage,
                **{name: float(value.detach()) for name, value in components.items()},
                **potential.values(),
            }
        )

    if config.adam_epochs:
        adam = torch.optim.Adam(parameters, lr=config.learning_rate)
        for epoch in range(config.adam_epochs):
            adam.zero_grad(set_to_none=True)
            total, components = objective()
            total.backward()
            adam.step()
            if epoch % config.history_every == 0 or epoch == config.adam_epochs - 1:
                record(epoch, "adam", components)

    if config.lbfgs_steps:
        lbfgs = torch.optim.LBFGS(
            parameters,
            lr=1.0,
            max_iter=config.lbfgs_steps,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
        closure_calls = 0

        def closure() -> Tensor:
            nonlocal closure_calls
            lbfgs.zero_grad(set_to_none=True)
            total, components = objective()
            total.backward()
            if closure_calls % config.history_every == 0:
                record(closure_calls, "lbfgs", components)
            closure_calls += 1
            return total

        lbfgs.step(closure)

    _, final_components = objective()
    with torch.no_grad():
        wavefunctions = tuple(normalise_on_grid(model(x), x).detach().cpu() for model in models)
    return InverseTrainingResult(
        potential=potential,
        models=models,
        x=x.detach().cpu(),
        wavefunctions=wavefunctions,
        history=history,
        final_loss={name: float(value.detach()) for name, value in final_components.items()},
    )


def relative_potential_l2(
    predicted: Tensor,
    reference: Tensor,
    x: Tensor,
) -> float:
    if predicted.shape != reference.shape or predicted.shape != x.shape:
        raise ValueError("predicted, reference, and x must have the same shape")
    numerator = torch.trapezoid((predicted[:, 0] - reference[:, 0]).square(), x[:, 0])
    denominator = torch.trapezoid(reference[:, 0].square(), x[:, 0])
    return float(torch.sqrt(numerator / denominator).detach().cpu())
