"""Deterministic full-batch training for stationary bound-state PINNs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .losses import LossComponents, LossWeights, TorchPotential, normalise_on_grid, pinn_loss
from .pinn import BoundStatePINN


@dataclass(frozen=True)
class TrainingConfig:
    num_collocation: int = 256
    adam_epochs: int = 2_000
    learning_rate: float = 1e-3
    lbfgs_steps: int = 200
    seed: int = 0
    history_every: int = 50
    loss_weights: LossWeights = field(default_factory=LossWeights)

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
class TrainingResult:
    model: BoundStatePINN
    x: Tensor
    wavefunction: Tensor
    energy: float
    history: list[dict[str, float | str]]
    final_loss: dict[str, float]


def _record(
    history: list[dict[str, float | str]], step: int, stage: str, loss: LossComponents
) -> None:
    record: dict[str, Any] = {"step": float(step), "stage": stage, **loss.detached()}
    history.append(record)


def train_bound_state(
    model: BoundStatePINN,
    potential_fn: TorchPotential,
    *,
    config: TrainingConfig = TrainingConfig(),
    reference_states: tuple[Tensor, ...] = (),
    hbar: float = 1.0,
    mass: float = 1.0,
    device: str | torch.device = "cpu",
) -> TrainingResult:
    """Train one eigenstate using Adam followed by optional L-BFGS refinement."""

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(device)
    model = model.to(device)
    dtype = model.energy.dtype
    x = torch.linspace(
        model.x_min,
        model.x_max,
        config.num_collocation,
        dtype=dtype,
        device=device,
    ).reshape(-1, 1)
    x.requires_grad_(True)
    references = tuple(state.detach().to(device=device, dtype=dtype) for state in reference_states)
    history: list[dict[str, float | str]] = []

    def objective() -> LossComponents:
        return pinn_loss(
            model,
            x,
            potential_fn,
            weights=config.loss_weights,
            reference_states=references,
            hbar=hbar,
            mass=mass,
        )

    if config.adam_epochs:
        adam = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        for epoch in range(config.adam_epochs):
            adam.zero_grad(set_to_none=True)
            components = objective()
            components.total.backward()
            adam.step()
            if epoch % config.history_every == 0 or epoch == config.adam_epochs - 1:
                _record(history, epoch, "adam", components)

    if config.lbfgs_steps:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
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
            components = objective()
            components.total.backward()
            if closure_calls % config.history_every == 0:
                _record(history, closure_calls, "lbfgs", components)
            closure_calls += 1
            return components.total

        lbfgs.step(closure)

    final_components = objective()
    with torch.no_grad():
        wavefunction = normalise_on_grid(model(x), x).detach().cpu()
        energy = float(model.energy.detach().cpu())
    return TrainingResult(
        model=model,
        x=x.detach().cpu(),
        wavefunction=wavefunction,
        energy=energy,
        history=history,
        final_loss=final_components.detached(),
    )

