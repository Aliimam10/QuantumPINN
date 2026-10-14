"""Neural representation of stationary one-dimensional bound states."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import torch
from torch import Tensor, nn

Parity = Literal["even", "odd"]


class WavefunctionMLP(nn.Module):
    """A small fully connected network with smooth activations.

    Smooth activations are required because the Schrödinger residual contains the
    second spatial derivative of the network output.
    """

    def __init__(
        self,
        *,
        hidden_layers: Sequence[int] = (64, 64, 64),
        activation_factory: Callable[[], nn.Module] = nn.Tanh,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if not hidden_layers or any(width < 1 for width in hidden_layers):
            raise ValueError("hidden_layers must contain positive widths")

        widths = (1, *hidden_layers, 1)
        layers: list[nn.Module] = []
        for index, (input_width, output_width) in enumerate(zip(widths[:-1], widths[1:])):
            layer = nn.Linear(input_width, output_width, dtype=dtype)
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if index < len(widths) - 2:
                layers.append(activation_factory())
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class BoundStatePINN(nn.Module):
    """Wavefunction network with a trainable energy eigenvalue.

    Coordinates are mapped to ``[-1, 1]``. The factor ``1 - xi**2`` imposes exact
    homogeneous Dirichlet conditions at both finite-domain endpoints. For symmetric
    potentials, an optional parity transform helps select the intended eigenstate.
    """

    def __init__(
        self,
        *,
        x_min: float,
        x_max: float,
        initial_energy: float,
        hidden_layers: Sequence[int] = (64, 64, 64),
        parity: Parity | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if x_max <= x_min:
            raise ValueError("x_max must be greater than x_min")
        if parity not in (None, "even", "odd"):
            raise ValueError("parity must be None, 'even', or 'odd'")

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.parity = parity
        self.wavefunction_network = WavefunctionMLP(hidden_layers=hidden_layers, dtype=dtype)
        self.energy = nn.Parameter(torch.tensor(float(initial_energy), dtype=dtype))

    def scaled_coordinate(self, x: Tensor) -> Tensor:
        return 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0

    def forward(self, x: Tensor) -> Tensor:
        xi = self.scaled_coordinate(x)
        network_input = xi.square() if self.parity is not None else xi
        raw = self.wavefunction_network(network_input)
        if self.parity == "odd":
            raw = xi * raw
        return (1.0 - xi.square()) * raw

