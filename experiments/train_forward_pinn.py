"""Train a forward PINN for an infinite-well or harmonic-oscillator eigenstate."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from quantumpinn.analytical import (
    harmonic_oscillator_energy,
    harmonic_oscillator_wavefunction,
    infinite_well_energy,
    infinite_well_wavefunction,
)
from quantumpinn.metrics import relative_energy_error, sign_invariant_l2
from quantumpinn.pinn import BoundStatePINN
from quantumpinn.train import TrainingConfig, train_bound_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("well", "oscillator"), default="well")
    parser.add_argument("--state", type=int, choices=range(3), default=0)
    parser.add_argument("--adam-epochs", type=int, default=2_000)
    parser.add_argument("--lbfgs-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)

    if args.system == "well":
        x_min, x_max = 0.0, 1.0

        def potential(x):
            return torch.zeros_like(x)

        exact_energy = infinite_well_energy(args.state)

        def exact_wavefunction(x):
            return infinite_well_wavefunction(x, args.state)

        initial_energy = 0.75 * exact_energy
    else:
        x_min, x_max = -8.0, 8.0

        def potential(x):
            return 0.5 * x.square()

        exact_energy = harmonic_oscillator_energy(args.state)

        def exact_wavefunction(x):
            return harmonic_oscillator_wavefunction(x, args.state)

        initial_energy = args.state + 0.25

    parity = "even" if args.state % 2 == 0 else "odd"
    model = BoundStatePINN(
        x_min=x_min,
        x_max=x_max,
        initial_energy=initial_energy,
        hidden_layers=(64, 64, 64),
        parity=parity,
    )

    reference_states: list[torch.Tensor] = []
    grid = np.linspace(x_min, x_max, 256)
    for state in range(args.state):
        if args.system == "well":
            values = infinite_well_wavefunction(grid, state)
        else:
            values = harmonic_oscillator_wavefunction(grid, state)
        reference_states.append(torch.as_tensor(values).reshape(-1, 1))

    result = train_bound_state(
        model,
        potential,
        config=TrainingConfig(
            num_collocation=256,
            adam_epochs=args.adam_epochs,
            lbfgs_steps=args.lbfgs_steps,
            seed=args.seed,
        ),
        reference_states=tuple(reference_states),
    )
    x = result.x[:, 0].numpy()
    reference = exact_wavefunction(x)
    print(f"system: {args.system}")
    print(f"state: {args.state}")
    print(f"energy: {result.energy:.10f}")
    print(f"reference energy: {exact_energy:.10f}")
    print(f"relative energy error: {relative_energy_error(result.energy, exact_energy):.3e}")
    print(
        "wavefunction L2 error: "
        f"{sign_invariant_l2(result.wavefunction[:, 0].numpy(), reference, x):.3e}"
    )
    print(f"final loss: {result.final_loss['total']:.3e}")


if __name__ == "__main__":
    main()

