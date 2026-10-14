"""Run analytical checks for the finite-difference baseline."""

from __future__ import annotations

from functools import partial

from quantumpinn.analytical import (
    harmonic_oscillator_energy,
    harmonic_oscillator_wavefunction,
    infinite_well_energy,
    infinite_well_wavefunction,
)
from quantumpinn.finite_difference import solve_bound_states
from quantumpinn.metrics import relative_energy_error, sign_invariant_l2
from quantumpinn.potentials import harmonic_oscillator, infinite_well


def report(name: str, solution, energy_fn, wavefunction_fn) -> None:
    print(f"\n{name}")
    print("state  reference E    numerical E    relative E error    wavefunction L2")
    for state, energy in enumerate(solution.energies):
        reference_energy = energy_fn(state)
        reference_wavefunction = wavefunction_fn(solution.x, state)
        print(
            f"{state:>5d}  {reference_energy:>11.7f}  {energy:>13.7f}  "
            f"{relative_energy_error(energy, reference_energy):>16.3e}  "
            f"{sign_invariant_l2(solution.wavefunctions[state], reference_wavefunction, solution.x):>15.3e}"
        )


def main() -> None:
    well = solve_bound_states(
        partial(infinite_well, left=0.0, right=1.0),
        x_min=0.0,
        x_max=1.0,
        num_points=1000,
        num_states=3,
    )
    report("Infinite square well", well, infinite_well_energy, infinite_well_wavefunction)

    oscillator = solve_bound_states(
        harmonic_oscillator,
        x_min=-8.0,
        x_max=8.0,
        num_points=1600,
        num_states=3,
    )
    report(
        "Harmonic oscillator",
        oscillator,
        harmonic_oscillator_energy,
        harmonic_oscillator_wavefunction,
    )


if __name__ == "__main__":
    main()
