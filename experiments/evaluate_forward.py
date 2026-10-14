"""Benchmark PINNs against analytical and finite-difference quantum solutions."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from torch import Tensor

from quantumpinn.analytical import (
    harmonic_oscillator_energy,
    harmonic_oscillator_wavefunction,
    infinite_well_energy,
    infinite_well_wavefunction,
)
from quantumpinn.evaluate import (
    EvaluationRecord,
    StatePrediction,
    aggregate_records,
    align_global_sign,
    evaluate_state,
    write_metadata,
    write_records_csv,
    write_summary_csv,
)
from quantumpinn.finite_difference import solve_bound_states
from quantumpinn.losses import DeflationState
from quantumpinn.pinn import BoundStatePINN
from quantumpinn.potentials import double_well, harmonic_oscillator, infinite_well
from quantumpinn.train import TrainingConfig, train_bound_state

matplotlib.use("Agg")

NumpyPotential = Callable[[NDArray[np.float64]], NDArray[np.float64]]
TorchPotential = Callable[[Tensor], Tensor]
EnergyReference = Callable[[int], float]
WavefunctionReference = Callable[[NDArray[np.float64], int], NDArray[np.float64]]


@dataclass(frozen=True)
class SystemSpec:
    name: str
    label: str
    x_min: float
    x_max: float
    num_states: int
    numpy_potential: NumpyPotential
    torch_potential: TorchPotential
    reference_energy: EnergyReference
    reference_wavefunction: WavefunctionReference
    initial_energy: EnergyReference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=("well", "oscillator", "double_well"),
        default=("well", "oscillator", "double_well"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0,))
    parser.add_argument("--adam-epochs", type=int, default=2_000)
    parser.add_argument("--lbfgs-steps", type=int, default=200)
    parser.add_argument("--collocation-points", type=int, default=256)
    parser.add_argument("--fd-points", type=int, default=512)
    parser.add_argument("--reference-points", type=int, default=4_000)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--results-dir", type=Path, default=Path("results/forward"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def build_systems(reference_points: int) -> dict[str, SystemSpec]:
    def zero_potential_torch(x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def oscillator_potential_torch(x: Tensor) -> Tensor:
        return 0.5 * x.square()

    def double_well_potential_torch(x: Tensor) -> Tensor:
        return (x.square() - 1.5**2).square()

    double_well_reference = solve_bound_states(
        double_well,
        x_min=-4.0,
        x_max=4.0,
        num_points=reference_points,
        num_states=2,
    )

    def double_well_energy_reference(state: int) -> float:
        return float(double_well_reference.energies[state])

    def double_well_wavefunction_reference(
        x: NDArray[np.float64], state: int
    ) -> NDArray[np.float64]:
        return np.interp(
            x,
            double_well_reference.x,
            double_well_reference.wavefunctions[state],
        )

    return {
        "well": SystemSpec(
            name="infinite_well",
            label="Infinite square well",
            x_min=0.0,
            x_max=1.0,
            num_states=3,
            numpy_potential=infinite_well,
            torch_potential=zero_potential_torch,
            reference_energy=infinite_well_energy,
            reference_wavefunction=infinite_well_wavefunction,
            initial_energy=lambda state: 3.0 * (state + 1) ** 2,
        ),
        "oscillator": SystemSpec(
            name="harmonic_oscillator",
            label="Harmonic oscillator",
            x_min=-8.0,
            x_max=8.0,
            num_states=3,
            numpy_potential=harmonic_oscillator,
            torch_potential=oscillator_potential_torch,
            reference_energy=harmonic_oscillator_energy,
            reference_wavefunction=harmonic_oscillator_wavefunction,
            initial_energy=lambda state: state + 0.25,
        ),
        "double_well": SystemSpec(
            name="double_well",
            label="Symmetric double well",
            x_min=-4.0,
            x_max=4.0,
            num_states=2,
            numpy_potential=double_well,
            torch_potential=double_well_potential_torch,
            reference_energy=double_well_energy_reference,
            reference_wavefunction=double_well_wavefunction_reference,
            # Local harmonic expansion around either minimum gives E ~= 2.1.
            initial_energy=lambda state: 2.0 + 0.1 * state,
        ),
    }


def benchmark_finite_difference(
    spec: SystemSpec,
    *,
    num_points: int,
) -> tuple[list[EvaluationRecord], list[StatePrediction]]:
    started = time.perf_counter()
    solution = solve_bound_states(
        spec.numpy_potential,
        x_min=spec.x_min,
        x_max=spec.x_max,
        num_points=num_points,
        num_states=spec.num_states,
    )
    elapsed = time.perf_counter() - started
    records: list[EvaluationRecord] = []
    predictions: list[StatePrediction] = []
    runtime_per_state = elapsed / spec.num_states
    for state in range(spec.num_states):
        reference = spec.reference_wavefunction(solution.x, state)
        records.append(
            evaluate_state(
                system=spec.name,
                state=state,
                method="finite_difference",
                seed=None,
                x=solution.x,
                energy=solution.energies[state],
                wavefunction=solution.wavefunctions[state],
                reference_energy=spec.reference_energy(state),
                reference_wavefunction=reference,
                potential=solution.potential,
                runtime_seconds=runtime_per_state,
            )
        )
        predictions.append(
            StatePrediction(
                system=spec.name,
                state=state,
                method="finite_difference",
                seed=None,
                x=solution.x,
                energy=float(solution.energies[state]),
                wavefunction=solution.wavefunctions[state],
            )
        )
    return records, predictions


def benchmark_pinn(
    spec: SystemSpec,
    *,
    seeds: tuple[int, ...],
    training_config: TrainingConfig,
    hidden_layers: tuple[int, ...],
) -> tuple[list[EvaluationRecord], list[StatePrediction]]:
    records: list[EvaluationRecord] = []
    predictions: list[StatePrediction] = []
    for seed in seeds:
        learned_states: list[DeflationState] = []
        for state in range(spec.num_states):
            model_seed = seed * 1_000 + state
            torch.manual_seed(model_seed)
            model = BoundStatePINN(
                x_min=spec.x_min,
                x_max=spec.x_max,
                initial_energy=spec.initial_energy(state),
                hidden_layers=hidden_layers,
                parity="even" if state % 2 == 0 else "odd",
            )
            state_config = TrainingConfig(
                num_collocation=training_config.num_collocation,
                adam_epochs=training_config.adam_epochs,
                learning_rate=training_config.learning_rate,
                lbfgs_steps=training_config.lbfgs_steps,
                seed=model_seed,
                history_every=training_config.history_every,
                loss_weights=training_config.loss_weights,
            )
            started = time.perf_counter()
            result = train_bound_state(
                model,
                spec.torch_potential,
                config=state_config,
                deflation_states=tuple(learned_states),
            )
            elapsed = time.perf_counter() - started
            learned_states.append(
                DeflationState(
                    wavefunction=result.wavefunction,
                    second_derivative=result.second_derivative,
                )
            )

            x = result.x[:, 0].numpy()
            wavefunction = result.wavefunction[:, 0].numpy()
            reference = spec.reference_wavefunction(x, state)
            records.append(
                evaluate_state(
                    system=spec.name,
                    state=state,
                    method="pinn",
                    seed=seed,
                    x=x,
                    energy=result.energy,
                    wavefunction=wavefunction,
                    reference_energy=spec.reference_energy(state),
                    reference_wavefunction=reference,
                    potential=spec.numpy_potential(x),
                    runtime_seconds=elapsed,
                    pde_residual_rms=np.sqrt(result.final_loss["residual"]),
                )
            )
            predictions.append(
                StatePrediction(
                    system=spec.name,
                    state=state,
                    method="pinn",
                    seed=seed,
                    x=x,
                    energy=result.energy,
                    wavefunction=wavefunction,
                )
            )
    return records, predictions


def _prediction(
    predictions: list[StatePrediction],
    *,
    system: str,
    state: int,
    method: str,
    seed: int | None,
) -> StatePrediction:
    return next(
        prediction
        for prediction in predictions
        if prediction.system == system
        and prediction.state == state
        and prediction.method == method
        and prediction.seed == seed
    )


def plot_system_states(
    spec: SystemSpec,
    predictions: list[StatePrediction],
    *,
    representative_seed: int,
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, spec.num_states, figsize=(4.2 * spec.num_states, 3.4))
    axes = np.atleast_1d(axes)
    for state, axis in enumerate(axes):
        x = np.linspace(spec.x_min, spec.x_max, 1_000)
        reference = spec.reference_wavefunction(x, state)
        axis.plot(x, reference, color="black", linewidth=2.0, label="Reference")
        for method, colour, style in (
            ("finite_difference", "#2878B5", "--"),
            ("pinn", "#D95319", "-."),
        ):
            seed = representative_seed if method == "pinn" else None
            prediction = _prediction(
                predictions,
                system=spec.name,
                state=state,
                method=method,
                seed=seed,
            )
            local_reference = spec.reference_wavefunction(prediction.x, state)
            aligned = align_global_sign(
                prediction.wavefunction,
                local_reference,
                prediction.x,
            )
            axis.plot(
                prediction.x,
                aligned,
                linestyle=style,
                color=colour,
                linewidth=1.6,
                label="Finite difference" if method == "finite_difference" else "PINN",
            )
        axis.set_title(f"State {state}")
        axis.set_xlabel("x")
        if state == 0:
            axis.set_ylabel(r"$\psi(x)$")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(spec.label, y=0.98, fontsize=13)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.83))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_energy_errors(records: list[EvaluationRecord], destination: Path) -> None:
    groups: dict[tuple[str, int, str], list[float]] = {}
    for record in records:
        groups.setdefault((record.system, record.state, record.method), []).append(
            record.relative_energy_error
        )
    state_keys = sorted({(record.system, record.state) for record in records})
    x_positions = np.arange(len(state_keys))
    figure, axis = plt.subplots(figsize=(max(7.0, len(state_keys) * 1.15), 4.0))
    for offset, method, colour, marker in (
        (-0.12, "finite_difference", "#2878B5", "o"),
        (0.12, "pinn", "#D95319", "s"),
    ):
        means = []
        lower = []
        upper = []
        for system, state in state_keys:
            values = np.asarray(groups[(system, state, method)])
            mean = max(float(values.mean()), 1e-16)
            means.append(mean)
            lower.append(max(mean - float(values.min()), 0.0))
            upper.append(max(float(values.max()) - mean, 0.0))
        axis.errorbar(
            x_positions + offset,
            means,
            yerr=np.asarray((lower, upper)),
            fmt=marker,
            color=colour,
            capsize=3,
            label="Finite difference" if method == "finite_difference" else "PINN",
        )
    axis.set_yscale("log")
    axis.set_ylabel("Relative energy error")
    axis.set_xticks(
        x_positions,
        [f"{system.replace('_', ' ')}\nstate {state}" for system, state in state_keys],
        rotation=20,
        ha="right",
    )
    axis.grid(axis="y", which="both", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_double_well_tunnelling(
    spec: SystemSpec,
    predictions: list[StatePrediction],
    *,
    representative_seed: int,
    destination: Path,
) -> None:
    x = np.linspace(spec.x_min, spec.x_max, 1_000)
    potential = spec.numpy_potential(x)
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(x, potential, color="0.65", linewidth=2.0, label="Potential")
    amplitude = 0.35
    for state, colour in ((0, "#2878B5"), (1, "#D95319")):
        reference_energy = spec.reference_energy(state)
        reference = spec.reference_wavefunction(x, state)
        axis.plot(
            x,
            reference_energy + amplitude * reference,
            color=colour,
            linewidth=2.0,
            label=f"Reference state {state}",
        )
        prediction = _prediction(
            predictions,
            system=spec.name,
            state=state,
            method="pinn",
            seed=representative_seed,
        )
        local_reference = spec.reference_wavefunction(prediction.x, state)
        aligned = align_global_sign(
            prediction.wavefunction,
            local_reference,
            prediction.x,
        )
        axis.plot(
            prediction.x,
            prediction.energy + amplitude * aligned,
            color=colour,
            linestyle="--",
            linewidth=1.5,
            label=f"PINN state {state}",
        )
    reference_split = spec.reference_energy(1) - spec.reference_energy(0)
    pinn_energies = [
        _prediction(
            predictions,
            system=spec.name,
            state=state,
            method="pinn",
            seed=representative_seed,
        ).energy
        for state in (0, 1)
    ]
    pinn_split = pinn_energies[1] - pinn_energies[0]
    axis.text(
        0.02,
        0.96,
        f"Energy splitting\nreference: {reference_split:.5f}\nPINN: {pinn_split:.5f}",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    axis.set_ylim(-0.1, 6.0)
    axis.set_xlabel("x")
    axis.set_ylabel("Energy / offset wavefunction")
    axis.set_title("Double-well tunnelling and low-state energy splitting")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right", frameon=False, ncol=2)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_tunnelling_table(
    spec: SystemSpec,
    predictions: list[StatePrediction],
    destination: Path,
) -> None:
    reference_split = spec.reference_energy(1) - spec.reference_energy(0)
    rows: list[dict[str, object]] = []
    methods_and_seeds = sorted(
        {
            (prediction.method, prediction.seed)
            for prediction in predictions
            if prediction.system == spec.name
        },
        key=lambda item: (item[0], -1 if item[1] is None else item[1]),
    )
    for method, seed in methods_and_seeds:
        energies = [
            _prediction(
                predictions,
                system=spec.name,
                state=state,
                method=method,
                seed=seed,
            ).energy
            for state in (0, 1)
        ]
        splitting = energies[1] - energies[0]
        rows.append(
            {
                "method": method,
                "seed": "" if seed is None else seed,
                "energy_0": energies[0],
                "energy_1": energies[1],
                "splitting": splitting,
                "reference_splitting": reference_split,
                "relative_splitting_error": abs(splitting - reference_split)
                / abs(reference_split),
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: list[dict[str, object]]) -> None:
    print("\nsystem               state  method              relative E error    wavefunction L2")
    for row in summary:
        print(
            f"{str(row['system']):<20} {int(row['state']):>5d}  "
            f"{str(row['method']):<18} "
            f"{float(row['relative_energy_error_mean']):>16.3e}  "
            f"{float(row['wavefunction_l2_error_mean']):>15.3e}"
        )


def main() -> None:
    args = parse_args()
    if args.hidden_width < 1 or args.hidden_layers < 1:
        raise ValueError("hidden-width and hidden-layers must be positive")
    if not args.seeds:
        raise ValueError("at least one seed is required")
    torch.set_default_dtype(torch.float64)
    systems = build_systems(args.reference_points)
    selected_specs = [systems[name] for name in args.systems]
    config = TrainingConfig(
        num_collocation=args.collocation_points,
        adam_epochs=args.adam_epochs,
        lbfgs_steps=args.lbfgs_steps,
    )
    hidden_layers = (args.hidden_width,) * args.hidden_layers
    records: list[EvaluationRecord] = []
    predictions: list[StatePrediction] = []

    for spec in selected_specs:
        print(f"Benchmarking {spec.label}...")
        fd_records, fd_predictions = benchmark_finite_difference(spec, num_points=args.fd_points)
        pinn_records, pinn_predictions = benchmark_pinn(
            spec,
            seeds=tuple(args.seeds),
            training_config=config,
            hidden_layers=hidden_layers,
        )
        records.extend(fd_records)
        records.extend(pinn_records)
        predictions.extend(fd_predictions)
        predictions.extend(pinn_predictions)
        plot_system_states(
            spec,
            predictions,
            representative_seed=args.seeds[0],
            destination=args.figures_dir / f"{spec.name}_states.png",
        )

    summary = aggregate_records(records)
    write_records_csv(records, args.results_dir / "metrics.csv")
    write_summary_csv(summary, args.results_dir / "summary.csv")
    write_metadata(
        {
            "systems": list(args.systems),
            "seeds": list(args.seeds),
            "adam_epochs": args.adam_epochs,
            "lbfgs_steps": args.lbfgs_steps,
            "collocation_points": args.collocation_points,
            "finite_difference_points": args.fd_points,
            "double_well_reference_points": args.reference_points,
            "hidden_layers": list(hidden_layers),
            "units": {"hbar": 1.0, "mass": 1.0},
            "double_well": {"scale": 1.0, "separation": 1.5, "domain": [-4.0, 4.0]},
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
        },
        args.results_dir / "metadata.json",
    )
    plot_energy_errors(records, args.figures_dir / "forward_energy_errors.png")
    if "double_well" in args.systems:
        write_tunnelling_table(
            systems["double_well"],
            predictions,
            args.results_dir / "tunnelling_splitting.csv",
        )
        plot_double_well_tunnelling(
            systems["double_well"],
            predictions,
            representative_seed=args.seeds[0],
            destination=args.figures_dir / "double_well_tunnelling.png",
        )
    print_summary(summary)
    print(f"\nMetrics written to {args.results_dir}")
    print(f"Figures written to {args.figures_dir}")


if __name__ == "__main__":
    main()
