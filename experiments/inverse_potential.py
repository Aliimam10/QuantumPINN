"""Recover a hidden even polynomial potential from sparse quantum observations."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

from quantumpinn.evaluate import write_metadata
from quantumpinn.finite_difference import solve_bound_states
from quantumpinn.inverse import (
    EvenPolynomialPotential,
    InverseTrainingConfig,
    WavefunctionObservation,
    relative_potential_l2,
    train_inverse_potential,
)
from quantumpinn.pinn import BoundStatePINN

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--noise-levels", nargs="+", type=float, default=(0.0, 0.005, 0.02))
    parser.add_argument("--observation-kind", choices=("wavefunction", "density"), default="wavefunction")
    parser.add_argument("--observations", type=int, default=25)
    parser.add_argument("--states", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--adam-epochs", type=int, default=3_000)
    parser.add_argument("--lbfgs-steps", type=int, default=300)
    parser.add_argument("--collocation-points", type=int, default=256)
    parser.add_argument("--reference-points", type=int, default=3_000)
    parser.add_argument("--results-dir", type=Path, default=Path("results/inverse"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def true_potential_numpy(x: np.ndarray) -> np.ndarray:
    return 0.5 * x**2 + 0.05 * x**4


def make_observations(
    reference,
    *,
    num_states: int,
    num_observations: int,
    noise_level: float,
    seed: int,
    kind: str,
) -> tuple[WavefunctionObservation, ...]:
    rng = np.random.default_rng(seed)
    observed_x = np.linspace(-3.5, 3.5, num_observations)
    observations: list[WavefunctionObservation] = []
    for state in range(num_states):
        wavefunction = np.interp(
            observed_x,
            reference.x,
            reference.wavefunctions[state],
        )
        if wavefunction[num_observations // 2 + (state % 2)] < 0.0:
            wavefunction = -wavefunction
        if kind == "density":
            values = wavefunction**2
        else:
            values = wavefunction
        scale = max(float(np.max(np.abs(values))), 1e-12)
        values = values + noise_level * scale * rng.normal(size=values.shape)
        energy = float(reference.energies[state])
        energy = energy + noise_level * abs(energy) * float(rng.normal())
        observations.append(
            WavefunctionObservation(
                state=state,
                x=torch.as_tensor(observed_x, dtype=torch.float64).reshape(-1, 1),
                values=torch.as_tensor(values, dtype=torch.float64).reshape(-1, 1),
                energy=energy,
                kind=kind,
            )
        )
    return tuple(observations)


def run_reconstruction(
    reference,
    *,
    noise_level: float,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    torch.manual_seed(seed)
    observations = make_observations(
        reference,
        num_states=args.states,
        num_observations=args.observations,
        noise_level=noise_level,
        seed=seed,
        kind=args.observation_kind,
    )
    models = tuple(
        BoundStatePINN(
            x_min=-5.0,
            x_max=5.0,
            initial_energy=observation.energy,
            hidden_layers=(48, 48, 48),
            parity="even" if state % 2 == 0 else "odd",
        )
        for state, observation in enumerate(observations)
    )
    potential = EvenPolynomialPotential(quadratic=0.3, quartic=0.02)
    started = time.perf_counter()
    result = train_inverse_potential(
        models,
        potential,
        observations,
        x_min=-5.0,
        x_max=5.0,
        config=InverseTrainingConfig(
            num_collocation=args.collocation_points,
            adam_epochs=args.adam_epochs,
            lbfgs_steps=args.lbfgs_steps,
            seed=seed,
        ),
    )
    elapsed = time.perf_counter() - started
    coefficients = result.potential.values()
    held_out_x = torch.linspace(-3.5, 3.5, 1_001, dtype=torch.float64).reshape(-1, 1)
    full_x = torch.linspace(-5.0, 5.0, 1_001, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        held_out_prediction = result.potential(held_out_x)
        full_prediction = result.potential(full_x)
    held_out_reference = 0.5 * held_out_x.square() + 0.05 * held_out_x.pow(4)
    full_reference = 0.5 * full_x.square() + 0.05 * full_x.pow(4)

    def reconstructed_numpy(x: np.ndarray) -> np.ndarray:
        return coefficients["quadratic"] * x**2 + coefficients["quartic"] * x**4

    reconstructed_solution = solve_bound_states(
        reconstructed_numpy,
        x_min=-5.0,
        x_max=5.0,
        num_points=1_000,
        num_states=args.states,
    )
    energy_errors = np.abs(
        (reconstructed_solution.energies - reference.energies[: args.states])
        / reference.energies[: args.states]
    )
    row: dict[str, float | int | str] = {
        "noise_level": noise_level,
        "seed": seed,
        "observation_kind": args.observation_kind,
        "num_observations_per_state": args.observations,
        "num_states": args.states,
        "quadratic": coefficients["quadratic"],
        "quartic": coefficients["quartic"],
        "quadratic_relative_error": abs(coefficients["quadratic"] - 0.5) / 0.5,
        "quartic_relative_error": abs(coefficients["quartic"] - 0.05) / 0.05,
        "potential_l2_observed_domain": relative_potential_l2(
            held_out_prediction,
            held_out_reference,
            held_out_x,
        ),
        "potential_l2_full_domain": relative_potential_l2(
            full_prediction,
            full_reference,
            full_x,
        ),
        "forward_energy_error_mean": float(energy_errors.mean()),
        "forward_energy_error_max": float(energy_errors.max()),
        "final_loss": result.final_loss["total"],
        "final_physics_loss": result.final_loss["physics"],
        "final_data_loss": result.final_loss["data"],
        "runtime_seconds": elapsed,
    }
    return row, full_prediction[:, 0].numpy()


def write_metrics(rows: list[dict[str, float | int | str]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise_metrics(
    rows: list[dict[str, float | int | str]], noise_levels: tuple[float, ...]
) -> list[dict[str, float | int]]:
    metrics = (
        "quadratic",
        "quartic",
        "potential_l2_observed_domain",
        "potential_l2_full_domain",
        "forward_energy_error_mean",
        "runtime_seconds",
    )
    summary: list[dict[str, float | int]] = []
    for level in noise_levels:
        selected = [row for row in rows if float(row["noise_level"]) == level]
        aggregate: dict[str, float | int] = {"noise_level": level, "runs": len(selected)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            output_name = (
                "forward_energy_error" if metric == "forward_energy_error_mean" else metric
            )
            aggregate[f"{output_name}_mean"] = float(values.mean())
            aggregate[f"{output_name}_std"] = float(values.std(ddof=0))
            aggregate[f"{output_name}_median"] = float(np.median(values))
            aggregate[f"{output_name}_min"] = float(values.min())
            aggregate[f"{output_name}_max"] = float(values.max())
        summary.append(aggregate)
    return summary


def plot_reconstructions(
    rows: list[dict[str, float | int | str]],
    curves: list[np.ndarray],
    noise_levels: tuple[float, ...],
    destination: Path,
) -> None:
    x = np.linspace(-5.0, 5.0, curves[0].size)
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(x, true_potential_numpy(x), color="black", linewidth=2.3, label="True potential")
    colours = ("#2878B5", "#E17C05", "#C33C54", "#6A4C93")
    for index, noise_level in enumerate(noise_levels):
        selected = [
            curve
            for row, curve in zip(rows, curves)
            if float(row["noise_level"]) == noise_level
        ]
        values = np.stack(selected)
        mean = values.mean(axis=0)
        axis.plot(
            x,
            mean,
            color=colours[index % len(colours)],
            linewidth=1.8,
            label=f"PINN, noise={100 * noise_level:g}%",
        )
        axis.fill_between(
            x,
            values.min(axis=0),
            values.max(axis=0),
            color=colours[index % len(colours)],
            alpha=0.14,
        )
    axis.set_xlim(-5.0, 5.0)
    axis.set_ylim(-0.5, 45.0)
    axis.set_xlabel("x")
    axis.set_ylabel("V(x)")
    axis.set_title("Hidden quartic potential reconstruction")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_noise_sensitivity(
    rows: list[dict[str, float | int | str]],
    noise_levels: tuple[float, ...],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.7))
    for axis, coefficient, truth in zip(axes, ("quadratic", "quartic"), (0.5, 0.05)):
        means = []
        minima = []
        maxima = []
        for level in noise_levels:
            values = np.asarray(
                [float(row[coefficient]) for row in rows if float(row["noise_level"]) == level]
            )
            means.append(float(values.mean()))
            minima.append(float(values.min()))
            maxima.append(float(values.max()))
        means_array = np.asarray(means)
        axis.errorbar(
            np.asarray(noise_levels) * 100.0,
            means_array,
            yerr=(means_array - minima, np.asarray(maxima) - means_array),
            marker="o",
            capsize=4,
            color="#2878B5",
            label="Recovered",
        )
        axis.axhline(truth, color="black", linestyle="--", label="True")
        axis.set_xlabel("Observation noise (%)")
        axis.set_ylabel(coefficient.title())
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle("Potential-coefficient sensitivity to measurement noise")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)


def print_summary(rows: list[dict[str, float | int | str]], noise_levels: tuple[float, ...]) -> None:
    print("\nnoise    quadratic       quartic         potential L2     forward E error")
    for level in noise_levels:
        selected = [row for row in rows if float(row["noise_level"]) == level]
        print(
            f"{100 * level:>4.1f}%  "
            f"{np.median([float(row['quadratic']) for row in selected]):>12.6f}  "
            f"{np.median([float(row['quartic']) for row in selected]):>12.6f}  "
            f"{np.median([float(row['potential_l2_observed_domain']) for row in selected]):>15.3e}  "
            f"{np.median([float(row['forward_energy_error_mean']) for row in selected]):>15.3e}"
        )


def main() -> None:
    args = parse_args()
    if args.observations < 3:
        raise ValueError("at least three observations per state are required")
    if any(level < 0.0 for level in args.noise_levels):
        raise ValueError("noise levels must be non-negative")
    if not args.seeds:
        raise ValueError("at least one seed is required")
    torch.set_default_dtype(torch.float64)
    reference = solve_bound_states(
        true_potential_numpy,
        x_min=-5.0,
        x_max=5.0,
        num_points=args.reference_points,
        num_states=args.states,
    )
    rows: list[dict[str, float | int | str]] = []
    curves: list[np.ndarray] = []
    for noise_level in args.noise_levels:
        for seed in args.seeds:
            print(f"Reconstructing noise={100 * noise_level:g}%, seed={seed}...")
            row, curve = run_reconstruction(
                reference,
                noise_level=noise_level,
                seed=seed,
                args=args,
            )
            rows.append(row)
            curves.append(curve)

    noise_levels = tuple(float(level) for level in args.noise_levels)
    write_metrics(rows, args.results_dir / "metrics.csv")
    write_metrics(summarise_metrics(rows, noise_levels), args.results_dir / "summary.csv")
    write_metadata(
        {
            "true_potential": {"quadratic": 0.5, "quartic": 0.05, "constant": 0.0},
            "gauge": "V(0) = 0",
            "domain": [-5.0, 5.0],
            "observation_domain": [-3.5, 3.5],
            "observation_kind": args.observation_kind,
            "observations_per_state": args.observations,
            "states": args.states,
            "noise_levels": list(noise_levels),
            "seeds": list(args.seeds),
            "adam_epochs": args.adam_epochs,
            "lbfgs_steps": args.lbfgs_steps,
            "collocation_points": args.collocation_points,
            "reference_points": args.reference_points,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
        },
        args.results_dir / "metadata.json",
    )
    plot_reconstructions(
        rows,
        curves,
        noise_levels,
        args.figures_dir / "inverse_potential_reconstruction.png",
    )
    plot_noise_sensitivity(
        rows,
        noise_levels,
        args.figures_dir / "inverse_noise_sensitivity.png",
    )
    print_summary(rows, noise_levels)
    print(f"\nMetrics written to {args.results_dir}")
    print(f"Figures written to {args.figures_dir}")


if __name__ == "__main__":
    main()
