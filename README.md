# QuantumPINN

QuantumPINN is a compact scientific machine-learning project for the one-dimensional,
time-independent Schrödinger equation

\[
-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi.
\]

The project asks two questions:

1. How accurately can a physics-informed neural network (PINN) recover low-lying
   eigenstates compared with analytical and finite-difference solutions?
2. When the potential is unknown, what information is needed to reconstruct it
   reliably from measured energies and probability densities?

The intended result is a reproducible comparison, not a claim that PINNs outperform
standard eigensolvers on simple forward problems. See [PROJECT_SCOPE.md](PROJECT_SCOPE.md)
for the scientific scope, acceptance criteria, and delivery plan.

## Current status

The numerical benchmark and forward-PINN layers are implemented:

- dimensionless quantum units by default (`hbar = m = 1`);
- reusable harmonic, infinite-well, finite-well, barrier, quartic, and double-well
  potentials;
- a finite-difference Dirichlet eigensolver;
- analytical infinite-well and harmonic-oscillator reference states;
- sign-invariant wavefunction and energy error metrics;
- forward-solver tests and a benchmark script.
- a smooth PyTorch wavefunction network with a trainable energy;
- exact finite-domain Dirichlet boundary conditions;
- Schrödinger residual, normalisation, and excited-state orthogonality losses;
- grid-normalised physics residuals that prevent collapse to the trivial zero state;
- deterministic Adam plus L-BFGS training and a forward-PINN experiment script.

PyTorch remains an optional dependency so the numerical reference solver can be used
without installing the ML stack. The inverse-potential stage is the next milestone.

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m pip install -e .
python experiments/benchmark_forward.py
python -m unittest discover -s tests -v
```

For later PINN development:

```bash
python -m pip install -e '.[ml,dev]'
```

Train a ground-state PINN after installing the ML extra:

```bash
python experiments/train_forward_pinn.py --system well --state 0
python experiments/train_forward_pinn.py --system oscillator --state 0
```

## Repository layout

```text
src/quantumpinn/       numerical and, later, PINN library code
experiments/           reproducible experiment entry points
tests/                 numerical and scientific regression tests
figures/               generated plots (not source data)
PROJECT_SCOPE.md       scope, milestones, and success criteria
```

## Conventions

Unless an experiment states otherwise, all quantities are dimensionless with
`hbar = m = 1`. Wavefunctions are normalised on the finite computational domain.
For analytical comparisons, their arbitrary global sign is aligned before computing
the error.
