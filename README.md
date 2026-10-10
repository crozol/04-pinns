# Physics-Informed Neural Networks (PINNs)

[![CI](https://img.shields.io/github/actions/workflow/status/crozol/04-pinns/tests.yml?label=CI&style=flat-square)](https://github.com/crozol/04-pinns/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)
[![Portfolio page](https://img.shields.io/badge/Portfolio-page-7C3AED?style=flat-square)](https://crozol.github.io/projects/04-pinns.html)

**TL;DR** — Two PDEs solved with physics-informed neural networks (Raissi, Perdikaris & Karniadakis, 2019): the loss combines the PDE residual, evaluated via automatic differentiation at collocation points, with the initial/boundary conditions — no mesh, no training data beyond those conditions. The 1D heat equation reaches **0.45% relative L2 error** against its exact analytical solution. The 1D viscous Burgers equation (`nu = 0.01/pi`, the classic benchmark with a thin internal layer at `x = 0`) reaches **2.04%** against a Chebyshev-spectral numerical reference — but only after L-BFGS fine-tuning on top of Adam; Adam alone plateaus around 9%. Stack: PyTorch (autograd) · NumPy/SciPy (reference solutions) · Matplotlib.

## Motivation

A PINN is a neural network `u_θ(x, t)` whose loss combines:

1. **Initial / boundary conditions** — the network is pushed to match known data at the edges of the domain.
2. **PDE residual** — evaluated at randomly sampled interior points ("collocation points") using `torch.autograd.grad` to compute the exact partial derivatives of the network's own output.

The model is trained to satisfy the PDE in the classical sense, without a mesh and without ever seeing a numerical solution as training data. This project implements two cases of increasing difficulty:

- **1D heat equation** — a smooth, linear, diffusive PDE with a closed-form solution. The minimal case to check the method works at all.
- **1D Burgers equation** — a nonlinear PDE (`u u_x` advection term) with a steep, physically-thin internal layer. The classic PINN stress test.

## Results

### Heat equation

`u_t = alpha * u_xx` on `x in [0, 1]`, `t in [0, 1]`, `alpha = 1`, with `u(x, 0) = sin(pi x)` and `u(0, t) = u(1, t) = 0`. The exact solution follows from separation of variables: `u(x, t) = sin(pi x) * exp(-pi^2 alpha t)`.

![Heat equation: PINN vs. analytical solution](figures/heat_comparison.png)

**Relative L2 error: 0.45%** (target was <1%). Adam alone, 5000 epochs, 10k collocation points — no L-BFGS fine-tuning was even needed to clear the target, though `main.py` applies a short one anyway for consistency with the Burgers run.

### Burgers equation

`u_t + u u_x = nu u_xx` on `x in [-1, 1]`, `t in [0, 1]`, `nu = 0.01/pi`, with `u(x, 0) = -sin(pi x)` and `u(-1, t) = u(1, t) = 0` — the standard benchmark from the original PINN paper. There is no closed-form solution, so the reference here is a Chebyshev-spectral numerical integration (`src/reference.py`): a plain central-difference finite-difference discretization was tried first and produced spurious oscillations, because the viscous layer that forms near `x = 0` is thinner than the grid spacing can resolve. A global spectral differentiation matrix resolves it without needing a prohibitively fine grid.

![Burgers equation: PINN vs. spectral reference](figures/burgers_comparison.png)

**Relative L2 error: 2.04%** (target was <5%). The error map (rightmost panel) is concentrated almost entirely in a thin vertical band right at the internal layer — exactly where the solution is hardest to fit, not spread randomly across the domain.

**Finding worth stating plainly**: Adam alone (8000 epochs, same architecture and collocation budget as the heat case) plateaus at **~9% relative L2 error**, above the 5% target. This is not a bug — it is the well-known slow asymptotic convergence of first-order optimizers on PINN losses, which is precisely why Raissi et al. (2019) follow Adam with L-BFGS in the original paper. Adding an L-BFGS refinement stage (`src/train.py::refine_lbfgs`, 1000 iterations on a large *fixed* batch of collocation/boundary/initial points) brought the error down to 2.04%. The "fixed batch" part matters: L-BFGS's line search assumes a deterministic objective, so reusing Adam's per-step resampling strategy silently breaks it — the line search fails and the optimizer makes no progress at all, without raising an error. That failure mode cost real debugging time and is worth documenting for exactly that reason.

## Stack

- Python 3.11+
- PyTorch 2.x (autograd for partial derivatives, `torch.optim.LBFGS` for fine-tuning)
- NumPy, SciPy (`solve_ivp` for the Chebyshev-spectral Burgers reference)
- Matplotlib

## Structure

```
04-pinns/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── main.py              # full pipeline: train, validate, save figures + metrics.json
├── src/
│   ├── pinn.py           # u_theta(x, t) — MLP with tanh activations (twice-differentiable)
│   ├── physics.py        # PDE residuals (heat, Burgers) + exact heat solution
│   ├── reference.py       # Chebyshev-spectral numerical reference for Burgers
│   └── train.py           # Adam training loop + L-BFGS fine-tuning, for both PDEs
├── tests/                # pytest: residual correctness, reference-solution sanity, training smoke test
└── figures/               # PINN-vs-reference comparison plots (tracked; committed by main.py)
```

## Reproducing

```bash
pip install -r requirements.txt
python main.py        # trains both models, saves figures/ and results/metrics.json
pytest tests/ -v       # fast correctness checks (no full training run)
```

Training is not bit-for-bit reproducible between runs (no fixed seed is enforced beyond `torch.manual_seed`, and collocation points are resampled every epoch by design), so the numbers above are from one frozen run, consistent with how the rest of this portfolio treats stochastic training. Expect run-to-run variation at the percent level, not a qualitative change in conclusions.

## References

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686-707.
