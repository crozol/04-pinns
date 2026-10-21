# Physics-Informed Neural Networks (PINNs)

[![CI](https://img.shields.io/github/actions/workflow/status/crozol/04-pinns/tests.yml?label=CI&style=flat-square)](https://github.com/crozol/04-pinns/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)
[![Portfolio page](https://img.shields.io/badge/Portfolio-page-7C3AED?style=flat-square)](https://crozol.github.io/projects/04-pinns.html)

**TL;DR** — Two PDEs solved with physics-informed neural networks (Raissi, Perdikaris & Karniadakis, 2019): the loss combines the PDE residual, computed via automatic differentiation at collocation points, with the initial/boundary conditions — no mesh, no training data beyond those conditions. The 1D heat equation reaches **0.23% relative L2 error** against its closed-form solution (Adam + a short L-BFGS pass), but that is a global figure: the PINN's absolute error stays near `10⁻³` while the solution decays by `5 × 10⁻⁵`, so the relative error per time slice passes 10% after `t ≈ 0.6` (Section 4). The 1D viscous Burgers equation (`nu = 0.01/pi`, the classic benchmark with a thin internal layer at `x = 0`) is harder: Adam alone plateaus at **~8.8%**; L-BFGS fine-tuning on a fixed point batch brings it to **1.78%** against the exact Cole-Hopf solution. Getting there surfaced a silent failure mode of L-BFGS with resampled collocation points (Section 5.4). Across 5 seeds the errors are **0.14% ± 0.05%** (heat) and **1.66% ± 0.49%** (Burgers). The original paper reports 0.067% on Burgers; following its recipe (full-batch L-BFGS in float64) gives `7.1 × 10⁻⁴`, `1.15 × 10⁻³` and `2.2 × 10⁻²` on three seeds (Section 9), and the result is sensitive to the untuned boundary-loss weight (Section 8). Stack: PyTorch (autograd, `torch.optim.LBFGS`) · NumPy/SciPy (reference solutions) · Matplotlib.

The document walks through both cases in the order they were actually solved —
including the point where the obvious approach quietly didn't work — rather
than only presenting the final, polished numbers.

---

## Contents

1. [Objective](#1--objective)
2. [Equipment (software stack)](#2--equipment-software-stack)
3. [Background: what is a physics-informed neural network?](#3--background-what-is-a-physics-informed-neural-network)
4. [The heat equation](#4--the-heat-equation)
5. [The Burgers equation](#5--the-burgers-equation)
   - [5.1 · Formulation](#51--formulation)
   - [5.2 · A numerical cross-check — why finite differences failed](#52--a-numerical-cross-check--why-finite-differences-failed)
   - [5.3 · First attempt: Adam alone](#53--first-attempt-adam-alone)
   - [5.4 · Diagnosing the plateau — and a silent L-BFGS failure](#54--diagnosing-the-plateau--and-a-silent-l-bfgs-failure)
   - [5.5 · The fix and the final result](#55--the-fix-and-the-final-result)
6. [Head-to-head: why Burgers is harder](#6--head-to-head-why-burgers-is-harder)
7. [Robustness: does it hold across seeds?](#7--robustness-does-it-hold-across-seeds)
8. [Sensitivity to the boundary weight](#8--sensitivity-to-the-boundary-weight)
9. [Comparison with the original paper](#9--comparison-with-the-original-paper)
10. [Reproducing](#10--reproducing)
11. [File map](#11--file-map)
12. [References](#12--references)

---

## 1 · Objective

Train a neural network `u_θ(x, t)` to satisfy a partial differential equation
directly — not by fitting it to a numerically-solved dataset, but by making
the PDE itself part of the loss function, evaluated via automatic
differentiation. Validate the result against a reference with no ambiguity
about ground truth: the closed-form solution for the heat equation, and the
exact Cole-Hopf solution for Burgers (an integral representation evaluated by
quadrature), with an independent Chebyshev-spectral solver as a cross-check.

## 2 · Equipment (software stack)

| Component | What it does in this experiment |
|---|---|
| Python 3.11+, PyTorch | The network and `torch.autograd.grad` for exact partial derivatives |
| `torch.optim.Adam` | First-order optimizer, resampled mini-batches every step |
| `torch.optim.LBFGS` | Second-order fine-tuning, strong-Wolfe line search |
| NumPy | Exact Cole-Hopf solution for Burgers (trapezoid quadrature) |
| SciPy (`solve_ivp`, `qmc`) | Chebyshev-spectral cross-check for Burgers; Latin-hypercube sampling in the paper reproduction |
| Matplotlib | Static figures |

No GPU is needed. Heat training takes a few minutes on a modern laptop CPU;
Burgers (Adam + L-BFGS) takes roughly 15.

## 3 · Background: what is a physics-informed neural network?

A PINN is a neural network `u_θ(x, t)` whose loss combines three pieces:

1. **Initial condition** — `u_θ(x, 0)` must match the known state at `t = 0`.
2. **Boundary condition** — `u_θ` must match known values at the edges of the
   spatial domain, for all `t`.
3. **PDE residual** — at randomly sampled interior points ("collocation
   points"), the PDE itself, evaluated using the network's own partial
   derivatives, should equal zero.

Each piece is a mean squared error over its own set of points, and the loss
is their weighted sum,

```
L = L_PDE + λ_B · L_boundary + λ_I · L_initial
```

with `λ_I = 1` for both equations and `λ_B = 2` (the boundary term is the
mean over both edges, weighted twice). Section 7 measures how sensitive the
Burgers result is to `λ_B`.

The key trick is `torch.autograd.grad(u, x, create_graph=True)`: since `u` is
the network's output and `x` is one of its inputs, this computes `∂u/∂x`
*exactly* (to float32 precision), not by finite differences — and because
`create_graph=True` keeps the computation differentiable, the same call can be
applied again to get `∂²u/∂x²`. This is why `pinn.py` uses `Tanh` activations
throughout: `ReLU` has a zero second derivative almost everywhere, which would
make any PDE residual involving a second-order term (both equations here)
collapse to a useless constant.

The network never sees a numerically-solved trajectory. Everything it learns
comes from the IC/BC data points and the PDE residual at randomly sampled
interior points — the physics *is* the training signal.

## 4 · The heat equation

`u_t = alpha · u_xx` on `x ∈ [0, 1]`, `t ∈ [0, 1]`, `alpha = 1`, with
`u(x, 0) = sin(pi x)` and `u(0, t) = u(1, t) = 0`. This is the textbook case:
separation of variables gives the exact solution in closed form,

```
u(x, t) = sin(pi x) · exp(-pi² alpha t)
```

a single decaying mode, smooth everywhere, nothing stiff or thin to resolve.
It is the sanity check that the PINN machinery — the residual, the autograd
chain, the composite loss — is wired correctly before attempting something
harder.

Adam, 5000 epochs, 10k collocation points, reaches **0.45% relative L2
error** on its own — already well past the <1% target, with no second-order
fine-tuning needed. The frozen run below adds a short L-BFGS pass anyway
(for consistency with the Burgers pipeline in `main.py`), which pushes it
further to **0.23%**:

![Heat equation: PINN vs. analytical solution](figures/heat_comparison.png)

The rightmost panel — the pointwise difference between the PINN and the
exact solution — is essentially blank on a colour scale set by the initial
amplitude.

That global number needs a qualification. The solution decays by a factor
`exp(-pi²) ≈ 5 × 10⁻⁵` over `t ∈ [0, 1]`, so a single relative L2 error over
the whole space-time grid is dominated by the early times, where `u` is of
order one. Resolving the error in time (`scripts/heat_time_error.py`) shows
what actually happens: the PINN's **absolute** error stays roughly constant,
below `1.9 × 10⁻³` at every `t`, while the solution keeps shrinking. The
**relative** error per time slice is 0.07% at `t = 0`, 1.1% at `t = 0.3` and
2.4% at `t = 0.5`, crosses 10% at `t ≈ 0.59`, and exceeds 100% from `t ≈ 0.8`
on, where `|u|` has fallen below the network's error floor.

![Heat equation: amplitude, absolute and relative error against time](figures/heat_error_vs_time.png)

The PINN therefore approximates the solution to a fixed absolute accuracy,
not a fixed relative one; the 0.23% figure describes the first half of the
time window well and the second half not at all.

## 5 · The Burgers equation

### 5.1 · Formulation

`u_t + u·u_x = nu·u_xx` on `x ∈ [-1, 1]`, `t ∈ [0, 1]`, `nu = 0.01/pi`, with
`u(x, 0) = -sin(pi x)` and `u(-1, t) = u(1, t) = 0` — the standard benchmark
from the original PINN paper (Raissi et al., 2019). The nonlinear advection
term `u·u_x` steepens the odd initial condition toward `x = 0` as `t`
increases; the small viscosity `nu` keeps the solution smooth but confines
that steepening to a thin internal layer rather than letting it form a true
discontinuity.

The equation does have an exact solution. The Cole-Hopf substitution
`u = -2 nu ∂_x ln φ` turns it into the heat equation for `φ`, and for this
initial condition (Basdevant et al., 1986)

```
u(x, t) = ∫ (η/t) w(x, η, t) dη / ∫ w(x, η, t) dη,
w(x, η, t) = exp( -cos(pi (x - η)) / (2 pi nu) - η² / (4 nu t) ).
```

The integrals have no elementary closed form, but the Gaussian factor
confines them to `|η| ≲ 12 √(4 nu t)`, and the trapezoid rule on 8001 nodes
evaluates them to roundoff (`src/reference.py::burgers_exact`). On the
256 × 100 evaluation grid, against 16001 nodes (`scripts/reference_convergence.py`):

| Nodes | Relative L2 change | Max. change |
|---|---|---|
| 1001 | 9.9 × 10⁻¹⁵ | 3.4 × 10⁻¹³ |
| 2001 | 2.5 × 10⁻¹⁵ | 8.4 × 10⁻¹⁴ |
| 4001 | 6.2 × 10⁻¹⁶ | 2.0 × 10⁻¹⁴ |
| 8001 | 2.5 × 10⁻¹⁶ | 4.1 × 10⁻¹⁵ |

Widening the window from 12 to 16 Gaussian widths changes the field by
`2.6 × 10⁻¹⁶`. Two independent checks: the computed field satisfies the PDE
with a finite-difference residual of `4 × 10⁻¹⁰`, and its maximum slope,
`|u_x(0, t)| = 152.00516` at `t = 1.6037/pi` (from the same integrals
differentiated under the integral sign), matches the value published by
Basdevant et al. (1986) to all the digits they give. Every Burgers error
reported below is measured against this solution; its own error, `~10⁻¹⁶`,
is negligible next to any of them.

### 5.2 · A numerical cross-check — why finite differences failed

The project first validated Burgers against a numerical solution, before
switching to the exact one above; that solver is kept as an independent
cross-check, and building it had its own failure worth recording. The first
attempt was the obvious one: discretize `x` on a uniform grid, use central
differences for `u_x` and `u_xx`, and integrate in `t` with a stiff implicit
solver (`scipy.integrate.solve_ivp`, method `Radau`). On a 201-point grid the
integration ran in under a second — and produced `max|u| ≈ 1.51`.

That number is wrong on its own terms: `u_t + u·u_x = nu·u_xx` is a parabolic
PDE with no source term, so the maximum principle guarantees
`max|u(·, t)| ≤ max|u(·, 0)| = 1` for every `t`. An overshoot above `1` is not
a borderline numerical-error case — it is a sign the discretization itself is
invalid. The cause: `nu = 0.01/pi ≈ 0.00318` is small, so the internal layer
that forms near `x = 0` is only a few thousandths of a unit wide — thinner
than the grid spacing could resolve. A central-difference stencil on a
too-coarse grid produces exactly this kind of spurious oscillation (Gibbs-type
ringing) near a thin feature it cannot see clearly.

The fix was to stop discretizing locally. A **Chebyshev spectral method**
(`src/reference.py`) differentiates via a global polynomial interpolant
instead of a local stencil, so it does not need the grid to be locally finer
than the feature — spectral differentiation converges exponentially for
smooth functions regardless of where a sharp-but-smooth feature sits. With
`N = 256` points, the same maximum-principle check now gives `max|u| ≈ 0.9999`
— inside the physical bound, as it must be — and the solution is odd-symmetric
(`u(-x, t) = -u(x, t)`) and exactly zero at the boundaries to machine
precision, both properties guaranteed by the problem's symmetry and checked
directly in `tests/test_physics.py`.

Compared with the exact solution, the spectral solution has a relative L2
error of **0.40%** (maximum pointwise error 0.035, inside the internal layer).
That is small in absolute terms but only a factor of ~5 below the PINN's own
error, which is why the reported numbers now use the exact solution instead.
The 0.40% is spatial under-resolution, not interpolation (the comparison is
on the solver's own nodes): Gauss-Lobatto nodes cluster at the boundaries and
are sparsest at `x = 0`, with spacing `≈ pi/N` — for `N = 256`, `0.0123`,
about twice the width of the layer at its steepest (`1/152 ≈ 0.0066`). The
largest error sits at the node next to `x = 0` at `t ≈ 0.52`, when the layer
is steepest, and doubling `N` removes most of it:

| N | Spacing at x = 0 | Relative L2 error | Max. error |
|---|---|---|---|
| 128 | 0.0245 | 2.06% | 0.11 |
| 256 | 0.0123 | 0.40% | 0.035 |
| 512 | 0.0061 | 0.013% | 0.0014 |
The 8.8% and 1.92% figures in Sections 5.3–5.5 are from the original
validation against this spectral solution and are labelled as such.

### 5.3 · First attempt: Adam alone

Same recipe as the heat equation — Adam, 8000 epochs, 10k collocation points,
resampled every step — applied to the harder problem:

```
[epoch      0] loss=4.932933e-01
[epoch   4000] loss=2.744396e-02
[epoch   8000] loss=1.040192e-02   (final)
```

Evaluated against the spectral reference: **relative L2 error ≈ 8.8%** — above
the 5% target, and the loss curve (Section 6 has the full plot) is not
smoothly converging by the end; it oscillates around `~0.01` instead of
continuing to drop. Something was leaving real accuracy on the table.

### 5.4 · Diagnosing the plateau — and a silent L-BFGS failure

Raissi et al.'s original paper does not stop at Adam: it follows a first-order
warm-up with **L-BFGS**, a quasi-Newton method whose curvature information
lets it take much larger, better-aimed steps near a minimum than Adam's
per-parameter first-order updates can. `torch.optim.LBFGS` was added
(`src/train.py::refine_lbfgs`) and pointed at the already-Adam-trained model.

The first run of this looked like it worked — the closure reported a loss of
`6.46 × 10⁻³`, well below Adam's `1.04 × 10⁻²` — but the relative L2 error
against the reference came back **identical to 15 significant figures**
(`0.08827312736777508`, both times) to the pre-refinement number. The model
had not actually changed at all.

The cause: `torch.optim.LBFGS` with `line_search_fn="strong_wolfe"` searches
for a step size by evaluating the loss (and its gradient) at several candidate
points, and both the line search and the quasi-Newton Hessian approximation
assume the objective is a **deterministic** function of the parameters —
evaluating it twice at the same point must return the same value. The
training loop's `loss_fn`, by design, resamples a fresh batch of collocation,
boundary and initial points on *every* call (ordinary and correct for Adam's
stochastic mini-batch training). Under L-BFGS that same design turns the
objective into a noisy function: the line search's step-size probes see
different random realizations of the loss at each trial point, can never
reliably satisfy the Wolfe conditions, and the step is effectively rejected —
silently, with no exception, no warning, just an optimizer that reports an
interesting intermediate number and then changes nothing. The giveaway was
exactly that bit-for-bit-identical error: a coincidence that precise does not
happen from two different training runs landing on the same loss basin — it
happens when the parameters never moved.

### 5.5 · The fix and the final result

The fix: sample one large batch of collocation, boundary and initial points
**once**, before calling L-BFGS, and reuse that exact batch on every internal
evaluation of the closure during the refinement step. This makes the
objective deterministic for as long as L-BFGS runs, which is what both the
line search and the curvature estimate require (`src/train.py` — `train_heat`
and `train_burgers` both take an `lbfgs_iter` argument that triggers this
fixed-batch refinement after the stochastic Adam phase).

With that single change, 1000 L-BFGS iterations on a fixed batch of 40k
collocation / 800 boundary / 800 initial points:

```
[L-BFGS] final loss = 4.525385e-04
```

![Burgers equation: PINN vs. exact solution](figures/burgers_comparison.png)

**Relative L2 error: 1.78%** against the exact solution on the 256 × 100 grid
of Raissi et al. (1.92% against the spectral solution; target was <5%), down
from ~8.8% with Adam alone. The error map (rightmost panel) is now concentrated almost entirely in
a thin vertical band right at the internal layer — exactly where the solution
is hardest to represent — rather than spread across the domain, which is the
qualitative signature of a method that is working correctly but running into
a genuinely hard local feature, not one that is broken.

## 6 · Head-to-head: why Burgers is harder

![Burgers PINN training loss: Adam plateaus, L-BFGS does not](figures/burgers_loss_curve.png)

The heat equation's loss (not pictured — it simply decays smoothly to
`6.7 × 10⁻⁵` with Adam alone, then to `3.1 × 10⁻⁶` after the same short
L-BFGS pass) never *needed* the second optimizer the way Burgers did. Two
things make Burgers
different: the nonlinear `u·u_x` term couples every collocation point's
residual to the local value and slope of `u` itself, unlike the heat
equation's linear residual; and the thin internal layer concentrates nearly
all of the PDE's "information content" into a small region of the domain,
which first-order per-parameter updates are slow to resolve. Both are
well-documented properties of the Burgers benchmark in the PINN literature,
not an artifact of this particular implementation — which is exactly why
Raissi et al.'s original recipe already includes the L-BFGS stage this
project needed to rediscover.

## 7 · Robustness: does it hold across seeds?

Every number above comes from one frozen run (seed 0). Training is genuinely
stochastic — collocation points resample every Adam step, the L-BFGS fixed
batch is itself a random draw, and the network's initial weights depend on
the seed — so a single run could in principle be a lucky draw. `scripts/seed_sweep.py`
retrains each PINN end-to-end (full Adam + L-BFGS pipeline) across 5
independent seeds, mirroring the n=5 seed error bars used for the Neural
ODEs project (02):

**Heat equation** — mean ± std: **0.14% ± 0.05%** (range 0.10%–0.23%)

| seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| relative L2 error | 0.23% | 0.13% | 0.10% | 0.14% | 0.12% |

**Burgers equation**, against the exact solution — mean ± std: **1.66% ± 0.49%**
(range 1.05%–2.33%)

| seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| relative L2 error | 1.88% | 1.66% | 1.34% | 2.33% | 1.05% |

Both equations clear their targets (<1% heat, <5% Burgers) on *every* one of
the 5 seeds, not just the featured run. For heat, seed 0 is the worst of the
five; for Burgers it sits in the middle, and the frozen run (1.78%) is close
to the mean. The heat errors are global over `t ∈ [0, 1]` and inherit the
qualification of Section 4. Seed 0 retrained here gives 1.88% rather than the
frozen run's 1.78% because training is not bit-for-bit reproducible
(Section 10); the difference is well inside the seed-to-seed spread.

```bash
python -m scripts.seed_sweep --equation heat --seeds 5
python -m scripts.seed_sweep --equation burgers --seeds 5
```

## 8 · Sensitivity to the boundary weight

Uniform loss weights are a known weak point of PINNs: the terms compete
during training, and their relative scale shapes what the optimizer fits
first (Wang, Teng & Perdikaris, 2021). The weight `λ_B = 2` used throughout
was not tuned — it is what the original code did implicitly, summing the two
per-edge means. `scripts/bc_weight_sweep.py` retrains the Burgers PINN (seed 0,
full Adam + L-BFGS pipeline) for four values of `λ_B`:

| `λ_B` | 0.5 | 1 | 2 (used) | 4 |
|---|---|---|---|---|
| relative L2 error vs. exact | 0.80% | 1.34% | 1.88% | 1.67% |

Lower boundary weights help here, by more than a factor of two between
`λ_B = 0.5` and `λ_B = 2`. Two caveats limit what this shows: it is a single
seed, and the seed-to-seed spread at fixed `λ_B` (Section 7) is of the same
order as some of these differences. The frozen configuration is left at
`λ_B = 2`: choosing the weight by the same test error that is being reported
would bias that error downward.

## 9 · Comparison with the original paper

Raissi et al. (2019) report a relative L2 error of `6.7 × 10⁻⁴` (0.067%) on
this benchmark, about 25 times lower than the 5-seed mean above. Their recipe
differs in three ways: a deeper, narrower network (20 neurons per hidden
layer), a fixed set of `N_u = 100` initial/boundary points and `N_f = 10 000`
Latin-hypercube collocation points, and full-batch L-BFGS run to convergence
(up to 50 000 iterations) with no Adam phase, in float64.
`scripts/raissi_repro.py` follows that recipe and measures the result
against the same exact solution, on the same 256 × 100 grid:

| seed | 0 | 1 | 2 |
|---|---|---|---|
| relative L2 error vs. exact | 2.2 × 10⁻² | 7.1 × 10⁻⁴ | 1.15 × 10⁻³ |
| final loss | 3.3 × 10⁻⁶ | 1.2 × 10⁻⁶ | 1.4 × 10⁻⁶ |

One seed matches the published number, the median is `1.15 × 10⁻³`, and one
seed stalls at 2.2%, no better than the much cheaper Adam + short L-BFGS
pipeline. All three runs used their full evaluation budget (62 501 loss
evaluations, ~8.3 h each on one CPU core) rather than stopping on a
convergence tolerance. The published figure is therefore reproducible, but it
is closer to a good draw than to a typical outcome of the recipe, and the
gap between it and this project's pipeline (about 14× against the median) comes from the
optimization budget and precision, not from a different formulation.

Two deviations from the paper are known. The paper's text describes "9
layers"; this reproduction uses 9 hidden layers (3 441 parameters), while the
authors' released code uses 8 (3 021). The `N_u` points are drawn from the
exact initial and boundary values rather than the authors' data file, which
contains the same values.

## 10 · Reproducing

```bash
pip install -r requirements.txt
python main.py        # trains both models, saves figures/ and results/metrics.json
pytest tests/ -v       # fast correctness checks (no full training run)

python -m scripts.heat_time_error    # heat error resolved in time (reads results/)
python -m scripts.bc_weight_sweep    # Section 8, ~1 h per weight on one core
python -m scripts.raissi_repro       # Section 9, ~8 h per seed on one core
```

Training is not bit-for-bit reproducible between runs — collocation points
are resampled every epoch by design, and no global seed is enforced beyond
`torch.manual_seed` at the start of each `train_*` call — so the numbers above
are from one frozen run, consistent with how the rest of this portfolio
treats stochastic training. Expect run-to-run variation at the percent level
on Burgers, not a qualitative change in the conclusions above.

| Stage | Time (CPU) |
|---|---|
| Heat (Adam, 5000 epochs) | ~6 min |
| Burgers (Adam, 8000 epochs) | ~9 min |
| Burgers (L-BFGS, 1000 iters, fixed batch) | ~7 min |

## 11 · File map

```
04-pinns/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── main.py              # full pipeline: train, validate, save figures + metrics.json
├── src/
│   ├── pinn.py            # u_theta(x, t) — MLP with Tanh activations (twice-differentiable)
│   ├── physics.py         # PDE residuals (heat, Burgers) + exact heat solution
│   ├── reference.py        # exact Cole-Hopf solution + Chebyshev-spectral cross-check for Burgers
│   └── train.py            # Adam training loop + fixed-batch L-BFGS fine-tuning
├── scripts/
│   ├── export_json.py        # arrays -> JSON for the portfolio page's interactive charts
│   ├── seed_sweep.py          # n=5 seed robustness check (mean +/- std relative L2 error)
│   ├── heat_time_error.py     # heat error per time slice + figure
│   ├── bc_weight_sweep.py     # Burgers sensitivity to the boundary weight λ_B
│   ├── reference_convergence.py  # error floor of the Cole-Hopf reference + Chebyshev resolution
│   └── raissi_repro.py        # Burgers with the original paper's recipe (L-BFGS, float64)
├── tests/                  # pytest: residual correctness, reference sanity, training smoke test
└── figures/                 # PINN-vs-reference comparison plots + loss curve (tracked)
```

## 12 · References

- Basdevant, C., Deville, M., Haldenwang, P., Lacroix, J. M., Ouazzani, J.,
  Peyret, R., Orlandi, P., & Patera, A. T. (1986). *Spectral and finite
  difference solutions of the Burgers equation.* Computers & Fluids, 14(1),
  23-41. (Exact Cole-Hopf solution, used in `src/reference.py`)
- Wang, S., Teng, Y., & Perdikaris, P. (2021). *Understanding and mitigating
  gradient flow pathologies in physics-informed neural networks.* SIAM
  Journal on Scientific Computing, 43(5), A3055-A3081.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed
  neural networks: A deep learning framework for solving forward and inverse
  problems involving nonlinear partial differential equations.* Journal of
  Computational Physics, 378, 686-707.
- Trefethen, L. N. (2000). *Spectral Methods in MATLAB.* SIAM. (Chebyshev
  differentiation matrices, used in `src/reference.py`)
- Liu, D. C. & Nocedal, J. (1989). *On the limited memory BFGS method for
  large scale optimization.* Mathematical Programming, 45, 503-528.
  (L-BFGS, used in the fine-tuning stage)
