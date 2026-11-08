"""Reproduce the Burgers result of Raissi, Perdikaris & Karniadakis (2019), §3.1, with the
authors' optimizer, and measure how often the recipe reaches the published error.

What the authors ran
--------------------
The paper reports a relative L2 error of 6.7e-4. Its code
(maziarraissi/PINNs, appendix/continuous_time_inference (Burgers)/Burgers.py, TensorFlow 1.x):

* a tanh network with 8 hidden layers of 20 neurons (3 021 parameters) in float32. The paper's
  "9 layers" counts the output layer (7 hidden -> 2 601 parameters, 9 -> 3 441). The input is
  mapped to [-1, 1] by H = 2 (X - lb) / (ub - lb) - 1;
* Xavier weights from a truncated normal (TensorFlow cuts at two standard deviations), zero biases;
* N_u = 100 initial/boundary points and N_f = 10 000 Latin-hypercube collocation points, to
  which the N_u points are appended;
* loss = MSE_u + MSE_f, minimised by SciPy's L-BFGS-B through TensorFlow's
  ScipyOptimizerInterface with maxiter = maxfun = 50 000, maxcor = 50, maxls = 50, ftol = eps.
  gtol is not set, so SciPy's default 1e-5 applies: TensorFlow passes the options through
  unchanged and only casts the float32 gradient to float64, as L-BFGS-B requires.

This script runs the same optimizer (scipy.optimize.minimize, method L-BFGS-B) with the same
options on a PyTorch network built the same way, and hands SciPy the loss and gradient as
TensorFlow did.

Why SciPy's L-BFGS-B and not torch.optim.LBFGS
----------------------------------------------
The question is how often the published recipe reaches the published error, so the optimizer,
its stopping rule and the precision have to be the recipe's. An earlier version of this script
used torch.optim.LBFGS in float64. In the regime this benchmark reaches it is a different
algorithm, not the same one in another library:

1. Memory. L-BFGS keeps a curvature pair (s, y) only if y's is safely positive. L-BFGS-B
   (Byrd, Lu, Nocedal & Zhu, 1995) skips the pair when y's <= eps * (-g's), a test relative to
   the step it was measured on. torch keeps it only when y's > 1e-10, an absolute threshold.
   A strong-Wolfe line search already guarantees y's > 0, so the test only guards against
   round-off, and an absolute threshold makes it depend on the units of the loss. With the
   loss near 1e-6, y's falls near 1e-12: torch discards every pair although y's / (-g's) is
   about 1, fifteen orders of magnitude above eps. Its memory freezes, and the method becomes
   a fixed-metric descent that barely moves. scripts/lbfgs_memory_test.py measures this.
2. Stopping. The recipe stops when max|g| <= 1e-5 or when an iteration lowers the loss by less
   than eps * max(|f|, 1). torch's tests are different (tolerance_grad, and tolerance_change on
   the loss and on the step), and the earlier version set them so low that every seed used its
   full budget, far past the point where the recipe stops.
3. Line search: Moré-Thuente in L-BFGS-B; in torch, a bracketing cubic search whose first step
   is scaled by min(1, 1/|g|_1).

Lowering torch's threshold fixes (1) only, and gives an optimizer that exists only in this
project. The stopping rule matters as much as the memory: once the loss is near 1e-6, lowering
it further does not lower the error against the exact solution, so where a run stops decides
the number it reports (scripts/lbfgs_memory_test.py also measures this).

Precision
---------
The recipe runs in float32, as the authors ran it, and again in float64 as a sensitivity check.
The precision decides which test ends a run: near a loss of 1e-6, float32 cannot represent
decreases much below 1e-13, so the relative-reduction test (ftol = eps) stops the run; in
float64 it continues until max|g| <= 1e-5.

Remaining differences from the authors' runs: PyTorch instead of TensorFlow (same arithmetic,
different rounding), SciPy's C translation of L-BFGS-B 3.0 instead of the Fortran original
(same update and stopping tests), initial/boundary values from the exact formulas instead of the
authors' data file (the same values), and the seeds.

The quantity reported is the failure rate over 20 seeds: how many runs stay above 1e-3. A seed
takes about 10 min in float32 and 15 min in float64 on a laptop CPU; each process writes its
own JSON (seeds already in it are skipped when a command is run again), and --summarize
combines them per precision.

Usage:
    python -m scripts.raissi_repro --dtype float32 --seeds 0 1 2 --out results/raissi/float32/p0.json
    python -m scripts.raissi_repro --summarize          # results/raissi/<dtype>/*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import qmc
from torch import nn

from src.physics import NU_BURGERS, burgers_residual
from src.reference import burgers_exact

LAYERS = (2,) + (20,) * 8 + (1,)
LB, UB = (-1.0, 0.0), (1.0, 1.0)                  # (x, t) domain
DTYPES = {"float32": torch.float32, "float64": torch.float64}
# the authors' ScipyOptimizerInterface options; gtol is left to SciPy's default (1e-5), as they did
AUTHORS_OPTIONS = {"maxiter": 50_000, "maxfun": 50_000, "maxcor": 50, "maxls": 50,
                   "ftol": float(np.finfo(float).eps)}
PAPER_ERROR = 6.7e-4
THRESHOLD = 1e-3


class RaissiNet(nn.Module):
    """tanh network with the authors' input scaling H = 2 (X - lb) / (ub - lb) - 1."""

    def __init__(self, layers: tuple[int, ...] = LAYERS):
        super().__init__()
        self.linears = nn.ModuleList(nn.Linear(a, b) for a, b in zip(layers[:-1], layers[1:]))
        self.register_buffer("lb", torch.tensor(LB))
        self.register_buffer("ub", torch.tensor(UB))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = 2.0 * (torch.cat([x, t], dim=-1) - self.lb) / (self.ub - self.lb) - 1.0
        for lin in self.linears[:-1]:
            h = torch.tanh(lin(h))
        return self.linears[-1](h)


def xavier_truncated_(model: RaissiNet) -> None:
    """tf.truncated_normal(stddev = sqrt(2 / (fan_in + fan_out))): redrawn beyond 2 std; zero biases."""
    for lin in model.linears:
        sd = float(np.sqrt(2.0 / (lin.in_features + lin.out_features)))
        nn.init.trunc_normal_(lin.weight, std=sd, a=-2 * sd, b=2 * sd)
        nn.init.zeros_(lin.bias)


def training_points(n_u: int, n_f: int, seed: int):
    rng = np.random.default_rng(seed)
    # initial (t = 0) and boundary (x = ±1) edges, as in the paper's data file
    x_i = np.linspace(-1, 1, 256)
    t_b = np.linspace(0, 1, 100)
    edge_xt = np.concatenate([
        np.stack([x_i, np.zeros_like(x_i)], 1),
        np.stack([-np.ones_like(t_b), t_b], 1),
        np.stack([np.ones_like(t_b), t_b], 1),
    ])
    edge_u = np.concatenate([-np.sin(np.pi * x_i), np.zeros_like(t_b), np.zeros_like(t_b)])
    pick = rng.choice(len(edge_xt), n_u, replace=False)
    lhs = qmc.LatinHypercube(d=2, seed=seed).random(n_f)
    xf = qmc.scale(lhs, [-1, 0], [1, 1])
    xf = np.concatenate([xf, edge_xt[pick]])        # the paper also adds X_u to X_f
    return edge_xt[pick], edge_u[pick], xf


class Problem:
    """The recipe's network and loss for one seed, as a function of a flat float64 vector."""

    def __init__(self, seed: int, dtype: str = "float32", n_u: int = 100, n_f: int = 10_000):
        self.dtype = DTYPES[dtype]
        torch.manual_seed(seed)
        self.model = RaissiNet().to(self.dtype)
        xavier_truncated_(self.model)
        self.params = list(self.model.parameters())
        xu, uu, xf = training_points(n_u, n_f, seed)
        self.x_u, self.t_u, self.u_u = self._t(xu[:, :1]), self._t(xu[:, 1:]), self._t(uu[:, None])
        self.x_f = self._t(xf[:, :1]).requires_grad_(True)
        self.t_f = self._t(xf[:, 1:]).requires_grad_(True)

    def _t(self, a) -> torch.Tensor:
        return torch.tensor(a, dtype=self.dtype)

    def loss(self) -> torch.Tensor:
        loss_u = ((self.model(self.x_u, self.t_u) - self.u_u) ** 2).mean()
        r = burgers_residual(self.model(self.x_f, self.t_f), self.x_f, self.t_f, nu=NU_BURGERS)
        return loss_u + (r ** 2).mean()

    def get_x(self) -> np.ndarray:
        return torch.cat([p.detach().reshape(-1) for p in self.params]).double().numpy()

    def set_x(self, v: np.ndarray) -> None:
        i = 0
        with torch.no_grad():
            for p in self.params:
                n = p.numel()
                p.copy_(torch.from_numpy(v[i:i + n]).view_as(p))     # cast to the network's dtype
                i += n

    def value_and_grad(self, v: np.ndarray) -> tuple[float, np.ndarray]:
        self.set_x(v)
        for p in self.params:
            p.grad = None
        loss = self.loss()
        loss.backward()
        g = torch.cat([p.grad.reshape(-1) for p in self.params]).numpy()
        return float(loss.item()), g.astype(np.float64)              # as TensorFlow's wrapper

    def relative_error(self) -> float:
        x, t = np.linspace(-1, 1, 256), np.linspace(0, 1, 100)       # the paper's grid
        X, T = np.meshgrid(x, t)
        with torch.no_grad():
            u = self.model(self._t(X.reshape(-1, 1)), self._t(T.reshape(-1, 1)))
        u = u.double().numpy().reshape(X.shape)
        exact = burgers_exact(x, t)
        return float(np.linalg.norm(u - exact) / np.linalg.norm(exact))


def run(seed: int, dtype: str = "float32", n_u: int = 100, n_f: int = 10_000,
        options: dict | None = None, problem: Problem | None = None) -> dict:
    """One seed of the recipe: L-BFGS-B with the authors' options until one of its tests stops it."""
    pb = problem or Problem(seed, dtype, n_u, n_f)
    t0 = time.time()
    res = minimize(pb.value_and_grad, pb.get_x(), jac=True, method="L-BFGS-B",
                   options=AUTHORS_OPTIONS if options is None else options)
    pb.set_x(res.x)
    out = {"seed": seed, "dtype": dtype, "relative_l2_error": pb.relative_error(),
           "final_loss": float(res.fun), "max_abs_grad": float(np.abs(res.jac).max()),
           "n_iter": int(res.nit), "n_evals": int(res.nfev), "stop": str(res.message),
           "seconds": time.time() - t0, "n_params": int(sum(p.numel() for p in pb.params))}
    print(f"  seed {seed} [{dtype}]: rel L2 vs exact = {out['relative_l2_error']:.3e}  "
          f"loss {out['final_loss']:.3e}  {out['n_iter']} it / {out['n_evals']} evals  "
          f"{out['seconds']:.0f} s  ({out['stop']})", flush=True)
    return out


def main(seeds: list[int], dtype: str, out: str) -> None:
    path = Path(out)
    runs = json.loads(path.read_text())["runs"] if path.exists() else []
    done = {r["seed"] for r in runs}
    for s in seeds:
        if s in done:
            print(f"  seed {s} [{dtype}]: already in {out}, skipped", flush=True)
            continue
        runs.append(run(s, dtype))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"dtype": dtype, "options": AUTHORS_OPTIONS,
                                    "paper_error": PAPER_ERROR, "runs": runs}, indent=2))
    print(f"[ok] {out}")


def summarize(root: str = "results/raissi", out: str = "results/raissi_repro.json",
              threshold: float = THRESHOLD) -> dict:
    summary = {"paper_error": PAPER_ERROR, "threshold": threshold, "options": AUTHORS_OPTIONS}
    for dtype in DTYPES:
        runs = {}
        for f in sorted(glob.glob(str(Path(root) / dtype / "*.json"))):
            for r in json.loads(Path(f).read_text())["runs"]:
                runs[r["seed"]] = r
        if not runs:
            continue
        runs = [runs[s] for s in sorted(runs)]
        errs = np.array([r["relative_l2_error"] for r in runs])
        fails = int((errs > threshold).sum())
        summary[dtype] = {
            "n_seeds": len(runs), "n_above_threshold": fails,
            "median": float(np.median(errs)),
            "quartiles": [float(np.quantile(errs, 0.25)), float(np.quantile(errs, 0.75))],
            "min": float(errs.min()), "max": float(errs.max()),
            "median_evals": float(np.median([r["n_evals"] for r in runs])),
            "stops": dict(Counter(r["stop"] for r in runs)),
            "n_params": runs[0]["n_params"], "runs": runs}
        print(f"{dtype}: {len(runs)} seeds, {fails}/{len(runs)} above {threshold:g}; median "
              f"{np.median(errs):.2e}, IQR [{np.quantile(errs, 0.25):.2e}, "
              f"{np.quantile(errs, 0.75):.2e}], min {errs.min():.2e}, max {errs.max():.2e}")
        for stop, n in Counter(r["stop"] for r in runs).items():
            print(f"    {n:2d} stopped by: {stop}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"[ok] {out}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    p.add_argument("--dtype", choices=list(DTYPES), default="float32")
    p.add_argument("--out", default=None, help="default: results/raissi/<dtype>/all.json")
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    if a.summarize:
        summarize()
    else:
        main(a.seeds, a.dtype, a.out or f"results/raissi/{a.dtype}/all.json")
