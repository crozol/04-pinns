"""Reproduce the Burgers setup of Raissi, Perdikaris & Karniadakis (2019), §3.1.

The paper reports a relative L2 error of 6.7e-4 for:

* a tanh network with 20 neurons per hidden layer. The paper's text says
  "9 layers"; this script uses 9 hidden layers (3 441 parameters), whereas
  the authors' released code uses 8 hidden layers (3 021 parameters),
* N_u = 100 initial/boundary points drawn from the t = 0 and x = ±1 edges,
* N_f = 10 000 collocation points from a Latin hypercube,
* loss = MSE_u + MSE_f, minimised by full-batch L-BFGS to convergence
  (their TensorFlow code calls SciPy's L-BFGS-B with maxiter = 50 000).

This script follows that recipe in float64 PyTorch and measures the error
against the exact Cole-Hopf solution on the paper's 256 x 100 grid.

Usage:
    python -m scripts.raissi_repro                    # seeds 0 1 2
    python -m scripts.raissi_repro --seeds 0 --max-iter 200   # smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from src.physics import NU_BURGERS, burgers_residual
from src.pinn import PINN
from src.reference import burgers_exact

LAYERS = (2,) + (20,) * 9 + (1,)


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


def run(seed: int, max_iter: int, n_u: int = 100, n_f: int = 10_000) -> dict:
    torch.manual_seed(seed)
    model = PINN(LAYERS).double()
    for m in model.modules():                         # Xavier init, as in the paper's code
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    xu, uu, xf = training_points(n_u, n_f, seed)
    T = lambda a: torch.tensor(a, dtype=torch.float64)          # noqa: E731
    x_u, t_u, u_u = T(xu[:, :1]), T(xu[:, 1:]), T(uu[:, None])
    x_f = T(xf[:, :1]).requires_grad_(True)
    t_f = T(xf[:, 1:]).requires_grad_(True)

    opt = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=max_iter, max_eval=int(1.25 * max_iter),
        tolerance_grad=1e-12, tolerance_change=np.finfo(float).eps,
        history_size=50, line_search_fn="strong_wolfe",
    )
    n_eval = [0]
    losses: list[float] = []

    def closure():
        opt.zero_grad()
        loss_u = ((model(x_u, t_u) - u_u) ** 2).mean()
        r = burgers_residual(model(x_f, t_f), x_f, t_f, nu=NU_BURGERS)
        loss = loss_u + (r ** 2).mean()
        loss.backward()
        n_eval[0] += 1
        losses.append(loss.item())
        return loss

    t0 = time.time()
    opt.step(closure)
    secs = time.time() - t0

    x = np.linspace(-1, 1, 256)
    t = np.linspace(0, 1, 100)
    X, Tt = np.meshgrid(x, t)
    with torch.no_grad():
        u = model(T(X.reshape(-1, 1)), T(Tt.reshape(-1, 1))).numpy().reshape(X.shape)
    exact = burgers_exact(x, t)
    err = float(np.linalg.norm(u - exact) / np.linalg.norm(exact))
    print(f"  seed {seed}: rel L2 vs exact = {err:.3e}  final loss {losses[-1]:.3e}  "
          f"{n_eval[0]} evals  {secs:.0f} s", flush=True)
    return {"seed": seed, "relative_l2_error": err, "final_loss": losses[-1],
            "n_evals": n_eval[0], "seconds": secs,
            "n_params": sum(p.numel() for p in model.parameters())}


def main(seeds: list[int], max_iter: int, out: str) -> None:
    runs = []
    for s in seeds:
        runs.append(run(s, max_iter))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        errs = np.array([r["relative_l2_error"] for r in runs])
        with open(out, "w") as fh:
            json.dump({"max_iter": max_iter, "paper_error": 6.7e-4,
                       "mean": float(errs.mean()),
                       "std": float(errs.std(ddof=1)) if len(errs) > 1 else None,
                       "runs": runs}, fh, indent=2)
    print(f"[ok] {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--max-iter", type=int, default=50_000)
    p.add_argument("--out", default="results/raissi_repro.json")
    a = p.parse_args()
    main(a.seeds, a.max_iter, a.out)
