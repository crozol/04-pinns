"""Sensitivity of the Burgers PINN to the boundary-loss weight lambda_bc.

The composite loss is  L = L_PDE + lambda_bc * L_BC + L_IC,  with L_BC the
mean squared boundary error over both edges. The default lambda_bc = 2 is
what the original code did implicitly (sum of the two per-edge means).
Uniform weighting is a known failure mode of PINNs (Wang, Teng & Perdikaris,
2021), so this script retrains the frozen Burgers configuration (seed 0,
Adam 8000 epochs + 1000 fixed-batch L-BFGS iterations) for a few weights and
measures each against the exact Cole-Hopf solution on a 256 x 100 grid.

Choosing the weight by that error would use the answer to tune the method.
The selection here uses only quantities a PINN user has without the exact
solution: on a fresh regular grid (201 x 101 interior points, 101 boundary
times per edge, 201 initial points; none of them training points) it
computes the unweighted validation loss

    V = mean r_PDE² + mean u(±1, t)² + mean (u(x, 0) + sin(pi x))²,

with the same weight 1 on each term whatever lambda_bc was used in
training, picks the lambda_bc with the smallest V, and reports the exact
error of that one choice (`selected`). The exact errors of the other weights
are kept in the output for transparency, not for selection.

Usage:
    python -m scripts.bc_weight_sweep                       # 0.5 1 2 4
    python -m scripts.bc_weight_sweep --weights 2 --out results/bc_sweep/w2.json
    python -m scripts.bc_weight_sweep --summarize           # select over results/bc_sweep/*.json
    python -m scripts.bc_weight_sweep --weights 1 --epochs 200 --lbfgs 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.physics import NU_BURGERS, burgers_residual
from src.reference import burgers_exact
from src.train import train_burgers


def evaluate(model, x: np.ndarray, t: np.ndarray, exact: np.ndarray) -> float:
    X, T = np.meshgrid(x, t)
    model.eval()
    with torch.no_grad():
        u = model(torch.tensor(X.reshape(-1, 1), dtype=torch.float32),
                  torch.tensor(T.reshape(-1, 1), dtype=torch.float32)).numpy().reshape(X.shape)
    return float(np.linalg.norm(u - exact) / np.linalg.norm(exact))


def validation_loss(model, nx: int = 201, nt: int = 101, chunk: int = 5000) -> dict:
    """Unweighted residual + boundary + initial loss on a fresh regular grid."""
    model.eval()
    xs = np.linspace(-1, 1, nx + 2)[1:-1]
    ts = np.linspace(0, 1, nt + 2)[1:-1]
    X, T = np.meshgrid(xs, ts)
    xf, tf = X.reshape(-1, 1), T.reshape(-1, 1)
    sq = []
    for k in range(0, len(xf), chunk):
        x = torch.tensor(xf[k:k + chunk], dtype=torch.float32, requires_grad=True)
        t = torch.tensor(tf[k:k + chunk], dtype=torch.float32, requires_grad=True)
        r = burgers_residual(model(x, t), x, t, nu=NU_BURGERS)
        sq.append((r.detach() ** 2).numpy().ravel())
    pde = float(np.concatenate(sq).mean())
    with torch.no_grad():
        tb = torch.linspace(0, 1, 101).reshape(-1, 1)
        bc = float(0.5 * ((model(-torch.ones_like(tb), tb) ** 2).mean()
                          + (model(torch.ones_like(tb), tb) ** 2).mean()))
        xi = torch.linspace(-1, 1, 201).reshape(-1, 1)
        ic = float(((model(xi, torch.zeros_like(xi)) + torch.sin(torch.pi * xi)) ** 2).mean())
    return {"pde": pde, "bc": bc, "ic": ic, "total": pde + bc + ic}


def main(weights: list[float], epochs: int, lbfgs: int, seed: int, out: str) -> None:
    x = np.linspace(-1, 1, 256)
    t = np.linspace(0, 1, 100)
    exact = burgers_exact(x, t)
    runs = []
    for w in weights:
        t0 = time.time()
        model, _, _ = train_burgers(epochs=epochs, lbfgs_iter=lbfgs, seed=seed, lambda_bc=w)
        val = validation_loss(model)
        err = evaluate(model, x, t, exact)
        runs.append({"lambda_bc": w, "validation_loss": val,
                     "relative_l2_error_vs_exact": err, "seconds": time.time() - t0})
        print(f"  lambda_bc = {w:<4}  validation loss = {val['total']:.3e} "
              f"(pde {val['pde']:.2e}, bc {val['bc']:.2e}, ic {val['ic']:.2e})", flush=True)
        best = _write(out, seed, epochs, lbfgs, runs)
    print(f"  selected lambda_bc = {best['lambda_bc']} -> rel L2 vs exact "
          f"{best['relative_l2_error_vs_exact']:.4%}", flush=True)
    print(f"[ok] {out}")


def _write(out: str, seed: int, epochs: int, lbfgs: int, runs: list[dict]) -> dict:
    best = min(runs, key=lambda r: r["validation_loss"]["total"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"seed": seed, "epochs": epochs, "lbfgs_iter": lbfgs,
                   "selection": "smallest unweighted validation loss on a fresh grid",
                   "selected": {"lambda_bc": best["lambda_bc"],
                                "relative_l2_error_vs_exact": best["relative_l2_error_vs_exact"]},
                   "runs": runs}, fh, indent=2)
    return best


def summarize(pattern: str = "results/bc_sweep/*.json",
              out: str = "results/bc_weight_sweep.json") -> dict:
    """Select lambda_bc over weights trained in separate processes (one JSON each)."""
    import glob

    parts = [json.load(open(f)) for f in sorted(glob.glob(pattern))]
    settings = {(d["seed"], d["epochs"], d["lbfgs_iter"]) for d in parts}
    if len(settings) != 1:
        raise SystemExit(f"expected one training setting in {pattern}, found {sorted(settings)}")
    runs = sorted((r for d in parts for r in d["runs"]), key=lambda r: r["lambda_bc"])
    best = _write(out, *settings.pop(), runs)
    for r in runs:
        print(f"  lambda_bc = {r['lambda_bc']:<4}  validation loss = "
              f"{r['validation_loss']['total']:.3e}")
    print(f"  selected lambda_bc = {best['lambda_bc']} -> rel L2 vs exact "
          f"{best['relative_l2_error_vs_exact']:.4%}")
    print(f"[ok] {out}")
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--epochs", type=int, default=8000)
    p.add_argument("--lbfgs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/bc_weight_sweep.json")
    p.add_argument("--summarize", action="store_true",
                   help="select over the per-weight runs in results/bc_sweep/*.json")
    a = p.parse_args()
    if a.summarize:
        summarize()
    else:
        main(a.weights, a.epochs, a.lbfgs, a.seed, a.out)
