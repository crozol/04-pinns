"""Sensitivity of the Burgers PINN to the boundary-loss weight lambda_bc.

The composite loss is  L = L_PDE + lambda_bc * L_BC + L_IC,  with L_BC the
mean squared boundary error over both edges. The default lambda_bc = 2 is
what the original code did implicitly (sum of the two per-edge means).
Uniform weighting is a known failure mode of PINNs (Wang, Teng & Perdikaris,
2021), so this script retrains the frozen Burgers configuration (seed 0,
Adam 8000 epochs + 1000 fixed-batch L-BFGS iterations) for a few weights and
measures each against the exact Cole-Hopf solution on a 256 x 100 grid.

Usage:
    python -m scripts.bc_weight_sweep                       # 0.5 1 2 4
    python -m scripts.bc_weight_sweep --weights 1 --epochs 200 --lbfgs 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.reference import burgers_exact
from src.train import train_burgers


def evaluate(model, x: np.ndarray, t: np.ndarray, exact: np.ndarray) -> float:
    X, T = np.meshgrid(x, t)
    model.eval()
    with torch.no_grad():
        u = model(torch.tensor(X.reshape(-1, 1), dtype=torch.float32),
                  torch.tensor(T.reshape(-1, 1), dtype=torch.float32)).numpy().reshape(X.shape)
    return float(np.linalg.norm(u - exact) / np.linalg.norm(exact))


def main(weights: list[float], epochs: int, lbfgs: int, seed: int, out: str) -> None:
    x = np.linspace(-1, 1, 256)
    t = np.linspace(0, 1, 100)
    exact = burgers_exact(x, t)
    runs = []
    for w in weights:
        t0 = time.time()
        model, _, _ = train_burgers(epochs=epochs, lbfgs_iter=lbfgs, seed=seed, lambda_bc=w)
        err = evaluate(model, x, t, exact)
        runs.append({"lambda_bc": w, "relative_l2_error_vs_exact": err,
                     "seconds": time.time() - t0})
        print(f"  lambda_bc = {w:<4}  rel L2 vs exact = {err:.4%}", flush=True)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump({"seed": seed, "epochs": epochs, "lbfgs_iter": lbfgs, "runs": runs},
                      fh, indent=2)
    print(f"[ok] {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--epochs", type=int, default=8000)
    p.add_argument("--lbfgs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/bc_weight_sweep.json")
    a = p.parse_args()
    main(a.weights, a.epochs, a.lbfgs, a.seed, a.out)
