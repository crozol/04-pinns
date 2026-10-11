"""Multi-seed robustness check for the PINNs: training is stochastic
(collocation points resampled every Adam step, and the L-BFGS fixed batch is
itself randomly sampled), so a single frozen run could in principle be a
lucky draw. This retrains from scratch across several seeds and reports the
relative L2 error as mean +/- std, mirroring the n=5 seed error bars used
for the Neural ODEs project (02).

Usage:
    python -m scripts.seed_sweep --equation burgers --seeds 5
    python -m scripts.seed_sweep --equation heat --seeds 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.physics import exact_heat
from src.reference import burgers_reference
from src.train import train_burgers, train_heat


def _predict_grid(model, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    X, T = np.meshgrid(x, t)
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X.reshape(-1, 1), dtype=torch.float32)
        tt = torch.tensor(T.reshape(-1, 1), dtype=torch.float32)
        u = model(xt, tt).numpy().reshape(X.shape)
    return u


def _run_burgers_seed(seed: int):
    x, t, u_ref = burgers_reference(n=256, nt=201)
    model, adam_hist, lbfgs_hist = train_burgers(
        n_collocation=10_000, n_boundary=200, n_initial=200,
        epochs=8000, lr=1e-3, lbfgs_iter=1000, seed=seed,
    )
    u_pred = _predict_grid(model, x, t)
    err = float(np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref))
    return err, adam_hist, lbfgs_hist


def _run_heat_seed(seed: int):
    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 1, 101)
    X, T = np.meshgrid(x, t)
    u_ref = exact_heat(X, T, alpha=1.0)
    model, adam_hist, lbfgs_hist = train_heat(
        n_collocation=10_000, n_boundary=200, n_initial=200,
        epochs=5000, lr=1e-3, lbfgs_iter=500, seed=seed,
    )
    u_pred = _predict_grid(model, x, t)
    err = float(np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref))
    return err, adam_hist, lbfgs_hist


RUNNERS = {"burgers": _run_burgers_seed, "heat": _run_heat_seed}


def main(equation: str, n_seeds: int, out_path: str) -> None:
    runner = RUNNERS[equation]

    results = []
    for seed in range(n_seeds):
        t0 = time.time()
        err, adam_hist, lbfgs_hist = runner(seed)
        elapsed = time.time() - t0
        print(f"[{equation} seed {seed}] relative L2 error = {err:.4%}  "
              f"(adam_final={adam_hist[-1]:.3e}, lbfgs_final={lbfgs_hist[-1]:.3e}, "
              f"{elapsed:.0f}s)")
        results.append({
            "seed": seed,
            "relative_l2_error": err,
            "adam_final_loss": adam_hist[-1],
            "lbfgs_final_loss": lbfgs_hist[-1],
            "elapsed_s": round(elapsed, 1),
        })

    errors = np.array([r["relative_l2_error"] for r in results])
    summary = {
        "equation": equation,
        "n_seeds": n_seeds,
        "mean_relative_l2_error": float(errors.mean()),
        "std_relative_l2_error": float(errors.std(ddof=1)) if n_seeds > 1 else 0.0,
        "min_relative_l2_error": float(errors.min()),
        "max_relative_l2_error": float(errors.max()),
        "runs": results,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[ok] {out}")
    print(f"mean +/- std = {summary['mean_relative_l2_error']:.4%} "
          f"+/- {summary['std_relative_l2_error']:.4%} "
          f"(range {summary['min_relative_l2_error']:.4%}-{summary['max_relative_l2_error']:.4%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equation", choices=["heat", "burgers"], default="burgers")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out_path = args.out or f"results/{args.equation}_seed_sweep.json"
    main(equation=args.equation, n_seeds=args.seeds, out_path=out_path)
