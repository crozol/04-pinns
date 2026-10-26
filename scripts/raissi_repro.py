"""Reproduce the Burgers setup of Raissi, Perdikaris & Karniadakis (2019), §3.1.

The paper reports a relative L2 error of 6.7e-4 for:

* a tanh network with 8 hidden layers of 20 neurons (3 021 parameters).
  The paper's "9 layers" counts the output layer: 8 hidden layers is the
  only depth that gives the 3 021 parameters of the authors' code
  (7 → 2 601, 9 → 3 441). An earlier version of this script used 9 hidden
  layers; its three seeds are kept in results/raissi_repro_9hidden.json,
* N_u = 100 initial/boundary points drawn from the t = 0 and x = ±1 edges,
* N_f = 10 000 collocation points from a Latin hypercube,
* loss = MSE_u + MSE_f, minimised by full-batch L-BFGS to convergence
  (their TensorFlow code calls SciPy's L-BFGS-B with maxiter = 50 000).

This script follows that recipe in float64 PyTorch and measures the error
against the exact Cole-Hopf solution on the paper's 256 x 100 grid.

With 20 seeds the quantity reported is the failure rate — how many runs of
the recipe stay above 10^-3 — rather than a median, because one run in three
stalled in the first, 3-seed attempt. Each seed takes about 8 h on one CPU
core; seeds are independent, so they are run as parallel processes, each
writing its own JSON, and combined with --summarize.

Every 200 L-BFGS iterations each seed saves its model and optimiser state to
results/raissi/checkpoints/, so an interrupted process resumes where it stopped
when the same command is launched again. The optimiser then runs in chunks of
200 iterations that carry over the evaluation budget and apply the stopping
tests of one uninterrupted call at every chunk boundary: a chunked or resumed
run follows the same trajectory as a single opt.step(closure)
(tests/test_raissi_repro.py).

Usage:
    python -m scripts.raissi_repro --seeds 0 8 16 --out results/raissi/p0.json
    python -m scripts.raissi_repro --summarize            # results/raissi/*.json
    python -m scripts.raissi_repro --seeds 0 --max-iter 200 --out results/smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from src.physics import NU_BURGERS, burgers_residual
from src.pinn import PINN
from src.reference import burgers_exact

LAYERS = (2,) + (20,) * 8 + (1,)


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


def _save_checkpoint(path: Path, model, opt, progress: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "progress": progress}, tmp)
    os.replace(tmp, path)                             # never leave a half-written checkpoint


def run(seed: int, max_iter: int, n_u: int = 100, n_f: int = 10_000,
        checkpoint_dir: str | None = None, checkpoint_every: int = 200) -> dict:
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

    max_eval = int(1.25 * max_iter)
    opt = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=max_iter, max_eval=max_eval,
        tolerance_grad=1e-12, tolerance_change=np.finfo(float).eps,
        history_size=50, line_search_fn="strong_wolfe",
    )
    # evals counts the evaluations as one uninterrupted opt.step(closure) would
    progress = {"evals": 0, "seconds": 0.0, "final_loss": None, "done": False}
    ckpt = Path(checkpoint_dir) / f"seed{seed}_iter{max_iter}.pt" if checkpoint_dir else None
    if ckpt is not None and ckpt.exists():
        saved = torch.load(ckpt, weights_only=False)  # our own file; the state holds a NumPy scalar
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        progress.update(saved["progress"])
        print(f"  seed {seed}: resuming from {ckpt} ({progress['evals']} evals done)", flush=True)

    def loss_fn():
        loss_u = ((model(x_u, t_u) - u_u) ** 2).mean()
        r = burgers_residual(model(x_f, t_f), x_f, t_f, nu=NU_BURGERS)
        return loss_u + (r ** 2).mean()

    n_calls = [0]

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        n_calls[0] += 1
        progress["final_loss"] = loss.item()
        return loss

    # L-BFGS in chunks of checkpoint_every iterations. Each chunk gets the evaluation budget
    # that one uninterrupted call would have left (it also sets the line-search budget), and
    # the stopping tests that call applies between iterations are applied at every chunk
    # boundary, so the chunked run follows the same trajectory as a single opt.step(closure).
    group = opt.param_groups[0]
    state = opt.state[group["params"][0]]
    t0 = time.time() - progress["seconds"]
    while not progress["done"]:
        it0, calls0 = state.get("n_iter", 0), n_calls[0]
        group["max_iter"] = min(checkpoint_every, max_iter - it0)
        group["max_eval"] = max_eval - max(progress["evals"] - 1, 0)
        opt.step(closure)
        # every chunk after the first starts by re-evaluating the point where the last one stopped
        progress["evals"] += n_calls[0] - calls0 - (1 if progress["evals"] else 0)
        it = state["n_iter"]
        if it - it0 < group["max_iter"] or it >= max_iter or progress["evals"] >= max_eval:
            progress["done"] = True
        else:
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            g = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
            progress["done"] = bool(
                g.abs().max() <= group["tolerance_grad"]
                or state["d"].mul(state["t"]).abs().max() <= group["tolerance_change"]
                or abs(loss.item() - state["prev_loss"]) < group["tolerance_change"])
        progress["seconds"] = time.time() - t0
        if ckpt is not None:
            _save_checkpoint(ckpt, model, opt, progress)

    x = np.linspace(-1, 1, 256)
    t = np.linspace(0, 1, 100)
    X, Tt = np.meshgrid(x, t)
    with torch.no_grad():
        u = model(T(X.reshape(-1, 1)), T(Tt.reshape(-1, 1))).numpy().reshape(X.shape)
    exact = burgers_exact(x, t)
    err = float(np.linalg.norm(u - exact) / np.linalg.norm(exact))
    print(f"  seed {seed}: rel L2 vs exact = {err:.3e}  final loss {progress['final_loss']:.3e}  "
          f"{progress['evals']} evals  {progress['seconds']:.0f} s", flush=True)
    return {"seed": seed, "relative_l2_error": err, "final_loss": progress["final_loss"],
            "n_evals": progress["evals"], "seconds": progress["seconds"],
            "n_params": sum(p.numel() for p in model.parameters())}


def main(seeds: list[int], max_iter: int, out: str, checkpoint_dir: str | None,
         checkpoint_every: int) -> None:
    runs = []
    for s in seeds:
        runs.append(run(s, max_iter, checkpoint_dir=checkpoint_dir,
                        checkpoint_every=checkpoint_every))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        errs = np.array([r["relative_l2_error"] for r in runs])
        with open(out, "w") as fh:
            json.dump({"max_iter": max_iter, "paper_error": 6.7e-4,
                       "mean": float(errs.mean()),
                       "std": float(errs.std(ddof=1)) if len(errs) > 1 else None,
                       "runs": runs}, fh, indent=2)
    print(f"[ok] {out}")


def summarize(pattern: str = "results/raissi/*.json", out: str = "results/raissi_repro.json",
              threshold: float = 1e-3) -> None:
    import glob

    runs = sorted((r for f in glob.glob(pattern) for r in json.load(open(f))["runs"]),
                  key=lambda r: r["seed"])
    errs = np.array([r["relative_l2_error"] for r in runs])
    fails = int((errs > threshold).sum())
    summary = {"n_seeds": len(runs), "paper_error": 6.7e-4, "threshold": threshold,
               "n_above_threshold": fails, "median": float(np.median(errs)),
               "quartiles": [float(np.quantile(errs, 0.25)), float(np.quantile(errs, 0.75))],
               "min": float(errs.min()), "max": float(errs.max()),
               "n_params": runs[0]["n_params"] if runs else None, "runs": runs}
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"{len(runs)} seeds: median {summary['median']:.3e}, IQR {summary['quartiles']}, "
          f"{fails}/{len(runs)} above {threshold:g}; min {errs.min():.2e}, max {errs.max():.2e}")
    print(f"[ok] {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    p.add_argument("--max-iter", type=int, default=50_000)
    p.add_argument("--out", default="results/raissi/all.json")
    p.add_argument("--checkpoint-dir", default="results/raissi/checkpoints",
                   help="where each seed saves its model and L-BFGS state ('' to disable)")
    p.add_argument("--checkpoint-every", type=int, default=200, help="L-BFGS iterations")
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    if a.summarize:
        summarize()
    else:
        main(a.seeds, a.max_iter, a.out, a.checkpoint_dir or None, a.checkpoint_every)
