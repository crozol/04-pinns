"""Why torch.optim.LBFGS stalls on the Burgers benchmark and SciPy's L-BFGS-B does not.

1. Runs the recipe of scripts/raissi_repro.py in float64 for one seed until L-BFGS-B stops on
   max|g| <= 1e-5. The loss is then near 1e-6, the regime where the earlier torch-based runs
   stalled.
2. From that point it gives each optimizer the same budget of loss evaluations:
   torch   torch.optim.LBFGS configured as the earlier raissi_repro.py (history 50, strong Wolfe,
           tolerance_grad 1e-12, tolerance_change eps), starting with an empty memory;
   lbfgsb  SciPy's L-BFGS-B with the authors' options and gtol = 0, so only the budget stops it.
3. For every accepted step it computes s = x_{k+1} - x_k and y = g_{k+1} - g_k and counts the
   pairs each memory test keeps: torch's y's > 1e-10 and L-BFGS-B's y's > eps * (-g's). It also
   reports the loss decrease per evaluation and the error against the exact solution before and
   after, i.e. what continuing past the recipe's stopping point is worth.

Usage:
    python -m scripts.lbfgs_memory_test                    # seed 0, 600 evaluations each
    python -m scripts.lbfgs_memory_test --seed 3 --evals 1000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from scripts.raissi_repro import AUTHORS_OPTIONS, Problem, run

EPS = float(np.finfo(float).eps)


def _flat_grad(pb: Problem) -> np.ndarray:
    return torch.cat([p.grad.reshape(-1) for p in pb.params]).numpy().astype(np.float64)


def torch_arm(pb: Problem, x0: np.ndarray, budget: int):
    """torch.optim.LBFGS one iteration per step() call (same trajectory as one call); the first
    evaluation of each call is the point the iteration starts from and is not counted."""
    pb.set_x(x0)
    opt = torch.optim.LBFGS(pb.params, lr=1.0, max_iter=1, max_eval=10**9,
                            tolerance_grad=1e-12, tolerance_change=EPS,
                            history_size=50, line_search_fn="strong_wolfe")
    iterates, evals, first = [], [0], [True]

    def closure():
        opt.zero_grad()
        loss = pb.loss()
        loss.backward()
        if first[0]:
            iterates.append((pb.get_x(), loss.item(), _flat_grad(pb)))
            first[0] = False
        else:
            evals[0] += 1
        return loss

    while evals[0] < budget and len(iterates) <= budget:
        first[0] = True
        opt.step(closure)
    f, g = pb.value_and_grad(pb.get_x())
    iterates.append((pb.get_x(), f, g))
    return iterates, evals[0]


def lbfgsb_arm(pb: Problem, x0: np.ndarray, budget: int):
    cache = {}

    def fun(v):
        f, g = pb.value_and_grad(v)
        cache[v.tobytes()] = (v.copy(), f, g)
        return f, g

    f0, g0 = pb.value_and_grad(x0)
    iterates = [(x0.copy(), f0, g0)]

    def callback(intermediate_result):
        iterates.append(cache[intermediate_result.x.tobytes()])

    res = minimize(fun, x0, jac=True, method="L-BFGS-B", callback=callback,
                   options=dict(AUTHORS_OPTIONS, maxfun=budget + 1, gtol=0.0))
    pb.set_x(res.x)
    if not np.array_equal(iterates[-1][0], res.x):
        iterates.append((res.x.copy(),) + pb.value_and_grad(res.x))
    return iterates, int(res.nfev) - 1                # the evaluation at x0 is not counted


def report(name: str, pb: Problem, iterates: list, evals: int, error_start: float,
           seconds: float) -> dict:
    ys, gs = [], []
    for (xa, _, ga), (xb, _, gb) in zip(iterates[:-1], iterates[1:]):
        s, y = xb - xa, gb - ga
        ys.append(float(y @ s))
        gs.append(float(-ga @ s))
    ys, gs = np.array(ys), np.array(gs)
    pos = gs > 0
    f = [it[1] for it in iterates]
    return {"optimizer": name, "iterations": len(ys), "evals": evals, "seconds": seconds,
            "loss_start": f[0], "loss_end": f[-1],
            "loss_drop_per_eval": (f[0] - f[-1]) / max(evals, 1),
            "error_start": error_start, "error_end": pb.relative_error(),
            "pairs_kept_by_torch_test": int((ys > 1e-10).sum()),
            "pairs_kept_by_lbfgsb_test": int((ys > EPS * gs).sum()),
            "median_ys": float(np.median(ys)) if len(ys) else None,
            "median_ys_over_minus_gs": float(np.median(ys[pos] / gs[pos])) if pos.any() else None}


def main(seed: int = 0, evals: int = 600, out: str = "results/lbfgs_memory_test.json",
         options: dict | None = None, n_u: int = 100, n_f: int = 10_000) -> dict:
    pb = Problem(seed, "float64", n_u, n_f)
    start = run(seed, "float64", options=options, problem=pb)
    x0 = pb.get_x()
    result = {"seed": seed, "budget_evals": evals, "start": start}
    for name, arm in (("torch", torch_arm), ("lbfgsb", lbfgsb_arm)):
        t0 = time.time()
        iterates, n = arm(pb, x0, evals)
        result[name] = report(name, pb, iterates, n, start["relative_l2_error"], time.time() - t0)
        r = result[name]
        print(f"{name:6s}: {r['iterations']} it / {r['evals']} evals; pairs kept "
              f"torch-test {r['pairs_kept_by_torch_test']}, L-BFGS-B-test "
              f"{r['pairs_kept_by_lbfgsb_test']}; median y's {r['median_ys']:.2e}, "
              f"y's/(-g's) {r['median_ys_over_minus_gs']:.3f}; loss {r['loss_start']:.4e} -> "
              f"{r['loss_end']:.4e} ({r['loss_drop_per_eval']:.2e}/eval); error "
              f"{r['error_start']:.3e} -> {r['error_end']:.3e}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"[ok] {out}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evals", type=int, default=600)
    p.add_argument("--out", default="results/lbfgs_memory_test.json")
    a = p.parse_args()
    main(a.seed, a.evals, a.out)
