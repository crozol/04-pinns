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
   after, i.e. what continuing past the recipe's stopping point is worth, and the loss against
   the number of evaluations for both (loss_curve: [evaluations, loss] at every accepted point).
4. Locates where torch's memory freezes: the last step whose pair passes torch's test, after
   which no pair does. It reports the loss and the number of evaluations at that point, and the
   loss decrease per evaluation of both optimizers before and after it, over the same
   evaluations, plus L-BFGS-B's rate from the loss at which torch froze ("freeze" in the JSON).
5. Plots both (figures/lbfgs_memory_test.png): the loss against evaluations, and y's at every
   accepted step against torch's threshold 1e-10, with the freeze marked on both panels.

Usage:
    python -m scripts.lbfgs_memory_test                    # seed 0, 600 evaluations each
    python -m scripts.lbfgs_memory_test --seed 3 --evals 1000
    python -m scripts.lbfgs_memory_test --plot-only        # redo 4 and 5 from the JSON
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
    iterates, evals, first, curve = [], [0], [True], []

    def closure():
        opt.zero_grad()
        loss = pb.loss()
        loss.backward()
        if first[0]:
            iterates.append((pb.get_x(), loss.item(), _flat_grad(pb)))
            curve.append((evals[0], loss.item()))
            first[0] = False
        else:
            evals[0] += 1
        return loss

    while evals[0] < budget and len(iterates) <= budget:
        first[0] = True
        opt.step(closure)
    f, g = pb.value_and_grad(pb.get_x())
    iterates.append((pb.get_x(), f, g))
    curve.append((evals[0], f))
    return iterates, evals[0], curve


def lbfgsb_arm(pb: Problem, x0: np.ndarray, budget: int):
    cache, calls = {}, [0]

    def fun(v):
        f, g = pb.value_and_grad(v)
        cache[v.tobytes()] = (v.copy(), f, g)
        calls[0] += 1
        return f, g

    f0, g0 = pb.value_and_grad(x0)
    iterates, curve = [(x0.copy(), f0, g0)], [(0, f0)]

    def callback(intermediate_result):
        iterates.append(cache[intermediate_result.x.tobytes()])
        curve.append((calls[0] - 1, iterates[-1][1]))     # the evaluation at x0 is not counted

    res = minimize(fun, x0, jac=True, method="L-BFGS-B", callback=callback,
                   options=dict(AUTHORS_OPTIONS, maxfun=budget + 1, gtol=0.0))
    pb.set_x(res.x)
    if not np.array_equal(iterates[-1][0], res.x):
        iterates.append((res.x.copy(),) + pb.value_and_grad(res.x))
        curve.append((int(res.nfev) - 1, iterates[-1][1]))
    return iterates, int(res.nfev) - 1, curve


def report(name: str, pb: Problem, iterates: list, evals: int, curve: list,
           error_start: float, seconds: float) -> dict:
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
            "median_ys_over_minus_gs": float(np.median(ys[pos] / gs[pos])) if pos.any() else None,
            "loss_curve": [[int(e), float(v)] for e, v in curve],
            "ys": ys.tolist(), "minus_gs": gs.tolist()}


def _loss_at(curve: np.ndarray, evals: float) -> float:
    """Loss after ``evals`` evaluations, interpolated linearly between accepted points."""
    return float(np.interp(evals, curve[:, 0], curve[:, 1]))


def _rate(curve: np.ndarray, e0: float, e1: float) -> float | None:
    return None if e1 <= e0 else (_loss_at(curve, e0) - _loss_at(curve, e1)) / (e1 - e0)


def freeze(result: dict) -> dict | None:
    """Where torch's memory freezes, and the loss decrease per evaluation of both optimizers
    before and after that point (None if torch still keeps a pair at its last step). Steps are
    numbered from 1; loss_curve[j] is the point reached after step j."""
    ys = np.array(result["torch"]["ys"], dtype=float)
    kept = np.flatnonzero(ys > 1e-10)
    last = int(kept[-1]) + 1 if len(kept) else 0          # last step whose pair torch keeps
    if last == len(ys):
        return None
    tc = np.array(result["torch"]["loss_curve"], dtype=float)
    lc = np.array(result["lbfgsb"]["loss_curve"], dtype=float)
    e, f = float(tc[last, 0]), float(tc[last, 1])
    end = float(min(tc[-1, 0], lc[-1, 0]))
    after = ys[last:]
    # the evaluation at which L-BFGS-B first reaches the loss at which torch froze
    reach = lc[lc[:, 1] <= f]
    e_same = float(reach[0, 0]) if len(reach) else None
    return {"last_step_kept": last, "steps": int(len(ys)), "evals": e, "loss": f,
            "evals_compared": [e, end],
            "ys_after": [float(after.min()), float(np.median(after)), float(after.max())],
            "rate_before": {"torch": _rate(tc, 0, e), "lbfgsb": _rate(lc, 0, e)},
            "rate_after": {"torch": _rate(tc, e, end), "lbfgsb": _rate(lc, e, end)},
            "lbfgsb_rate_from_same_loss": None if e_same is None else _rate(lc, e_same, end),
            "lbfgsb_evals_to_same_loss": e_same}


def describe_freeze(fr: dict | None) -> str:
    if fr is None:
        return "torch keeps pairs up to its last step: no freeze within the budget"
    msg = (f"torch keeps no pair after step {fr['last_step_kept']} of {fr['steps']} "
           f"(evaluation {fr['evals']:.0f}, loss {fr['loss']:.4e}); y's after it "
           f"{fr['ys_after'][0]:.1e}-{fr['ys_after'][2]:.1e} (median {fr['ys_after'][1]:.1e})")
    b, a = fr["rate_before"], fr["rate_after"]
    if b["torch"] is not None:
        msg += (f"\n  loss decrease per evaluation before: torch {b['torch']:.2e}, "
                f"L-BFGS-B {b['lbfgsb']:.2e}")
    def ratio(r):
        return f" ({r / a['torch']:.0f}x)" if a["torch"] and a["torch"] > 0 else ""

    msg += (f"\n  after (evaluations {fr['evals_compared'][0]:.0f}-{fr['evals_compared'][1]:.0f}): "
            f"torch {a['torch']:.2e}, L-BFGS-B {a['lbfgsb']:.2e}{ratio(a['lbfgsb'])}")
    if fr["lbfgsb_rate_from_same_loss"] is not None:
        msg += (f"\n  L-BFGS-B from the same loss (reached at evaluation "
                f"{fr['lbfgsb_evals_to_same_loss']:.0f}): {fr['lbfgsb_rate_from_same_loss']:.2e}"
                f"{ratio(fr['lbfgsb_rate_from_same_loss'])}")
    return msg


LABELS = {"torch": "torch.optim.LBFGS (earlier version)",
          "lbfgsb": "SciPy L-BFGS-B (authors' optimizer)"}


def plot(result: dict, out_fig: str = "figures/lbfgs_memory_test.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in ("torch", "lbfgsb"):
        r = result[name]
        e, f = np.array(r["loss_curve"]).T
        ax1.plot(e, f, lw=2, label=f"{LABELS[name]}\nerror {r['error_start']:.2e} -> "
                                     f"{r['error_end']:.2e}")
        ys = np.array(r["ys"])
        k = np.arange(1, len(ys) + 1)
        ax2.semilogy(k[ys > 0], ys[ys > 0], ".", ms=3,
                     label=f"{LABELS[name]}: torch keeps {r['pairs_kept_by_torch_test']}"
                           f"/{r['iterations']}, L-BFGS-B keeps "
                           f"{r['pairs_kept_by_lbfgsb_test']}/{r['iterations']}")
    ax1.set_xlabel("loss evaluations after the recipe's stopping point")
    ax1.set_ylabel("training loss")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax2.axhline(1e-10, color="grey", lw=1, ls="--",
                label="torch's threshold: a pair is kept only if y's > 1e-10")
    fr = result.get("freeze")
    if fr is not None and fr["last_step_kept"] > 0:
        ax1.axvline(fr["evals"], color="grey", lw=1, ls=":",
                    label=f"torch keeps no pair from here on (evaluation {fr['evals']:.0f})")
        ax2.axvline(fr["last_step_kept"] + 0.5, color="grey", lw=1, ls=":",
                    label=f"torch's last kept pair: step {fr['last_step_kept']}")
    ax2.set_xlabel("accepted step")
    ax2.set_ylabel("y's of the step")
    # Legends go under the axes so they do not cover the data.
    for ax in (ax1, ax2):
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.suptitle(f"Burgers, seed {result['seed']}: both optimizers from the point where the "
                 f"float64 recipe stops (loss {result['start']['final_loss']:.2e})", fontsize=10)
    fig.tight_layout()
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_fig}")


def main(seed: int = 0, evals: int = 600, out: str = "results/lbfgs_memory_test.json",
         options: dict | None = None, n_u: int = 100, n_f: int = 10_000,
         out_fig: str = "figures/lbfgs_memory_test.png") -> dict:
    pb = Problem(seed, "float64", n_u, n_f)
    start = run(seed, "float64", options=options, problem=pb)
    x0 = pb.get_x()
    result = {"seed": seed, "budget_evals": evals, "start": start}
    for name, arm in (("torch", torch_arm), ("lbfgsb", lbfgsb_arm)):
        t0 = time.time()
        iterates, n, curve = arm(pb, x0, evals)
        result[name] = report(name, pb, iterates, n, curve, start["relative_l2_error"],
                              time.time() - t0)
        r = result[name]
        print(f"{name:6s}: {r['iterations']} it / {r['evals']} evals; pairs kept "
              f"torch-test {r['pairs_kept_by_torch_test']}, L-BFGS-B-test "
              f"{r['pairs_kept_by_lbfgsb_test']}; median y's {r['median_ys']:.2e}, "
              f"y's/(-g's) {r['median_ys_over_minus_gs']:.3f}; loss {r['loss_start']:.4e} -> "
              f"{r['loss_end']:.4e} ({r['loss_drop_per_eval']:.2e}/eval); error "
              f"{r['error_start']:.3e} -> {r['error_end']:.3e}", flush=True)
    return analyse(result, out, out_fig)


def analyse(result: dict, out: str, out_fig: str) -> dict:
    """Locate the freeze, store it in the JSON and draw the figure."""
    result["freeze"] = freeze(result)
    print(describe_freeze(result["freeze"]), flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"[ok] {out}")
    plot(result, out_fig)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evals", type=int, default=600)
    p.add_argument("--out", default="results/lbfgs_memory_test.json")
    p.add_argument("--fig", default="figures/lbfgs_memory_test.png")
    p.add_argument("--plot-only", action="store_true")
    a = p.parse_args()
    if a.plot_only:
        analyse(json.loads(Path(a.out).read_text()), a.out, a.fig)
    else:
        main(a.seed, a.evals, a.out, out_fig=a.fig)
