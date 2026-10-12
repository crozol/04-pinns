"""Heat-equation error resolved in time, for the frozen run.

u(x, t) = sin(pi x) exp(-pi^2 t) decays by a factor exp(-pi^2) ~ 5e-5 over
t in [0, 1], so a single relative L2 error over the whole space-time grid
is dominated by the early times. This script reads the frozen run's arrays
(results/heat_arrays.npz, written by main.py) and reports, for each time
slice, the solution amplitude, the PINN's maximum absolute error and the
relative L2 error, plus a figure of all three against t.

Usage:
    python -m scripts.heat_time_error
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(arrays: str = "results/heat_arrays.npz",
         out_json: str = "results/heat_time_error.json",
         out_fig: str = "figures/heat_error_vs_time.png") -> None:
    d = np.load(arrays)
    t, u_pred, u_exact = d["t"], d["u_pred"], d["u_exact"]
    amp = np.abs(u_exact).max(axis=1)
    abs_err = np.abs(u_pred - u_exact).max(axis=1)
    rel = np.linalg.norm(u_pred - u_exact, axis=1) / np.linalg.norm(u_exact, axis=1)
    global_rel = float(np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact))
    first_10pct = float(t[np.argmax(rel > 0.10)]) if (rel > 0.10).any() else None

    payload = {
        "global_relative_l2": global_rel,
        "first_t_with_relative_error_above_10pct": first_10pct,
        "max_abs_error_overall": float(abs_err.max()),
        "slices": [{"t": float(ti), "amplitude": float(a), "max_abs_error": float(e),
                    "relative_l2": float(r)} for ti, a, e, r in zip(t, amp, abs_err, rel)],
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.semilogy(t, amp, label=r"solution amplitude $\max_x |u|$", lw=2)
    ax.semilogy(t, abs_err, label=r"PINN max abs. error $\max_x |u_\theta - u|$", lw=2)
    ax.semilogy(t, rel, label=r"relative $L^2$ error per slice", lw=2, ls="--")
    ax.axhline(0.1, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("t")
    ax.set_title(f"Heat equation, error resolved in time (global rel. L2 = {global_rel:.2%})")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=140)
    plt.close(fig)

    for ti, a, e, r in list(zip(t, amp, abs_err, rel))[::10]:
        print(f"  t = {ti:.2f}  amplitude {a:.2e}  max abs err {e:.2e}  rel L2 {r:.2%}")
    print(f"  global rel L2 {global_rel:.2%}; relative error first exceeds 10% at t = {first_10pct}")
    print(f"[ok] {out_json}  {out_fig}")


if __name__ == "__main__":
    main()
