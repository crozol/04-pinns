"""Full pipeline: train both PINNs, validate against their reference
solutions, and save space-time figures + a metrics.json summary.

Usage: python main.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.physics import exact_heat
from src.reference import burgers_reference
from src.train import train_burgers, train_heat

RESULTS = Path("results")
FIGURES = Path("figures")


def _predict_grid(model, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    X, T = np.meshgrid(x, t)
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X.reshape(-1, 1), dtype=torch.float32)
        tt = torch.tensor(T.reshape(-1, 1), dtype=torch.float32)
        u = model(xt, tt).numpy().reshape(X.shape)
    return u


def _relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def _space_time_figure(x, t, u_pred, u_ref, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    vmax = max(np.abs(u_pred).max(), np.abs(u_ref).max())
    for ax, field, name in [
        (axes[0], u_ref, "Reference"),
        (axes[1], u_pred, "PINN"),
        (axes[2], u_pred - u_ref, "PINN - Reference"),
    ]:
        im = ax.imshow(
            field, extent=[x.min(), x.max(), t.max(), t.min()], aspect="auto",
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        )
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("t" if ax is axes[0] else "")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(title)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_heat() -> dict:
    model, history = train_heat(
        n_collocation=10_000, n_boundary=200, n_initial=200, epochs=5000, lr=1e-3, lbfgs_iter=500
    )
    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 1, 101)
    X, T = np.meshgrid(x, t)
    u_pred = _predict_grid(model, x, t)
    u_exact = exact_heat(X, T, alpha=1.0)
    err = _relative_l2(u_pred, u_exact)
    _space_time_figure(x, t, u_pred, u_exact, f"Heat equation (rel. L2 error = {err:.2%})", FIGURES / "heat_comparison.png")
    RESULTS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), RESULTS / "heat_model.pt")
    return {"relative_l2_error": err, "final_loss": history[-1]}


def run_burgers() -> dict:
    # L-BFGS refinement matters a lot more here than for heat: Adam alone
    # plateaus around 9% relative L2 error (the thin viscous layer at x=0 is
    # hard to fit with first-order updates); L-BFGS brings it to ~2%.
    model, history = train_burgers(
        n_collocation=10_000, n_boundary=200, n_initial=200, epochs=8000, lr=1e-3, lbfgs_iter=1000
    )
    x, t, u_ref = burgers_reference(n=256, nt=201)
    u_pred = _predict_grid(model, x, t)
    err = _relative_l2(u_pred, u_ref)
    _space_time_figure(x, t, u_pred, u_ref, f"Burgers equation (rel. L2 error = {err:.2%})", FIGURES / "burgers_comparison.png")
    RESULTS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), RESULTS / "burgers_model.pt")
    return {"relative_l2_error": err, "final_loss": history[-1]}


if __name__ == "__main__":
    metrics = {"heat": run_heat(), "burgers": run_burgers()}
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
