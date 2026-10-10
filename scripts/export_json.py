"""Export space-time fields + loss curves to JSON for the portfolio website.

Output is consumed by Plotly charts on
``website/projects/04-pinns.html``.

Usage:
    python -m scripts.export_json
    python -m scripts.export_json --out ../website/assets/data/04-pinns.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _downsample_grid(x: np.ndarray, t: np.ndarray, field: np.ndarray, max_points: int = 80):
    nx = max(1, len(x) // max_points) or 1
    nt = max(1, len(t) // max_points) or 1
    return x[::nx], t[::nt], field[::nt, ::nx]


def _round_grid(field: np.ndarray, ndigits: int = 4) -> list:
    return [[round(float(v), ndigits) for v in row] for row in field]


def _thin_loss(values: np.ndarray, max_points: int = 400) -> list:
    if len(values) <= max_points:
        return [round(float(v), 6) for v in values]
    step = max(1, len(values) // max_points)
    return [round(float(v), 6) for v in values[::step]]


def _snapshot_slices(x: np.ndarray, t: np.ndarray, pred: np.ndarray, ref: np.ndarray,
                      fractions=(0.0, 1 / 3, 2 / 3, 1.0)) -> dict:
    """Full-resolution u(x) curves (PINN vs. reference) at a few fixed times —
    a direct, legible view of fit quality, independent of the downsampled
    heatmap grid used elsewhere."""
    idx = [int(round(f * (len(t) - 1))) for f in fractions]
    return {
        "x": [round(float(v), 5) for v in x],
        "t_values": [round(float(t[i]), 4) for i in idx],
        "pred": [[round(float(v), 5) for v in pred[i]] for i in idx],
        "ref": [[round(float(v), 5) for v in ref[i]] for i in idx],
    }


def _field_bundle(npz_path: Path, pred_key: str, ref_key: str) -> dict:
    data = np.load(npz_path)
    x, t = data["x"], data["t"]
    pred_full, ref_full = data[pred_key], data[ref_key]
    x_ds, t_ds, pred_ds = _downsample_grid(x, t, pred_full)
    _, _, ref_ds = _downsample_grid(x, t, ref_full)
    diff_ds = pred_ds - ref_ds

    adam_loss = data["adam_loss"]
    lbfgs_loss = data["lbfgs_loss"]
    loss = {
        "adam_step": list(range(0, len(adam_loss), max(1, len(adam_loss) // 400))),
        "adam": _thin_loss(adam_loss),
    }
    if len(lbfgs_loss) > 0:
        offset = len(adam_loss)
        loss["lbfgs_step"] = [offset + i for i in range(0, len(lbfgs_loss), max(1, len(lbfgs_loss) // 400))]
        loss["lbfgs"] = _thin_loss(lbfgs_loss)

    return {
        "x": [round(float(v), 5) for v in x_ds],
        "t": [round(float(v), 5) for v in t_ds],
        "pred": _round_grid(pred_ds),
        "ref": _round_grid(ref_ds),
        "diff": _round_grid(diff_ds),
        "slices": _snapshot_slices(x, t, pred_full, ref_full),
        "loss": loss,
    }


def main(out_path: str = "../website/assets/data/04-pinns.json", results_dir: str = "results") -> None:
    root = Path(results_dir)
    with open(root / "metrics.json") as fh:
        metrics = json.load(fh)

    bundle = {
        "metrics": metrics,
        "heat": _field_bundle(root / "heat_arrays.npz", "u_pred", "u_exact"),
        "burgers": _field_bundle(root / "burgers_arrays.npz", "u_pred", "u_ref"),
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    size_kb = out.stat().st_size / 1024
    print(f"[ok] {out}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="../website/assets/data/04-pinns.json")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(out_path=args.out, results_dir=args.results_dir)
