"""How accurate are the Burgers reference solutions?

Three checks, written to results/reference_convergence.json:

1. Cole-Hopf quadrature: `burgers_exact` on the 256 x 100 evaluation grid
   with n, 2n, 4n, 8n trapezoid nodes (1001 ... 8001) against 16001 nodes,
   plus the integration window widened from 12 to 16 Gaussian widths. This
   is the error floor under every Burgers number in the README.
2. An independent published value: the maximum slope |u_x(0, t)| and the
   time at which it occurs, which Basdevant et al. (1986) give as 152.00516
   at t = 1.6037/pi. u_x is computed from the same Cole-Hopf integrals
   differentiated under the integral sign, not by finite differences.
3. The Chebyshev cross-check `burgers_reference` at N = 128, 256, 512
   against the exact solution on its own Gauss-Lobatto nodes (no
   interpolation), with the location of the largest error. Gauss-Lobatto
   nodes are sparsest at x = 0 (spacing ~ pi/N), exactly where the internal
   layer sits, so the 0.40% at N = 256 is a resolution error.

Runs on one core in about ten minutes (most of it the N = 512 solve).

Usage:
    python -m scripts.reference_convergence
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from src.physics import NU_BURGERS
from src.reference import burgers_exact, burgers_reference

BASDEVANT_MAX_SLOPE = 152.00516
BASDEVANT_PI_T = 1.6037


def _exact_window(x: np.ndarray, t: np.ndarray, n_quad: int, widths: float,
                  nu: float = NU_BURGERS) -> np.ndarray:
    """burgers_exact with the half-window set to `widths` Gaussian widths."""
    out = np.empty((t.size, x.size))
    for i, ti in enumerate(t):
        if ti == 0.0:
            out[i] = -np.sin(np.pi * x)
            continue
        half = min(1.0, widths * np.sqrt(4.0 * nu * ti))
        eta = np.linspace(-half, half, n_quad)
        expo = (-np.cos(np.pi * (x[:, None] - eta[None, :])) / (2.0 * np.pi * nu)
                - eta[None, :] ** 2 / (4.0 * nu * ti))
        w = np.exp(expo - expo.max(axis=1, keepdims=True))
        out[i] = np.trapezoid(eta * w, eta, axis=1) / (ti * np.trapezoid(w, eta, axis=1))
    return out


def slope_at_origin(t: float, nu: float = NU_BURGERS, n_quad: int = 16001) -> float:
    """u_x(0, t) from the Cole-Hopf integrals: u = N/D with
    N = ∫ (eta/t) w, D = ∫ w and ∂_x w = w sin(pi (x - eta)) / (2 nu)."""
    half = min(1.0, 12.0 * np.sqrt(4.0 * nu * t))
    eta = np.linspace(-half, half, n_quad)
    expo = -np.cos(np.pi * eta) / (2.0 * np.pi * nu) - eta**2 / (4.0 * nu * t)
    w = np.exp(expo - expo.max())
    s = np.sin(-np.pi * eta) / (2.0 * nu)
    n_, d_ = np.trapezoid(eta * w, eta) / t, np.trapezoid(w, eta)
    n_x, d_x = np.trapezoid(eta * w * s, eta) / t, np.trapezoid(w * s, eta)
    return float((n_x * d_ - n_ * d_x) / d_**2)


def max_slope() -> tuple[float, float]:
    """(max |u_x(0, t)|, pi * t at the maximum)."""
    r = minimize_scalar(slope_at_origin, bounds=(0.4, 0.6), method="bounded",
                        options={"xatol": 1e-10})
    return float(-r.fun), float(np.pi * r.x)


def _diff(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    return {"relative_l2": float(np.linalg.norm(d) / np.linalg.norm(b)),
            "max_abs": float(np.abs(d).max())}


def main(out: str = "results/reference_convergence.json") -> None:
    x, t = np.linspace(-1, 1, 256), np.linspace(0, 1, 100)
    finest = burgers_exact(x, t, n_quad=16001)
    quadrature = [{"n_quad": n, **_diff(burgers_exact(x, t, n_quad=n), finest)}
                  for n in (1001, 2001, 4001, 8001)]
    for q in quadrature:
        print(f"  Cole-Hopf n_quad {q['n_quad']:>5} vs 16001: rel L2 {q['relative_l2']:.1e}"
              f"  max {q['max_abs']:.1e}")
    window = _diff(burgers_exact(x, t), _exact_window(x, t, 16001, 16.0))
    print(f"  window 12 vs 16 widths: rel L2 {window['relative_l2']:.1e}  max {window['max_abs']:.1e}")

    slope, pi_t = max_slope()
    print(f"  max |u_x(0, t)| = {slope:.5f} at t = {pi_t:.4f}/pi "
          f"(Basdevant et al.: {BASDEVANT_MAX_SLOPE} at {BASDEVANT_PI_T}/pi)")

    chebyshev = []
    for n in (128, 256, 512):
        xc, tc, uc = burgers_reference(n=n, nt=201)
        ue = burgers_exact(xc, tc)
        i, j = np.unravel_index(np.abs(uc - ue).argmax(), uc.shape)
        chebyshev.append({"n": n, **_diff(uc, ue), "center_spacing": float(np.pi / n),
                          "argmax_x": float(xc[j]), "argmax_t": float(tc[i])})
        c = chebyshev[-1]
        print(f"  Chebyshev N {n:>3}: rel L2 {c['relative_l2']:.2%}  max {c['max_abs']:.1e}"
              f" at (x, t) = ({c['argmax_x']:+.4f}, {c['argmax_t']:.3f}); spacing at x=0 {c['center_spacing']:.4f}")

    payload = {
        "grid": "256 x 100 (Raissi et al. 2019)",
        "cole_hopf_quadrature_vs_16001": quadrature,
        "cole_hopf_window_12_vs_16_widths": window,
        "max_slope": {"value": slope, "pi_t": pi_t,
                      "basdevant_value": BASDEVANT_MAX_SLOPE, "basdevant_pi_t": BASDEVANT_PI_T},
        "chebyshev_vs_exact": chebyshev,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
