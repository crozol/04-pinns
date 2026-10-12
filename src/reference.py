"""Reference solutions used to validate the PINNs.

Heat equation has a closed-form analytical solution (see physics.exact_heat).
For Burgers, `burgers_exact` evaluates the exact Cole-Hopf solution and is
the ground truth every reported error is measured against. `burgers_reference`
is an independent numerical cross-check: it integrates the semi-discretized
PDE with a Chebyshev spectral method in x (as in Raissi et al. 2019) and a
stiff implicit time integrator, and differs from the exact solution by 0.40%
relative L2 (max. 0.035, inside the viscous layer). A plain central-difference finite-difference
discretization was tried first and produced spurious oscillations: the
viscous layer that forms near x=0 (nu = 0.01/pi is small) is thinner than the
grid spacing can resolve without numerical dispersion. The spectral method
differentiates via a global polynomial, so it resolves the layer without
needing a prohibitively fine grid.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .physics import NU_BURGERS


def _cheb(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Chebyshev differentiation matrix and Gauss-Lobatto points on [-1, 1]
    (Trefethen, *Spectral Methods in MATLAB*, 2000). x[0] = 1, x[n] = -1."""
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.ones(n + 1)
    c[0] = 2.0
    c[n] = 2.0
    c *= (-1.0) ** np.arange(n + 1)
    dx = x[:, None] - x[None, :]
    d = np.outer(c, 1.0 / c) / (dx + np.eye(n + 1))
    d -= np.diag(d.sum(axis=1))
    return d, x


def burgers_reference(
    n: int = 256, nt: int = 201, nu: float = NU_BURGERS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (x, t, U) with U[i, j] = u(x[j], t[i]). x is ascending on [-1, 1]."""
    d, x_desc = _cheb(n)
    d2 = d @ d
    interior = slice(1, n)
    u0 = -np.sin(np.pi * x_desc)
    t_eval = np.linspace(0.0, 1.0, nt)

    def rhs(_t: float, u_int: np.ndarray) -> np.ndarray:
        u = np.zeros(n + 1)
        u[interior] = u_int
        du = d @ u
        d2u = d2 @ u
        return (-u * du + nu * d2u)[interior]

    sol = solve_ivp(
        rhs, (0.0, 1.0), u0[interior], t_eval=t_eval, method="Radau", rtol=1e-9, atol=1e-11
    )
    if not sol.success:
        raise RuntimeError(f"Burgers reference solve failed: {sol.message}")

    u_desc = np.zeros((nt, n + 1))
    u_desc[:, interior] = sol.y.T

    order = np.argsort(x_desc)
    x = x_desc[order]
    u = u_desc[:, order]
    return x, t_eval, u


def burgers_exact(
    x: np.ndarray, t: np.ndarray, nu: float = NU_BURGERS, n_quad: int = 8001
) -> np.ndarray:
    """Exact solution via the Cole-Hopf transformation (Basdevant et al., 1986).

    With u = -2 nu (ln phi)_x the equation becomes the heat equation for phi,
    and for u(x, 0) = -sin(pi x) (whose 2-periodic extension satisfies the
    boundary conditions) the solution is the ratio of two Gaussian integrals:

        u(x, t) = ∫ (eta / t) w(eta) d eta / ∫ w(eta) d eta,
        w(eta)  = exp( -cos(pi (x - eta)) / (2 pi nu) - eta^2 / (4 nu t) ).

    Both integrals are evaluated by the trapezoid rule on a fine eta grid,
    with the exponent shifted by its maximum for numerical stability.
    Returns U[i, j] = u(x[j], t[i]).
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.empty((t.size, x.size))
    for i, ti in enumerate(t):
        if ti == 0.0:
            out[i] = -np.sin(np.pi * x)
            continue
        half = min(1.0, 12.0 * np.sqrt(4.0 * nu * ti))   # Gaussian is negligible beyond
        eta = np.linspace(-half, half, n_quad)
        expo = (-np.cos(np.pi * (x[:, None] - eta[None, :])) / (2.0 * np.pi * nu)
                - eta[None, :] ** 2 / (4.0 * nu * ti))
        w = np.exp(expo - expo.max(axis=1, keepdims=True))
        out[i] = np.trapezoid(eta * w, eta, axis=1) / (ti * np.trapezoid(w, eta, axis=1))
    return out
