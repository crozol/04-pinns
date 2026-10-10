"""High-resolution numerical reference solutions used to validate the PINNs.

Heat equation has a closed-form analytical solution (see physics.exact_heat).
Burgers equation does not for this IC/BC, so we integrate the semi-discretized
PDE with a Chebyshev spectral method in x (as in Raissi et al. 2019) and a
stiff implicit time integrator. A plain central-difference finite-difference
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
