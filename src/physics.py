"""PDE residuals computed with torch.autograd.grad, plus the reference solutions
used to validate each PINN."""

from __future__ import annotations

import numpy as np
import torch

PI = np.pi
NU_BURGERS = 0.01 / PI


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def heat_residual(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Residual of u_t - alpha * u_xx = 0."""
    u_t = _grad(u, t)
    u_x = _grad(u, x)
    u_xx = _grad(u_x, x)
    return u_t - alpha * u_xx


def burgers_residual(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, nu: float = NU_BURGERS) -> torch.Tensor:
    """Residual of u_t + u * u_x - nu * u_xx = 0."""
    u_t = _grad(u, t)
    u_x = _grad(u, x)
    u_xx = _grad(u_x, x)
    return u_t + u * u_x - nu * u_xx


def exact_heat(x: np.ndarray, t: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Analytical solution on domain x in [0, 1], t in [0, 1] for the IC
    u(x, 0) = sin(pi x) with homogeneous Dirichlet BC u(0, t) = u(1, t) = 0:
    u(x, t) = sin(pi x) * exp(-pi^2 alpha t) (single-mode separation of variables)."""
    return np.sin(PI * x) * np.exp(-(PI**2) * alpha * t)
