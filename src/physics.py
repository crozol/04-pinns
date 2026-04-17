"""Residuales PDE calculados con torch.autograd.grad."""

from __future__ import annotations

import torch


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def heat_residual(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Residual de u_t - alpha * u_xx = 0."""
    u_t = _grad(u, t)
    u_x = _grad(u, x)
    u_xx = _grad(u_x, x)
    return u_t - alpha * u_xx


def burgers_residual(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, nu: float = 0.01 / 3.14159) -> torch.Tensor:
    """Residual de u_t + u * u_x - nu * u_xx = 0."""
    u_t = _grad(u, t)
    u_x = _grad(u, x)
    u_xx = _grad(u_x, x)
    return u_t + u * u_x - nu * u_xx
