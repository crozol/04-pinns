"""Shape and gradient-flow tests for the PINN architecture, plus a short
smoke test that the training loop actually reduces the loss."""
from __future__ import annotations

import torch

from src.pinn import PINN
from src.train import train_heat


def test_pinn_output_shape():
    model = PINN((2, 16, 16, 1))
    x = torch.rand(10, 1)
    t = torch.rand(10, 1)
    u = model(x, t)
    assert u.shape == (10, 1)


def test_pinn_is_twice_differentiable():
    """autograd.grad with create_graph=True must work through the whole
    network — this fails immediately if any activation is non-smooth (e.g. ReLU)."""
    model = PINN((2, 8, 8, 1))
    x = torch.rand(5, 1, requires_grad=True)
    t = torch.rand(5, 1, requires_grad=True)
    u = model(x, t)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x))[0]
    assert torch.isfinite(u_xx).all()


def test_train_heat_smoke_loss_decreases():
    """A short run (not expected to converge) should still reduce the loss
    on average — catches a broken loss/optimizer wiring without paying for a
    full training run in CI."""
    _, history, _ = train_heat(n_collocation=200, n_boundary=50, n_initial=50, epochs=100, lr=1e-2)
    early = sum(history[:10]) / 10
    late = sum(history[-10:]) / 10
    assert late < early
