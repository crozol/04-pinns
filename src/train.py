"""Training loops for the heat-equation and Burgers-equation PINNs, each with a
composite loss: PDE residual (collocation points) + boundary condition + initial
condition. Adam uses freshly resampled points every step (stochastic
mini-batches); L-BFGS instead needs a single fixed batch, because its line
search assumes the objective is deterministic from one evaluation to the
next — resampled points would make the line search fail."""

from __future__ import annotations

import torch
from torch import optim

from .pinn import PINN
from .physics import NU_BURGERS, burgers_residual, heat_residual


def _optimize(model: PINN, loss_fn, epochs: int, lr: float, log_every: int) -> list[float]:
    opt = optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss, parts = loss_fn()
        loss.backward()
        opt.step()
        history.append(loss.item())
        if epoch % log_every == 0 or epoch == epochs - 1:
            parts_str = " ".join(f"{k}={v:.3e}" for k, v in parts.items())
            print(f"[epoch {epoch:6d}] loss={loss.item():.6e}  {parts_str}")
    return history


def refine_lbfgs(model: PINN, loss_fn, max_iter: int = 500) -> float:
    """Second-order fine-tuning after Adam, as in Raissi et al. (2019) —
    Adam gets close quickly, L-BFGS squeezes out the remaining error that
    Adam's first-order updates converge to only very slowly. `loss_fn` must
    be deterministic (fixed points), not resample on every call."""
    opt = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=max_iter,
        tolerance_grad=1e-10, tolerance_change=1e-12,
        history_size=50, line_search_fn="strong_wolfe",
    )
    last_loss = {"value": float("nan")}

    def closure():
        opt.zero_grad()
        loss, parts = loss_fn()
        loss.backward()
        last_loss["value"] = loss.item()
        return loss

    opt.step(closure)
    print(f"[L-BFGS] final loss = {last_loss['value']:.6e}")
    return last_loss["value"]


def _heat_loss_fn(model, x_f, t_f, x_b0, x_b1, t_b, x_i, t_i, alpha):
    u_f = model(x_f, t_f)
    r = heat_residual(u_f, x_f, t_f, alpha=alpha)
    loss_pde = (r**2).mean()

    u_b0 = model(x_b0, t_b)
    u_b1 = model(x_b1, t_b)
    loss_bc = (u_b0**2).mean() + (u_b1**2).mean()

    t_i = torch.zeros_like(x_i)
    u_i = model(x_i, t_i)
    target_i = torch.sin(torch.pi * x_i)
    loss_ic = ((u_i - target_i) ** 2).mean()

    loss = loss_pde + loss_bc + loss_ic
    return loss, {"pde": loss_pde.item(), "bc": loss_bc.item(), "ic": loss_ic.item()}


def _sample_heat_batch(n_collocation, n_boundary, n_initial, device):
    x_f = torch.rand(n_collocation, 1, device=device, requires_grad=True)
    t_f = torch.rand(n_collocation, 1, device=device, requires_grad=True)
    t_b = torch.rand(n_boundary, 1, device=device)
    x_b0 = torch.zeros_like(t_b)
    x_b1 = torch.ones_like(t_b)
    x_i = torch.rand(n_initial, 1, device=device)
    t_i = torch.zeros_like(x_i)
    return x_f, t_f, x_b0, x_b1, t_b, x_i, t_i


def train_heat(
    n_collocation: int = 10_000,
    n_boundary: int = 200,
    n_initial: int = 200,
    epochs: int = 5000,
    lr: float = 1e-3,
    alpha: float = 1.0,
    device: str = "cpu",
    seed: int = 0,
    lbfgs_iter: int = 0,
) -> tuple[PINN, list[float]]:
    """Domain x in [0, 1], t in [0, 1]. IC u(x, 0) = sin(pi x). BC u(0, t) = u(1, t) = 0."""
    torch.manual_seed(seed)
    model = PINN((2, 64, 64, 64, 1)).to(device)

    def loss_fn():
        batch = _sample_heat_batch(n_collocation, n_boundary, n_initial, device)
        return _heat_loss_fn(model, *batch, alpha=alpha)

    history = _optimize(model, loss_fn, epochs, lr, log_every=500)

    if lbfgs_iter > 0:
        fixed_batch = _sample_heat_batch(4 * n_collocation, 4 * n_boundary, 4 * n_initial, device)
        refine_lbfgs(model, lambda: _heat_loss_fn(model, *fixed_batch, alpha=alpha), max_iter=lbfgs_iter)
    return model, history


def _burgers_loss_fn(model, x_f, t_f, x_bm1, x_bp1, t_b, x_i, nu):
    u_f = model(x_f, t_f)
    r = burgers_residual(u_f, x_f, t_f, nu=nu)
    loss_pde = (r**2).mean()

    u_bm1 = model(x_bm1, t_b)
    u_bp1 = model(x_bp1, t_b)
    loss_bc = (u_bm1**2).mean() + (u_bp1**2).mean()

    t_i = torch.zeros_like(x_i)
    u_i = model(x_i, t_i)
    target_i = -torch.sin(torch.pi * x_i)
    loss_ic = ((u_i - target_i) ** 2).mean()

    loss = loss_pde + loss_bc + loss_ic
    return loss, {"pde": loss_pde.item(), "bc": loss_bc.item(), "ic": loss_ic.item()}


def _sample_burgers_batch(n_collocation, n_boundary, n_initial, device):
    x_f = (torch.rand(n_collocation, 1, device=device) * 2 - 1).requires_grad_(True)
    t_f = torch.rand(n_collocation, 1, device=device, requires_grad=True)
    t_b = torch.rand(n_boundary, 1, device=device)
    x_bm1 = -torch.ones_like(t_b)
    x_bp1 = torch.ones_like(t_b)
    x_i = torch.rand(n_initial, 1, device=device) * 2 - 1
    return x_f, t_f, x_bm1, x_bp1, t_b, x_i


def train_burgers(
    n_collocation: int = 10_000,
    n_boundary: int = 200,
    n_initial: int = 200,
    epochs: int = 8000,
    lr: float = 1e-3,
    nu: float = NU_BURGERS,
    device: str = "cpu",
    seed: int = 0,
    lbfgs_iter: int = 0,
) -> tuple[PINN, list[float]]:
    """Domain x in [-1, 1], t in [0, 1]. IC u(x, 0) = -sin(pi x). BC u(-1, t) = u(1, t) = 0.
    Classic setup from Raissi, Perdikaris & Karniadakis (2019), nu = 0.01 / pi."""
    torch.manual_seed(seed)
    model = PINN((2, 64, 64, 64, 1)).to(device)

    def loss_fn():
        batch = _sample_burgers_batch(n_collocation, n_boundary, n_initial, device)
        return _burgers_loss_fn(model, *batch, nu=nu)

    history = _optimize(model, loss_fn, epochs, lr, log_every=500)

    if lbfgs_iter > 0:
        fixed_batch = _sample_burgers_batch(4 * n_collocation, 4 * n_boundary, 4 * n_initial, device)
        refine_lbfgs(model, lambda: _burgers_loss_fn(model, *fixed_batch, nu=nu), max_iter=lbfgs_iter)
    return model, history
