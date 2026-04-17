"""Training loop del PINN con loss compuesto (residual + boundary + initial)."""

from __future__ import annotations

import torch
from torch import optim

from .pinn import PINN
from .physics import heat_residual


def train_heat(
    n_collocation: int = 10_000,
    n_boundary: int = 200,
    n_initial: int = 200,
    epochs: int = 5000,
    lr: float = 1e-3,
    device: str = "cpu",
) -> PINN:
    model = PINN((2, 64, 64, 64, 1)).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    def sample_collocation() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(n_collocation, 1, device=device, requires_grad=True)
        t = torch.rand(n_collocation, 1, device=device, requires_grad=True)
        return x, t

    for epoch in range(epochs):
        opt.zero_grad()
        x, t = sample_collocation()
        u = model(x, t)
        r = heat_residual(u, x, t, alpha=1.0)
        loss_pde = (r**2).mean()
        # TODO: añadir términos de boundary e initial según el problema concreto
        loss = loss_pde
        loss.backward()
        opt.step()
        if epoch % 500 == 0:
            print(f"[epoch {epoch:5d}] loss_pde = {loss_pde.item():.6e}")

    return model


if __name__ == "__main__":
    raise NotImplementedError("Completar condiciones iniciales/frontera para un problema concreto.")
