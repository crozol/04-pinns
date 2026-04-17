"""Red neuronal PINN: u_theta(x, t). Requiere activaciones suaves para usar autograd de orden 2."""

from __future__ import annotations

import torch
from torch import nn


class PINN(nn.Module):
    def __init__(self, layers: tuple[int, ...] = (2, 64, 64, 64, 1)):
        super().__init__()
        modules: list[nn.Module] = []
        for i in range(len(layers) - 2):
            modules += [nn.Linear(layers[i], layers[i + 1]), nn.Tanh()]
        modules += [nn.Linear(layers[-2], layers[-1])]
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)
