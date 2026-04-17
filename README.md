# Physics-Informed Neural Networks (PINNs)

Solving partial differential equations (PDEs) with neural networks whose loss includes the PDE residual (Raissi, Perdikaris & Karniadakis, 2019).

## Motivation

A PINN is a neural network `u_θ(x, t)` whose loss combines:

1. **Initial / boundary conditions** (observed or imposed data).
2. **PDE residual** evaluated at collocation points in the domain, computed with `autograd`.

The model learns to satisfy the PDE in the classical sense, without needing a mesh like a traditional numerical method. This project implements:

- **1D heat equation**: a minimal case to validate against the analytical solution.
- **1D Burgers equation**: classic nonlinear case, with emergent discontinuity.

## Stack

- Python 3.11+
- PyTorch 2.x (autograd for partial derivatives)
- NumPy, SciPy (reference solutions)
- Matplotlib

## Structure

```
04-pinns/
├── README.md
├── requirements.txt
├── src/
│   ├── pinn.py           # u_theta(x, t) neural network
│   ├── physics.py        # PDE residuals (heat, Burgers)
│   └── train.py          # Training loop with composite loss
└── notebooks/
```

## Roadmap

- [ ] Implement MLP with `tanh` activation (twice-differentiable).
- [ ] `residual_heat(u, x, t)` function using `torch.autograd.grad`.
- [ ] `residual_burgers(u, x, t, nu)` function.
- [ ] Training loop sampling collocation points and enforcing boundary / initial conditions.
- [ ] Validation against analytical solution (heat) and reference numerical solution (Burgers).
- [ ] Space-time figures + blog post.

## Expected results

Relative L² error < 1% against the analytical heat equation solution, and < 5% against the high-resolution numerical reference for Burgers.
