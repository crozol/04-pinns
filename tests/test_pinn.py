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


def test_validation_loss_is_the_unweighted_sum_on_a_fresh_grid():
    """The selection criterion of scripts/bc_weight_sweep.py: finite terms
    whose sum is the total, with the IC term sin² on a (numerically) zero model."""
    import numpy as np
    import torch

    from scripts.bc_weight_sweep import validation_loss

    class Zero(torch.nn.Module):          # cubic so that u_xx keeps a graph
        def forward(self, x, t):
            return 1e-20 * (x ** 3 + t)

    v = validation_loss(Zero(), nx=21, nt=11)
    assert np.isclose(v["total"], v["pde"] + v["bc"] + v["ic"])
    assert v["pde"] < 1e-30 and v["bc"] < 1e-30
    xi = np.linspace(-1, 1, 201)
    assert np.isclose(v["ic"], np.mean(np.sin(np.pi * xi) ** 2), rtol=1e-5)


def test_bc_weight_summary_selects_over_separate_runs(tmp_path):
    """Weights trained in separate processes are selected together, by validation loss only."""
    import json

    import pytest

    from scripts.bc_weight_sweep import summarize

    def part(w, val, err, epochs=8000):
        run = {"lambda_bc": w, "validation_loss": {"total": val},
               "relative_l2_error_vs_exact": err, "seconds": 1.0}
        (tmp_path / f"w{w}.json").write_text(json.dumps(
            {"seed": 0, "epochs": epochs, "lbfgs_iter": 1000, "runs": [run]}))

    part(0.5, 3e-4, 0.008)                # smallest exact error, not the smallest validation loss
    part(2.0, 1e-4, 0.019)
    out = tmp_path / "summary.json"
    best = summarize(str(tmp_path / "w*.json"), str(out))
    assert best["lambda_bc"] == 2.0
    assert [r["lambda_bc"] for r in json.loads(out.read_text())["runs"]] == [0.5, 2.0]
    part(4.0, 5e-5, 0.017, epochs=200)    # a run with other settings is not merged silently
    with pytest.raises(SystemExit):
        summarize(str(tmp_path / "w*.json"), str(out))
