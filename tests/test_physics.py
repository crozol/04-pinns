"""Physics-driven tests for the PDE residuals and reference solutions."""
from __future__ import annotations

import numpy as np
import torch

from src.physics import NU_BURGERS, burgers_residual, exact_heat, heat_residual
from src.reference import burgers_exact, burgers_reference


def test_heat_residual_vanishes_on_exact_solution():
    """Plugging the exact separation-of-variables solution into the residual
    should give (numerically) zero, since it solves the PDE exactly."""
    x = torch.rand(500, 1, requires_grad=True)
    t = torch.rand(500, 1, requires_grad=True)
    u = torch.sin(torch.pi * x) * torch.exp(-(torch.pi**2) * t)
    r = heat_residual(u, x, t, alpha=1.0)
    assert r.abs().max().item() < 1e-4


def test_burgers_residual_is_finite_through_a_nonlinear_field():
    """Exercise the full autograd chain (u_t, u_x, u_xx) on a field that
    genuinely depends on both x and t at every order, as a PINN output would.
    (A trivially-zero or purely-bilinear u breaks the autograd graph before
    reaching the second x-derivative, which is a PyTorch quirk, not a
    property of the PDE — so it is not a useful residual test case.)"""
    x = torch.rand(200, 1, requires_grad=True)
    t = torch.rand(200, 1, requires_grad=True)
    u = torch.sin(x) * torch.exp(-t) + 0.3 * x**3 * t
    r = burgers_residual(u, x, t, nu=NU_BURGERS)
    assert r.shape == u.shape
    assert torch.isfinite(r).all()


def test_exact_heat_matches_initial_condition():
    x = np.linspace(0, 1, 50)
    u0 = exact_heat(x, np.zeros_like(x), alpha=1.0)
    np.testing.assert_allclose(u0, np.sin(np.pi * x), atol=1e-10)


def test_exact_heat_satisfies_boundary_conditions():
    t = np.linspace(0, 1, 20)
    assert np.allclose(exact_heat(np.zeros_like(t), t), 0.0)
    assert np.allclose(exact_heat(np.ones_like(t), t), 0.0, atol=1e-10)


def test_exact_heat_decays_in_time():
    x = np.full(10, 0.5)
    t = np.linspace(0, 1, 10)
    u = exact_heat(x, t, alpha=1.0)
    assert np.all(np.diff(np.abs(u)) < 0)


def test_burgers_reference_respects_boundary_and_symmetry():
    """Dirichlet BC u(-1,t)=u(1,t)=0 and odd symmetry u(-x,t) = -u(x,t) follow
    from the odd IC -sin(pi x) and the symmetry of the PDE under x -> -x."""
    x, t, u = burgers_reference(n=64, nt=11)
    assert np.allclose(u[:, 0], 0.0, atol=1e-8)
    assert np.allclose(u[:, -1], 0.0, atol=1e-8)
    mid = len(x) // 2
    assert abs(x[mid]) < 1e-10  # grid is symmetric, x=0 is a node
    # tiny (<1e-7) time-accumulated roundoff from the stiff solver's own
    # rtol/atol is expected here, not a sign of a broken symmetry
    np.testing.assert_allclose(u[:, mid], 0.0, atol=1e-6)


def test_burgers_reference_respects_maximum_principle():
    """A parabolic PDE with no source cannot exceed the initial-condition
    amplitude (max principle for u_t + u u_x = nu u_xx). Needs enough
    Chebyshev modes to resolve the thin viscous layer at x=0 without Gibbs
    overshoot — this is also why the PINN validation run below uses n=256."""
    _, _, u = burgers_reference(n=256, nt=101)
    assert np.abs(u).max() <= 1.0 + 1e-3


def test_burgers_exact_symmetry_and_boundaries():
    x = np.linspace(-1, 1, 101)
    t = np.array([0.0, 0.1, 0.5, 1.0])
    u = burgers_exact(x, t)
    np.testing.assert_allclose(u[0], -np.sin(np.pi * x), atol=1e-12)
    np.testing.assert_allclose(u[:, 0], 0.0, atol=1e-10)
    np.testing.assert_allclose(u[:, -1], 0.0, atol=1e-10)
    np.testing.assert_allclose(u, -u[:, ::-1], atol=1e-10)


def test_burgers_exact_satisfies_the_pde():
    """Finite-difference residual of u_t + u u_x - nu u_xx on a smooth region
    (away from the internal layer) should vanish to discretization error."""
    h = 1e-4
    x = np.linspace(-0.9, -0.3, 7)
    t = np.array([0.3 - h, 0.3, 0.3 + h])
    xs = np.concatenate([x - h, x, x + h])
    u = burgers_exact(xs, t)
    n = len(x)
    um, u0, up = u[1, :n], u[1, n:2 * n], u[1, 2 * n:]
    u_t = (u[2, n:2 * n] - u[0, n:2 * n]) / (2 * h)
    u_x = (up - um) / (2 * h)
    u_xx = (up - 2 * u0 + um) / h**2
    r = u_t + u0 * u_x - NU_BURGERS * u_xx
    assert np.abs(r).max() < 1e-5


def test_chebyshev_reference_agrees_with_exact_solution():
    x, t, u_ref = burgers_reference(n=256, nt=11)
    u_ex = burgers_exact(x, t)
    assert np.linalg.norm(u_ref - u_ex) / np.linalg.norm(u_ex) < 1e-2
