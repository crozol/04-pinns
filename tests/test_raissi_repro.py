"""scripts/raissi_repro.py follows the authors' recipe, and the reason it uses L-BFGS-B."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch
from scipy.optimize import _lbfgsb_py, minimize

from scripts import raissi_repro

SMALL = dict(n_u=20, n_f=100)


@pytest.fixture(autouse=True)
def fixed_reference(monkeypatch):
    # the recipe tests need a reference, not the exact one (tested in test_physics.py)
    monkeypatch.setattr(raissi_repro, "burgers_exact", lambda x, t: np.ones((len(t), len(x))))


def test_authors_options_leave_gtol_to_scipys_default():
    assert raissi_repro.AUTHORS_OPTIONS == {"maxiter": 50_000, "maxfun": 50_000, "maxcor": 50,
                                            "maxls": 50, "ftol": np.finfo(float).eps}
    default = inspect.signature(_lbfgsb_py._minimize_lbfgsb).parameters["gtol"].default
    assert default == 1e-5          # what the authors' call (no gtol) ran with


def test_network_matches_the_authors_code():
    net = raissi_repro.RaissiNet()
    assert sum(p.numel() for p in net.parameters()) == 3_021
    seen = []
    net.linears[0].register_forward_pre_hook(lambda m, inp: seen.append(inp[0]))
    net(torch.tensor([[-1.0], [1.0]]), torch.tensor([[0.0], [0.99]]))
    assert torch.allclose(seen[0], torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))   # domain -> [-1, 1]
    # the authors' domain is their data grid's bounding box: t = 0, 0.01, ..., 0.99
    assert raissi_repro.UB == (1.0, 0.99)
    xu, _, xf = raissi_repro.training_points(100, 1_000, seed=0)
    assert xu[:, 1].max() <= 0.99 and xf[:, 1].max() <= 0.99
    raissi_repro.xavier_truncated_(net)
    for lin in net.linears:
        sd = np.sqrt(2.0 / (lin.in_features + lin.out_features))
        assert lin.weight.abs().max() <= 2 * sd and not lin.bias.any()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_run_trains_in_the_requested_precision(dtype, tmp_path):
    pb = raissi_repro.Problem(0, dtype, **SMALL)
    assert all(p.dtype == raissi_repro.DTYPES[dtype] for p in pb.params)
    f, g = pb.value_and_grad(pb.get_x())
    assert g.dtype == np.float64                         # what L-BFGS-B receives
    r = raissi_repro.run(0, dtype, options=dict(raissi_repro.AUTHORS_OPTIONS, maxiter=20),
                         problem=pb, params_dir=str(tmp_path))
    assert r["dtype"] == dtype and r["final_loss"] < f and r["n_params"] == 3_021
    v = r["validation_loss"]                             # Section 8's V, in the run's precision
    assert np.isclose(v["total"], v["pde"] + v["bc"] + v["ic"]) and v["total"] > 0
    saved = np.load(tmp_path / "seed00.npy")             # the optimizer's final vector
    pb2 = raissi_repro.Problem(1, dtype, **SMALL)
    pb2.set_x(saved)                                     # reloads the same network
    assert np.array_equal(pb2.get_x(), pb.get_x())


def test_main_skips_seeds_already_in_the_output(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(raissi_repro, "run",
                        lambda s, dtype, params_dir=None: calls.append(s) or {"seed": s, "relative_l2_error": 1.0})
    out = tmp_path / "p.json"
    raissi_repro.main([0], "float32", str(out))
    raissi_repro.main([0, 1], "float32", str(out))
    assert calls == [0, 1]
    assert [r["seed"] for r in json.loads(out.read_text())["runs"]] == [0, 1]


def test_lbfgs_memory_test_reports_both_optimizers(tmp_path):
    from scripts import lbfgs_memory_test

    r = lbfgs_memory_test.main(seed=0, evals=8, out=str(tmp_path / "m.json"),
                               options=dict(raissi_repro.AUTHORS_OPTIONS, maxiter=15),
                               out_fig=str(tmp_path / "m.png"), **SMALL)
    assert (tmp_path / "m.png").stat().st_size > 0
    for name in ("torch", "lbfgsb"):
        assert r[name]["evals"] >= 8 and r[name]["iterations"] > 0
        assert r[name]["pairs_kept_by_lbfgsb_test"] <= r[name]["iterations"]
        assert r[name]["loss_end"] <= r[name]["loss_start"]
        assert r[name]["loss_curve"][0] == [0, r[name]["loss_start"]]
    assert json.loads((tmp_path / "m.json").read_text())["seed"] == 0


def _quadratic(scale: float, n: int = 50):
    a = np.linspace(1.0, 100.0, n)
    return lambda x: (0.5 * scale * float(a @ x**2), scale * a * x), np.ones(n)


def test_torch_memory_test_depends_on_the_scale_of_the_loss_and_lbfgsb_does_not():
    """The same quadratic, multiplied by 2^-40 (~1e-12, exact in floating point), 10 iterations.
    torch.optim.LBFGS keeps a pair only if y's > 1e-10: it stores none and does not progress.
    L-BFGS-B keeps it if y's > eps * (-g's): it stores the same pairs and reaches the same loss
    in the loss's own units."""
    def torch_lbfgs(scale):
        fun, x0 = _quadratic(scale)
        x = torch.tensor(x0, requires_grad=True)
        opt = torch.optim.LBFGS([x], lr=1.0, max_iter=10, history_size=10,
                                tolerance_grad=0.0, tolerance_change=0.0,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            f, g = fun(x.detach().numpy())
            x.grad = torch.from_numpy(g)
            return torch.tensor(f)

        opt.step(closure)
        return len(opt.state[x]["old_dirs"]), fun(x.detach().numpy())[0] / scale

    def lbfgsb(scale):
        fun, x0 = _quadratic(scale)
        res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 10, "maxcor": 10, "ftol": 0.0, "gtol": 0.0})
        return res.hess_inv.sk.shape[0], res.fun / scale

    (mem1, f1), (mem2, f2) = torch_lbfgs(1.0), torch_lbfgs(2.0**-40)
    assert mem1 > 0 and mem2 == 0
    assert f2 > 100 * f1
    (mem1, f1), (mem2, f2) = lbfgsb(1.0), lbfgsb(2.0**-40)
    assert mem1 == mem2 > 0
    assert f2 == pytest.approx(f1, rel=1e-5)
