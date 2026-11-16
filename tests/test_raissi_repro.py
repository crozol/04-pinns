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
    assert r["validation_t_end"] == 0.99                 # on the authors' domain
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
    saved = json.loads((tmp_path / "m.json").read_text())
    assert saved["seed"] == 0 and "freeze" in saved


def test_freeze_is_located_and_both_rates_compared():
    from scripts import lbfgs_memory_test as m

    res = {"torch": {"ys": [1e-9, 1e-9, 1e-12, 1e-12],
                     "loss_curve": [[0, 1.0], [2, 0.8], [4, 0.6], [6, 0.599], [8, 0.598]]},
           "lbfgsb": {"loss_curve": [[0, 1.0], [2, 0.9], [4, 0.8], [6, 0.6], [8, 0.4]]}}
    fr = m.freeze(res)
    assert (fr["last_step_kept"], fr["evals"], fr["loss"]) == (2, 4, 0.6)
    assert fr["rate_before"]["torch"] == pytest.approx(0.1)        # (1.0 - 0.6) / 4
    assert fr["rate_before"]["lbfgsb"] == pytest.approx(0.05)      # (1.0 - 0.8) / 4
    assert fr["rate_after"]["torch"] == pytest.approx(0.0005)      # (0.6 - 0.598) / 4
    assert fr["rate_after"]["lbfgsb"] == pytest.approx(0.1)        # (0.8 - 0.4) / 4
    assert fr["lbfgsb_evals_to_same_loss"] == 6                    # L-BFGS-B reaches 0.6 there
    assert fr["lbfgsb_rate_from_same_loss"] == pytest.approx(0.1)  # (0.6 - 0.4) / 2
    assert "after step 2" in m.describe_freeze(fr)
    res["torch"]["ys"] = [1e-9] * 4                                # keeps a pair at every step
    assert m.freeze(res) is None
    res["torch"]["ys"] = [1e-12] * 4                               # frozen from the first step
    fr = m.freeze(res)
    assert fr["last_step_kept"] == 0 and fr["rate_before"]["torch"] is None


def test_clopper_pearson_matches_its_closed_forms():
    lo, hi = raissi_repro.clopper_pearson(0, 20)
    assert lo == 0 and hi == pytest.approx(1 - 0.025 ** (1 / 20))   # 0.168
    lo, hi = raissi_repro.clopper_pearson(20, 20)
    assert hi == 1 and lo == pytest.approx(0.025 ** (1 / 20))       # 0.832
    lo, hi = raissi_repro.clopper_pearson(10, 20)
    assert lo + hi == pytest.approx(1.0)


def test_best_of_k_enumerates_every_set():
    errs = [1.0, 2.0, 3.0, 4.0]
    b = raissi_repro.best_of_k(errs, errs, 2, paper=1.0, threshold=2.5)    # a perfect score
    assert b["n_sets"] == 6 and b["median"] == 1.5                         # picks 1 1 1 2 2 3
    assert b["share_at_or_below_paper"] == pytest.approx(3 / 6)
    assert b["share_above_threshold"] == pytest.approx(1 / 6)
    w = raissi_repro.best_of_k(errs, [-e for e in errs], 2, paper=1.0, threshold=2.5)  # backwards
    assert w["median"] == 3.5 and w["share_at_or_below_paper"] == 0          # picks 2 3 3 4 4 4


def test_summarize_reports_rates_intervals_and_rankings(tmp_path):
    def runs(errs, v, loss):
        return [{"seed": s, "relative_l2_error": e, "final_loss": l, "max_abs_grad": 1e-5,
                 "n_evals": 100, "stop": "STOP", "n_params": 3021, "validation_t_end": 0.99,
                 "validation_loss": {"total": vv}} for s, (e, vv, l) in enumerate(zip(errs, v, loss))]
    for dtype, errs in (("float32", [2e-3, 3e-3, 1.5e-3, 5e-3]), ("float64", [5e-4, 2e-3, 8e-4, 6e-4])):
        (tmp_path / dtype).mkdir()
        (tmp_path / dtype / "p.json").write_text(json.dumps(
            {"runs": runs(errs, v=errs, loss=[1.0, 1.0, 2.0, 0.5])}))
    s = raissi_repro.summarize(str(tmp_path), str(tmp_path / "s.json"))
    f64 = s["float64"]
    assert f64["n_above_threshold"] == 1 and f64["n_at_or_below_paper"] == 2
    assert f64["ci95_above_threshold"] == raissi_repro.clopper_pearson(1, 4)
    assert (f64["best_seed"], f64["worst_seed"]) == (0, 1)
    assert f64["spearman_validation_loss_vs_error"]["rho"] == pytest.approx(1.0)
    assert f64["lowest_validation_loss"] == {"seed": 0, "error": 5e-4, "rank": 1}
    assert [b["k"] for b in f64["best_of_k_by_validation_loss"]] == [3]       # C(4, 5) is empty
    assert s["paired_by_seed"] == {"n": 4, "float64_lower": 4}


def test_rescore_recomputes_v_on_the_authors_domain_from_saved_parameters(tmp_path):
    d = tmp_path / "float32"
    pb = raissi_repro.Problem(0, "float32", **SMALL)
    r = raissi_repro.run(0, "float32", options=dict(raissi_repro.AUTHORS_OPTIONS, maxiter=5),
                         problem=pb, params_dir=str(d / "params"))
    expected = r["validation_loss"]["total"]
    r["validation_loss"] = {"total": 1.0}                 # as a run scored on another grid
    del r["validation_t_end"]
    (d / "p.json").write_text(json.dumps({"runs": [r]}))
    raissi_repro.rescore(str(tmp_path))
    r2 = json.loads((d / "p.json").read_text())["runs"][0]
    assert r2["validation_t_end"] == 0.99
    assert r2["validation_loss"]["total"] == pytest.approx(expected, rel=1e-5)
    r2["relative_l2_error"] *= 2                          # a saved network that does not match
    (d / "p.json").write_text(json.dumps({"runs": [r2]}))
    with pytest.raises(ValueError, match="saved network"):
        raissi_repro.rescore(str(tmp_path))


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
