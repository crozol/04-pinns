"""scripts/authors_reference.py reads the authors' file in the right layout (no network)."""

import numpy as np
import pytest
from scipy.io import savemat

from scripts import authors_reference
from src.reference import burgers_exact


def _fake_mat(path, transform=lambda u: u):
    """A file laid out like burgers_shock.mat: x (n, 1), t (m, 1), usol[j, i] = u(x[j], t[i])."""
    x = np.linspace(-1, 1, 9)
    t = np.array([0.0, 0.25, 0.5, 0.75])
    usol = transform(burgers_exact(x, t).T).astype(complex)       # stored as complex, like theirs
    savemat(path, {"x": x[:, None], "t": t[:, None], "usol": usol})
    return x, t


def test_a_file_holding_the_exact_solution_compares_at_rounding(tmp_path):
    f = tmp_path / "ref.mat"
    _fake_mat(f)
    res = authors_reference.main(str(f), out=str(tmp_path / "r.json"), expected_sha256=None)
    assert res["grid"] == [9, 4] and res["t_max"] == 0.75
    assert res["relative_l2"] < 1e-14


def test_a_perturbed_file_is_located(tmp_path):
    f = tmp_path / "ref.mat"

    def bump(u):
        u = u.copy()
        u[6, 2] += 1e-3          # x[6] = 0.5, t[2] = 0.5
        return u

    _fake_mat(f, bump)
    res = authors_reference.compare(*authors_reference.load(f))
    assert res["max_abs"] == pytest.approx(1e-3, rel=1e-9)
    assert (res["at_x"], res["at_t"]) == (0.5, 0.5)


def test_a_file_with_another_hash_is_refused(tmp_path):
    f = tmp_path / "ref.mat"
    _fake_mat(f)
    with pytest.raises(ValueError, match="SHA-256"):
        authors_reference.main(str(f), out=str(tmp_path / "r.json"))
