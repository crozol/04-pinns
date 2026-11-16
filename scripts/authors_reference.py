"""The authors' reference solution for the Burgers benchmark, against the exact Cole-Hopf one.

Raissi, Perdikaris & Karniadakis (2019) score their network against burgers_shock.mat, a
numerical solution shipped with their code (maziarraissi/PINNs, appendix/Data/), on its
256 x 100 grid: x in [-1, 1], t = 0, 0.01, ..., 0.99. This project scores against the
Cole-Hopf solution (src/reference.py), whose quadrature is converged to rounding
(scripts/reference_convergence.py). Section 9 compares errors measured against one with the
published error measured against the other, which is like for like only if the two references
agree far below 6.7e-4.

This script downloads the file (unless --mat gives a local copy), checks its SHA-256 against
the copy used in this project, and compares it with Cole-Hopf on the file's own grid.
Result: a relative L2 difference of 3.1e-12, the largest pointwise difference 4.2e-11, at
x = -0.0039, t = 0.66, next to the steep layer at x = 0.

Usage:
    python -m scripts.authors_reference                       # downloads to results/
    python -m scripts.authors_reference --mat path/to/burgers_shock.mat
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

from src.reference import burgers_exact

URL = ("https://raw.githubusercontent.com/maziarraissi/PINNs/master/"
       "appendix/Data/burgers_shock.mat")
SHA256 = "f4d11e6798b8d29b6829984952b04adcc470d9bf70dd7e56c7985cd8e316c4bf"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x, t and U with U[i, j] = u(x[j], t[i]), the layout of src.reference.burgers_exact.
    The file stores usol[j, i] = u(x[j], t[i]) as complex numbers with zero imaginary part."""
    from scipy.io import loadmat

    m = loadmat(str(path))
    x, t = m["x"].ravel(), m["t"].ravel()
    usol = m["usol"]
    if usol.shape != (x.size, t.size):
        raise ValueError(f"usol has shape {usol.shape}, expected (len(x), len(t)) = "
                         f"{(x.size, t.size)}")
    if np.iscomplexobj(usol) and np.abs(usol.imag).max() > 0:
        raise ValueError("usol has a non-zero imaginary part")
    return x, t, np.real(usol).T


def compare(x: np.ndarray, t: np.ndarray, U: np.ndarray) -> dict:
    """Relative L2 and largest pointwise difference between U and Cole-Hopf on the same grid."""
    exact = burgers_exact(x, t)
    diff = np.abs(U - exact)
    i, j = np.unravel_index(int(np.argmax(diff)), diff.shape)
    return {"grid": [int(x.size), int(t.size)], "t_max": float(t.max()),
            "relative_l2": float(np.linalg.norm(U - exact) / np.linalg.norm(exact)),
            "max_abs": float(diff.max()), "at_x": float(x[j]), "at_t": float(t[i])}


def main(mat: str | None = None, out: str = "results/authors_reference.json",
         expected_sha256: str | None = SHA256) -> dict:
    path = Path(mat) if mat else Path("results/burgers_shock.mat")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, path)
    digest = sha256(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"{path}: SHA-256 {digest}, expected {expected_sha256}")
    res = {"file": str(path), "sha256": digest, "url": URL, **compare(*load(path))}
    print(f"burgers_shock.mat ({res['grid'][0]} x {res['grid'][1]}, t up to {res['t_max']:g}) "
          f"against Cole-Hopf: relative L2 {res['relative_l2']:.3e}, largest difference "
          f"{res['max_abs']:.3e} at x = {res['at_x']:.4f}, t = {res['at_t']:.2f}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    print(f"[ok] {out}")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mat", default=None, help="a local copy of burgers_shock.mat")
    p.add_argument("--out", default="results/authors_reference.json")
    a = p.parse_args()
    main(a.mat, a.out)
