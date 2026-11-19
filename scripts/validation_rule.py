"""How much the validation rule (keep the run with the lowest V) is worth, with its uncertainty.

Two questions from the external review of 2026-11-19, on the 20 runs per precision of the
Raissi recipe (results/raissi/<dtype>/*.json, the runs of README §9):

1. The rule "of k runs, keep the one with the lowest V" was measured over every set of k of the
   same 20 runs, so its rates carry the uncertainty of n = 20. Here: a bootstrap over the runs
   (B = 2000 resamples of the 20, with replacement; the exact enumeration of every set of k inside
   each resample), for the share of kept runs at or below the paper's 6.7e-4 and above 1e-3, and
   for the share of the possible gain it recovers, (rule - random) / (oracle - random), where
   random keeps any one run and the oracle keeps the run with the lowest error.
2. If the training loss does not order the runs and V does, the bad runs may overfit their
   collocation points. Spearman correlation with the error of the gap V - L_train and of the ratio
   V / L_train, next to those of V and of L_train alone.

    python -m scripts.validation_rule            # -> results/validation_rule.json
"""

from __future__ import annotations

import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PAPER, THRESHOLD = 6.7e-4, 1e-3
B, SEED = 2000, 0


def load(root: str, dtype: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs = {}
    for f in sorted(glob.glob(str(Path(root) / dtype / "*.json"))):
        for r in json.loads(Path(f).read_text())["runs"]:
            runs[r["seed"]] = r
    runs = [runs[s] for s in sorted(runs)]
    return (np.array([r["relative_l2_error"] for r in runs]),
            np.array([r["validation_loss"]["total"] for r in runs]),
            np.array([r["final_loss"] for r in runs]))


def rates(err: np.ndarray, v: np.ndarray, comb: np.ndarray) -> dict:
    """Over every set of k (the rows of comb): the run kept by V, by the error (oracle) and any
    one run (random)."""
    kept = err[comb[np.arange(len(comb)), np.argmin(v[comb], axis=1)]]
    oracle = err[comb].min(axis=1)
    rule, best, rand = np.mean(kept <= PAPER), np.mean(oracle <= PAPER), np.mean(err <= PAPER)
    return {"at_or_below_paper": float(rule), "above_threshold": float(np.mean(kept > THRESHOLD)),
            "median": float(np.median(kept)), "oracle_at_or_below_paper": float(best),
            "random_at_or_below_paper": float(rand),
            "gain_recovered": float((rule - rand) / (best - rand)) if best > rand else None}


def analyse(err: np.ndarray, v: np.ndarray, loss: np.ndarray, ks=(3, 5)) -> dict:
    n = len(err)
    rng = np.random.default_rng(SEED)
    out = {"n_runs": n}
    for k in ks:
        comb = np.array(list(combinations(range(n), k)))
        point = rates(err, v, comb)
        boot = []
        for _ in range(B):
            i = rng.integers(0, n, n)
            boot.append(rates(err[i], v[i], comb))
        ci = {key: [float(np.nanquantile([b[key] for b in boot if b[key] is not None], q)) for q in (0.025, 0.975)]
              for key in ("at_or_below_paper", "above_threshold", "gain_recovered")}
        out[f"lowest V of {k}"] = {"point": point, "bootstrap_95": ci, "n_sets": int(len(comb))}
    out["spearman_vs_error"] = {}
    for name, x in (("V", v), ("L_train", loss), ("V - L_train", v - loss), ("V / L_train", v / loss)):
        r = spearmanr(x, err)
        out["spearman_vs_error"][name] = {"rho": float(r.statistic), "p": float(r.pvalue)}
    return out


def main(root: str = "results/raissi", out: str = "results/validation_rule.json") -> dict:
    res = {"bootstrap_resamples": B, "paper_error": PAPER, "threshold": THRESHOLD}
    for dtype in ("float32", "float64"):
        err, v, loss = load(root, dtype)
        res[dtype] = a = analyse(err, v, loss)
        print(f"{dtype}: {a['n_runs']} runs")
        for k in (3, 5):
            b = a[f"lowest V of {k}"]
            p, ci = b["point"], b["bootstrap_95"]
            g = "-" if p["gain_recovered"] is None else f"{p['gain_recovered']:.2f}"
            print(f"  lowest V of {k}: <= paper {p['at_or_below_paper']:.0%} "
                  f"[{ci['at_or_below_paper'][0]:.0%}, {ci['at_or_below_paper'][1]:.0%}] (random "
                  f"{p['random_at_or_below_paper']:.0%}, oracle {p['oracle_at_or_below_paper']:.1%}); "
                  f"> 1e-3 {p['above_threshold']:.0%} [{ci['above_threshold'][0]:.0%}, "
                  f"{ci['above_threshold'][1]:.0%}]; gain recovered {g} "
                  f"[{ci['gain_recovered'][0]:.2f}, {ci['gain_recovered'][1]:.2f}]")
        for name, r in a["spearman_vs_error"].items():
            print(f"  Spearman({name}, error) = {r['rho']:+.2f} (p = {r['p']:.3f})")
    Path(out).write_text(json.dumps(res, indent=2))
    print(f"[ok] {out}")
    return res


if __name__ == "__main__":
    main()
