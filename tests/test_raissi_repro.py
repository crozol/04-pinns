"""Checkpointed L-BFGS in scripts/raissi_repro.py follows the uninterrupted trajectory."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import raissi_repro

SMALL = dict(n_u=40, n_f=200)


class Crash(Exception):
    pass


@pytest.fixture(autouse=True)
def fixed_reference(monkeypatch):
    # the comparison needs a fixed reference, not the exact one (tested in test_physics.py)
    monkeypatch.setattr(raissi_repro, "burgers_exact", lambda x, t: np.ones((len(t), len(x))))


def _trajectory(r: dict) -> tuple:
    return r["relative_l2_error"], r["final_loss"], r["n_evals"]


def test_chunked_lbfgs_matches_a_single_call(tmp_path):
    single = raissi_repro.run(0, 40, checkpoint_every=10**9, **SMALL)
    chunked = raissi_repro.run(0, 40, checkpoint_dir=str(tmp_path), checkpoint_every=7, **SMALL)
    assert _trajectory(chunked) == _trajectory(single)


def test_interrupted_run_resumes_on_the_same_trajectory(tmp_path, monkeypatch):
    uninterrupted = raissi_repro.run(1, 40, checkpoint_every=10**9, **SMALL)
    save, saves = raissi_repro._save_checkpoint, []

    def save_then_crash(*args):
        save(*args)
        saves.append(1)
        if len(saves) == 2:
            raise Crash

    monkeypatch.setattr(raissi_repro, "_save_checkpoint", save_then_crash)
    with pytest.raises(Crash):
        raissi_repro.run(1, 40, checkpoint_dir=str(tmp_path), checkpoint_every=7, **SMALL)
    monkeypatch.setattr(raissi_repro, "_save_checkpoint", save)
    resumed = raissi_repro.run(1, 40, checkpoint_dir=str(tmp_path), checkpoint_every=7, **SMALL)
    assert _trajectory(resumed) == _trajectory(uninterrupted)
