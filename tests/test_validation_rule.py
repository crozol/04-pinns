"""The rates of the validation rule against random choice and the oracle."""

import numpy as np
import pytest

from scripts.validation_rule import PAPER, rates


def test_rule_random_and_oracle_rates_on_a_toy_set():
    from itertools import combinations

    err = np.array([1e-4, 5e-4, 2e-3, 3e-3])          # two runs at or below the paper's 6.7e-4
    comb = np.array(list(combinations(range(4), 2)))
    perfect = rates(err, err, comb)                    # V that orders the runs like the error
    assert perfect["at_or_below_paper"] == pytest.approx(perfect["oracle_at_or_below_paper"])
    assert perfect["random_at_or_below_paper"] == pytest.approx(0.5)
    assert perfect["oracle_at_or_below_paper"] == pytest.approx(5 / 6)   # every pair but (2, 3)
    assert perfect["gain_recovered"] == pytest.approx(1.0)
    reverse = rates(err, -err, comb)                   # V that keeps the worst run of each pair
    assert reverse["at_or_below_paper"] == pytest.approx(1 / 6)          # only the pair (0, 1)
    assert reverse["gain_recovered"] < 0
    assert PAPER == 6.7e-4
