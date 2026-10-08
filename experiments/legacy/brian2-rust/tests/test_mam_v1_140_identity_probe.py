from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mam_v1_140_identity_probe import certain_ids


def test_terminal_bit_cannot_change_admitted_lowest_ids():
    strict = np.array([55, 57, 58, 56, 57], dtype=np.int64)
    chosen, certain, uncertain = certain_ids(strict, 2)
    np.testing.assert_array_equal(chosen, [1, 2])
    assert (certain, uncertain) == (3, 1)
    for terminal in product((0, 1), repeat=len(strict)):
        actual = np.flatnonzero(strict + np.array(terminal) >= 57)[:2]
        np.testing.assert_array_equal(actual, chosen)


def test_terminal_ambiguity_before_cutoff_rejected():
    with pytest.raises(ValueError, match="terminal tick"):
        certain_ids(np.array([57, 56, 57], dtype=np.int64), 2)
    with pytest.raises(ValueError, match="fewer than required"):
        certain_ids(np.array([55, 56, 57], dtype=np.int64), 2)
