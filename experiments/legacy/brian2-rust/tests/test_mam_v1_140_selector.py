from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mam_v1_140_selector import select_population, BIN_COUNT


def fixture_blocks():
    # ID 1 has exactly 56 eligible spikes; IDs 2 and 3 have 57 each.
    # ID 0 spikes only during warmup. ID 2's lower half-ms events count
    # toward eligibility but not the shifted histogram.
    ticks = np.array([5000] + [5001] * 56 + [5001] * 55 + [5005, 1005000]
                     + [5005] * 56 + [1005000], dtype=np.uint32)
    cells = np.array([0] + [1] * 56 + [2] * 55 + [2, 2]
                     + [3] * 56 + [3], dtype=np.uint32)
    assert len(ticks) == len(cells)

    def blocks():
        for start in range(0, len(ticks), 43):
            yield ticks[start:start + 43], cells[start:start + 43]

    return blocks, len(ticks)


def test_lowest_eligible_ids_and_physical_endpoint_grid():
    blocks, raw = fixture_blocks()
    result = select_population(blocks, neurons=4, sample_count=2,
                               global_offset=100, expected_raw=raw)
    np.testing.assert_array_equal(result["local_ids"], [2, 3])
    np.testing.assert_array_equal(result["global_ids"], [102, 103])
    np.testing.assert_array_equal(result["eligibility_spikes"], [57, 57])
    assert result["cell_counts"].shape == (2, BIN_COUNT)
    assert result["cell_counts"][0, 0] == 1
    assert result["cell_counts"][1, 0] == 56
    assert result["cell_counts"][0, -1] == 1
    assert result["cell_counts"][1, -1] == 1
    assert result["population_counts"].sum() == 59
    assert result["source_bound"] is False


def test_missing_cells_or_changed_second_pass_rejected():
    blocks, raw = fixture_blocks()
    with pytest.raises(ValueError, match="insufficient eligible"):
        select_population(blocks, neurons=4, sample_count=3,
                          global_offset=0, expected_raw=raw)

    visits = 0
    def changed():
        nonlocal visits
        visits += 1
        for ticks, cells in blocks():
            altered = cells.copy()
            if visits == 2:
                altered[altered == 3] = 1
            yield ticks, altered

    with pytest.raises(ValueError, match="differs between passes"):
        select_population(changed, neurons=4, sample_count=2,
                          global_offset=0, expected_raw=raw)
