from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mam_native_v1_140_selector import select_v1_populations
from mam_v1_140_selector import select_population


def fixture():
    groups = [(10 + 3 * i, 3, 1) for i in range(8)]
    ticks, cells = [4000], [0]  # Valid non-V1 event.
    for start, _, _ in groups:
        ticks.extend([5001] * 56 + [5001] * 55 + [5005, 1005000] + [5000])
        cells.extend([start] * 56 + [start + 1] * 57 + [start + 2])
    ticks = np.asarray(ticks, dtype=np.uint32)
    cells = np.asarray(cells, dtype=np.uint32)

    def blocks():
        for offset in range(0, len(ticks), 31):
            yield ticks[offset:offset + 31], cells[offset:offset + 31]

    return groups, blocks, len(ticks)


def test_native_global_ids_and_shifted_endpoints_across_eight_groups():
    groups, blocks, total = fixture()
    result = select_v1_populations(blocks, groups=groups,
                                   total_neurons=40, expected_raw=total)
    assert [ids.tolist() for ids in result["local_ids"]] == [[1]] * 8
    assert [ids.tolist() for ids in result["global_ids"]] == [
        [start + 1] for start, _, _ in groups]
    assert [counts.tolist() for counts in result["eligibility_spikes"]] == [[57]] * 8
    assert result["cell_counts"].shape == (8, 100000)
    np.testing.assert_array_equal(result["population_counts"][:, 0], [1] * 8)
    np.testing.assert_array_equal(result["population_counts"][:, -1], [1] * 8)
    assert int(result["population_counts"].sum()) == 16
    assert result["source_bound"] is False
    # The native global-ID path must match the established population-local
    # selector when both receive precisely the same synthetic events.
    for i, (start, count, wanted) in enumerate(groups):
        def local_blocks():
            for ticks, cells in blocks():
                mask = (cells >= start) & (cells < start + count)
                yield ticks[mask], cells[mask] - start

        raw = sum(len(ticks) for ticks, _ in local_blocks())
        local = select_population(local_blocks, neurons=count,
                                  sample_count=wanted, global_offset=start,
                                  expected_raw=raw)
        np.testing.assert_array_equal(result["local_ids"][i], local["local_ids"])
        np.testing.assert_array_equal(result["population_counts"][i],
                                      local["population_counts"])


def test_changed_second_pass_and_bad_source_contract_rejected():
    groups, blocks, total = fixture()
    visits = 0

    def changed():
        nonlocal visits
        visits += 1
        for ticks, cells in blocks():
            altered = cells.copy()
            if visits == 2:
                altered[altered == 11] = 12
            yield ticks, altered

    with pytest.raises(ValueError, match="differs between passes"):
        select_v1_populations(changed, groups=groups,
                              total_neurons=40, expected_raw=total)
    with pytest.raises(ValueError, match="raw event count differs"):
        select_v1_populations(blocks, groups=groups,
                              total_neurons=40, expected_raw=total + 1)
    with pytest.raises(ValueError, match="noncontiguous"):
        select_v1_populations(blocks, groups=groups[:3] + [(20, 3, 1)] + groups[4:],
                              total_neurons=40, expected_raw=total)
