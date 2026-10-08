"""Two-pass, bounded V1 cell selection from unsorted native NEST events.

The caller supplies fresh blocks of physical 0.1-ms ticks and zero-based
global cell IDs on each pass. This pure component neither opens nor attests
raw files. A production caller must independently pin and verify every source
rank, the population geometry, resource limits, and its output destination.
"""

import numpy as np

from mam_v1_140_selector import BIN_COUNT, END_TICK, MAX_BLOCK, START_TICK


def select_v1_populations(blocks, *, groups, total_neurons, expected_raw):
    """Apply one deterministic eligibility/selection rule to eight V1 groups.

    ``groups`` is an ordered sequence of eight ``(global_start, neuron_count,
    sample_count)`` triples. The groups must be contiguous. The selection is
    per realization: it does not reuse the chosen IDs from another simulator.
    """
    if (len(groups) != 8 or type(total_neurons) is not int
            or not 1 <= total_neurons <= 4_200_000
            or type(expected_raw) is not int or not 0 <= expected_raw <= 2**36):
        raise ValueError("invalid V1 geometry or raw-event contract")
    starts, ends, samples = [], [], []
    previous_end = None
    for row in groups:
        if (len(row) != 3 or any(type(v) is not int for v in row)):
            raise ValueError("invalid V1 group")
        start, count, sample_count = row
        if (start < 0 or count <= 0 or not 1 <= sample_count <= count
                or start + count > total_neurons
                or (previous_end is not None and start != previous_end)):
            raise ValueError("noncontiguous or out-of-range V1 group")
        starts.append(start)
        ends.append(start + count)
        samples.append(sample_count)
        previous_end = start + count
    if sum(samples) > 140:
        raise ValueError("V1 sample exceeds 140 cells")

    def checked_blocks():
        raw = 0
        for ticks, cells in blocks():
            ticks, cells = np.asarray(ticks), np.asarray(cells)
            if (ticks.ndim != 1 or cells.shape != ticks.shape
                    or ticks.dtype.kind not in "iu" or cells.dtype.kind not in "iu"
                    or len(ticks) > MAX_BLOCK or np.any(ticks < 0)
                    or np.any(ticks > END_TICK) or np.any(cells < 0)
                    or np.any(cells >= total_neurons)):
                raise ValueError("invalid physical native event block")
            raw += len(ticks)
            if raw > expected_raw:
                raise ValueError("native raw event count exceeded")
            yield ticks, cells
        if raw != expected_raw:
            raise ValueError("native raw event count differs")

    eligibility = [np.zeros(end - start, dtype=np.int64)
                   for start, end in zip(starts, ends, strict=True)]
    for ticks, cells in checked_blocks():
        v1 = (cells >= starts[0]) & (cells < ends[-1]) & (ticks > START_TICK)
        candidate = cells[v1].astype(np.int64)
        index = np.searchsorted(ends, candidate, side="right")
        for group_index in range(8):
            local = candidate[index == group_index] - starts[group_index]
            eligibility[group_index] += np.bincount(
                local, minlength=len(eligibility[group_index]))

    local_ids = []
    global_ids = []
    selected_eligibility = []
    for start, count, wanted in zip(starts, eligibility, samples, strict=True):
        local = np.flatnonzero(count > 56)[:wanted]
        if len(local) != wanted:
            raise ValueError("insufficient eligible native V1 cells")
        local_ids.append(local)
        global_ids.append(local + start)
        selected_eligibility.append(count[local])

    chosen = np.concatenate(global_ids)
    per_cell = np.zeros((len(chosen), BIN_COUNT), dtype=np.uint64)
    second_eligibility = np.zeros(len(chosen), dtype=np.int64)
    for ticks, cells in checked_blocks():
        # All 48 rank streams are still counted and hash-checked by the
        # caller, but only V1 cells need the selected-ID binary search.
        in_v1 = (cells >= starts[0]) & (cells < ends[-1])
        candidate_cells, candidate_ticks = cells[in_v1], ticks[in_v1]
        positions = np.searchsorted(chosen, candidate_cells)
        possible = positions < len(chosen)
        selected = np.zeros(len(candidate_cells), dtype=bool)
        selected[possible] = chosen[positions[possible]] == candidate_cells[possible]
        physical = selected & (candidate_ticks > START_TICK)
        second_eligibility += np.bincount(
            positions[physical], minlength=len(chosen))
        histogram = selected & (candidate_ticks >= START_TICK + 5)
        indices = (positions[histogram] * BIN_COUNT
                   + (candidate_ticks[histogram].astype(np.int64) - START_TICK - 5) // 10)
        np.add.at(per_cell.reshape(-1), indices, 1)
    if not np.array_equal(second_eligibility, np.concatenate(selected_eligibility)):
        raise ValueError("native selected-cell eligibility differs between passes")

    population_counts = []
    offset = 0
    for wanted in samples:
        summed = per_cell[offset:offset + wanted].sum(axis=0, dtype=np.uint64)
        if np.any(summed > 2**32 - 1):
            raise ValueError("native population bin overflows downstream contract")
        population_counts.append(summed.astype(np.uint32))
        offset += wanted
    return dict(local_ids=tuple(local_ids), global_ids=tuple(global_ids),
                eligibility_spikes=tuple(selected_eligibility),
                cell_counts=per_cell, population_counts=np.stack(population_counts),
                source_bound=False)
