"""Bounded two-pass selection for the pinned modern V1 rate wrapper.

The caller supplies a repeatable iterator of *physical* 0.1-ms ticks and
population-local neuron indices. This pure adapter does not open raw files or
attest their identity; a source-bound collector must do that separately.
"""

import numpy as np


END_TICK = 1_005_000
START_TICK = 5_000
BIN_COUNT = 100_000
MAX_BLOCK = 131_072


def select_population(blocks, *, neurons, sample_count, global_offset, expected_raw):
    """Select lowest eligible IDs, then count their spikes on the 1-ms grid.

    ``blocks`` is a zero-argument factory returning fresh event blocks for
    each of two passes. Eligibility uses (500,100500] ms, hence ticks
    5001..1005000. The wrapper's histogram uses ticks 5005..1005000 with
    index ``(tick-5005)//10``. The strict >0.56 Hz threshold means >=57
    spikes in the 100-s eligibility interval.
    """
    if (type(neurons) is not int or not 1 <= neurons <= 4_200_000
            or type(sample_count) is not int or not 1 <= sample_count <= neurons
            or type(global_offset) is not int or not 0 <= global_offset <= 4_200_000
            or global_offset + neurons > 4_200_000
            or type(expected_raw) is not int or not 0 <= expected_raw <= 2**36):
        raise ValueError("invalid bounded population contract")

    def checked():
        total = 0
        for ticks, cells in blocks():
            ticks, cells = np.asarray(ticks), np.asarray(cells)
            if (ticks.ndim != 1 or cells.shape != ticks.shape
                    or len(ticks) > MAX_BLOCK or ticks.dtype.kind not in "iu"
                    or cells.dtype.kind not in "iu" or np.any(ticks < 0)
                    or np.any(ticks > END_TICK) or np.any(cells < 0)
                    or np.any(cells >= neurons)):
                raise ValueError("invalid physical event block")
            total += len(ticks)
            if total > expected_raw:
                raise ValueError("raw event count exceeded")
            yield ticks, cells
        if total != expected_raw:
            raise ValueError("raw event count mismatch")

    eligibility = np.zeros(neurons, dtype=np.int64)
    for ticks, cells in checked():
        keep = ticks > START_TICK
        eligibility += np.bincount(cells[keep].astype(np.int64), minlength=neurons)
    chosen = np.flatnonzero(eligibility > 56)[:sample_count]
    if len(chosen) != sample_count:
        raise ValueError("insufficient eligible cells for pinned sample")

    # Keep individual series so a future collector can hash and independently
    # verify each selected cell, not only the eight population aggregates.
    per_cell = np.zeros((sample_count, BIN_COUNT), dtype=np.uint64)
    second_eligibility = np.zeros(sample_count, dtype=np.int64)
    for ticks, cells in checked():
        positions = np.searchsorted(chosen, cells)
        possible = positions < sample_count
        selected = np.zeros(len(cells), dtype=bool)
        selected[possible] = chosen[positions[possible]] == cells[possible]
        physical = selected & (ticks > START_TICK)
        second_eligibility += np.bincount(positions[physical], minlength=sample_count)
        histogram = selected & (ticks >= START_TICK + 5)
        indices = positions[histogram] * BIN_COUNT + (ticks[histogram].astype(np.int64) - START_TICK - 5) // 10
        np.add.at(per_cell.reshape(-1), indices, 1)
    if not np.array_equal(second_eligibility, eligibility[chosen]):
        raise ValueError("selected-cell eligibility differs between passes")
    aggregate = per_cell.sum(axis=0, dtype=np.uint64)
    if np.any(aggregate > 2**32 - 1):
        raise ValueError("selected population bin overflows downstream contract")
    return dict(local_ids=chosen, global_ids=chosen + global_offset,
                eligibility_spikes=eligibility[chosen],
                cell_counts=per_cell, population_counts=aggregate,
                source_bound=False)
