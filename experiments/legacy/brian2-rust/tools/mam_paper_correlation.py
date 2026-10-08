"""Bounded integer-grid implementation of the available MAM correlation view.

Candidate IDs start at the lowest ID with any recorded spike, including warmup.
The toolbox uses an inclusive 3001-ID range, removes every constant histogram,
then keeps the first 2000 rows. NumPy histogram includes the final time edge.
"""
import hashlib
import numpy as np

HELPER_SHA = '9db8c61b7e56c2ab36c819b0059c0618a0eec236ea8fdf7a0073693e5f47a201'
WRAPPER_SHA = 'c97f4ff2fdf558939026bf031f6f43c19c16e652111b1b1d73143287ad3c4ed4'
TOOLBOX_COMMIT = '26b9e999069990a8b756d8a4d880bd152f95149f'


def candidate_histogram(ticks, cells, first_id, *, end_tick=25000):
    ticks, cells = np.asarray(ticks), np.asarray(cells)
    if (ticks.ndim != 1 or cells.shape != ticks.shape
            or ticks.dtype.kind not in 'iu' or cells.dtype.kind not in 'iu'
            or len(ticks) > 50_000_000 or type(first_id) is not int or first_id < 0
            or type(end_tick) is not int or end_tick not in (25000,105000)
            or np.any(ticks < 0) or np.any(ticks > end_tick) or np.any(cells < 0)):
        raise ValueError('invalid or excessive physical event arrays')
    nbins = (end_tick-5000)//10
    counts = np.zeros((3001, nbins), dtype=np.int64)
    for offset in range(0, len(ticks), 131072):
        t = ticks[offset:offset + 131072].astype(np.int64)
        c = cells[offset:offset + 131072].astype(np.int64)
        keep = (c >= first_id) & (c <= first_id + 3000) & (t >= 5000) & (t <= end_tick)
        idx = (c[keep] - first_id) * nbins + np.minimum((t[keep] - 5000) // 10, nbins-1)
        # add.at avoids allocating a full 3001 x 2000 bincount on every block.
        np.add.at(counts.reshape(-1), idx, 1)
    return counts


def summarize_histogram(counts, first_id):
    if (counts.shape not in ((3001, 2000),(3001,10000)) or counts.dtype.kind not in 'iu'
            or np.any(counts < 0) or type(first_id) is not int or first_id < 0):
        raise ValueError('requires bounded nonnegative candidate histograms')
    varying = np.ptp(counts, axis=1) > 0
    chosen = np.flatnonzero(varying)[:2000]
    selected = counts[chosen]
    n = len(chosen)
    value = None
    if n >= 2:
        # Mean of off-diagonal Pearson coefficients using the Gram-sum identity.
        # O(cells * bins) storage/work; no 2000 x 2000 correlation matrix.
        x = selected.astype(np.float64)
        x -= x.mean(axis=1, keepdims=True)
        norm = np.sqrt(np.einsum('ij,ij->i', x, x))
        assert np.all(norm > 0)
        x /= norm[:, None]
        summed = x.sum(axis=0)
        value = float((np.dot(summed, summed) - np.einsum('ij,ij->', x, x)) / (n * (n - 1)))
        assert np.isfinite(value) and -1 <= value <= 1
    report = dict(first_recorded_local_id=first_id, candidate_id_max_inclusive=first_id + 3000,
        candidate_ids=3001, varying_candidate_cells=int(varying.sum()), selected_cells=n,
        silent_candidates=int(np.count_nonzero(counts.sum(axis=1) == 0)),
        constant_nonzero_candidates=int(np.count_nonzero((~varying) & (counts.sum(axis=1) > 0))),
        candidate_events=int(counts.sum()), selected_events=int(selected.sum()),
        mean_pairwise_correlation=value, available=n >= 2,
        unavailable_reason=None if n >= 2 else 'Official wrapper cannot compute an off-diagonal mean with fewer than two varying cells.',
        selected_counts_sha256=hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest())
    return report, chosen + first_id, selected


MAX_STREAM_EVENTS = 2**32 - 1
PAPER_END_TICKS = (25000, 105000, 505000, 1005000)


class CandidateHistogram:
    """Stream physical event blocks into checked u32 candidate counts.

    ``first_id`` must be determined from the entire recording, including warmup,
    before accumulation. Total admitted input events bound every individual bin
    below u32 overflow. This is a resource ceiling, not a firing-rate assumption.
    The legacy array helper above retains its original API and limits.
    """
    def __init__(self, first_id, *, end_tick=25000):
        if (type(first_id) is not int or not 0 <= first_id <= 2**63-3001
                or type(end_tick) is not int or end_tick not in PAPER_END_TICKS):
            raise ValueError('invalid candidate ID or declared paper duration')
        self.first_id = first_id
        self.end_tick = end_tick
        self.nbins = (end_tick-5000)//10
        self.counts = np.zeros((3001, self.nbins), dtype='<u4')
        self.raw_events = 0

    def add(self, ticks, cells):
        ticks, cells = np.asarray(ticks), np.asarray(cells)
        if (ticks.ndim != 1 or cells.shape != ticks.shape
                or ticks.dtype.kind not in 'iu' or cells.dtype.kind not in 'iu'
                or len(ticks) > 131072 or self.raw_events + len(ticks) > MAX_STREAM_EVENTS
                or np.any(ticks < 0) or np.any(ticks > self.end_tick)
                or np.any(cells < 0) or np.any(cells > 2**63-1)):
            raise ValueError('invalid or excessive physical event block')
        t, c = ticks.astype(np.int64), cells.astype(np.int64)
        keep = ((c >= self.first_id) & (c <= self.first_id+3000)
                & (t >= 5000) & (t <= self.end_tick))
        index = ((c[keep]-self.first_id)*self.nbins
                 + np.minimum((t[keep]-5000)//10, self.nbins-1))
        np.add.at(self.counts.reshape(-1), index, 1)
        self.raw_events += len(ticks)


def summarize_histogram_bounded(counts, first_id, *, row_block_size=32):
    """Same selection/Gram identity with bounded floating-point/hash scratch.

    Selected integer rows are returned, retaining the input dtype. Canonical
    selected-count hashes still encode little-endian signed 64-bit row values.
    Blocked floating-point reduction is checked against the pre-existing 1e-12
    correlation tolerance; it is not claimed to preserve bitwise float sums.
    """
    counts = np.asarray(counts)
    if (counts.shape not in tuple((3001, (end-5000)//10) for end in PAPER_END_TICKS)
            or counts.dtype.kind not in 'iu' or type(first_id) is not int
            or not 0 <= first_id <= 2**63-3001 or type(row_block_size) is not int
            or not 1 <= row_block_size <= 128):
        raise ValueError('invalid bounded candidate histogram configuration')
    totals = np.zeros(3001, dtype=np.uint64)
    varying = np.zeros(3001, dtype=np.bool_)
    for start in range(0, 3001, row_block_size):
        block = counts[start:start+row_block_size]
        if np.any(block < 0) or np.any(block > MAX_STREAM_EVENTS):
            raise ValueError('candidate histogram exceeds nonnegative event budget')
        totals[start:start+row_block_size] = block.sum(axis=1, dtype=np.uint64)
        varying[start:start+row_block_size] = np.ptp(block, axis=1) > 0
    total_events = int(totals.sum())
    if total_events > MAX_STREAM_EVENTS:
        raise ValueError('candidate histogram exceeds total event budget')
    chosen = np.flatnonzero(varying)[:2000]
    selected = counts[chosen]
    n = len(chosen)
    summed = np.zeros(counts.shape[1], dtype=np.float64)
    energy = 0.
    canonical_hash = hashlib.sha256()
    for start in range(0, n, row_block_size):
        block = selected[start:start+row_block_size]
        canonical_hash.update(memoryview(np.ascontiguousarray(block, dtype='<i8')).cast('B'))
        if n >= 2:
            x = block.astype(np.float64)
            x -= x.mean(axis=1, keepdims=True)
            norm = np.sqrt(np.einsum('ij,ij->i', x, x))
            assert np.all(norm > 0)
            x /= norm[:, None]
            summed += x.sum(axis=0)
            energy += float(np.einsum('ij,ij->', x, x))
    value = None
    if n >= 2:
        value = float((np.dot(summed, summed)-energy)/(n*(n-1)))
        assert np.isfinite(value) and -1 <= value <= 1
    report = dict(first_recorded_local_id=first_id, candidate_id_max_inclusive=first_id+3000,
        candidate_ids=3001, varying_candidate_cells=int(varying.sum()), selected_cells=n,
        silent_candidates=int(np.count_nonzero(totals == 0)),
        constant_nonzero_candidates=int(np.count_nonzero((~varying) & (totals > 0))),
        candidate_events=total_events, selected_events=int(totals[chosen].sum()),
        mean_pairwise_correlation=value, available=n >= 2,
        unavailable_reason=None if n >= 2 else 'Official wrapper cannot compute an off-diagonal mean with fewer than two varying cells.',
        selected_counts_sha256=canonical_hash.hexdigest())
    return report, chosen+first_id, selected
