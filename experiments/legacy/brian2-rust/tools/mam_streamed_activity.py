"""Stream the frozen diagnostic sampling conventions without full event copies.

This auxiliary activity view is distinct from the pinned paper correlation
selection. Callers provide a repeatable physical event-block factory. Sampled LvR
neurons must have advancing selected timestamps between blocks; global rank-file order may
reset for different neurons. Existing baseline builders remain unchanged.
"""
import numpy as np
from mam_streamed_cell_metrics import CellMetrics, CHUNK, MAX_EVENTS


def mean_pairwise_correlation_bounded(counts, *, row_block_size=32):
    counts = np.asarray(counts)
    if (counts.ndim != 2 or not 0 <= counts.shape[0] <= 2000
            or not 2 <= counts.shape[1] <= 100000 or counts.dtype.kind not in 'iu'
            or type(row_block_size) is not int or not 1 <= row_block_size <= 128):
        raise ValueError('invalid bounded diagnostic correlation matrix')
    summed = np.zeros(counts.shape[1], dtype=np.float64)
    energy, n, events = 0., 0, 0
    for start in range(0, len(counts), row_block_size):
        block = counts[start:start+row_block_size]
        if np.any(block < 0) or np.any(block > MAX_EVENTS):
            raise ValueError('diagnostic counts exceed event budget')
        events += int(block.sum(dtype=np.uint64))
        if events > MAX_EVENTS:
            raise ValueError('diagnostic counts exceed event budget')
        varying = np.ptp(block, axis=1) > 0
        n += int(varying.sum())
        if not varying.any():
            continue
        x = block[varying].astype(np.float64)
        x -= x.mean(axis=1, keepdims=True)
        norms = np.sqrt(np.einsum('ij,ij->i', x, x))
        assert np.all(norms > 0)
        x /= norms[:, None]
        summed += x.sum(axis=0)
        energy += float(np.einsum('ij,ij->', x, x))
    if n < 2:
        return None, n
    result = float((np.dot(summed, summed)-energy)/(n*(n-1)))
    # Like the frozen diagnostic helper, retain ordinary rounding at +/-1.
    # Identical trains can yield 1 + epsilon; do not clamp or reject them.
    assert np.isfinite(result)
    return result, n


def population_activity_streamed(blocks, neuron_count, *, start_tick=5000,
        end_tick=25000, dt_seconds=.0001, bin_ticks=10, seed=20260908,
        sample_size=2000, refractory_ms=2.0, expected_raw):
    if (type(neuron_count) is not int or not 1 <= neuron_count <= 4200000
            or type(start_tick) is not int or start_tick != 5000
            or type(end_tick) is not int or end_tick not in (25000,105000,505000,1005000)
            or dt_seconds != .0001 or type(bin_ticks) is not int or bin_ticks != 10
            or type(seed) is not int or seed < 0
            or type(sample_size) is not int or not 1 <= sample_size <= 2000
            or type(expected_raw) is not int or not 0 <= expected_raw <= MAX_EVENTS
            or not np.isfinite(refractory_ms) or refractory_ms < 0):
        raise ValueError('invalid declared streamed activity contract')
    nbins = (end_tick-start_tick)//bin_ticks
    duration = (end_tick-start_tick)*dt_seconds
    rng = np.random.default_rng(seed)
    # The first draw is independent of event counts, so it can precede scanning.
    # The second draw still occurs only after the complete active set is known.
    lvr_ids = np.sort(rng.choice(neuron_count,min(neuron_count,sample_size),replace=False))
    lvr_lookup = np.full(neuron_count,-1,dtype=np.int32)
    lvr_lookup[lvr_ids] = np.arange(len(lvr_ids),dtype=np.int32)
    cell = CellMetrics(len(lvr_ids),start=start_tick,end=end_tick,
                       dt_ms=dt_seconds*1000,refractory_ms=refractory_ms)
    counts = np.zeros(neuron_count,dtype=np.int64)
    histogram = np.zeros(nbins,dtype=np.int64)
    raw_count = 0
    for ticks,ids in blocks():
        if (ticks.ndim != 1 or ids.shape != ticks.shape or len(ticks)>CHUNK
                or ticks.dtype.kind not in 'iu' or ids.dtype.kind not in 'iu'
                or np.any(ticks<0) or np.any(ticks>end_tick)
                or np.any(ids<0) or np.any(ids>=neuron_count)):
            raise ValueError('invalid activity event block')
        raw_count += len(ticks)
        if raw_count > expected_raw:
            raise ValueError('activity event count exceeded')
        keep = (ticks>=start_tick)&(ticks<end_tick)
        t,c = ticks[keep].astype(np.int64),ids[keep].astype(np.int64)
        counts += np.bincount(c,minlength=neuron_count)
        np.add.at(histogram,(t-start_tick)//bin_ticks,1)
        local = lvr_lookup[c];sampled = local>=0
        cell.add(t[sampled],local[sampled])
    if raw_count != expected_raw:
        raise ValueError('activity event count mismatch')
    _,per_cell = cell.summarize()
    np.testing.assert_array_equal(per_cell['half_open_cell_counts'],counts[lvr_ids])
    assert int(histogram.sum()) == int(counts.sum())
    rates = counts/duration
    active = np.flatnonzero(counts)
    corr_ids = np.sort(rng.choice(active,min(len(active),sample_size),replace=False))
    lvr = per_cell['cell_lvr']
    eligible = counts[lvr_ids] >= 3
    del cell,per_cell,lvr_lookup
    matrix = np.zeros((len(corr_ids),nbins),dtype='<u4')
    lookup = np.full(neuron_count,-1,dtype=np.int32)
    lookup[corr_ids] = np.arange(len(corr_ids),dtype=np.int32)
    seen = np.zeros(neuron_count,dtype=np.int64)
    replay_histogram = np.zeros(nbins,dtype=np.int64)
    total = 0
    for ticks,ids in blocks():
        if (ticks.ndim != 1 or ids.shape != ticks.shape or len(ticks)>CHUNK
                or ticks.dtype.kind not in 'iu' or ids.dtype.kind not in 'iu'
                or np.any(ticks<0) or np.any(ticks>end_tick)
                or np.any(ids<0) or np.any(ids>=neuron_count)):
            raise ValueError('invalid activity replay block')
        total += len(ticks)
        if total > expected_raw:
            raise ValueError('activity replay count exceeded')
        keep = (ticks>=start_tick)&(ticks<end_tick)
        t,c = ticks[keep].astype(np.int64),ids[keep].astype(np.int64)
        bins = (t-start_tick)//bin_ticks
        seen += np.bincount(c,minlength=neuron_count)
        np.add.at(replay_histogram,bins,1)
        local = lookup[c];sampled = local>=0
        np.add.at(matrix.reshape(-1),local[sampled].astype(np.int64)*nbins+bins[sampled],1)
    if total != expected_raw:
        raise ValueError('activity replay count mismatch')
    np.testing.assert_array_equal(seen,counts)
    np.testing.assert_array_equal(replay_histogram,histogram)
    correlation,variable = mean_pairwise_correlation_bounded(matrix)
    summary = dict(neurons=neuron_count,observed_spikes=int(histogram.sum()),
        mean_rate_hz=float(rates.mean()),rate_std_hz=float(rates.std()),
        rate_quantiles_hz=np.quantile(rates,[0,.25,.5,.75,.95,.99,1]).tolist(),
        silent_fraction=float(np.count_nonzero(counts==0)/neuron_count),
        lvr_zero_padded_mean=float(lvr.mean()),
        lvr_eligible_mean=float(lvr[eligible].mean()) if eligible.any() else None,
        lvr_sample_size=len(lvr_ids),lvr_eligible_cells=int(eligible.sum()),
        pairwise_corr_mean=correlation,corr_sample_size=len(corr_ids),
        corr_nonconstant_cells=variable,seed=int(seed))
    return summary,histogram,dict(lvr_ids=lvr_ids,lvr_values=lvr,lvr_eligible=eligible,
                                  corr_ids=corr_ids,single_cell_rates_hz=rates)
