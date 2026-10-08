"""Bounded paper-helper rate/LvR accumulation on per-neuron ordered streams.

Blocks may be globally unordered (e.g. separate NEST virtual processes). Each
neuron's selected spikes must advance between blocks. A backwards/duplicate
spike is rejected rather than silently producing different ISI triples.
"""
import numpy as np

CHUNK = 131072
MAX_EVENTS = 2**32-1


class CellMetrics:
    def __init__(self, neurons, *, start=5000, end=25000, dt_ms=.1, refractory_ms=2.):
        if (type(neurons) is not int or not 1 <= neurons <= 4200000
                or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= 1005000
                or not np.isfinite(dt_ms) or dt_ms <= 0
                or not np.isfinite(refractory_ms) or refractory_ms < 0):
            raise ValueError('invalid bounded cell-metric configuration')
        self.neurons, self.start, self.end = neurons, start, end
        self.dt_ms, self.refractory_ms = dt_ms, refractory_ms
        self.counts = np.zeros(neurons, dtype=np.int64)
        self.boundary = np.zeros(neurons, dtype=np.int64)
        self.sums = np.zeros(neurons, dtype=np.float64)
        self.last = np.full(neurons, -1, dtype=np.int64)
        self.penultimate = np.full(neurons, -1, dtype=np.int64)
        self.raw_events = 0

    def add(self, ticks, ids):
        ticks, ids = np.asarray(ticks), np.asarray(ids)
        if (ticks.ndim != 1 or ids.shape != ticks.shape
                or ticks.dtype.kind not in 'iu' or ids.dtype.kind not in 'iu'
                or len(ticks) > CHUNK or self.raw_events+len(ticks) > MAX_EVENTS
                or np.any(ids < 0) or np.any(ids >= self.neurons)
                or np.any(ticks < 0) or np.any(ticks > self.end)):
            raise ValueError('invalid or excessive physical event block')
        selected = (ticks >= self.start) & (ticks < self.end)
        t, c = ticks[selected].astype(np.int64), ids[selected].astype(np.int64)
        if not len(t):
            self.raw_events += len(ticks)
            return
        order = np.lexsort((t, c))
        t, c = t[order], c[order]
        begins = np.r_[0, np.flatnonzero(c[1:] != c[:-1])+1]
        ends = np.r_[begins[1:], len(c)]
        groups, sizes = c[begins], ends-begins
        same = c[1:] == c[:-1]
        if (np.any(same & (t[1:] <= t[:-1]))
                or np.any((self.counts[groups] > 0) & (t[begins] <= self.last[groups]))):
            raise ValueError('duplicate or backwards per-neuron spike across stream blocks')
        # Insert at most two preceding spikes into each already-sorted group.
        # This forms every new ISI triple exactly once, including both kinds
        # of triple that straddle a block boundary, without a second sort.
        history = np.minimum(self.counts[groups], 2)
        cumulative = np.cumsum(history)
        positions = begins+np.r_[0, cumulative[:-1]]
        combined_t = np.empty(len(t)+int(cumulative[-1]), dtype=np.int64)
        combined_t[np.arange(len(t))+np.repeat(cumulative, sizes)] = t
        one, two = history == 1, history == 2
        combined_t[positions[one]] = self.last[groups[one]]
        combined_t[positions[two]] = self.penultimate[groups[two]]
        combined_t[positions[two]+1] = self.last[groups[two]]
        combined_c = np.repeat(groups, sizes+history)
        triples = combined_c[2:] == combined_c[:-2]
        left = (combined_t[1:-1]-combined_t[:-2])[triples].astype(float)*self.dt_ms
        right = (combined_t[2:]-combined_t[1:-1])[triples].astype(float)*self.dt_ms
        total = left+right
        terms = 3*((left-right)/total)**2*(1+4*self.refractory_ms/total)
        sums = np.bincount(combined_c[:-2][triples], weights=terms, minlength=self.neurons)
        boundary = np.bincount(c[t == self.start], minlength=self.neurons)
        self.penultimate[groups] = np.where(sizes >= 2, t[np.maximum(ends-2, begins)], self.last[groups])
        self.last[groups] = t[ends-1]
        self.counts[groups] += sizes
        self.boundary += boundary
        self.sums += sums
        self.raw_events += len(ticks)

    def summarize(self):
        eligible = self.counts >= 3
        lvr = np.zeros(self.neurons)
        lvr[eligible] = self.sums[eligible]/(self.counts[eligible]-2)
        seconds = (self.end-self.start)*self.dt_ms/1000
        summary = dict(neurons=self.neurons, half_open_spikes=int(self.counts.sum()),
            lower_boundary_spikes=int(self.boundary.sum()), strict_spikes=int((self.counts-self.boundary).sum()),
            paper_rate_hz=float((self.counts-self.boundary).sum()/self.neurons/seconds),
            half_open_rate_hz=float(self.counts.sum()/self.neurons/seconds),
            paper_lvr_mean=float(lvr.mean()), lvr_eligible_cells=int(eligible.sum()),
            lvr_eligible_mean=float(lvr[eligible].mean()) if eligible.any() else None)
        return summary, dict(half_open_cell_counts=self.counts.copy(),
            lower_boundary_cell_counts=self.boundary.copy(), cell_lvr=lvr)
