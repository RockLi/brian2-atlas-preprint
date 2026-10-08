"""Bound raster storage before allocating selected point payloads."""
import numpy as np
from mam_streamed_cell_metrics import CHUNK


class RasterBudget:
    def __init__(self, maximum=3000000):
        if type(maximum) is not int or not 1 <= maximum <= 3000000:
            raise ValueError('raster point budget must be in 1..3000000')
        self.maximum, self.used = maximum, 0

    def reserve(self, points):
        if type(points) is not int or points < 0 or self.used+points > self.maximum:
            raise ValueError('raster point budget exceeded before payload allocation')
        self.used += points


class RasterSample:
    def __init__(self, neurons, seed, budget, *, start_tick=5000, end_tick=25000, dt_seconds=.0001):
        if (type(neurons) is not int or not 1 <= neurons <= 4200000
                or type(seed) is not int or seed < 0 or not isinstance(budget,RasterBudget)
                or type(start_tick) is not int or type(end_tick) is not int
                or not 0 <= start_tick < end_tick <= 1005000 or dt_seconds != .0001):
            raise ValueError('invalid bounded raster contract')
        self.neurons, self.budget = neurons,budget
        self.start, self.end, self.dt = start_tick,end_tick,dt_seconds
        self.selected = np.sort(np.random.default_rng(seed).choice(neurons,max(1,int(np.ceil(.03*neurons))),replace=False))
        self.lookup = np.full(neurons,-1,dtype=np.int32)
        self.lookup[self.selected] = np.arange(len(self.selected),dtype=np.int32)
        self.ticks, self.indices, self.finished = [],[],False

    def add(self,ticks,ids):
        if (self.finished or ticks.ndim != 1 or ids.shape != ticks.shape
                or ticks.dtype.kind not in 'iu' or ids.dtype.kind not in 'iu'
                or len(ticks)>CHUNK or np.any(ticks<0) or np.any(ticks>self.end)
                or np.any(ids<0) or np.any(ids>=self.neurons)):
            raise ValueError('invalid raster event block or finished sample')
        local = self.lookup[ids]
        keep = (ticks>=self.start)&(ticks<self.end)&(local>=0)
        points = int(keep.sum())
        self.budget.reserve(points)
        if points:
            self.ticks.append(ticks[keep].astype(np.int64))
            self.indices.append(local[keep])

    def finish(self):
        if self.finished:
            raise ValueError('raster sample already finished')
        ticks = np.concatenate(self.ticks) if self.ticks else np.empty(0,dtype=np.int64)
        ids = np.concatenate(self.indices) if self.indices else np.empty(0,dtype=np.int32)
        # Sorting only the bounded selected points exactly preserves the old
        # native global stable-sort followed by point selection, including ties.
        order = np.argsort(ticks,kind='stable')
        result = dict(selected=len(self.selected),times=ticks[order]*self.dt,indices=ids[order])
        self.ticks.clear();self.indices.clear();self.lookup=None;self.finished=True
        return result
