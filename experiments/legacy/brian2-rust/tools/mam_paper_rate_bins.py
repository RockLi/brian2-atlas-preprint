"""Full-rate bins matching the pinned modern MAM wrapper's physical grid.

This separate view does not change the frozen [500,2500) diagnostic bins.
The supported contract is 0.1 ms spike ticks and 1 ms bins, after 500 ms.
"""
import numpy as np


def full_rate_bins(physical_ticks, neurons, end_tick=25000):
    ticks = np.asarray(physical_ticks)
    if ticks.ndim != 1 or ticks.dtype.kind not in 'iu':
        raise ValueError('physical ticks must be a one-dimensional integer array')
    if (type(neurons) is not int or not 0 < neurons <= 4200000 or type(end_tick) is not int
            or not 5000 < end_tick <= 1005000 or (end_tick-5000) % 10):
        raise ValueError('positive neuron count and whole-millisecond duration required')
    if ticks.dtype.kind == 'i' and np.any(ticks < 0):
        raise ValueError('negative physical spike tick')
    # Wrapper retains (500,T] ms, helper bins [500.5,T+0.5] ms.
    # The intersection is [500.5,T]; never recover this from old 1 ms bins.
    selected = ticks[(ticks >= 5005) & (ticks <= end_tick)].astype(np.int64)
    counts = np.bincount((selected-5005)//10, minlength=(end_tick-5000)//10)
    # Preserve the official arithmetic order as well as the bin counts.
    return counts.astype(np.float64)/(neurons*1.0/1000.0)
