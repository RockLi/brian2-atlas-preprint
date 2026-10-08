"""Pinned modern Fig. 8 synaptic-input FC convention, not a BOLD/causality test.

Reference: INM-6/multi-area-model, commit
0a658be40bef3249cbe452f38809edf7d2f524ba, figures/Schmidt2018_dyn/
compute_synaptic_input.py and compute_functional_connectivity.py.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

SETTINGS = dict(
    source_commit='0a658be40bef3249cbe452f38809edf7d2f524ba',
    bin_ms=1., kernel_samples=20, convolution='numpy.convolve(mode=same)',
    weights='absolute mean recurrent weights times unrounded mean indegrees',
    area_weighting='unrounded official target population neuron numbers',
    external_input=False, conduction_delay_shift=False,
    fc='1 - squareform(pdist(centered_area_synaptic_inputs, metric=correlation))',
    bold=False, causal_inference=False,
)


def _series(values, maximum_rows):
    data = np.asarray(values, dtype=np.float64)
    if (data.ndim != 2 or not 1 <= data.shape[0] <= maximum_rows
            or not 20 <= data.shape[1] <= 100000 or not np.isfinite(data).all()):
        raise ValueError('finite bounded row-by-millisecond series required')
    return data


def functional_connectivity(area_inputs):
    """Fig. 8 zero-lag FC. Undefined off-diagonals remain NaN for flat series."""
    data = _series(area_inputs, 32)
    centered = data - data.mean(axis=1, keepdims=True)
    # Match official pdist/squareform (including its diagonal convention).
    matrix = 1. - squareform(pdist(centered, metric='correlation'))
    return matrix, np.flatnonzero(np.all(centered == 0., axis=1))


def synaptic_area_inputs(population_rates, weights, indegrees, neuron_numbers,
                         area_indices, tau_syn_ms):
    """Fig. 8 recurrent-input approximation, preserving source summation order.

    W/K are complete official [target, source] matrices, not sampled graph
    counts. N are unrounded M.N. All inputs must have the same population order.
    The figure script uses tau_syn_ex for its single common filter. It sums
    before filtering with a length-20 *centered* convolution, including its edge
    convention. Do not replace this with a causal recursive filter or delays.
    """
    rates = _series(population_rates, 254)
    n = len(rates)
    w, k, neurons = (np.asarray(x, dtype=np.float64)
                     for x in (weights, indegrees, neuron_numbers))
    areas = np.asarray(area_indices)
    if (w.shape != (n, n) or k.shape != (n, n) or neurons.shape != (n,)
            or not np.isfinite(w).all() or not np.isfinite(k).all()
            or not np.isfinite(neurons).all() or np.any(k < 0)
            or np.any(neurons <= 0) or np.any(rates < 0)
            or areas.shape != (n,) or areas.dtype.kind not in 'iu'
            or not np.array_equal(np.unique(areas), np.arange(len(np.unique(areas))))
            or len(np.unique(areas)) > 32
            or not np.isfinite(tau_syn_ms) or tau_syn_ms <= 0):
        raise ValueError('invalid full synaptic-input matrices or population metadata')
    kernel = np.exp(-np.arange(20., dtype=np.float64) / tau_syn_ms)
    result = np.empty((len(np.unique(areas)), rates.shape[1]), dtype=np.float64)
    for area in range(len(result)):
        indices = np.flatnonzero(areas == area)
        currents = []
        for target in indices:
            total = np.zeros(rates.shape[1], dtype=np.float64)
            for source in range(n):
                total += rates[source] * abs(w[target, source]) * k[target, source]
            currents.append(np.convolve(kernel, total, mode='same'))
        result[area] = np.average(currents, axis=0, weights=neurons[indices])
    return result
