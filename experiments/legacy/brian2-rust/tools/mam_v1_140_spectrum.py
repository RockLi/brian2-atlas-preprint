"""Four explicitly named V1 140-cell spectrum views for a future source-bound analysis.

Inputs are complete, already selected 1-ms cell-count aggregates. This module
does not choose cells, read raw events, infer historical IDs, or decide scientific
acceptance. Selection provenance must be established by a separate collector.
"""

import numpy as np
import scipy
from scipy.signal import welch

from mam_paper_spectrum import SETTINGS, spectrum


V1_POPULATIONS = (
    "mam_V1_23E", "mam_V1_23I", "mam_V1_4E", "mam_V1_4I",
    "mam_V1_5E", "mam_V1_5I", "mam_V1_6E", "mam_V1_6I",
)
V1_SAMPLE_COUNTS = (34, 9, 50, 12, 15, 3, 14, 3)
OBSERVATION_BINS = 100_000
OBSERVATION_SECONDS = 100


def four_views(population_counts, official_neurons, selected_ids, eligibility_spikes):
    """Return equal-cell/full-population-weighted rates and boxcar/Hann PSDs.

    ``population_counts`` is 8 x 100000 selected-cell spike counts on the
    caller's declared 1-ms grid. ``eligibility_spikes`` counts each selected
    cell on the wrapper's physical (500,T] ms interval, which can differ at
    endpoints from its shifted histogram grid. Both must be source-bound by
    the caller; this pure function checks their shape and eligibility only.
    """
    if scipy.__version__ != "1.18.1":
        raise ValueError("unverified SciPy Welch environment")
    counts = np.asarray(population_counts)
    neurons = np.asarray(official_neurons, dtype=np.float64)
    if (counts.shape != (8, OBSERVATION_BINS) or counts.dtype.kind not in "iu"
            or np.any(counts < 0) or np.any(counts > 2**32 - 1)
            or neurons.shape != (8,) or not np.isfinite(neurons).all()
            or np.any(neurons <= 0)):
        raise ValueError("invalid V1 selected-cell counts or population sizes")
    samples = np.rint(140 * neurons / neurons.sum()).astype(np.int64)
    if tuple(samples) != V1_SAMPLE_COUNTS:
        raise ValueError("official V1 sizes do not yield the pinned 140-cell allocation")
    if len(selected_ids) != 8 or len(eligibility_spikes) != 8:
        raise ValueError("eight selected-ID and eligibility groups required")
    all_ids = []
    for ids, spikes, required in zip(selected_ids, eligibility_spikes, samples, strict=True):
        ids = np.asarray(ids)
        spikes = np.asarray(spikes)
        if (ids.shape != (required,) or ids.dtype.kind not in "iu"
                or np.any(ids < 0) or spikes.shape != (required,)
                or spikes.dtype.kind not in "iu" or np.any(spikes < 0)
                or np.any(spikes / OBSERVATION_SECONDS <= 0.56)
                or np.unique(ids).size != required):
            raise ValueError("selected IDs or strict rate eligibility invalid")
        all_ids.extend(int(i) for i in ids)
    if len(set(all_ids)) != 140:
        raise ValueError("selected neuron IDs overlap across populations")

    # Modern wrapper weights by selected counts, reducing to an equal-cell rate.
    equal_cell = counts.sum(axis=0, dtype=np.uint64).astype(np.float64) * (1000. / 140.)
    # Inferred historical convention weights each sampled population rate by
    # its full, unrounded official M.N size. It remains a distinct diagnostic.
    full_population = np.sum(
        counts.astype(np.float64) * (1000. * neurons / samples)[:, None],
        axis=0,
    ) / neurons.sum()
    rates = {"modern_equal_cell": equal_cell,
             "inferred_full_population_weighted": full_population}
    spectra = {}
    frequency = None
    for rate_name, rate in rates.items():
        f_hann, hann = spectrum(rate)
        f_boxcar, boxcar = welch(rate - rate.mean(), **{**SETTINGS, "window": "boxcar"})
        if not np.array_equal(f_hann, f_boxcar):
            raise ValueError("Welch frequency grids differ")
        if frequency is None:
            frequency = f_hann
        elif not np.array_equal(frequency, f_hann):
            raise ValueError("V1 frequency grids differ")
        spectra[f"{rate_name}__declared_boxcar"] = boxcar
        spectra[f"{rate_name}__effective_hann"] = hann
    return dict(population_names=V1_POPULATIONS, sample_counts=V1_SAMPLE_COUNTS,
                selected_ids=tuple(tuple(int(i) for i in group) for group in selected_ids),
                rates_hz=rates, frequency_hz=frequency,
                power_hz2_per_hz=spectra, scipy_version=scipy.__version__,
                scientific_acceptance=False)
