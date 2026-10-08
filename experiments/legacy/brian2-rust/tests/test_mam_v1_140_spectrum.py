from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mam_v1_140_spectrum import four_views, V1_SAMPLE_COUNTS, OBSERVATION_BINS


# Official unrounded M.N values from the pinned provenance audit, not fitted to
# any observed Rust/NEST outcome or published power spectrum.
OFFICIAL_V1 = np.array([
    47386.686717394696, 13366.239438634659, 70387.2575552122,
    17597.61734633847, 20740.873834372647, 4554.439305898323,
    19839.788666495304, 4063.0564076990745,
])


def fixture():
    counts = np.zeros((8, OBSERVATION_BINS), dtype=np.uint32)
    counts[:, 200] = 1
    counts[0, 1000] = 2
    ids = []
    cursor = 0
    for size in V1_SAMPLE_COUNTS:
        ids.append(np.arange(cursor, cursor + size, dtype=np.uint32))
        cursor += size
    eligibility = [np.full(size, 57, dtype=np.uint32) for size in V1_SAMPLE_COUNTS]
    return counts, ids, eligibility


def test_four_distinct_views_match_external_formula_and_explicit_welch():
    counts, ids, eligibility = fixture()
    result = four_views(counts, OFFICIAL_V1, ids, eligibility)
    equal = result["rates_hz"]["modern_equal_cell"]
    weighted = result["rates_hz"]["inferred_full_population_weighted"]
    assert equal[200] == pytest.approx(1000 * 8 / 140)
    assert weighted[200] == pytest.approx(
        sum(1000 * n / s for n, s in zip(OFFICIAL_V1, V1_SAMPLE_COUNTS))
        / OFFICIAL_V1.sum())
    # The pinned source audit predicts these eight single-event amplitudes.
    expected = [6.842375386870734, 6.985718650500879, 7.041298029166194,
                7.11212432688607, 7.159526581750643, 7.4088008949366415,
                7.503122327371861, 7.669886971941684]
    amplitudes = 1000 * OFFICIAL_V1 / (np.array(V1_SAMPLE_COUNTS) * OFFICIAL_V1.sum())
    np.testing.assert_allclose(np.sort(amplitudes), expected, rtol=0, atol=1e-12)
    assert set(result["power_hz2_per_hz"]) == {
        "modern_equal_cell__declared_boxcar", "modern_equal_cell__effective_hann",
        "inferred_full_population_weighted__declared_boxcar",
        "inferred_full_population_weighted__effective_hann"}
    frequency, reference = welch(equal - equal.mean(), fs=1000., window="boxcar",
                                 nperseg=1024, noverlap=1000, nfft=1024,
                                 detrend="constant", return_onesided=True,
                                 scaling="density", average="mean")
    np.testing.assert_array_equal(result["frequency_hz"], frequency)
    np.testing.assert_allclose(
        result["power_hz2_per_hz"]["modern_equal_cell__declared_boxcar"],
        reference, rtol=0, atol=0)
    assert not np.array_equal(
        result["power_hz2_per_hz"]["modern_equal_cell__declared_boxcar"],
        result["power_hz2_per_hz"]["modern_equal_cell__effective_hann"])
    assert result["scientific_acceptance"] is False


def test_missing_or_ineligible_cells_rejected():
    counts, ids, eligibility = fixture()
    eligibility[0][0] = 56  # exactly 0.56 Hz, but source requires strictly greater
    with pytest.raises(ValueError, match="eligibility"):
        four_views(counts, OFFICIAL_V1, ids, eligibility)
    eligibility[0][0] = 57
    ids[1][0] = ids[0][0]
    with pytest.raises(ValueError, match="overlap"):
        four_views(counts, OFFICIAL_V1, ids, eligibility)
