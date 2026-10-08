import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from mam_paper_correlation import candidate_histogram, summarize_histogram


@pytest.mark.parametrize('end_tick',[25000,105000])
def test_closed_endpoint_and_inclusive_candidate_range(end_tick):
    counts = candidate_histogram(np.array([4999, 5000, end_tick, end_tick]), np.array([0, 0, 3000, 3001]), 0,end_tick=end_tick)
    assert counts.sum() == 2 and counts[0, 0] == 1 and counts[3000, -1] == 1
    report, ids, _ = summarize_histogram(counts, 0)
    assert ids.tolist() == [0, 3000] and report['available']


def test_constant_nonzero_and_insufficient_selection():
    counts = np.zeros((3001, 2000), dtype=int)
    counts[0] = 1
    counts[1, 4] = 1
    report, ids, _ = summarize_histogram(counts, 0)
    assert report['constant_nonzero_candidates'] == 1
    assert ids.tolist() == [1] and report['mean_pairwise_correlation'] is None


def test_invalid_input_rejected():
    for ticks, cells in [(np.array([-1]), np.array([0])), (np.array([25001]), np.array([0])),
                         (np.array([5000.]), np.array([0])), (np.array([5000]), np.array([-1]))]:
        with pytest.raises(ValueError):candidate_histogram(ticks, cells, 0)
