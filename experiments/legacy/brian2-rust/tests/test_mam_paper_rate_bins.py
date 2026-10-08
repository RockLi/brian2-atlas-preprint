from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from mam_paper_rate_bins import full_rate_bins


def test_wrapper_and_shifted_helper_endpoint_intersection():
    ticks=np.array([5000,5001,5004,5005,5014,5015,5020,5021,5025])
    np.testing.assert_array_equal(full_rate_bins(ticks,1000,5020),[2.,2.])
    np.testing.assert_array_equal(full_rate_bins(np.array([],dtype=np.int64),1000,5020),[0.,0.])
    np.testing.assert_array_equal(full_rate_bins(np.array([5005,5020,2**64-1],dtype=np.uint64),1000,5020),[1.,1.])


def test_invalid_or_unbounded_grid_rejected_before_histogram():
    for ticks,n,end in [(np.array([5005.]),1000,5020),(np.array([-1]),1000,5020),
                        (np.array([[5005]]),1000,5020),(np.array([5005]),0,5020),
                        (np.array([5005]),1000,5021),(np.array([5005]),1000,10**12),
                        (np.array([5005]),True,5020)]:
        with pytest.raises(ValueError):full_rate_bins(ticks,n,end)
