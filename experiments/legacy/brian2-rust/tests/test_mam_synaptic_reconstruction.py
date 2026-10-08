from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_mam_original_synaptic_inputs import comparison


def test_numerical_reconstruction_threshold_is_not_scientific_margin():
    reference=np.array([1.,2.,3.,4.])
    assert comparison(reference.copy(),reference)['passed']
    result=comparison(reference*1.01,reference)
    assert not result['passed']
    assert result['relative_l2_error']==pytest.approx(.01)
    assert result['relative_mean_error']==pytest.approx(.01)


@pytest.mark.parametrize('actual', [np.array([1.]), np.array([np.nan, 2.]), np.array([np.inf, 2.])])
def test_comparison_rejects_malformed_inputs(actual):
    with pytest.raises(ValueError):comparison(actual,np.array([1.,2.]))


def test_fc_comparison_uses_unordered_off_diagonal_pairs():
    from analyze_mam_paper_fc import compare_fc
    x=np.add.outer(np.arange(32.),np.arange(32.))/100
    y=x+.1
    np.fill_diagonal(x,1);np.fill_diagonal(y,-1)
    result=compare_fc(x,y)
    assert result['unordered_area_pairs']==496
    assert result['mean_absolute_difference']==pytest.approx(.1)
    assert result['maximum_absolute_difference']==pytest.approx(.1)
    assert result['rmse']==pytest.approx(.1)
    assert result['entry_correlation']==pytest.approx(1.)


def test_fc_rejects_missing_or_nonfinite_matrix():
    from analyze_mam_paper_fc import compare_fc
    with pytest.raises(ValueError):compare_fc(np.ones((31,31)),np.ones((32,32)))
    bad=np.ones((32,32));bad[0,1]=np.nan
    with pytest.raises(ValueError):compare_fc(bad,np.ones((32,32)))
