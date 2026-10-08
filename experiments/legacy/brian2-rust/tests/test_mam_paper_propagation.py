from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from mam_paper_propagation import time_axis,transforms,smoothed_curve,select_peak,fit_hierarchy


def test_fft_covariance_against_direct_circular_sum_and_even_kernel():
    n=10000;x=np.zeros(n);y=np.zeros(n);x[[10,23,31]]=[2,3,5];y[[7,19,26]]=[4,6,2]
    centered=np.array([x-x.mean(),y-y.mean()]);f=transforms([x,y]);actual=smoothed_curve(f[0],f[1]);times=time_axis(n)
    kernel=np.exp(-np.arange(-5.,5.)**2/8)/np.sqrt(8*np.pi)
    for lag in [-25,-1,0,1,25]:
        # same-mode on length-10 kernel slices full convolution from index 4.
        expected=sum(kernel[k]*np.dot(centered[0],np.roll(centered[1],lag+4-k)) for k in range(10))*1000/n
        position=int(np.flatnonzero(np.arange(-n/2+1,n/2+1)==lag)[0])
        assert actual[position]==pytest.approx(expected,rel=3e-13,abs=1e-14)
    assert times[n//2-1]==0
    assert times[n//2]==pytest.approx(n/(n-2))


def test_shift_sign_and_auto_peak_rule():
    n=10000;rng=np.random.default_rng(51);b=rng.normal(size=n);a=np.roll(b,15)
    f=transforms([a,b]);t=time_axis(n)
    lag,_,status=select_peak(t,smoothed_curve(f[0],f[1]))
    assert status=='selected' and 14<lag<18
    assert select_peak(t,smoothed_curve(f[0],f[0]),same_area=True)[0]==0
    assert select_peak(t,np.zeros(n),excluded=True)[2]=='excluded_MDP'
    assert select_peak(t,np.zeros(n))[2]=='no_selected_extremum'


def test_hierarchy_is_same_complete_pair_objective_even_when_asymmetric():
    rng=np.random.default_rng(29);c=rng.normal(size=(7,7));np.fill_diagonal(c,0)
    got=fit_hierarchy(c);h=got['levels_ms']
    # Independently solve the incidence-matrix least-squares problem with sum h=0.
    incidence=np.array([np.eye(7)[i]-np.eye(7)[j] for i in range(7) for j in range(7)])
    reference=np.linalg.lstsq(incidence,c.ravel(),rcond=None)[0]
    np.testing.assert_allclose(h,reference,rtol=1e-13,atol=1e-13)
    for seed in [1,2,3]:
        start=np.random.default_rng(seed).random(7)
        result=minimize(lambda x:np.linalg.norm(x[:,None]-x[None,:]-c),start)
        assert abs(result.fun-got['residual_norm'])<1e-8
    assert got['normalized_levels'].min()==0 and got['normalized_levels'].max()==1


@pytest.mark.parametrize('shape',[(33,10000),(2,2000),(2,100001)])
def test_rate_boundaries(shape):
    with pytest.raises(ValueError):transforms(np.zeros(shape))


def test_hierarchy_does_not_impute_missing_pairs():
    c=np.zeros((3,3));c[0,1]=np.nan
    with pytest.raises(ValueError):fit_hierarchy(c)
    result=fit_hierarchy(np.zeros((3,3)))
    assert not result['normalization_defined'] and np.isnan(result['normalized_levels']).all()


def test_zero_reciprocal_is_recomputed_instead_of_forced_symmetric(monkeypatch):
    import mam_paper_propagation as module
    spectra=np.array([np.zeros(10000,dtype=complex),np.ones(10000,dtype=complex)])
    monkeypatch.setattr(module,'transforms',lambda _:spectra)
    def curve(a,b):return np.array([a[0].real,b[0].real])
    def peak(time,values,**kwargs):
        i,j=map(int,values)
        return (1. if (i,j)==(1,0) else 0.),1.,'selected'
    monkeypatch.setattr(module,'smoothed_curve',curve);monkeypatch.setattr(module,'select_peak',peak)
    matrix,diag=module.lag_matrix(np.zeros((2,10000)),['A','B'])
    assert matrix.tolist()==[[0.,0.],[1.,0.]]
    assert diag['computed_curves']==4


def test_short_window_plot_refuses_primary_duration_before_block_analysis():
    from compare_mam_paper_lags import require_short_window_cases
    require_short_window_cases(100.,10.,10.)
    for values in [(100.,100.,100.),(100.,10.,100.),(10.,10.,10.)]:
        with pytest.raises(ValueError):require_short_window_cases(*values)
