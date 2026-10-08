from pathlib import Path
import sys,importlib.util
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import mam_streamed_activity as m
spec=importlib.util.spec_from_file_location('frozen_activity',ROOT/'python/brian2_rust/multi_area_analysis.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def compare(ticks,ids,n,block,seed=20260908,end=25000,sample=2000):
    order=np.argsort(ticks,kind='stable')
    a,h,x=old.population_activity(ticks[order],ids[order],n,start_tick=5000,end_tick=end,dt_seconds=.0001,bin_ticks=10,seed=seed,sample_size=sample)
    def blocks():
        for lo in range(0,len(ticks),block):yield ticks[lo:lo+block],ids[lo:lo+block]
    b,j,y=m.population_activity_streamed(blocks,n,end_tick=end,seed=seed,sample_size=sample,expected_raw=len(ticks))
    np.testing.assert_array_equal(h,j);assert x.keys()==y.keys() and a.keys()==b.keys()
    for k in x:
        assert x[k].dtype==y[k].dtype
        if k=='lvr_values':np.testing.assert_allclose(x[k],y[k],rtol=1e-12,atol=1e-12)
        else:np.testing.assert_array_equal(x[k],y[k])
    for k in a:
        if k in ['pairwise_corr_mean','lvr_zero_padded_mean','lvr_eligible_mean'] and a[k] is not None:np.testing.assert_allclose(a[k],b[k],rtol=1e-12,atol=1e-12)
        else:assert a[k]==b[k]


@pytest.mark.parametrize('block',[1,7,131072])
@pytest.mark.parametrize('end',[25000,105000])
@pytest.mark.parametrize('sample',[3,2000])
def test_frozen_draw_order_boundaries_counts_and_sampled_statistics(block,end,sample):
    rng=np.random.default_rng(1750);ticks=[];ids=[]
    for cell in range(21):
        t=np.unique(np.r_[4999,5000,end-1,end,np.sort(rng.choice(np.arange(5010,end-10),70,replace=False))])
        ticks.extend(t);ids.extend([cell]*len(t))
    compare(np.array(ticks),np.array(ids),30,block,end=end,sample=sample)


@pytest.mark.parametrize('mode',['empty','one','constant','equal-isi'])
def test_silent_sparse_and_constant_correlation_rows(mode):
    ticks=np.array([],dtype=int);ids=np.array([],dtype=int)
    if mode=='one':ticks=np.array([5000]);ids=np.array([1])
    if mode=='constant':ticks=np.r_[np.arange(5000,25000,10),5011,5012];ids=np.r_[np.zeros(2000,dtype=int),1,2]
    if mode=='equal-isi':ticks=np.arange(5000,25000,21);ids=np.zeros(len(ticks),dtype=int)
    order=np.argsort(ticks,kind='stable');compare(ticks[order],ids[order],5,7)


@pytest.mark.parametrize('rows',[0,1,2,2000])
@pytest.mark.parametrize('block',[1,32,128])
def test_blocked_diagnostic_gram_matches_legacy(rows,block):
    rng=np.random.default_rng(1729);counts=rng.integers(0,4,(rows,2000),dtype=np.uint32)
    if rows:counts[0]=2
    a,n=old.mean_pairwise_correlation(counts);b,j=m.mean_pairwise_correlation_bounded(counts,row_block_size=block)
    assert n==j
    if a is None:assert b is None
    else:np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('end',[505000,1005000])
def test_primary_grid_keeps_selected_ids_and_exact_integer_counts(end):
    ticks=np.array([4999,5000,5010,end-1,end]);ids=np.array([0,0,0,1,1])
    s,h,a=m.population_activity_streamed(lambda:iter([(ticks,ids)]),3,end_tick=end,expected_raw=5)
    assert s['observed_spikes']==3 and h.shape==((end-5000)//10,) and h[0]==1 and h[1]==1 and h[-1]==1
    np.testing.assert_array_equal(a['single_cell_rates_hz'],np.array([2,1,0])/((end-5000)*.0001))
    assert s['lvr_eligible_cells']==0


def test_nonrepeatable_input_and_invalid_contracts_fail():
    empty=lambda:iter([])
    for kwargs in [dict(expected_raw=-1),dict(expected_raw=2**32),dict(expected_raw=0,end_tick=25001),dict(expected_raw=0,sample_size=2001),dict(expected_raw=0,start_tick=4999)]:
        with pytest.raises(ValueError):m.population_activity_streamed(empty,4,**kwargs)
    calls=0
    def changed():
        nonlocal calls
        calls+=1
        yield np.array([5000]),np.array([calls-1])
    with pytest.raises(AssertionError):m.population_activity_streamed(changed,4,expected_raw=1)
    with pytest.raises(ValueError):m.mean_pairwise_correlation_bounded(np.array([[-1,0]]))
    with pytest.raises(ValueError):m.mean_pairwise_correlation_bounded(np.zeros((2001,2),dtype=int))


def test_only_lvr_sample_requires_positive_per_neuron_intervals():
    # The frozen diagnostic permits duplicate timestamps in a neuron outside
    # the LvR sample: it contributes counts/correlation, not an ISI statistic.
    chosen=np.random.default_rng(7).choice(8,2,replace=False)
    other=next(i for i in range(8) if i not in chosen)
    compare(np.array([5000,5000,5010]),np.full(3,other),8,1,seed=7,sample=2)
