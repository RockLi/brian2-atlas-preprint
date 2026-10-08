from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_mam_paper_time_series import Counts
from mam_paper_rate_bins import full_rate_bins
from mam_paper_spectrum import spectrum


@pytest.mark.parametrize('end_tick',[25000,105000,505000,1005000])
def test_chunked_histograms_match_independent_helper_and_boundary_identity(end_tick):
    ticks=np.array([0,4999,5000,5004,5005,5014,5015,end_tick-1,end_tick]*3)
    groups=np.arange(len(ticks))%3;c=Counts(3,end=end_tick)
    for i in range(0,len(ticks),4):c.add(ticks[i:i+4],groups[i:i+4])
    for pop in range(3):
        np.testing.assert_array_equal(c.shifted[pop],full_rate_bins(ticks[groups==pop],1000,end_tick))
        np.testing.assert_array_equal(c.frozen[pop],np.histogram(ticks[groups==pop][ticks[groups==pop]<end_tick],bins=np.arange(5000,end_tick+1,10))[0])
    np.testing.assert_array_equal(c.shifted.sum(axis=1),c.frozen.sum(axis=1)-c.lower+c.terminal)


def test_bad_or_unbounded_blocks_and_spectra_are_rejected():
    c=Counts(2)
    for t,g in [(np.array([-1]),np.array([0])),(np.array([25001]),np.array([0])),(np.array([5005]),np.array([2])),(np.array([5005.]),np.array([0])),(np.zeros(131073,dtype=int),np.zeros(131073,dtype=int))]:
        with pytest.raises(ValueError):c.add(t,g)
    for x in [np.zeros(100),np.zeros(100001),np.array([np.nan]*2000),np.zeros((2,2000))]:
        with pytest.raises(ValueError):spectrum(x)


@pytest.mark.parametrize('end_tick',[25000,105000,505000,1005000])
@pytest.mark.parametrize('block_size',[1,7,131072])
def test_unsorted_repeated_bins_preserve_chunk_independent_integer_counts(end_tick,block_size):
    rng=np.random.default_rng(1729)
    ticks=np.concatenate([rng.integers(0,end_tick+1,size=513),np.tile([5000,5004,5005,end_tick-1,end_tick],25)])
    groups=rng.integers(0,4,size=len(ticks));order=rng.permutation(len(ticks));ticks=ticks[order];groups=groups[order]
    counts=Counts(4,end_tick)
    for start in range(0,len(ticks),block_size):counts.add(ticks[start:start+block_size],groups[start:start+block_size])
    assert counts.raw==len(ticks)
    for group in range(4):
        t=ticks[groups==group]
        np.testing.assert_array_equal(counts.shifted[group],full_rate_bins(t,1000,end_tick))
        expected=np.zeros((end_tick-5000)//10,dtype=np.int64)
        for tick in t:
            if 5000<=tick<end_tick:expected[(tick-5000)//10]+=1
        np.testing.assert_array_equal(counts.frozen[group],expected)
        assert counts.lower[group]==sum(5000<=tick<5005 for tick in t)
        assert counts.terminal[group]==sum(tick==end_tick for tick in t)


def test_primary_duration_never_allocates_whole_grid_temporary(monkeypatch):
    counts=Counts(3,1005000)
    original=np.bincount
    def bounded_bincount(values,*args,**kwargs):
        assert kwargs.get('minlength',0)<=3
        return original(values,*args,**kwargs)
    monkeypatch.setattr(np,'bincount',bounded_bincount)
    ticks=np.array([5005]*131072,dtype=np.uint32);groups=np.full(len(ticks),2,dtype=np.uint32)
    counts.add(ticks,groups)
    assert counts.shifted[2,0]==counts.frozen[2,0]==131072
    assert counts.shifted.sum()==counts.frozen.sum()==131072
    assert counts.shifted.dtype==counts.frozen.dtype==np.dtype('int64')


@pytest.mark.parametrize('populations,end',[(True,25000),(0,25000),(255,25000),(1,True),(1,1005001),(1,1005000.)])
def test_histogram_configuration_limits(populations,end):
    with pytest.raises(ValueError):Counts(populations,end)
