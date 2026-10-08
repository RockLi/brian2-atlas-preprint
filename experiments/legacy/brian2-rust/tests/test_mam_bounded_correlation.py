from pathlib import Path
import sys
import hashlib
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_paper_correlation as m


@pytest.mark.parametrize('end',[25000,105000,505000,1005000])
@pytest.mark.parametrize('block',[1,7,131072])
def test_streaming_candidate_selection_matches_independent_event_counts(end,block):
    rng=np.random.default_rng(1750);ticks=np.r_[rng.integers(0,end+1,128),[4999,5000,end,end]]
    cells=np.r_[rng.integers(6,3020,128),[7,7,3007,3008]]
    order=rng.permutation(len(ticks));ticks=ticks[order];cells=cells[order]
    counter=m.CandidateHistogram(7,end_tick=end)
    for start in range(0,len(ticks),block):counter.add(ticks[start:start+block],cells[start:start+block])
    assert counter.raw_events==len(ticks) and counter.counts.dtype==np.dtype('<u4')
    expected={}
    for t,c in zip(ticks,cells,strict=True):
        if 7<=c<=3007 and 5000<=t<=end:
            key=(c-7,min((t-5000)//10,counter.nbins-1));expected[key]=expected.get(key,0)+1
    assert int(counter.counts.sum())==sum(expected.values())
    for key,value in expected.items():assert counter.counts[key]==value


@pytest.mark.parametrize('row_block_size',[1,7,32,128])
@pytest.mark.parametrize('dtype',['int64','uint32'])
def test_row_blocks_keep_exact_selection_and_declared_correlation_tolerance(row_block_size,dtype):
    counts=np.zeros((3001,2000),dtype=dtype)
    counts[0]=2;counts[1,20]=3;counts[2,30]=4
    rng=np.random.default_rng(1729);counts[3:2010]=rng.integers(0,3,(2007,2000))
    old,old_ids,old_selected=m.summarize_histogram(counts,3)
    new,new_ids,new_selected=m.summarize_histogram_bounded(counts,3,row_block_size=row_block_size)
    np.testing.assert_array_equal(old_ids,new_ids);np.testing.assert_array_equal(old_selected,new_selected)
    assert new_selected.dtype==np.dtype(dtype)
    assert abs(old['mean_pairwise_correlation']-new['mean_pairwise_correlation'])<=1e-12
    for k in old:
        if k!='mean_pairwise_correlation':assert old[k]==new[k]


@pytest.mark.parametrize('n',[0,1,2])
def test_insufficient_or_sparse_candidates_match_legacy(n):
    counts=np.zeros((3001,2000),dtype='<u4');counts[0]=1
    for i in range(n):counts[i+1,i]=2
    a,ids,x=m.summarize_histogram(counts,0);b,ids2,y=m.summarize_histogram_bounded(counts,0)
    assert a==b;np.testing.assert_array_equal(ids,ids2);np.testing.assert_array_equal(x,y)


def test_rejected_over_budget_block_does_not_wrap_or_mutate(monkeypatch):
    monkeypatch.setattr(m,'MAX_STREAM_EVENTS',5)
    c=m.CandidateHistogram(0);c.add(np.full(5,5000),np.zeros(5,dtype=int))
    with pytest.raises(ValueError):c.add(np.array([5000]),np.array([0]))
    assert c.raw_events==5 and c.counts[0,0]==5 and c.counts.sum()==5
    c.counts[0,0]=6
    with pytest.raises(ValueError):m.summarize_histogram_bounded(c.counts,0)


@pytest.mark.parametrize('ticks,cells',[(np.array([-1]),np.array([0])),(np.array([25001]),np.array([0])),(np.array([5000]),np.array([-1])),(np.array([5000.]),np.array([0])),(np.array([5000],dtype='<u8'),np.array([2**64-1],dtype='<u8')),(np.zeros(131073,dtype=int),np.zeros(131073,dtype=int))])
def test_bad_physical_blocks_rejected(ticks,cells):
    c=m.CandidateHistogram(0)
    with pytest.raises(ValueError):c.add(ticks,cells)
    assert c.raw_events==0 and c.counts.sum()==0


@pytest.mark.parametrize('first,end',[(True,25000),(-1,25000),(2**63,25000),(0,25001),(0,1005000.)])
def test_invalid_stream_configuration(first,end):
    with pytest.raises(ValueError):m.CandidateHistogram(first,end_tick=end)


def test_invalid_row_block_budget():
    counts=np.zeros((3001,2000),dtype='<u4')
    for size in [0,129,1.5,True]:
        with pytest.raises(ValueError):m.summarize_histogram_bounded(counts,0,row_block_size=size)
