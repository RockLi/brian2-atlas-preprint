from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_mam_paper_cell_metrics import cell_metrics
import mam_streamed_cell_metrics as m


def compare(ticks,ids,n,block,**kwargs):
    expected,old=cell_metrics(ticks,ids,n,**kwargs)
    stream=m.CellMetrics(n,**kwargs)
    for lo in range(0,len(ticks),block):stream.add(ticks[lo:lo+block],ids[lo:lo+block])
    actual,new=stream.summarize()
    assert stream.raw_events==len(ticks)
    for key in old:
        if key=='cell_lvr':np.testing.assert_allclose(new[key],old[key],rtol=1e-12,atol=1e-12)
        else:assert new[key].dtype==old[key].dtype;np.testing.assert_array_equal(new[key],old[key])
    for key in expected:
        if key in ['paper_lvr_mean','lvr_eligible_mean'] and expected[key] is not None:
            np.testing.assert_allclose(actual[key],expected[key],rtol=1e-12,atol=1e-12)
        else:assert actual[key]==expected[key]
    return stream


@pytest.mark.parametrize('block',[1,2,3,7,131072])
@pytest.mark.parametrize('end',[25000,105000,505000,1005000])
def test_cross_block_triples_and_declared_long_windows(block,end):
    rng=np.random.default_rng(1750);ticks=[];ids=[]
    for cell in range(21):
        t=np.sort(rng.choice(np.arange(4990,end+1),size=50+cell,replace=False))
        if cell==0:t=np.unique(np.r_[4999,5000,5001,end-1,end,t])
        ticks.extend(t);ids.extend([cell]*len(t))
    # Each cell's train is monotonic, global ticks reset when changing cell.
    compare(np.array(ticks),np.array(ids),25,block,end=end)


@pytest.mark.parametrize('block',[1,2,5,131072])
def test_mixed_cells_and_boundary_triples(block):
    ticks=np.array([4999,5000,5000,5001,5010,5020,5040,5070,5090,25000])
    ids=np.array([0,0,2,1,2,0,2,0,2,0])
    compare(ticks,ids,4,block)


def test_unsorted_within_one_block_and_silent_sparse_cells():
    compare(np.array([5040,5000,5020,4999,25000]),np.array([1,1,1,0,0]),4,131072)
    compare(np.array([],dtype=int),np.array([],dtype=int),4,1)
    compare(np.array([5000,5001,5010]),np.array([0,1,1]),4,1)


@pytest.mark.parametrize('bad',[(np.array([5000,5000]),np.array([0,0])),(np.array([25001]),np.array([0])),(np.array([-1]),np.array([0])),(np.array([5000]),np.array([3])),(np.array([5000.]),np.array([0])),(np.zeros(131073,dtype=int),np.zeros(131073,dtype=int))])
def test_rejections_leave_state_unchanged(bad):
    c=m.CellMetrics(3)
    with pytest.raises(ValueError):c.add(*bad)
    assert c.raw_events==0 and not c.counts.any() and not c.sums.any() and np.all(c.last==-1)


def test_cross_block_backwards_duplicate_and_budget_rejected_before_mutation(monkeypatch):
    c=m.CellMetrics(3);c.add(np.array([5000,5020]),np.array([0,0]))
    for t in [5020,5010]:
        with pytest.raises(ValueError,match='backwards'):c.add(np.array([t]),np.array([0]))
        assert c.raw_events==2 and c.counts.tolist()==[2,0,0] and c.last[0]==5020
    monkeypatch.setattr(m,'MAX_EVENTS',2)
    with pytest.raises(ValueError):c.add(np.array([5040]),np.array([0]))
    assert c.raw_events==2


def test_summary_is_a_snapshot_and_all_cells_remain_in_mean():
    c=compare(np.array([5000,5100,5300]),np.array([0,0,0]),4,1)
    r,a=c.summarize();expected=cell_metrics(np.array([5000,5100,5300]),np.array([0,0,0]),4)[0]
    assert r['paper_lvr_mean']==expected['paper_lvr_mean']
    c.add(np.array([5600]),np.array([0]));assert a['half_open_cell_counts'][0]==3


@pytest.mark.parametrize('kwargs',[dict(neurons=0),dict(neurons=4200001),dict(neurons=True),dict(neurons=2,end=1005001),dict(neurons=2,start=-1),dict(neurons=2,dt_ms=float('nan')),dict(neurons=2,refractory_ms=-1)])
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError):m.CellMetrics(**kwargs)
