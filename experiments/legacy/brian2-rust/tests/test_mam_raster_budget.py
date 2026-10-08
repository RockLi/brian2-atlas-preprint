from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from mam_raster_budget import RasterBudget,RasterSample


@pytest.mark.parametrize('block',[1,7,131072])
def test_exact_legacy_selection_stable_ties_and_boundaries(block):
    n=100;seed=20270908;b=RasterBudget();c=RasterSample(n,seed,b)
    selected=np.sort(np.random.default_rng(seed).choice(n,int(np.ceil(.03*n)),replace=False))
    ticks=np.array([25000,5001,5000,4999,5000,24999]);ids=np.array([selected[0],selected[1],selected[0],selected[0],selected[1],selected[0]])
    for lo in range(0,len(ticks),block):c.add(ticks[lo:lo+block],ids[lo:lo+block])
    r=c.finish();np.testing.assert_array_equal(c.selected,selected)
    np.testing.assert_array_equal(r['times'],np.array([5000,5000,5001,24999])*.0001)
    np.testing.assert_array_equal(r['indices'],np.array([0,1,1,0],dtype=np.int32))
    assert b.used==4 and r['indices'].dtype==np.dtype('int32')
    with pytest.raises(ValueError):c.add(np.array([],dtype=int),np.array([],dtype=int))


def test_shared_budget_rejects_before_payload_and_retains_completed_points():
    b=RasterBudget(3);a=RasterSample(1,1,b);c=RasterSample(1,2,b)
    a.add(np.array([5000,5001]),np.array([0,0]))
    with pytest.raises(ValueError,match='before payload'):c.add(np.array([5000,5001]),np.array([0,0]))
    assert b.used==2 and not c.ticks and not c.indices
    c.add(np.array([5002]),np.array([0]));assert b.used==3
    assert len(a.finish()['times'])==2 and len(c.finish()['times'])==1


def test_empty_input_and_strict_limits():
    c=RasterSample(1,1,RasterBudget());r=c.finish();assert r['times'].shape==r['indices'].shape==(0,)
    for limit in [0,-1,3000001,True,1.5]:
        with pytest.raises(ValueError):RasterBudget(limit)
    with pytest.raises(ValueError):RasterSample(1,1,RasterBudget(),end_tick=1005001)
    with pytest.raises(ValueError):RasterSample(1,1,RasterBudget()).add(np.array([5000]),np.array([1]))
