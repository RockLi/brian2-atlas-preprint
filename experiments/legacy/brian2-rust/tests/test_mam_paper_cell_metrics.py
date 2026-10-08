import importlib.util
from pathlib import Path
import numpy as np
import pytest

p=Path(__file__).resolve().parents[1]/'tools/analyze_mam_paper_cell_metrics.py'
s=importlib.util.spec_from_file_location('paper_cells',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def test_rate_and_lvr_use_distinct_endpoints_and_all_cells():
    # Cell 1: ISIs 10/20 ms => LvR = (1/3)*(1+8/30); cell 0 sparse; cell 2 silent.
    ticks=np.array([800,500,1000,600,499,501]);ids=np.array([1,1,1,1,0,0])
    result,a=m.cell_metrics(ticks,ids,3,start=500,end=1000,dt_ms=.1)
    assert result['half_open_spikes']==4 and result['strict_spikes']==3
    assert result['lower_boundary_spikes']==1
    assert result['paper_rate_hz']==20
    assert result['lvr_eligible_cells']==1
    np.testing.assert_allclose(a['cell_lvr'],[0,(1/3)*(1+8/30),0],rtol=1e-14)
    assert result['paper_lvr_mean']==pytest.approx((1+8/30)/9)


def test_silence_reordering_and_sparse_trains():
    result,a=m.cell_metrics(np.array([],dtype=int),np.array([],dtype=int),4)
    assert result['paper_rate_hz']==result['paper_lvr_mean']==0
    assert result['lvr_eligible_mean'] is None
    ticks=np.array([5000,5020,5040,5010,5050,5070]);ids=np.array([2,2,2,0,0,0])
    expected,_=m.cell_metrics(ticks,ids,4)
    actual,_=m.cell_metrics(ticks[::-1],ids[::-1],4)
    assert actual==expected
    assert expected['lvr_eligible_cells']==2


def test_invalid_event_contract():
    with pytest.raises(ValueError,match='invalid'):
        m.cell_metrics(np.array([1.5]),np.array([0]),2)
    with pytest.raises(ValueError,match='invalid'):
        m.cell_metrics(np.array([1]),np.array([2]),2)
    with pytest.raises(ValueError,match='duplicate'):
        m.cell_metrics(np.array([5000,5000,5010]),np.array([0,0,0]),2)


def test_long_observation_contract_and_byte_budget():
    window=dict(start_tick=5000,end_tick=105000,dt_seconds=.0001,endpoint='[start,end)')
    assert m.observation_contract(window)==(105000,10.,12288000000)
    assert m.observation_contract(dict(window,end_tick=25000))==(25000,2.,3072000000)
    for change in [dict(end_tick=105001),dict(end_tick=105000.),dict(start_tick=0),dict(endpoint='(start,end]')]:
        with pytest.raises(ValueError):m.observation_contract(dict(window,**change))


def test_long_tail_counts_and_rate_denominator():
    ticks=np.array([5000,5001,25000,104999,105000]);ids=np.zeros(len(ticks),dtype=int)
    result,arrays=m.cell_metrics(ticks,ids,2,end=105000)
    assert result['half_open_spikes']==4 and result['strict_spikes']==3
    assert result['half_open_rate_hz']==.2 and result['paper_rate_hz']==.15
    np.testing.assert_array_equal(arrays['half_open_cell_counts'],[4,0])
