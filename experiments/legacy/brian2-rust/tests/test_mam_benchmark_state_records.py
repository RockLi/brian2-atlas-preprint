from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from mam_benchmark_state_records import voltage_records, final_state_records, MAX_ROWS


def f(values):
    return np.array(values, dtype=np.float64)


def test_physical_times_initial_row_endpoint_and_input_order_preserved():
    values=f([-65, -64, -63]);before=values.copy()
    result=voltage_records(f([100500., 0., .1]), np.array([3,1,2]), values,
                           cell_count=3, last_tick=1005000)
    assert result['tick'].tolist()==[1005000,0,1]
    assert result['cell'].tolist()==[2,0,1]
    expected=b''.join(struct.pack('<IId',tick,cell,mv*1e-3)
                      for tick,cell,mv in [(1005000,2,-65.),(0,0,-64.),(1,1,-63.)])
    assert result.tobytes()==expected
    assert np.array_equal(values,before)


@pytest.mark.parametrize('times', [[.05],[-.1],[.3],[float('nan')]])
def test_invalid_state_grid_and_domain_are_not_shifted_or_clipped(times):
    with pytest.raises(ValueError):
        voltage_records(f(times),np.array([1]),f([-65]),cell_count=1,last_tick=2)


def test_duplicate_sample_refused_instead_of_averaged():
    with pytest.raises(ValueError,match='Duplicate'):
        voltage_records(f([.1,.1]),np.array([1,1]),f([-65,-64]),cell_count=1,last_tick=1)


def test_empty_block_preserves_dtype():
    result=voltage_records(f([]),np.array([],dtype=np.int64),f([]),cell_count=1,last_tick=0)
    assert len(result)==0 and result.dtype.itemsize==16


@pytest.mark.parametrize('bad', [np.array([-65],dtype=np.float32), None, f([float('inf')])])
def test_missing_or_reduced_precision_state_not_substituted(bad):
    with pytest.raises(ValueError):
        final_state_records(np.array([1]),f([-65]),bad,f([-2]),cell_count=1,
                            tau_syn_ex_ms=.5,tau_syn_in_ms=.5)


def test_final_current_exact_float64_conversion_and_no_reordering():
    result=final_state_records(np.array([3,1]),f([-65,-64]),f([100,20]),f([-40,-30]),
                              cell_count=3,tau_syn_ex_ms=.5,tau_syn_in_ms=.5)
    assert result.tobytes()==struct.pack('<IddIdd',2,-65.*1e-3,60.*1e-12,
                                       0,-64.*1e-3,-10.*1e-12)
    assert result.dtype.itemsize==20


def test_unequal_synaptic_time_constants_rejected():
    with pytest.raises(ValueError,match='equal'):
        final_state_records(np.array([1]),f([-65]),f([1]),f([-2]),cell_count=1,
                            tau_syn_ex_ms=.5,tau_syn_in_ms=1.)


def test_duplicate_final_cell_rejected():
    with pytest.raises(ValueError,match='Duplicate'):
        final_state_records(np.array([1,1]),f([-65,-64]),f([1,2]),f([-2,-3]),cell_count=1,
                            tau_syn_ex_ms=.5,tau_syn_in_ms=.5)


def test_block_bound_before_output_allocation():
    with pytest.raises(ValueError,match='block bound'):
        voltage_records(np.zeros(MAX_ROWS+1),np.array([1]),f([-65]),cell_count=1,last_tick=1)


@pytest.mark.parametrize('ids', [np.array([0]),np.array([4]),np.array([1.5])])
def test_invalid_sender_cannot_wrap_or_truncate(ids):
    with pytest.raises(ValueError):
        voltage_records(f([.1]),ids,f([-65]),cell_count=3,last_tick=1)
