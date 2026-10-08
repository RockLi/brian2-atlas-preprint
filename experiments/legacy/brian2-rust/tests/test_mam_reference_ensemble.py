import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mam_ensemble',ROOT/'tools/compare_mam_reference_ensemble.py')
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_small_reference_range_is_descriptive_and_missing_data_remains_missing():
    d=m.describe_metric([1.,3.],4.)
    assert d['native_mean']==2 and d['native_sample_std']==pytest.approx(2**.5)
    assert d['rust_outside_observed_range'] is True
    assert m.describe_metric([1.,3.],3.)['rust_outside_observed_range'] is False
    d=m.describe_metric([None,.2],.5)
    assert d['native_valid_count']==1 and d['native_sample_std'] is None
    assert d['rust_outside_observed_range'] is None
    assert m.describe_metric([None,None],None)['native_mean'] is None
    with pytest.raises(ValueError,match='nonfinite'):m.describe_metric([float('nan'),1.],0.)


def test_identity_and_physical_window_validation():
    protocol=json.loads((ROOT/'mpi-evidence/nest-reference-ensemble/protocol-v1.json').read_text())
    base=dict(simulation=dict(seed=1729,ranks=48,threads=4,duration_ms=2500,dt_ms=.1,nest_version='3.10.0'),
        parameters_sha256=protocol['parameters_sha256'],neurons=protocol['neurons'],
        populations=[dict(name=f'p{i}',neurons=protocol['neurons']-253 if i==0 else 1) for i in range(254)],
        areas=[dict(area=f'a{i}') for i in range(32)],sampling=dict(seed=20260908),
        window=dict(start_tick=5000,end_tick=25000,raw_start_tick=5000,raw_end_tick=25000,
            spike_tick_offset=0,endpoint='[start,end)',dt_seconds=.0001,bin_ticks=10,seconds=2.))
    second=copy.deepcopy(base);second['simulation']['seed']=1730
    rust=copy.deepcopy(base);rust['window'].update(spike_tick_offset=1,raw_start_tick=4999,raw_end_tick=24999)
    assert m.validate_reports([base,second],rust,protocol)==[1729,1730]
    with pytest.raises(ValueError,match='duplicate'):m.validate_reports([base,base],rust,protocol)
    bad=copy.deepcopy(second);bad['window']['raw_start_tick']=4999
    with pytest.raises(ValueError,match='mapping'):m.validate_reports([base,bad],rust,protocol)
    bad=copy.deepcopy(second);bad['window']['end_tick']=25001
    with pytest.raises(ValueError,match='physical window'):m.validate_reports([base,bad],rust,protocol)
    bad=copy.deepcopy(second);bad['parameters_sha256']='wrong'
    with pytest.raises(ValueError,match='parameter'):m.validate_reports([base,bad],rust,protocol)
    for r in [base,second,rust]:r['window']['bin_ticks']=20
    with pytest.raises(ValueError,match='units'):m.validate_reports([base,second],rust,protocol)
