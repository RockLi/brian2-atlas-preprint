"""Fail-closed source ablation and independent observed-event predictions."""
import gzip
import json
from pathlib import Path
import numpy as np
import pytest
from brian2genn_precision_diagnostic import round_decay_factor, plasticity_from_spikes, OPTIONS
from gpu_brian2genn_stdp_adapter import transform_model
from gpu_stdp_compare import oracle, FIELDS


@pytest.fixture(params=['l4','a100'])
def generated(request):
    evidence=Path(__file__).resolve().parents[1]/'execution-plan-evidence/dense-stdp'
    manifest=json.loads((evidence/'manifest.json').read_text())['artifacts']
    row=manifest[request.param+'/report.json']
    data=(evidence/'blobs'/row['blob']).read_bytes()
    if row['encoding']=='gzip':data=gzip.decompress(data)
    report=json.loads(data)
    worker=report['cases']['dense-1024']['workers']['brian2genn-corrected']
    return transform_model(worker['adapter_evidence']['magicnetwork_model.cpp.original'],
        1024,{'plastic'+str(q):7-q for q in range(8)},drive=1/16,post_delay=3)


def test_cast_only_changes_decay_factor(generated):
    adapted=round_decay_factor(generated,8)
    assert adapted.count(' * (float)exp(')==32
    assert adapted.replace(' * (float)exp(',' * exp(')==generated
    with pytest.raises(ValueError):round_decay_factor(adapted,8)


@pytest.mark.parametrize('corruption',['time','lastupdate','expression','count'])
def test_rejects_unexpected_generated_program(generated,corruption):
    bad=generated
    if corruption=='time':bad=bad.replace('TimePrecision::DOUBLE','TimePrecision::FLOAT')
    if corruption=='lastupdate':bad=bad.replace('{"lastupdate", "double"}','{"lastupdate", "float"}',1)
    if corruption=='expression':bad=bad.replace(' * exp(',' * expf(',1)
    with pytest.raises(ValueError):round_decay_factor(bad,7 if corruption=='count' else 8)


@pytest.mark.parametrize('steps',[1,16,128])
def test_observed_f32_prediction_matches_full_recurrence(steps):
    result=oracle(17,7,steps,np.float32,**OPTIONS)
    predicted=plasticity_from_spikes(17,7,steps,result['ticks'],result['indices'],mode='f32-factor',**OPTIONS)
    for key in FIELDS:np.testing.assert_array_equal(predicted[key],result[key])


@pytest.mark.parametrize('ticks,indices',[(np.array([0,0]),np.array([1,1])),
    (np.array([-1]),np.array([0])),(np.array([16]),np.array([0])),
    (np.array([0]),np.array([17])),(np.array([0.]),np.array([0]))])
def test_invalid_spikes_fail(ticks,indices):
    with pytest.raises(ValueError):
        plasticity_from_spikes(17,7,16,ticks,indices,mode='f32-factor',**OPTIONS)
