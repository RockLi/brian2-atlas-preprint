from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'validation'),str(ROOT/'experiments'),str(ROOT/'python')]
import flywire_mnist_state_probe as r


def test_collector_preserves_subthreshold_float_signal_without_spikes():
    states={f'count{k}':np.zeros(3) for k in range(4)}
    states.update(v=np.array([-.052,-.05199,-.06]),ge=np.array([.1,.100001,0.]),gi=np.array([.2,.20001,0.]))
    result=dict(populations=[dict(states=states)])
    counts,analog=r.collect(result,0,np.array([1,0]))
    assert counts.dtype==np.uint16 and not counts.any()
    assert analog.dtype==np.float64
    np.testing.assert_array_equal(analog[0],[-.05199,-.052])
    assert analog[1,0]!=analog[1,1]
    states['v'][:]=0  # The original result buffer can now be released/reused.
    np.testing.assert_array_equal(analog[0],[-.05199,-.052])


def test_nonfinite_state_and_negative_conductance_rejected():
    states={f'count{k}':np.zeros(2) for k in range(4)}
    states.update(v=np.array([-.052,-.051]),ge=np.array([0.,-.01]),gi=np.zeros(2))
    with pytest.raises(ValueError,match='invalid terminal'):
        r.collect(dict(populations=[dict(states=states)]),0,np.array([0,1]))
    states['ge'][:]=0;states['v'][0]=np.nan
    with pytest.raises(ValueError,match='invalid terminal'):
        r.collect(dict(populations=[dict(states=states)]),0,np.array([0,1]))


def test_state_shard_rejects_changed_data_and_wrong_image_order(tmp_path):
    ids=np.array([3,8],np.int64);counts=np.zeros((2,4,2),np.uint16);analog=np.zeros((2,3,2),np.float64)
    analog[:,0]=-.052
    data=dict(ids=ids,counts=counts,analog=analog,identity='id',payload_sha256=r.r.payload_hash(ids,counts,analog))
    path=tmp_path/'sample.npz';np.savez(path,**data)
    r.load_shard(path,ids,'id',2)
    with pytest.raises(ValueError,match='corrupt'):
        r.load_shard(path,ids[::-1],'id',2)
    analog[0,0,0]=-.05199;np.savez(path,**data)
    with pytest.raises(ValueError,match='corrupt'):
        r.load_shard(path,ids,'id',2)
