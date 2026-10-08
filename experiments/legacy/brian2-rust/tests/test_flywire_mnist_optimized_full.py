from pathlib import Path
import json
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'validation'),str(ROOT/'experiments'),str(ROOT/'python')]
import flywire_mnist_optimized_full as f
from flywire_mnist.protocol import file_sha


def test_complete_official_ids_and_smoke_train_only():
    p=dict(preset='standard')
    np.testing.assert_array_equal(f.phase_ids(p,'train'),np.arange(60000))
    np.testing.assert_array_equal(f.phase_ids(p,'test'),np.arange(60000,70000))
    p=dict(preset='smoke',smoke_ids=dict(train=[1,2],test=[3,4]))
    assert not set(f.phase_ids(p,'train')) & set(f.phase_ids(p,'test'))
    assert f.phase_ids(p,'test').max()<60000


def test_test_gate_rejects_small_fit_missing_models_and_mutations(tmp_path):
    p=dict(preset='standard',sha256='protocol')
    def write(name,value): (tmp_path/name).write_text(json.dumps(value))
    for name in f.GROUPS:(tmp_path/f'model-{name}.npz').write_bytes(name.encode())
    write('train-manifest.json',dict(protocol='protocol',phase='train',count=60000))
    lock=dict(protocol='protocol',fit_count=60000,models={name:file_sha(tmp_path/f'model-{name}.npz') for name in f.GROUPS},
              training_manifest_sha256=file_sha(tmp_path/'train-manifest.json'))
    write('test-lock.json',lock);f.require_lock(tmp_path,p)
    lock['fit_count']=20;write('test-lock.json',lock)
    with pytest.raises(ValueError,match='incomplete test lock'):f.require_lock(tmp_path,p)
    lock['fit_count']=60000;removed=lock['models'].pop('pixels');write('test-lock.json',lock)
    with pytest.raises(ValueError,match='incomplete test lock'):f.require_lock(tmp_path,p)
    lock['models']['pixels']=removed;write('test-lock.json',lock)
    (tmp_path/'model-neural.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='readout changed'):f.require_lock(tmp_path,p)


def test_manifest_cannot_disguise_partial_training_as_full(tmp_path):
    p=dict(preset='standard',sha256='p')
    for name in f.GROUPS:(tmp_path/f'model-{name}.npz').write_bytes(b'model')
    (tmp_path/'train-manifest.json').write_text(json.dumps(dict(protocol='p',phase='train',count=20)))
    lock=dict(protocol='p',fit_count=60000,models={name:file_sha(tmp_path/f'model-{name}.npz') for name in f.GROUPS},
              training_manifest_sha256=file_sha(tmp_path/'train-manifest.json'))
    (tmp_path/'test-lock.json').write_text(json.dumps(lock))
    with pytest.raises(ValueError,match='required full training set'):f.require_lock(tmp_path,p)


def test_verified_cache_reuse_emits_normal_shards_without_running_simulator(tmp_path,monkeypatch):
    ids=np.array([3,17],dtype=np.int64);lookup=np.full(70000,-1,dtype=np.int64);lookup[ids]=[0,1]
    counts=np.arange(24,dtype=np.uint16).reshape(2,4,3);projected=np.arange(16,dtype=np.uint16).reshape(2,4,2)
    (tmp_path/'shards').mkdir()
    # object() has no run method: taking the simulation branch would fail.
    state=(object(),None,None,None,None,None,None,2,lookup,[counts,projected],tmp_path,
           dict(preset='standard',sha256='frozen'),'train',False)
    monkeypatch.setattr(f,'_WORKER',state)
    assert f.write_shard(0,ids)==(2,2)
    actual,control=f.r.load_shard(tmp_path/'shards/train-000000.npz',ids,'frozen',3,2)
    np.testing.assert_array_equal(actual,counts);np.testing.assert_array_equal(control,projected)
