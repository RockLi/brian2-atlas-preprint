"""Full-run gates and frozen input optimization, independent of remote resources."""
import json
import struct
import subprocess
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'experiments')]
from flywire_mnist.config import Config
from flywire_mnist.encoding import encode
from flywire_mnist.graph import load_graph
from flywire_mnist.model import build,instance
from flywire_mnist.backends import CPU
from flywire_mnist.backends.frozen_cpu import FrozenCPU
from flywire_mnist.full import load_shard,shard_digest,require_test_lock
from flywire_mnist.readout import fit,ridge_path,predict


def test_frozen_input_matches_validated_binary_and_states(tmp_path):
    c=Config(warmup_ms=10,stimulus_ms=20,tail_ms=5,fanout=2)
    g=load_graph(ROOT/'wasm/flywire-circuit.json');template=build(g,c,monitor=True)
    normal=CPU(template,tmp_path/'normal');fast=FrozenCPU(template,normal.native,tmp_path/'fast')
    outputs=[]
    for key,pixels in [('a',255),('b',0),('a-repeat',255)]:
        image=np.full((28,28),pixels,dtype=np.uint8);indices,ticks=encode(image,7,c)
        actual=fast.run(indices,ticks,key)
        expected=normal.run(instance(template,image,7,c),key)
        fast.write(indices,ticks,fast.directory/f'{key}.bin')
        assert (fast.directory/f'{key}.bin').read_bytes()==(normal.directory/f'{key}.bin').read_bytes()
        for p,q in zip(actual['populations'],expected['populations']):
            for state in p['states']:np.testing.assert_array_equal(p['states'][state],q['states'][state])
            np.testing.assert_array_equal(p['spike_ticks'],q['spike_ticks'])
            np.testing.assert_array_equal(p['indices'],q['indices'])
        outputs.append(actual)
    for p,q in zip(outputs[0]['populations'],outputs[-1]['populations']):
        for state in p['states']:np.testing.assert_array_equal(p['states'][state],q['states'][state])
    for indices,ticks in [(np.array([784]),np.array([1])),(np.array([0,0]),np.array([1,1])),
                          (np.array([0,1]),np.array([2,1])),(np.array([0]),np.array([c.ticks]))]:
        with pytest.raises(ValueError):fast.write(indices,ticks,tmp_path/'invalid.bin')
    assert not (tmp_path/'invalid.bin').exists()
    # Native validation is independent of the Python producer's checks.
    for position,(population,indices,ticks) in enumerate([
            (fast.population,[784],[1]),(fast.population,[0,0],[1,1]),
            (fast.population,[0],[c.ticks]),(999,[0],[1])]):
        sidecar=tmp_path/f'bad-{position}.spikes'
        sidecar.write_bytes(b'B2SPIK01'+struct.pack('<Q',len(ticks))+
                            np.array(ticks,dtype='<u8').tobytes()+np.array(indices,dtype='<u8').tobytes())
        result=subprocess.run([str(fast.binary),str(fast.base),str(tmp_path/f'bad-out-{position}'),
                               '--spike-input',str(population),str(sidecar)],capture_output=True)
        assert result.returncode!=0
        assert not (tmp_path/f'bad-out-{position}').exists()
    with fast.base.open('r+b') as stream:stream.seek(25);stream.write(b'bad')
    with pytest.raises(ValueError,match='snapshot changed'):
        fast.write(np.array([],dtype=int),np.array([],dtype=int),tmp_path/'corrupt.bin')
    assert not (tmp_path/'corrupt.bin').exists()


def test_shared_gram_ridge_matches_original():
    rng=np.random.default_rng(52);x=rng.normal(size=(130,18));x[:,0]=1;y=rng.integers(0,10,130)
    for actual in ridge_path(x,y,[.1,10],block_size=17):
        expected=fit(x,y,float(actual['alpha']))
        np.testing.assert_allclose(actual['weights'],expected['weights'],rtol=1e-10,atol=1e-12)
        np.testing.assert_array_equal(predict(actual,x),predict(expected,x))


def test_shard_corruption_and_test_lock_are_rejected(tmp_path):
    ids=np.array([9,2]);x=np.array([[1,2],[3,4]],dtype=np.uint16);control=x[:,:1].copy()
    protocol={'sha256':'fixed','feature_dimension':2,'projected_dimension':1}
    path=tmp_path/'shard.npz'
    np.savez(path,sample_ids=ids,features=x,projected=control,protocol='fixed',
             payload_sha256=shard_digest(ids,x,control))
    np.testing.assert_array_equal(load_shard(path,ids,protocol)[0],x)
    with pytest.raises(ValueError,match='identity'):load_shard(path,ids[::-1],protocol)
    np.savez(path,sample_ids=ids,features=x+1,projected=control,protocol='fixed',
             payload_sha256=shard_digest(ids,x,control))
    with pytest.raises(ValueError,match='checksum'):load_shard(path,ids,protocol)
    with pytest.raises(FileNotFoundError):require_test_lock(tmp_path,protocol)
    (tmp_path/'test-lock.json').write_text(json.dumps({'protocol_sha256':'wrong'}))
    with pytest.raises(ValueError,match='protocol'):require_test_lock(tmp_path,protocol)
