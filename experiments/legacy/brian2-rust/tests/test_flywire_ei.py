"""Transmitter policy, frozen inputs, and stateful EI model conformance."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

EXAMPLES=Path(__file__).resolve().parents[1]/'examples'
sys.path.insert(0,str(EXAMPLES))
from flywire_ei_import import transmitter_sign
from flywire_device import poisson_schedule,read_snapshot
from flywire_benchmark import compare,replay


def test_exported_source_does_not_require_git_metadata(tmp_path, monkeypatch):
    from performance_suite import source_provenance
    assert source_provenance(tmp_path) == {
        'source_commit': None, 'source_dirty': None}
    monkeypatch.setenv('PATH', '')
    assert source_provenance(tmp_path) == {
        'source_commit': None, 'source_dirty': None}


def test_curated_fast_transmitters_and_ambiguous_predictions():
    assert transmitter_sign('acetylcholine','')==(1,'EM_prediction')
    assert transmitter_sign('glutamate','')==(-1,'EM_prediction')
    assert transmitter_sign('dopamine','acetylcholine, sNPF')==(1,'curated_fast_transmitter')
    assert transmitter_sign('acetylcholine','glutamate, gaba')==(-1,'curated_fast_transmitter')
    assert transmitter_sign('acetylcholine','acetylcholine, gaba')==(0,'curated_EI_conflict')
    assert transmitter_sign('dopamine','')==(0,'EM_prediction')
    assert transmitter_sign('','')==(0,'EM_prediction')


def test_frozen_poisson_is_repeatable_ordered_and_windowed():
    a,t=poisson_schedule(8,80,.1,1000,783,300,700)
    b,u=poisson_schedule(8,80,.1,1000,783,300,700)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(t,u)
    assert len(a)>100 and np.all((t>=3000)&(t<7000))
    assert np.all(np.diff(t.astype(np.int64)*8+a)>0)
    _,other=poisson_schedule(8,80,.1,1000,784,300,700)
    assert not np.array_equal(t,other)


@pytest.mark.parametrize('condition',['rest','odor','cut_rest','cut'])
def test_ei_dynamics_and_frozen_sensory_inputs_match_cpp_and_replays(tmp_path,condition):
    # Small, signed empirical-format graph with both fast-current channels.
    from brian2_rust.binary_topology import HEADER,MAGIC,file_hash
    graph=tmp_path/'graph';graph.mkdir()
    n=12;targets=np.array([(i+1)%n for i in range(n)],dtype='<u4')
    weights=np.array([12 if i%3 else -7 for i in range(n)],dtype='<f8')
    path=graph/'connectome.b2csr'
    with path.open('wb') as f:
        f.write(HEADER.pack(MAGIC,n,n,n,1));f.write(np.arange(n+1,dtype='<u8').tobytes())
        f.write(targets.tobytes());f.write(weights.tobytes())
    (graph/'manifest.json').write_text(json.dumps({'csr_sha256':file_hash(path)}))
    np.savez(graph/'annotations.npz',sensory=[1,2],pn=[3,4],kc=[5,6],mbon=[7,8],cx=[9,10])
    cfg=tmp_path/'config.json'
    cfg.write_text(json.dumps({'duration_ms':100.,'stimulus_start_ms':20.,'stimulus_end_ms':70.,
                              'background_weight_mv':5.,'background_channels':8}))
    for backend in ('aot','cpp'):
        subprocess.run([sys.executable,str(EXAMPLES/'flywire_device.py'),'--graph',str(graph),
                        '--output',str(tmp_path/backend),'--backend',backend,'--condition',condition,'--config',str(cfg)],
                       check=True,capture_output=True,text=True)
    subprocess.run([sys.executable,str(EXAMPLES/'flywire_device.py'),'--graph',str(graph),
                    '--output',str(tmp_path/'mpi-export'),'--backend','mpi','--export-only',
                    '--condition',condition,'--config',str(cfg)],
                   check=True,capture_output=True,text=True)
    # Deferring execution for MPI must retain the exact existing CPU model.
    assert json.loads((tmp_path/'mpi-export/model.json').read_text()) == json.loads(
        (tmp_path/'aot/project/model.json').read_text())
    with np.load(tmp_path/'aot/snapshot.npz') as d:expected={k:d[k] for k in d.files}
    assert len(expected['spike_i'])>0
    assert np.any(expected['ge']>0) and np.any(expected['gi']>0)
    assert np.all(expected['v']>=-.070) and np.all(expected['v']<=0)
    with np.load(tmp_path/'cpp/snapshot.npz') as d:compare({k:d[k] for k in d.files},expected)
    for backend in ('aot','cpp'):
        output=tmp_path/f'replay-{backend}'
        replay(backend,tmp_path/backend,output,1,expected,reader=read_snapshot,cleanup=True)
        assert not output.exists()
