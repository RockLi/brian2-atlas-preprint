"""Explicit bitmap benchmark workers must execute and reset their actual plan."""
import json
import numpy as np
import pytest
from test_gpu_spike_generator import BACKENDS
from test_metal_delays import device


@pytest.mark.parametrize('backend',BACKENDS[1:])
@pytest.mark.parametrize('prefix',[False,True])
def test_bitmap_worker_replays_selected_plan(device,tmp_path,backend,prefix):
    from gpu_stdp_precompiled import Replay,artifact_hashes
    from gpu_stdp_compare import oracle,checks
    opts=dict(drive=1/16,delay_span=8,post_delay=3,topology_kind='random-fixed-outdegree',topology_seed=42)
    name=backend+('-prefix' if prefix else '')+'-bitset'
    control_dir=tmp_path/'control';control_dir.mkdir()
    control=Replay('cpu-f32',256,48,64,control_dir,**opts)
    expected=control.bootstrap;control.close();device.reinit()
    native_dir=tmp_path/'native';native_dir.mkdir()
    ex=Replay(name,256,48,64,native_dir,**opts)
    try:
        assert ex.backend==backend and ex.requested_backend==name
        assert ex.synapse_sparse=='bitset' and ex.synapse_prefix==prefix
        evidence=ex.adapter_evidence;plan=evidence['execution_plan']
        assert evidence['synapse_sparse']=='bitset' and evidence['synapse_prefix']==prefix
        assert any(d['role']=='target-owned-bitset-synapse-pathway' for d in plan['dispatches'])
        assert any(d['role']=='edge-synapse-prefix' for d in plan['dispatches'])==prefix
        assert any(n.endswith('/target_sparse_bitmap_words') for n in plan['buffers'])
        reference=oracle(256,64,48,dtype=np.float32,**opts)
        snapshots={}
        for i in range(3):
            actual=ex.bootstrap if i==0 else ex.run()[0]
            assert checks(actual,reference)['passed']
            for key,value in actual.items():
                np.testing.assert_array_equal(value,expected[key])
                snapshots[f'actual/{i}/{key}']=value
                snapshots[f'reference/{i}/{key}']=expected[key]
            assert artifact_hashes(native_dir)==ex.artifacts
        np.savez_compressed(tmp_path/'bitset-worker-results.npz',**snapshots)
        (tmp_path/'bitset-worker-plan.json').write_text(json.dumps(evidence)+'\n')
    finally:ex.close()


def test_explicit_workers_keep_existing_defaults_and_qualification(monkeypatch):
    import modal_f32_stdp_compare as runner
    import modal_stdp_precompiled as compare
    from gpu_stdp_precompiled import BITSET_BACKENDS,BACKENDS as defaults
    assert len(BITSET_BACKENDS)==4 and not set(BITSET_BACKENDS)&set(defaults)
    assert not set(BITSET_BACKENDS)&set(runner.BACKENDS)
    wanted=list(runner.BACKENDS)+['brian2genn-f32-factor','cuda-bitset','cuda-prefix-bitset']
    def run_case(neurons,steps,degree,repeats,**kwargs):
        assert (neurons,steps,degree,repeats)==(4096,1024,32,5)
        assert kwargs['requested_backends']==wanted and kwargs['continue_on_gate_failure']
        assert kwargs['numeric_contract']=='explicit-f32-v1'
        return dict(status='completed-with-gate-failures',nvidia_smi='same GPU',
            payloads={'result.npz':b'kept'},excluded_by_gate={'brian2genn-corrected':{'passed':False}})
    monkeypatch.setattr(compare,'compare',run_case)
    report=runner.verify(wanted,scenario='wide')
    assert report['backends']==wanted and not report['passed']
    assert report['cases']['wide-4096']['excluded_by_gate']
    for name in BITSET_BACKENDS:
        good=dict(passed=True,original_f64_gate={'w':False},independent_f32_gate={'passed':True})
        assert compare.qualified(good,name,'explicit-f32-v1')
        assert not compare.qualified(good,name,'f64-compatible')
        for bad in (dict(good,passed=False),dict(good,independent_f32_gate={'passed':False})):
            assert not compare.qualified(bad,name,'explicit-f32-v1')
