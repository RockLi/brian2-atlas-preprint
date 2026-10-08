"""An optimized benchmark label must execute and reset the selected plan."""
from pathlib import Path
import numpy as np
import pytest
from test_gpu_spike_generator import BACKENDS
from test_metal_delays import device


@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_prefix_worker_replays_actual_selected_plan(device,tmp_path,monkeypatch,backend):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    from gpu_stdp_precompiled import Replay,artifact_hashes
    from gpu_stdp_compare import oracle,checks
    options=dict(drive=.0625,delay_span=8,post_delay=3,topology_kind='random-fixed-outdegree',topology_seed=42)
    ex=Replay(backend+'-prefix',256,48,64,tmp_path,**options)
    try:
        assert ex.synapse_prefix and ex.backend==backend
        assert ex.adapter_evidence['synapse_prefix']
        assert any(d['role']=='edge-synapse-prefix' for d in ex.adapter_evidence['execution_plan']['dispatches'])
        assert checks(ex.bootstrap,oracle(256,64,48,**options))['passed']
        for _ in range(2):
            actual,timing=ex.run()
            for name,a in actual.items():np.testing.assert_array_equal(a,ex.bootstrap[name])
            assert timing['details']['runtime'] is not None
            assert artifact_hashes(tmp_path)==ex.artifacts
        np.savez_compressed(tmp_path/'worker-results.npz',**{'actual/'+k:v for k,v in actual.items()},
                            **{'reference/'+k:v for k,v in oracle(256,64,48,**options).items()})
    finally:ex.close()
