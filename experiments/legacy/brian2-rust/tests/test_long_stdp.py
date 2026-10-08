"""Long-horizon cases preserve failed gates and enforce a finite worker limit."""
from pathlib import Path
import subprocess,sys
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    import modal_long_stdp_compare as sweep
    import modal_stdp_precompiled as compare
    return sweep,compare


def test_declared_long_case_runs_even_when_gate_fails(modules,monkeypatch):
    sweep,compare=modules;calls=[]
    def run(*args,**kwargs):
        calls.append((args,kwargs))
        return dict(nvidia_smi='GPU-1',status='passed-matched-numeric-gates' if args[1]==256 else 'completed-with-gate-failures',
            excluded_by_gate={} if args[1]==256 else {'cuda':{'original_f64_gate':{'ticks':False}}},payloads={'bootstrap.npz':b'evidence'})
    monkeypatch.setattr(compare,'compare',run)
    report=sweep.verify()
    assert [c[0] for c in calls]==[(4096,256,8,5),(4096,4096,8,5)]
    assert all(c[1]['continue_on_gate_failure'] for c in calls)
    assert not report['passed'] and report['status']=='completed-with-gate-failures'
    assert report['cases']['long-4096']['excluded_by_gate']['cuda']['original_f64_gate']['ticks'] is False
    assert set(report['payloads'])=={'short-256/bootstrap.npz','long-4096/bootstrap.npz'}


@pytest.mark.parametrize('backends',[[],['metal'],['cpu-f32','cpu-f32'],['cpu-f32','invalid']])
def test_bad_selection_never_launches_workers(modules,monkeypatch,backends):
    sweep,compare=modules
    monkeypatch.setattr(compare,'compare',lambda *a,**k:pytest.fail('worker launched'))
    with pytest.raises(ValueError):sweep.verify(backends)


@pytest.mark.parametrize('steps',['0','4097'])
def test_cli_limit_rejects_before_output_or_gpu_setup(tmp_path,steps):
    script=Path(__file__).resolve().parents[1]/'examples/gpu_stdp_precompiled.py'
    output=tmp_path/'never-created'
    result=subprocess.run([sys.executable,str(script),'--backend','cuda','--neurons','4096','--steps',steps,'--output',str(output)],capture_output=True,text=True)
    assert result.returncode==2 and '1..4096 replay steps' in result.stderr
    assert not output.exists()


@pytest.fixture
def spike_difference():
    import runpy
    verifier=Path(__file__).resolve().parents[1]/'execution-plan-evidence/long-stdp/verify.py'
    return runpy.run_path(str(verifier))['spike_difference']


def test_spike_coordinates_detect_shift_despite_equal_counts(spike_difference):
    import numpy as np
    a=dict(ticks=np.array([1,3]),indices=np.array([0,1]))
    b=dict(ticks=np.array([1,2]),indices=np.array([0,1]))
    assert spike_difference(a,b,2)==dict(actual_spikes=2,reference_spikes=2,
        extra_coordinates=1,missing_coordinates=1,first=dict(tick=2,neuron=1))


def test_spike_coordinates_reject_duplicate_emissions(spike_difference):
    import numpy as np
    a=dict(ticks=np.array([1,1]),indices=np.array([0,0]))
    with pytest.raises(AssertionError):spike_difference(a,a,2)


def test_spike_coordinates_handle_empty_and_reordered_records(spike_difference):
    import numpy as np
    empty=dict(ticks=np.array([],dtype=int),indices=np.array([],dtype=int))
    assert spike_difference(empty,empty,2)['first'] is None
    a=dict(ticks=np.array([1,2]),indices=np.array([0,1]))
    b={k:v[::-1] for k,v in a.items()}
    assert spike_difference(a,b,2)['first'] is None
    assert spike_difference(a,empty,2)['extra_coordinates']==2
