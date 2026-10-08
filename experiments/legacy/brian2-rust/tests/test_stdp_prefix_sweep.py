"""A failed numerical gate remains visible in the two-case comparison."""
import sys
from pathlib import Path
import pytest

@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    import modal_stdp_precompiled as compare
    import modal_stdp_prefix_sweep as sweep
    return compare,sweep

def test_case_gates_and_original_data_are_preserved(modules,monkeypatch):
    compare,sweep=modules;calls=[]
    def run(*args,**kwargs):
        calls.append((args,kwargs))
        return dict(nvidia_smi='GPU-1',status='completed-with-gate-failures' if len(calls)==1 else 'passed-matched-numeric-gates',
                    payloads={'cuda-prefix-bootstrap.npz':b'evidence'},excluded_by_gate={'cuda-prefix':{'passed':False}} if len(calls)==1 else {})
    monkeypatch.setattr(compare,'compare',run)
    report=sweep.verify()
    assert len(calls)==2 and all(k['continue_on_gate_failure'] for _,k in calls)
    assert report['status']=='completed-with-gate-failures' and not report['passed']
    assert list(report['cases'])==list(sweep.CASES)
    assert report['cases']['sparse-low-drive']['excluded_by_gate']['cuda-prefix']=={'passed':False}
    assert len(report['payloads'])==2
    assert all(args==(4096,256,8,5) for args,_ in calls)

@pytest.mark.parametrize('bad',[[],['cuda'],['cpu-f32','cpu-f32'],['cpu-f32','unknown']])
def test_invalid_backend_selection_never_starts_worker(modules,monkeypatch,bad):
    compare,sweep=modules
    monkeypatch.setattr(compare,'compare',lambda *a,**kw:pytest.fail('worker started'))
    with pytest.raises(ValueError):sweep.verify(bad)

@pytest.mark.parametrize('case',['gpu-change','payload-path'])
def test_invalid_provenance_or_payload_rejected(modules,monkeypatch,case):
    compare,sweep=modules;ids=iter(['GPU-1','GPU-2'])
    monkeypatch.setattr(compare,'compare',lambda *a,**kw:dict(nvidia_smi=next(ids) if case=='gpu-change' else 'GPU-1',
        status='passed-matched-numeric-gates',payloads={'../escape':b'bad'} if case=='payload-path' else {}))
    with pytest.raises((ValueError,RuntimeError)):sweep.verify()
