"""Explicit precision selection never turns self-equality into independent evidence."""
import copy
from pathlib import Path
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    import modal_stdp_precompiled as compare
    import modal_f32_stdp_compare as runner
    return compare,runner


@pytest.mark.parametrize('backend',['cpu-f32','cuda','metal','genn','brian2cuda','brian2genn-corrected'])
def test_explicit_selection_and_independent_oracle_required(modules,backend):
    compare,_=modules
    result=dict(passed=True,original_f64_gate={'ticks':False},independent_f32_gate={'passed':True})
    assert not compare.qualified(result,backend)
    assert compare.qualified(result,backend,'explicit-f32-v1')
    failed=copy.deepcopy(result);failed['independent_f32_gate']['passed']=False
    assert not compare.qualified(failed,backend,'explicit-f32-v1')
    failed=copy.deepcopy(result);failed['passed']=False
    assert not compare.qualified(failed,backend,'explicit-f32-v1')
    assert result['original_f64_gate']=={'ticks':False}


@pytest.mark.parametrize('backend',['rust-f64','cpp-f64-t1','cpp-f64-t4'])
def test_f64_worker_still_requires_f64_in_explicit_contract(modules,backend):
    compare,_=modules
    result=dict(passed=True,original_f64_gate={'ticks':False},independent_f32_gate={'passed':True})
    assert not compare.qualified(result,backend,'explicit-f32-v1')
    result['original_f64_gate']['ticks']=True;result['independent_f32_gate']['passed']=False
    assert compare.qualified(result,backend,'explicit-f32-v1')


def test_unknown_contract_rejected_before_any_worker(modules,monkeypatch):
    compare,_=modules
    import subprocess
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('process launched'))
    with pytest.raises(ValueError,match='contract'):compare.compare(4,10,2,1,numeric_contract='f32-ish')


def test_explicit_summary_revokes_all_credit_after_late_failure(modules):
    compare,_=modules
    good=dict(passed=True,original_f64_gate={'ticks':False},independent_f32_gate={'passed':True})
    bad=copy.deepcopy(good);bad['independent_f32_gate']['passed']=False
    samples=[dict(backend=b,round=i,wall_seconds=.1,checks=bad if b=='genn' and i==1 else good)
             for b in ['genn','cuda'] for i in [0,1]]
    report=dict(excluded_by_gate={})
    compare.retain_gate_failure(report,'genn',bad,'replay',True)
    result=compare.timing_summary(['genn','cuda'],samples,2,report['excluded_by_gate'],'explicit-f32-v1')
    assert result[0]['samples_seconds']==[] and result[0]['median_seconds'] is None
    assert result[1]['matched_gate_passed'] and result[1]['median_seconds']==.1
    assert not compare.timing_summary(['cuda'],samples,2,{})[0]['matched_gate_passed']


def test_prospective_driver_declares_duration_and_contract(modules,monkeypatch):
    compare,runner=modules
    def run(*args,**kwargs):
        assert args==(4096,4096,8,5)
        assert kwargs['numeric_contract']=='explicit-f32-v1'
        assert kwargs['continue_on_gate_failure']
        assert set(kwargs['requested_backends'])==set(runner.BACKENDS)
        return dict(status='completed-with-gate-failures',nvidia_smi='GPU',payloads={'reference-f32.npz':b'kept'},excluded_by_gate={'genn':{'passed':False}})
    monkeypatch.setattr(compare,'compare',run)
    report=runner.verify()
    assert report['numeric_contract']=='explicit-f32-v1' and not report['passed']
    assert report['case_order']==['long-4096']
    assert report['payloads']=={'long-4096/reference-f32.npz':b'kept'}


def test_dense_scenario_keeps_failed_backends_and_full_results(modules,monkeypatch):
    compare,runner=modules
    def run(*args,**kwargs):
        assert args==(1024,1024,128,5)
        assert kwargs['numeric_contract']=='explicit-f32-v1' and kwargs['continue_on_gate_failure']
        assert kwargs['workload']==dict(drive=1/16,delay_span=8,post_delay=3,topology_kind='random-fixed-outdegree',topology_seed=42)
        return dict(status='completed-with-gate-failures',nvidia_smi='GPU',payloads={'failed.npz':b'failed'},excluded_by_gate={'genn':{'passed':False}})
    monkeypatch.setattr(compare,'compare',run)
    report=runner.verify(scenario='dense')
    assert not report['passed'] and report['scenario']=='dense'
    assert report['case_order']==['dense-1024']
    assert report['cases']['dense-1024']['excluded_by_gate']=={'genn':{'passed':False}}
    assert report['payloads']=={'dense-1024/failed.npz':b'failed'}


def test_unknown_scenario_cannot_launch_a_comparison(modules,monkeypatch):
    compare,runner=modules
    monkeypatch.setattr(compare,'compare',lambda *a,**kw:pytest.fail('comparison launched'))
    with pytest.raises(ValueError,match='scenario'):runner.verify(scenario='typo')


def test_activity_scenario_keeps_quiet_and_active_results_after_gate_failure(modules,monkeypatch):
    compare,runner=modules;drives=[]
    def run(*args,**kwargs):
        assert args==(4096,1024,8,5)
        assert kwargs['numeric_contract']=='explicit-f32-v1' and kwargs['continue_on_gate_failure']
        options=kwargs['workload'];drives.append(options['drive'])
        assert {k:v for k,v in options.items() if k!='drive'}==dict(delay_span=16,post_delay=16,topology_kind='random-fixed-outdegree',topology_seed=42)
        failed=len(drives)==1
        return dict(status='completed-with-gate-failures' if failed else 'passed-matched-numeric-gates',
            nvidia_smi='same GPU',payloads={'result.npz':b'full arrays'},excluded_by_gate={'genn':{'passed':False}} if failed else {})
    monkeypatch.setattr(compare,'compare',run)
    report=runner.verify(scenario='activity')
    assert drives==[1/64,17/512] and not report['passed']
    assert report['case_order']==['quiet-4096','low-4096']
    assert set(report['payloads'])=={'quiet-4096/result.npz','low-4096/result.npz'}
    assert report['cases']['low-4096']['status']=='passed-matched-numeric-gates'


def test_activity_scenario_rejects_changed_gpu_between_cases(modules,monkeypatch):
    compare,runner=modules;seen=[]
    def run(*a,**kw):
        seen.append(1)
        return dict(status='passed-matched-numeric-gates',nvidia_smi=str(len(seen)),payloads={})
    monkeypatch.setattr(compare,'compare',run)
    with pytest.raises(RuntimeError,match='identity'):runner.verify(scenario='activity')
