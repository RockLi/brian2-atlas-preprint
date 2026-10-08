"""Correction must be explicit, outside timing, and never rescue a failed gate."""
import sys
import types
from pathlib import Path
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    import gpu_genn_barrier_adapter as adapter
    import modal_stdp_precompiled as compare
    return adapter,compare


@pytest.mark.parametrize('fails',[False,True])
@pytest.mark.parametrize('delays',[8,16])
def test_correction_build_precedes_bootstrap_and_restores_builder(modules,monkeypatch,tmp_path,fails,delays):
    adapter,_=modules;events=[]
    class Model:
        def build(self):events.append('build')
    original=Model.build
    monkeypatch.setitem(sys.modules,'pygenn',types.SimpleNamespace(GeNNModel=Model))
    def patch(path,*,groups):
        assert groups==delays
        events.append('patch-and-rebuild')
        if fails:raise RuntimeError('rebuild failed')
        return {'barriers_added':delays}
    monkeypatch.setattr(adapter,'patch_and_build',patch)
    def genn(*args,**kwargs):
        Model().build();events.append('load-and-bootstrap')
        assert kwargs['trace_mode']=='device'
        return {'result':True},{}
    monkeypatch.setattr(adapter,'genn_run',genn)
    if fails:
        with pytest.raises(RuntimeError):adapter.run(4096,8,256,tmp_path,delay_span=delays)
        assert events==['build','patch-and-rebuild']
    else:
        _,timing=adapter.run(4096,8,256,tmp_path,delay_span=delays)
        assert events==['build','patch-and-rebuild','load-and-bootstrap']
        assert 'not stock' in timing['label'] and timing['generated_code_correction']['barriers_added']==delays
    assert Model.build is original


def test_unvalidated_delay_layout_rejected_before_build(modules,tmp_path):
    adapter,_=modules
    with pytest.raises(ValueError):adapter.run(4096,8,256,tmp_path,delay_span=17)


@pytest.mark.parametrize('keep_going',[True,False])
def test_late_failure_cancels_all_timing_credit_but_preserves_samples(modules,keep_going):
    _,compare=modules
    passed=dict(passed=True,original_f64_gate={'spikes':True})
    failed=dict(passed=False,original_f64_gate={'spikes':False})
    samples=[dict(backend=b,round=i,wall_seconds=.1,checks=passed if i==0 or b=='cuda' else failed)
             for b in ['genn','cuda'] for i in [0,1]]
    report=dict(excluded_by_gate={},samples=samples)
    if keep_going:compare.retain_gate_failure(report,'genn',failed,'replay',True)
    else:
        with pytest.raises(RuntimeError):compare.retain_gate_failure(report,'genn',failed,'replay',False)
    assert report['samples'] is samples and len(samples)==4
    summary=compare.timing_summary(['genn','cuda'],samples,2,report['excluded_by_gate'])
    assert summary[0]['median_seconds'] is None and not summary[0]['matched_gate_passed']
    assert summary[1]['complete'] and summary[1]['matched_gate_passed']
    assert report['exclusion_phase']=={'genn':'replay'}


def test_declared_eight_way_comparison_keeps_stock_failure(modules,monkeypatch):
    _,compare=modules
    import modal_genn_barrier_compare as runner
    def run(*args,**kwargs):
        assert args==(4096,256,8,5)
        assert {'genn','genn-barrier','brian2cuda','brian2genn-corrected','cuda','cuda-prefix','rust-f64','cpu-f32'}==set(kwargs['requested_backends'])
        assert kwargs['continue_on_gate_failure']
        return dict(status='completed-with-gate-failures',excluded_by_gate={'genn':{'passed':False}})
    monkeypatch.setattr(compare,'compare',run)
    result=runner.verify()
    assert not result['passed'] and result['excluded_by_gate']['genn']['passed'] is False
