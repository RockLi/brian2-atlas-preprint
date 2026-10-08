"""Explicit comparator integration preserves earlier model sources and gates."""
import gzip
import json
from pathlib import Path
import numpy as np
import pytest
from gpu_brian2genn_stdp_adapter import transform_model, run


def artifact(folder,name):
    e=Path(__file__).resolve().parents[1]/'execution-plan-evidence'/folder
    meta=json.loads((e/'manifest.json').read_text())['artifacts'][name]
    data=(e/'blobs'/meta['blob']).read_bytes()
    return gzip.decompress(data) if meta['encoding']=='gzip' else data


@pytest.mark.parametrize('host',['l4','a100'])
def test_integrated_mode_reproduces_both_previous_native_programs(host):
    report=json.loads(artifact('dense-stdp',host+'/report.json'))
    original=report['cases']['dense-1024']['workers']['brian2genn-corrected']['adapter_evidence']['magicnetwork_model.cpp.original']
    opts=dict(drive=1/16,post_delay=3)
    mapping={'plastic'+str(q):7-q for q in range(8)}
    expected=artifact('brian2genn-precision','history/'+host+'-generated.cpp').decode()
    assert transform_model(original,1024,mapping,**opts)==expected
    assert transform_model(original,1024,mapping,decay_mode='generated',**opts)==expected
    corrected=transform_model(original,1024,mapping,decay_mode='f32-factor',**opts)
    assert corrected==artifact('brian2genn-precision','history/'+host+'-f32-factor.cpp').decode()


@pytest.mark.parametrize('mode',['float32','f64','',None,True])
def test_unknown_mode_rejected_before_source_or_native_use(mode):
    with pytest.raises(ValueError,match='decay mode'):
        transform_model('',1,{},decay_mode=mode)
    with pytest.raises(ValueError,match='decay mode'):
        run(17,7,16,Path('/must-not-create'),decay_mode=mode)


def test_new_worker_is_explicit_and_retains_failed_old_adapter(monkeypatch):
    import modal_stdp_precompiled as compare
    import modal_f32_stdp_compare as runner
    from gpu_stdp_precompiled import BACKENDS,F32_GENN_BACKENDS
    name='brian2genn-f32-factor'
    assert name in F32_GENN_BACKENDS and name not in BACKENDS and name not in runner.BACKENDS
    wanted=list(runner.BACKENDS)+[name]
    def run_case(*args,**kwargs):
        assert kwargs['requested_backends']==wanted
        assert kwargs['numeric_contract']=='explicit-f32-v1'
        assert kwargs['continue_on_gate_failure']
        return dict(status='completed-with-gate-failures',nvidia_smi='same GPU',
                    payloads={'kept.npz':b'kept'},excluded_by_gate={'brian2genn-corrected':{'passed':False}})
    monkeypatch.setattr(compare,'compare',run_case)
    report=runner.verify(wanted,scenario='dense')
    assert report['backends']==wanted
    assert not report['passed'] and report['cases']['dense-1024']['excluded_by_gate']


def test_new_worker_still_requires_both_f32_gates():
    from modal_stdp_precompiled import qualified
    good=dict(passed=True,original_f64_gate={'w':False},independent_f32_gate={'passed':True})
    assert qualified(good,'brian2genn-f32-factor','explicit-f32-v1')
    assert not qualified(good,'brian2genn-f32-factor','f64-compatible')
    for field in ('compiled','independent'):
        bad=dict(good,passed=field!='compiled',independent_f32_gate={'passed':field!='independent'})
        assert not qualified(bad,'brian2genn-f32-factor','explicit-f32-v1')
