"""Both timed modes use the same model and exclude all compilation."""
from pathlib import Path
import subprocess
import random
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    import genn_same_model_readback
    return genn_same_model_readback


def context():
    class Variable:
        @property
        def values(self):return None
    return dict(model=object(),host_gather=dict(variable_class=Variable),
        last_replay_timings={'phase':0},last_gather={'calls':64})


@pytest.mark.parametrize('mode',['public','gather'])
def test_no_compile_and_same_context_for_both_modes(module,monkeypatch,mode):
    state=context();model=state['model'];getter=state['host_gather']['variable_class'].values
    def run(c,steps):
        assert c is state and c['model'] is model and steps==256
        with pytest.raises(RuntimeError,match='Compilation attempted'):
            subprocess.run(['make'],check=True)
        return {'result':True}
    monkeypatch.setattr(module,'genn_replay',run if mode=='public' else lambda *a:pytest.fail('wrong mode'))
    monkeypatch.setattr(module,'gather_replay',run if mode=='gather' else lambda *a:pytest.fail('wrong mode'))
    result,timing=module.timed_replay(state,mode,256)
    assert result=={'result':True} and timing['wall_seconds']>=0
    assert state['host_gather']['variable_class'].values is getter


def test_invalid_mode_never_executes(module,monkeypatch):
    monkeypatch.setattr(module,'genn_replay',lambda *a:pytest.fail('executed'))
    with pytest.raises(ValueError):module.timed_replay(context(),'unknown',256)


@pytest.mark.parametrize('change',['model','getter'])
def test_replay_cannot_silently_replace_identity(module,monkeypatch,change):
    state=context()
    def run(c,steps):
        if change=='model':c['model']=object()
        else:c['host_gather']['variable_class'].values=property(lambda self:None)
        return {}
    monkeypatch.setattr(module,'genn_replay',run)
    with pytest.raises(RuntimeError,match='identity'):module.timed_replay(state,'public',256)


def test_balanced_reproducible_rounds_without_global_rng_mutation(module):
    before=random.getstate();rows=list(module.order())
    assert rows==list(module.order()) and random.getstate()==before
    assert len(rows)==12
    for i in range(-1,5):
        group=[r for r in rows if r[0]==i]
        assert {r[2] for r in group}=={'public','gather'} and [r[1] for r in group]==[0,1]
    assert {r[2] for r in rows if r[1]==0}=={'public','gather'}
