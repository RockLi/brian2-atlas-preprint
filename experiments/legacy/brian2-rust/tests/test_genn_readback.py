"""Sparse host gather preserves row order and never retains unloaded buffers."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    import gpu_genn_readback
    return gpu_genn_readback


def group():
    return SimpleNamespace(connections_set=True,_any_ccu_references=False,
        row_lengths=np.array([2,0,1,3],np.uint32),max_connections=3,weight_update_var_size=12)


@pytest.mark.parametrize('batched',[False,True])
def test_ragged_empty_rows_and_reloaded_host_view(module,batched):
    g=group();plan=module.SparseRows(g)
    class Variable:
        @property
        def values(self):
            return np.concatenate([self._view[...,i*3:i*3+int(n)] for i,n in enumerate(g.row_lengths)],axis=-1)
    var=Variable();original=Variable.values
    state=dict(plans=[plan],variables={id(var):(var,plan)},variable_class=Variable)
    for offset in [0,100]:
        var._view=np.arange(12,dtype=np.float32)+offset
        if batched:var._view=var._view.reshape(1,-1)
        with module.gather_values(state,verify=True) as calls:
            actual=var.values
            np.testing.assert_array_equal(actual.reshape(-1),np.array([0,1,6,9,10,11])+offset)
            assert not np.shares_memory(actual,var._view)
        assert calls=={id(var):1} and Variable.values is original
        var._view=None


@pytest.mark.parametrize('mutation',[lambda g:setattr(g,'_any_ccu_references',True),
    lambda g:setattr(g,'connections_set',False),lambda g:setattr(g,'max_connections',4),
    lambda g:g.row_lengths.__setitem__(0,1)])
def test_changed_static_layout_rejected(module,mutation):
    g=group();plan=module.SparseRows(g);mutation(g)
    with pytest.raises(ValueError):plan.validate()


@pytest.mark.parametrize('view',[None,np.zeros(12,np.float64),np.zeros((2,12),np.float32),np.zeros(11,np.float32)])
def test_wrong_precision_unloaded_or_multibatch_rejected(module,view):
    plan=module.SparseRows(group())
    with pytest.raises(ValueError):plan.gather(SimpleNamespace(_view=view))


def test_getter_restored_when_crosscheck_detects_wrong_result(module):
    class Variable:
        @property
        def values(self):return np.zeros(6,np.float32)
    var=Variable();var._view=np.arange(12,dtype=np.float32);original=Variable.values
    plan=module.SparseRows(group())
    state=dict(plans=[plan],variables={id(var):(var,plan)},variable_class=Variable)
    with pytest.raises(RuntimeError,match='differs'):
        with module.gather_values(state,verify=True):var.values
    assert Variable.values is original


def test_unknown_vendor_source_rejected_before_private_view_access(module,monkeypatch,tmp_path):
    import sys,types
    source=tmp_path/'unknown.py';source.write_text('changed package')
    parent=types.ModuleType('pygenn');vendor=types.ModuleType('pygenn.model_preprocessor')
    vendor.__file__=str(source);parent.model_preprocessor=vendor
    monkeypatch.setitem(sys.modules,'pygenn',parent);monkeypatch.setitem(sys.modules,'pygenn.model_preprocessor',vendor)
    with pytest.raises(ValueError,match='Unrecognized'):module.prepare({})


def test_nine_worker_ablation_keeps_original_and_corrected_readbacks(module,monkeypatch):
    import modal_stdp_precompiled as compare
    import modal_genn_readback_compare as runner
    def run(*args,**kwargs):
        assert args==(4096,256,8,5)
        labels=kwargs['requested_backends']
        assert len(labels)==9 and len(set(labels))==9
        assert {'genn','genn-barrier','genn-barrier-gather','cuda','brian2cuda','brian2genn-corrected'}.issubset(labels)
        return dict(status='completed-with-gate-failures')
    monkeypatch.setattr(compare,'compare',run)
    assert not runner.verify()['passed']
