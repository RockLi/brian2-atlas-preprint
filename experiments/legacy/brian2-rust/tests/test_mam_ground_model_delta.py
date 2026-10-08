import copy
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from mam_prepare_ground_rerun import audit_model_delta,bits
s=importlib.util.spec_from_file_location('ground_parameter_fixture',Path(__file__).with_name('test_mam_ground_parameters.py'))
f=importlib.util.module_from_spec(s);s.loader.exec_module(f)


def fixture():
    bp,gp=f.fixture()
    def model(p):
        return dict(schema='test',definition=dict(synapses=[dict(name=f'mam_projection_{i}') for i in range(3)],external='frozen'),run=dict(duration=bits(2.5)),
                    instance=dict(rng_seed=1729,populations=[dict(v=['frozen'],refractory=21)],synapses=[dict(topology=dict(edge_count=q['count'],seed=i,initializers=dict(w=dict(mean=bits(q['weight_mean_pA']*1e-12),std=bits(q['weight_sd_pA']*1e-12),minimum=bits(0.)))),delay=[1,2]) for i,q in enumerate(p['projections'])]))
    return bp,gp,model(bp),model(gp)


def test_only_frozen_condition_weight_changes_allowed():
    bp,gp,old,new=fixture()
    result=audit_model_delta(old,new,bp,gp)
    assert result['weight_initializer_fields_changed']==4
    for mutate in [lambda m:m['definition'].update(external='changed'),
                   lambda m:m['instance']['populations'][0].update(v=['changed']),
                   lambda m:m['instance']['synapses'][1]['topology'].update(seed=99),
                   lambda m:m['instance']['synapses'][1].update(delay=[1,3]),
                   lambda m:m['instance']['synapses'][1]['topology']['initializers']['w'].update(minimum=bits(1.))]:
        bad=copy.deepcopy(new);mutate(bad)
        with pytest.raises(ValueError):audit_model_delta(old,bad,bp,gp)


def test_parameter_mapping_and_initializer_mean_are_checked():
    bp,gp,old,new=fixture()
    new['instance']['synapses'][1]['topology']['initializers']['w']['mean']=bits(1.)
    with pytest.raises(ValueError,match='initializer'):audit_model_delta(old,new,bp,gp)
