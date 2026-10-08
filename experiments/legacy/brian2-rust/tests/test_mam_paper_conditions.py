import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('paper_conditions',ROOT/'tools/audit_mam_paper_conditions.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_data_only_condition_extraction_preserves_snapshot_times():
    code='''
import forbidden_simulation_module
sim_params = {'t_sim': 10500.}
params_stab = {}
network_params = {'connection_params': {'cc_weights_factor': 1.9}}
params_stab['1.9_spikes'] = (network_params, copy.deepcopy(sim_params))
sim_params.update({'t_sim': 100500.})
params_stab[1.9] = (network_params, copy.deepcopy(sim_params))
NEW_SIM_PARAMS = dangerous_call()
'''
    env,lines=m.extract_conditions(code)
    assert env['params_stab']['1.9_spikes'][1]['t_sim']==10500
    assert env['params_stab'][1.9][1]['t_sim']==100500
    assert lines['1.9']>lines['1.9_spikes']


def test_arbitrary_calls_and_excessive_loops_are_rejected():
    with pytest.raises(ValueError,match='unsupported experiment expression'):
        m.extract_conditions("x = __import__('os').system('touch never-execute')")
    with pytest.raises(ValueError,match='excessive experiment loop'):
        m.extract_conditions('values='+repr(list(range(21)))+'\nfor x in values:\n y = x\n')
