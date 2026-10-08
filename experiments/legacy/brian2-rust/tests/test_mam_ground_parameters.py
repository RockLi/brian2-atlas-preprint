import copy
import math
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name):
    s=importlib.util.spec_from_file_location(name,ROOT/'tools'/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
exporter=load('export_multi_area_parameters');auditor=load('audit_mam_ground_parameters')


def fixture():
    pops=[dict(area='A',population='23E'),dict(area='B',population='23E'),dict(area='B',population='23I')]
    conn=dict(g=-11.,fac_nu_ext_TH=1.2,fac_nu_ext_5E=1.125,fac_nu_ext_6E=1.41666667,av_indegree_V1=3950.,K_stable='baseline.npy',cc_weights_factor=1.9,cc_weights_I_factor=2.)
    p=dict(state='metastable',params=dict(connection_params=conn,input_params=dict(rate_ext=10.),neuron_params=dict(V0_mean=-150.,V0_sd=50.)),populations=pops,
           projections=[dict(source=0,target=t,count=101,delay_mean_ms=2.,weight_mean_pA=100.*f,weight_sd_pA=10.*f) for t,f in enumerate([1.,1.9,3.8])])
    g=copy.deepcopy(p);g['state']='stabilized-ground';g['params']['connection_params'].update(cc_weights_factor=1.,cc_weights_I_factor=1.,K_stable='ground.npy')
    for row in g['projections']:row.update(weight_mean_pA=100.,weight_sd_pA=10.)
    return p,g


def test_ground_does_not_inherit_modern_metastable_factors():
    p,g=fixture()
    with pytest.raises(ValueError,match='resolved'):
        exporter.resolved_condition(p['params'],'stabilized-ground')
    assert exporter.condition_overrides('metastable')=={}
    conn=copy.deepcopy(p['params']['connection_params']);conn.update(exporter.condition_overrides('stabilized-ground')['connection_params'])
    params=dict(p['params'],connection_params=conn)
    assert exporter.resolved_condition(params,'stabilized-ground')['cc_weights_I_factor']==1
    with pytest.raises(ValueError):exporter.condition_overrides('ground-ish')


def test_all_projection_quantities_are_audited():
    p,g=fixture();r=auditor.compare_parameters(p,g)
    assert r['projection_classes']==dict(local=1,cc_to_E=1,cc_to_I=1)
    for key,value in [('count',102),('delay_mean_ms',2.1),('weight_sd_pA',11.)]:
        bad=copy.deepcopy(g);bad['projections'][2][key]=value
        with pytest.raises(ValueError):auditor.compare_parameters(p,bad)
    bad=copy.deepcopy(g);bad['params']['input_params']['rate_ext']=11
    with pytest.raises(ValueError,match='parameter'):auditor.compare_parameters(p,bad)
    bad=copy.deepcopy(g);bad['populations'][0]['external_indegree']=9
    with pytest.raises(ValueError,match='population'):auditor.compare_parameters(p,bad)


def test_label_alone_cannot_select_ground():
    p,g=fixture();bad=copy.deepcopy(p);bad['state']='stabilized-ground'
    with pytest.raises(ValueError,match='factors'):auditor.compare_parameters(p,bad)


def test_scaled_ground_rejected_before_source_or_output_access(tmp_path):
    with pytest.raises(ValueError,match='fullscale rate reference'):
        exporter.export(tmp_path/'missing-source',tmp_path/'output',.5,.5,condition='stabilized-ground')
    assert not (tmp_path/'output').exists()


def test_even_single_ulp_undeclared_changes_are_rejected():
    p,g=fixture()
    bad=copy.deepcopy(g)
    bad['projections'][1]['weight_mean_pA']=math.nextafter(100.,math.inf)
    with pytest.raises(ValueError,match='weight'):auditor.compare_parameters(p,bad)
    p['populations'][0]['external_indegree']=100.
    g['populations'][0]['external_indegree']=math.nextafter(100.,math.inf)
    with pytest.raises(ValueError,match='population'):auditor.compare_parameters(p,g)
