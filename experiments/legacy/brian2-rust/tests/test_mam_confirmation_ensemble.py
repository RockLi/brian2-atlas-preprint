"""Synthetic adapter regression; no new neural observation is manufactured."""
import ast,copy,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import compare_mam_confirmation_ensemble as new
import compare_mam_full_reference_ensemble as old
from test_mam_full_reference_ensemble import actual_reports
from test_mam_confirmation_terminal import controls


@pytest.fixture
def reports(controls):
    v=controls[0];r=copy.deepcopy(actual_reports('rust1729'))
    r['activity']['model_sha256']=v['model_sha256']
    for stage in ['cell','correlation','series','fc','lags']:
        r[stage]['identity']['model_sha256']=v['model_sha256']
        if stage!='cell':r[stage]['identity']['seed']=v['random_keys']['runtime_input']
    return r,v


def test_new_identity_is_explicit_and_legacy_gate_stays_closed(reports):
    r,v=reports;new.validate_identity(r,v)
    with pytest.raises(ValueError):old.validate_identity(r,1729,'Rust')


@pytest.mark.parametrize('stage,key,value',[
    ('series','seed',1750),('lags','seed',1729),('fc','seed',16757147634959265529.0),
    ('cell','model_sha256','0'*64),('correlation','simulator','NEST')])
def test_mixed_model_or_seed_rejected(reports,stage,key,value):
    r,v=reports;r[stage]['identity'][key]=value
    with pytest.raises(ValueError):new.validate_identity(r,v)


@pytest.mark.parametrize('key,value',[('end_tick',20000),('spike_tick_offset',0),('raw_start_tick',5000)])
def test_observation_convention_not_relaxed(reports,key,value):
    r,v=reports;r['activity']['window'][key]=value
    with pytest.raises(ValueError):new.validate_identity(r,v)


def test_missing_stage_model_is_rejected(reports):
    r,v=reports;del r['series']['identity']['model_sha256']
    with pytest.raises(ValueError):new.validate_identity(r,v)


def test_waiting_for_science_does_not_load_reference_or_arrays(tmp_path,monkeypatch):
    def fail(*a,**k):pytest.fail('no evidence or arrays before science completion')
    monkeypatch.setattr(new.legacy,'readiness',fail);monkeypatch.setattr(new,'load_cohort',fail)
    out=tmp_path/'output';r=new.run(tmp_path,tmp_path,tmp_path,out)
    assert not r['ready'] and not r['summary_written'] and not out.exists()


def test_summary_arithmetic_and_missingness_unchanged():
    cohort={}
    for case,value in [('rust1729',10),('native1729',1),('native1730',2),('native1731',3)]:
        missing=case=='native1730'
        activity=dict(mean_rate_hz=value,sampling=dict(synthetic=True),populations=[dict(name='p',neurons=10,mean_rate_hz=value,pairwise_corr_mean=.04,silent_fraction=.1,lvr_eligible_cells=3,lvr_sample_size=4,corr_sample_size=5,corr_nonconstant_cells=4)])
        cell=dict(window=dict(synthetic=True),populations=[dict(name='p',strict_spikes=1250*value,half_open_spikes=1250*value,lower_boundary_spikes=0,paper_lvr_mean=value/2,lvr_eligible_mean=value,diagnostic_sampled_lvr_mean=.2,diagnostic_sampled_eligible_lvr_mean=.3,lvr_eligible_cells=7)])
        corr=dict(observation_ms=[500,100500],endpoint='synthetic fixture',bin_ms=1,selection='synthetic fixture',calculation='synthetic fixture',populations=[dict(name='p',mean_pairwise_correlation=None if missing else value/10,available=not missing,selected_cells=0 if missing else 4,unavailable_reason='insufficient varying cells' if missing else None)])
        a=dict(area_rates_hz=np.full((1,100000),float(value)),power_hz2_per_hz=np.full((1,513),float(value)),frequency_hz=np.arange(513)*.9765625,functional_connectivity=np.ones((1,1)),reference_fc=np.ones((1,1)),retained_lag_ms=np.zeros((1,1)),levels_ms=np.zeros(1))
        cohort[case]=dict(arrays=a,reports=dict(activity=activity,cell=cell,correlation=corr,series=dict(area_names=['V1']),fc=dict(area_names=['V1']),lags=dict(hierarchy_area_names=['V1'])))
    original=dict(cases=dict(metastable100=dict(populations=[dict(name='p',modern_normalization_neurons=12.5,rate_hz=7,lvr=.5,correlation=.2)])))
    old_result,old_arrays=old.summarize(cohort,original)
    adapted={'rust1750':copy.deepcopy(cohort['rust1729']),**{k:v for k,v in cohort.items() if k!='rust1729'}}
    adapted['rust1750']['reports']['series']['identity']={'seed':16757147634959265529}
    result,arrays=new.summarize(adapted,original)
    assert result.pop('rust_replicate')==1750
    assert result.pop('rust_runtime_random_key')==16757147634959265529
    result['rust_seed']=1729
    for p in result['populations']:p['eligibility']['rust1729']=p['eligibility'].pop('rust1750')
    assert result==old_result
    assert arrays.keys()==old_arrays.keys()
    for key in arrays:np.testing.assert_array_equal(arrays[key],old_arrays[key])


def test_no_partial_reference_cohort_or_old_rust_label():
    for keys in [('rust1750','native1729','native1730'),('rust1729','native1729','native1730','native1731')]:
        with pytest.raises(ValueError):new.summarize(dict.fromkeys(keys,{}),{})


def test_full_cohort_convention_and_array_validation_unchanged():
    def block(module):
        tree=ast.parse(Path(module.__file__).read_text())
        body=next(n.body for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_cohort')
        start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='base' for t in n.targets))
        return [ast.dump(n) for n in body[start:start+5]]
    assert block(new)==block(old)
