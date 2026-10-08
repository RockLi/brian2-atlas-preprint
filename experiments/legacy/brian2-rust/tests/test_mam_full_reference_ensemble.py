"""Cohort safety and missing-value contracts; no fixture is simulation evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import compare_mam_full_reference_ensemble as m


def test_three_seed_mean_and_sample_std_are_descriptive():
    result=m.describe([1.,2.,3.],7.)
    assert result['native_mean']==2 and result['native_sample_std']==1
    assert result['native_min']==1 and result['native_max']==3 and result['rust']==7
    assert not any(k in result for k in ['passed','equivalent','rust_outside_observed_range'])


def test_missing_observables_are_not_zero_padded_or_imputed():
    r=m.describe([None,2.,None],None)
    assert r['native_values']==[None,2.,None] and r['native_valid_count']==1
    assert r['native_mean']==2 and r['native_sample_std'] is None
    r=m.describe([None,None,None],5.)
    assert r['native_valid_count']==0
    assert all(r[k] is None for k in ['native_mean','native_sample_std','native_min','native_max'])


@pytest.mark.parametrize('values,rust',[([1,2],1),([1,2,3,4],1),([1,np.nan,3],1),([1,2,3],np.inf)])
def test_partial_or_nonfinite_native_summary_rejected(values,rust):
    with pytest.raises(ValueError):m.describe(values,rust)


def test_missing_third_completion_does_not_read_arrays_or_write_output(tmp_path,monkeypatch):
    evidence=tmp_path/'evidence';root=evidence/m.CAMPAIGN
    (root/'seed1730').mkdir(parents=True);(root/'seed1730/completion.json').write_text('{}')
    protocol=ROOT/'mpi-evidence'/m.CAMPAIGN/'protocol.json'
    (root/'protocol.json').write_bytes(protocol.read_bytes())
    def fail(*args,**kwargs):raise AssertionError('must not load data or audit completed cohort before third completion')
    monkeypatch.setattr(m,'load_cohort',fail);monkeypatch.setattr(m,'protocol_gate',fail)
    output=tmp_path/'not-created'
    r=m.run(evidence,tmp_path/'t7',output)
    assert r['ready'] is False and r['summary_written'] is False
    assert r['missing_completed_seeds']==[1731] and not output.exists()


def test_changed_protocol_rejected_even_when_waiting(tmp_path):
    p=tmp_path/m.CAMPAIGN/'protocol.json';p.parent.mkdir();p.write_text('{}')
    with pytest.raises(ValueError,match='protocol'):m.readiness(tmp_path,tmp_path/'t7')


def actual_reports(case):
    evidence=ROOT/'mpi-evidence'
    if case=='native1730':
        root=evidence/m.CAMPAIGN/'seed1730/analysis'
        return {s:json.loads((root/s/name).read_text()) for s,name in m.REPORTS.items()}
    root=evidence/('primary-postrun' if case=='rust1729' else 'primary-native-postrun/run')
    reports={s:json.loads((root/s/m.REPORTS[s]).read_text()) for s in ['activity','cell','correlation','series']}
    stem='rust' if case=='rust1729' else 'native'
    for stage in ['fc','lags']:
        reports[stage]=json.loads((evidence/'primary-interarea-v1/run'/(stem+'-'+stage)/m.REPORTS[stage]).read_text())
    return reports


@pytest.mark.parametrize('case,seed,simulator',[('rust1729',1729,'Rust'),('native1729',1729,'NEST'),('native1730',1730,'NEST')])
def test_existing_real_metadata_satisfies_identity_contract(case,seed,simulator):
    # Read actual completed scalar reports only; this does not claim a full
    # three-reference result or rerun any scientific calculation.
    m.validate_identity(actual_reports(case),seed,simulator)


@pytest.mark.parametrize('stage,key,value',[
    ('series','seed',1731),('cell','simulator','Rust'),('fc','parameters_sha256','changed'),
    ('lags','seed',1750)])
def test_cross_seed_stage_or_parameter_rejected(stage,key,value):
    reports=deepcopy(actual_reports('native1730'))
    reports[stage]['identity'][key]=value
    with pytest.raises(ValueError):m.validate_identity(reports,1730,'NEST')


@pytest.mark.parametrize('key,value',[('end_tick',25000),('spike_tick_offset',1),('raw_start_tick',4999),('seconds',2.)])
def test_short_window_or_wrong_physical_tick_mapping_rejected(key,value):
    reports=deepcopy(actual_reports('native1730'));reports['activity']['window'][key]=value
    with pytest.raises(ValueError,match='physical'):m.validate_identity(reports,1730,'NEST')


def test_two_references_cannot_be_summarized_as_full_cohort():
    with pytest.raises(ValueError,match='complete ordered cohort'):
        m.summarize(dict(rust1729={},native1729={},native1730={}),{})


def test_legacy_rust_cell_identity_requires_exact_pinned_model():
    reports=deepcopy(actual_reports('rust1729'))
    reports['cell']['identity']['model_sha256']='other-model'
    with pytest.raises(ValueError,match='legacy Rust cell'):m.validate_identity(reports,1729,'Rust')


def test_summary_preserves_metric_denominators_nulls_and_seed_axis():
    # Minimal synthetic arithmetic fixture. Production loading separately
    # requires 254 populations and 32 areas; no artifact is published here.
    cohort={}
    for case,value in [('rust1729',10),('native1729',1),('native1730',2),('native1731',3)]:
        missing=case=='native1730'
        activity=dict(mean_rate_hz=value,sampling=dict(synthetic=True),populations=[dict(name='p',neurons=10,mean_rate_hz=value,pairwise_corr_mean=.04,silent_fraction=.1,lvr_eligible_cells=3,lvr_sample_size=4,corr_sample_size=5,corr_nonconstant_cells=4)])
        cell=dict(window=dict(synthetic=True),populations=[dict(name='p',strict_spikes=1250*value,half_open_spikes=1250*value,lower_boundary_spikes=0,paper_lvr_mean=value/2,lvr_eligible_mean=value,diagnostic_sampled_lvr_mean=.2,diagnostic_sampled_eligible_lvr_mean=.3,lvr_eligible_cells=7)])
        corr=dict(observation_ms=[500,100500],endpoint='synthetic fixture',bin_ms=1,selection='synthetic fixture',calculation='synthetic fixture',populations=[dict(name='p',mean_pairwise_correlation=None if missing else value/10,available=not missing,selected_cells=0 if missing else 4,unavailable_reason='insufficient varying cells' if missing else None)])
        a=dict(area_rates_hz=np.full((1,100000),float(value)),power_hz2_per_hz=np.full((1,513),float(value)),frequency_hz=np.arange(513)*.9765625,functional_connectivity=np.ones((1,1)),reference_fc=np.ones((1,1)),retained_lag_ms=np.zeros((1,1)),levels_ms=np.zeros(1))
        cohort[case]=dict(arrays=a,reports=dict(activity=activity,cell=cell,correlation=corr,series=dict(area_names=['V1']),fc=dict(area_names=['V1']),lags=dict(hierarchy_area_names=['V1'])))
    original=dict(cases=dict(metastable100=dict(populations=[dict(name='p',modern_normalization_neurons=12.5,rate_hz=7,lvr=.5,correlation=.2)])))
    result,arrays=m.summarize(cohort,original)
    p=result['populations'][0]
    assert p['metrics']['strict_unrounded_rate_hz']['native_values']==[1,2,3]
    assert p['metrics']['lvr_all_cells']['native_values']==[.5,1,1.5]
    assert p['metrics']['lvr_eligible_only']['native_values']==[1,2,3]
    assert p['metrics']['paper_correlation']['native_values']==[.1,None,.3]
    assert p['metrics']['paper_correlation']['native_valid_count']==2
    assert p['eligibility']['native1730']['correlation_available'] is False
    assert p['eligibility']['native1730']['lvr_eligible_cells']==7
    assert p['eligibility']['native1730']['sampled_lvr_eligible_cells']==3
    assert p['eligibility']['native1730']['sampled_lvr_cells']==4
    assert p['eligibility']['native1730']['sampled_correlation_cells']==5
    assert p['eligibility']['native1730']['sampled_correlation_nonconstant_cells']==4
    assert arrays['psd_native_values'].shape==(3,1,513)
    np.testing.assert_array_equal(arrays['psd_native_mean'],np.full((1,513),2.))
    np.testing.assert_array_equal(arrays['psd_native_sample_std'],np.ones((1,513)))
    assert result['scientific_acceptance'] is result['performance_cost_acceptance'] is False
    json.dumps(result,allow_nan=False)


def test_excluded_mdp_nan_mask_is_preserved():
    names=[str(i) for i in range(32)];names[14]='MDP'
    full=np.zeros((32,32));full[14,:]=np.nan;full[:,14]=np.nan
    keep=m.validate_lag_missingness(full,names,['MDP'])
    assert len(keep)==31 and 14 not in keep and np.isnan(full).sum()==63


@pytest.mark.parametrize('row,col,value',[(0,1,np.nan),(14,0,0.),(0,14,np.inf)])
def test_unexpected_nan_or_imputed_mdp_lag_is_rejected(row,col,value):
    names=[str(i) for i in range(32)];names[14]='MDP'
    full=np.zeros((32,32));full[14,:]=np.nan;full[:,14]=np.nan;full[row,col]=value
    with pytest.raises(ValueError,match='do not impute'):
        m.validate_lag_missingness(full,names,['MDP'])
