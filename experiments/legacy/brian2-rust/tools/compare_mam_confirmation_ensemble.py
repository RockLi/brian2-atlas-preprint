"""Fixed six-metric descriptive comparison of Rust1750 with three NEST references.

Consumes accepted complete 100 s observations. No seed relabeling, added
replicates, imputation, fitted tolerances or formal equivalence claims.
"""
from mam_confirmation_profile import REPLICATE,selected,worker_environment,IDENTITY_SHA_1751
import argparse,json,os,time
from pathlib import Path
import numpy as np
import compare_mam_full_reference_ensemble as legacy
from compare_mam_full_reference_ensemble import SEEDS,PARAMETERS,POP_METRICS,REPORTS,CAMPAIGN,PROTOCOL_SHA,describe,validate_lag_missingness
from compare_mam_confirmation_lags import accepted_science
from mam_launch_confirmation_run import CASE
from mam_confirmation_terminal_sync import IDENTITY_SHA
from mam_primary_analysis_pipeline import check,read,sha
from mam_paper_propagation import fit_hierarchy
IDENTITY_SHA=selected(IDENTITY_SHA,IDENTITY_SHA_1751)


def validate_identity(reports, admitted):
    simulator='Rust';seed=admitted['random_keys']['runtime_input']
    activity=reports['activity']
    check(activity['model_sha256']==admitted['model_sha256'],'confirmation model differs')
    for stage in ['cell', 'correlation', 'series', 'fc', 'lags']:
        ident = reports[stage]['identity']
        if stage == 'cell':
            # This retained helper emits only the model hash. The exact
            # immutable full model, raw audit and catalog bind it to replicate1750;
            # do not add a fabricated seed to its report.
            check(ident == dict(simulator='Rust',model_sha256=admitted['model_sha256']), 'Rust cell identity differs')
        else:
            check(type(ident.get('seed')) is int and ident.get('seed') == seed and ident.get('simulator') == simulator, 'cross-seed/stage report')
        if 'parameters_sha256' in ident: check(ident['parameters_sha256'] == PARAMETERS, 'parameters differ')
        check(ident.get('model_sha256') == admitted['model_sha256'], 'Rust model differs')
    w = activity['window']; offset = 1
    check(w['start_tick'] == 5000 and w['end_tick'] == 1005000 and w['seconds'] == 100
          and w['dt_seconds'] == .0001 and w['bin_ticks'] == 10 and w['endpoint'] == '[start,end)'
          and w['spike_tick_offset'] == offset and w['raw_start_tick']+offset == 5000
          and w['raw_end_tick']+offset == 1005000, 'physical activity window differs')
    check(reports['cell']['window'] == dict(start_tick=5000, end_tick=1005000, dt_ms=.1,
          rate_endpoint='(start,end)', lvr_endpoint='[start,end)', refractory_ms=2), 'cell window differs')
    check(reports['correlation']['observation_ms'] == [500,100500]
          and reports['correlation']['frozen_histograms_exact'], 'correlation observation differs')
    check(reports['series']['frozen_histogram_exact'] and reports['series']['endpoint_count_identity_exact'],
          'series reconciliation failed')
    check(reports['fc']['observation_seconds'] == reports['lags']['observation_seconds'] == 100,
          'interarea duration differs')



def load_cohort(evidence,t7,data):
    case=evidence/CASE;admitted,science=accepted_science(case)
    baseline,inputs=legacy.load_cohort(evidence,t7)
    root=case/'science'
    reports={stage:inputs.report(root/stage,name) for stage,name in REPORTS.items()}
    for stage,digest in science['catalogs'].items():
        check(sha(root/stage/'catalog.json')==digest,'confirmation scalar catalog changed')
    validate_identity(reports,admitted)
    arrays={}
    specs=[('series','time-series.npz',['area_rates_hz','frequency_hz','power_hz2_per_hz']),
        ('fc','fc.npz',['functional_connectivity','reference_fc']),
        ('lags','lags.npz',['lag_ms','retained_lag_ms','levels_ms','normalized_levels'])]
    for stage,name,keys in specs:
        check(sha(data/stage/'catalog.json')==science['catalogs'][stage], 'confirmation array catalog changed')
        arrays.update(inputs.arrays(data/stage,name,keys))
    cohort={f'rust{REPLICATE}':dict(reports=reports,arrays=arrays)}
    cohort.update({f'native{s}':baseline[f'native{s}'] for s in SEEDS})
    base = cohort['native1729']['reports']
    names = [p['name'] for p in base['activity']['populations']]
    sizes = [p['neurons'] for p in base['activity']['populations']]
    check(len(names) == len(set(names)) == 254 and sum(sizes) == 4129924, 'population universe differs')
    for entry in cohort.values():
        r,a = entry['reports'],entry['arrays']
        check([p['name'] for p in r['activity']['populations']] == names
              and [p['neurons'] for p in r['activity']['populations']] == sizes, 'population ordering/size differs')
        for stage in ['cell','correlation']:
            check(len(r[stage]['populations']) == 254 and {p['name'] for p in r[stage]['populations']} == set(names), 'incomplete population metric')
        checks = dict(activity=['sampling'],correlation=['endpoint','bin_ms','selection','calculation','helper_sha256','wrapper_sha256','toolbox_commit'],
            series=['area_names','population_names','welch','unrounded_population_sum','wrapper_window_ms','helper_histogram_range_ms','bin_ms','welch_segments','normalization','area_weighting'],
            fc=['area_names','settings','parameters_sha256','reference_observation_seconds'],
            lags=['area_names','settings','hierarchy_area_names','excluded_from_hierarchy'])
        for stage, keys in checks.items():
            for key in keys: check(r[stage][key] == base[stage][key], 'cohort convention differs: '+stage+'/'+key)
        check(a['area_rates_hz'].shape == (32,100000) and a['power_hz2_per_hz'].shape == (32,513), 'incomplete area series')
        check(np.array_equal(a['frequency_hz'],np.arange(513)*.9765625), 'frequency grid differs')
        check(a['functional_connectivity'].shape == (32,32) and a['retained_lag_ms'].shape == (31,31)
              and a['lag_ms'].shape == (32,32) and a['levels_ms'].shape == (31,), 'incomplete interarea arrays')
        check(all(np.isfinite(v).all() for k,v in a.items() if k != 'lag_ms'), 'nonfinite full-cohort arrays')
        np.testing.assert_array_equal(a['reference_fc'],cohort['native1729']['arrays']['reference_fc'])
        keep=validate_lag_missingness(a['lag_ms'],r['lags']['area_names'],r['lags']['excluded_from_hierarchy'])
        np.testing.assert_array_equal(a['retained_lag_ms'],a['lag_ms'][np.ix_(keep,keep)])
        np.testing.assert_array_equal(a['levels_ms'],r['lags']['levels_ms'])
        np.testing.assert_array_equal(a['normalized_levels'],r['lags']['normalized_levels'])
        np.testing.assert_allclose(a['levels_ms'],fit_hierarchy(a['retained_lag_ms'])['levels_ms'],rtol=1e-12,atol=1e-12)
        for stage,keys in [('fc',['reference_report','matrices','matrices_metadata']),('lags',['reference_audit','matrices_metadata'])]:
            for key in keys: check(r[stage]['source_sha256'][key] == base[stage]['source_sha256'][key], 'interarea source differs')
    inputs.pins[str(root/'report.json')]=sha(root/'report.json')
    inputs.pins[str(root/'complete.json')]=sha(root/'complete.json')
    inputs.pins[str(case/'identity.json')]=IDENTITY_SHA
    return cohort,inputs


def summarize(cohort, original):
    check(list(cohort) == [f'rust{REPLICATE}','native1729','native1730','native1731'], 'complete ordered cohort required')
    native = [cohort[f'native{s}'] for s in SEEDS]; rust = cohort[f'rust{REPLICATE}']
    norms = {p['name']:p for p in original['cases']['metastable100']['populations']}
    values = {}; counts = {}
    for case, entry in cohort.items():
        reports = entry['reports']; cells = {p['name']:p for p in reports['cell']['populations']}; corrs = {p['name']:p for p in reports['correlation']['populations']}
        values[case] = {}; counts[case] = {}
        for p in reports['activity']['populations']:
            name=p['name'];c=cells[name];q=corrs[name]
            check(c['strict_spikes'] == c['half_open_spikes']-c['lower_boundary_spikes'], 'strict rate reconciliation differs')
            values[case][name]=dict(activity_rate_hz=p['mean_rate_hz'],strict_unrounded_rate_hz=c['strict_spikes']/(100*norms[name]['modern_normalization_neurons']),
                lvr_all_cells=c['paper_lvr_mean'],lvr_eligible_only=c['lvr_eligible_mean'],sampled_lvr_all_cells=c['diagnostic_sampled_lvr_mean'],
                sampled_lvr_eligible_only=c['diagnostic_sampled_eligible_lvr_mean'],paper_correlation=q['mean_pairwise_correlation'] if q['available'] else None,
                sampled_correlation=p['pairwise_corr_mean'],silent_fraction=p['silent_fraction'])
            counts[case][name]=dict(simulated_neurons=p['neurons'],lvr_eligible_cells=c['lvr_eligible_cells'],sampled_lvr_eligible_cells=p['lvr_eligible_cells'],
                sampled_lvr_cells=p['lvr_sample_size'],sampled_correlation_cells=p['corr_sample_size'],sampled_correlation_nonconstant_cells=p['corr_nonconstant_cells'],
                correlation_selected_cells=q['selected_cells'],correlation_available=q['available'],correlation_unavailable_reason=q['unavailable_reason'])
    populations=[]
    for name in values[f'rust{REPLICATE}']:
        populations.append(dict(name=name,metrics={k:describe([values[f'native{s}'][name][k] for s in SEEDS],values[f'rust{REPLICATE}'][name][k]) for k in POP_METRICS},
            eligibility={case:counts[case][name] for case in cohort},published={k:norms[name][k] for k in ['rate_hz','lvr','correlation']}))
    areas=[]
    for i,name in enumerate(native[0]['reports']['series']['area_names']):
        summaries={}
        for case,entry in cohort.items():
            rate=entry['arrays']['area_rates_hz'][i];power=entry['arrays']['power_hz2_per_hz'][i];frequency=entry['arrays']['frequency_hz']
            summaries[case]=dict(mean_rate_hz=float(rate.mean()),first_half_mean_hz=float(rate[:50000].mean()),second_half_mean_hz=float(rate[50000:].mean()),
                integrated_psd=float(power.sum()*.9765625),peak_nonzero_frequency_hz=float(frequency[1+np.argmax(power[1:])]))
        areas.append(dict(area=name,metrics={k:describe([summaries[f'native{s}'][k] for s in SEEDS],summaries[f'rust{REPLICATE}'][k]) for k in summaries[f'rust{REPLICATE}']}))
    small={};summary_arrays={}
    for name,key in [('fc','functional_connectivity'),('lags','retained_lag_ms'),('hierarchy','levels_ms'),('psd','power_hz2_per_hz')]:
        x=np.stack([r['arrays'][key] for r in native])
        summary_arrays.update({name+'_native_values':x,name+'_native_mean':x.mean(axis=0),name+'_native_sample_std':x.std(axis=0,ddof=1),
            name+'_native_min':x.min(axis=0),name+'_native_max':x.max(axis=0),name+'_rust':rust['arrays'][key]})
        small[name]=dict(shape=list(x.shape[1:]),native_realizations=3,dependent_entries=True)
    summary_arrays['frequency_hz']=native[0]['arrays']['frequency_hz'];summary_arrays['published_fc']=native[0]['arrays']['reference_fc']
    return dict(native_seeds=SEEDS,rust_replicate=REPLICATE,rust_runtime_random_key=rust['reports']['series']['identity']['seed'],scientific_acceptance=False,performance_cost_acceptance=False,
        inference='Descriptive sample means, ddof=1 standard deviations and observed ranges; not confidence, prediction or equivalence intervals. No outside-range failure flag or fitted acceptance margin.',
        global_activity_rate_hz=describe([r['reports']['activity']['mean_rate_hz'] for r in native],rust['reports']['activity']['mean_rate_hz']),
        activity_sampling=native[0]['reports']['activity']['sampling'],cell_window=native[0]['reports']['cell']['window'],
        correlation_convention={k:native[0]['reports']['correlation'][k] for k in ['observation_ms','endpoint','bin_ms','selection','calculation']},
        populations=populations,areas=areas,matrices_and_spectra=small,series_area_names=native[0]['reports']['series']['area_names'],
        fc_area_names=native[0]['reports']['fc']['area_names'],hierarchy_area_names=native[0]['reports']['lags']['hierarchy_area_names'],
        excluded_from_lag_hierarchy=['MDP'],excluded_lag_values='The source 32x32 lag matrices retain NaN for the MDP row/column; summaries use only the unchanged finite 31x31 matrices.',
        conventions='Historical LvR denominator remains unresolved; all-cell and eligible-only views are separate. Unavailable correlations are null, never zero. Native realizations are independent; cells, pairs, time bins and Welch segments are not seed replicates. Lowest nonzero PSD grid bin is not proof of a 1 Hz oscillation. Full traces remain in pinned input artifacts.'),summary_arrays

def run(evidence,t7,data,output):
    if not (evidence/CASE/'science/complete.json').is_file():
        return dict(ready=False,summary_written=False,reason='Full confirmation science is pending.')
    state=legacy.readiness(evidence,t7)
    if not state['ready']:return dict(**state,summary_written=False)
    check(not output.exists(),'cohort summary already exists')
    check(Path('/Volumes/T7').is_mount() and output.resolve().is_relative_to((t7/'artifacts').resolve())
          and t7.resolve().is_relative_to(Path('/Volumes/T7').resolve()), 'mounted T7 output required')
    disk=os.statvfs(t7)
    check(disk.f_bavail*disk.f_frsize >= 128*2**30+64*2**20, 'T7 reserve')
    started=time.monotonic();cohort,inputs=load_cohort(evidence,t7,data)
    original=evidence/'mam-original-statistics/original-statistics-v1.json'
    check(sha(original)=='d37e1be7a896e19e60004bb93ab73ca19614077436a894b92b00c1ccff86b15b','published scalar source changed')
    result,arrays=summarize(cohort,read(original));inputs.pins[str(original)]=sha(original)
    for seed in SEEDS[1:]:
        p=evidence/CAMPAIGN/f'seed{seed}/completion.json';inputs.pins[str(p)]=sha(p)
    result.update(schema='b2-mam-confirmation-reference-ensemble-description-v1',identity_sha256=IDENTITY_SHA,formal_equivalence_acceptance=False,protocol_sha256=PROTOCOL_SHA,source_sha256=inputs.pins,
                  implementation_sha256={p.name:sha(p) for p in [Path(__file__),Path(legacy.__file__),Path(__file__).with_name('compare_mam_confirmation_lags.py')]},analysis_seconds=time.monotonic()-started)
    payload=json.dumps(result,indent=2,allow_nan=False)+'\n'
    output.mkdir();(output/'comparison.json').write_text(payload)
    np.savez_compressed(output/'ensemble-arrays.npz',**arrays)
    catalog={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir() if p.is_file() and not p.name.startswith('._')}
    (output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    return dict(ready=True,summary_written=True,native_seeds=SEEDS,scientific_acceptance=False,performance_cost_acceptance=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['evidence','t7','data','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=run(a.evidence,a.t7,a.data,a.output)
    print(json.dumps(r,indent=2));raise SystemExit(0 if r['ready'] else 2)
