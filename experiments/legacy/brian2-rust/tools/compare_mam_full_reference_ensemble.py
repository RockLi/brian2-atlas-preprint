"""Frozen three-reference 100 s descriptive cohort; never an equivalence test.

This consumes completed analyses only. Missing seed1731 completion or compact
arrays returns not-ready without creating an output or launching any work.
"""
import argparse
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np

from mam_launch_native_full_reference import CAMPAIGN, PROTOCOL_SHA, label_for, protocol_gate
from mam_complete_native_reference import REPORTS, validate_analysis
from mam_launch_native_analysis import prerequisites
from mam_primary_analysis_pipeline import check, read, sha
from mam_paper_propagation import fit_hierarchy
from mam_launch_native_primary import rust_gate

SEEDS = [1729, 1730, 1731]
PARAMETERS = 'ee1a642c94b9d6e808627c54f78162e7f5d87f4140ba56496c5dfee3ed900d3e'
RUST_MODEL = '9526a75e00e7cd4c3457691610fce4072c2014c6e6c6b53ea8a62f588dcac6bd'
POP_METRICS = ['activity_rate_hz', 'strict_unrounded_rate_hz', 'lvr_all_cells',
               'lvr_eligible_only', 'sampled_lvr_all_cells', 'sampled_lvr_eligible_only',
               'paper_correlation', 'sampled_correlation', 'silent_fraction']


def describe(values, rust=None):
    check(len(values) == 3, 'exactly three planned native values required')
    valid = np.asarray([v for v in values if v is not None], dtype=float)
    check(np.isfinite(valid).all() and (rust is None or np.isfinite(rust)), 'nonfinite observable')
    n = len(valid)
    return dict(native_values=values, native_valid_count=n,
                native_mean=float(valid.mean()) if n else None,
                native_sample_std=float(valid.std(ddof=1)) if n > 1 else None,
                native_min=float(valid.min()) if n else None,
                native_max=float(valid.max()) if n else None, rust=rust)


def readiness(evidence, t7):
    check(sha(evidence/CAMPAIGN/'protocol.json') == PROTOCOL_SHA, 'frozen protocol changed')
    missing = [s for s in SEEDS[1:] if not (evidence/CAMPAIGN/f'seed{s}/completion.json').exists()]
    if missing:
        return dict(ready=False, missing_completed_seeds=missing,
                    reason='All three full native references require terminal, raw and six-stage audits')
    protocol = protocol_gate(evidence, 1731)
    check(protocol is not None, 'full-reference protocol gate closed')
    rust = rust_gate(evidence,t7/'artifacts/primary-terminal-resources-v1')
    check(rust is not None, 'completed Rust resource/raw prerequisite missing')
    for folder in ['primary-postrun','primary-native-postrun/run']:
        root=evidence/folder; report=read(root/'report.json')
        check(report['analysis_complete'] is True and report['scientific_acceptance'] is False,
              'primary analysis is incomplete')
        if folder == 'primary-postrun':
            check(report['output_audit_sha256'] == rust['output_report_sha256'], 'Rust raw-analysis linkage differs')
        for stage in ['activity','cell','correlation','series']:
            check(sha(root/stage/'catalog.json') == report['catalogs'][stage], 'primary stage pin differs')
    for seed in SEEDS[1:]:
        root = evidence/CAMPAIGN/f'seed{seed}'
        completion = read(root/'completion.json')
        check(completion['seed'] == seed and completion['label'] == label_for(seed)
              and completion['protocol_sha256'] == PROTOCOL_SHA, 'completion identity differs')
        check(all(completion[k] is True for k in ['terminal_resource_audit_passed',
              'raw_output_audit_passed', 'analysis_complete']), 'incomplete diagnostic seed')
        check(completion['scientific_acceptance'] is False
              and completion['performance_cost_acceptance'] is False, 'completion scope changed')
        check(set(completion['reports']) == {'resources','raw','analysis'}, 'completion gate coverage differs')
        for row in completion['reports'].values():
            p = root/row['path']
            check(p.resolve().is_relative_to(root.resolve()) and sha(p) == row['sha256'],
                  'completion source changed')
        check((t7/'artifacts'/CAMPAIGN/f'seed{seed}/completion.json').read_bytes()
              == (root/'completion.json').read_bytes(), 'completion backup differs')
        gate = prerequisites(t7, reference_seed=seed)
        check(gate is not None, 'resource and collection audit gate closed')
        source = t7/'artifacts'/f'native-reference{seed}-postrun-v1'
        required = validate_analysis(source, gate, seed)
        check(all((root/'analysis'/name).read_bytes() == (source/name).read_bytes()
                  for name in required), 'terminal scientific evidence differs')
        needed = ['series/time-series.npz', 'fc/fc.npz', 'lags/lags.npz']
        absent = [n for n in needed if not (source/n).is_file()]
        if absent:
            return dict(ready=False, missing_compact_arrays={str(seed):absent},
                        reason='Verified compact arrays must be collected before cohort aggregation')
    return dict(ready=True)


class Inputs:
    def __init__(self): self.pins = {}

    def checked(self, root, name):
        catalog = root/'catalog.json'
        check(catalog.is_file() and catalog.stat().st_size < 2**20, 'missing/oversized catalog')
        row = read(catalog)[name]; p = root/name
        check(p.is_file() and not p.is_symlink() and p.stat().st_size == row['bytes']
              and p.stat().st_size <= 256*2**20 and sha(p) == row['sha256'], 'artifact hash/size differs')
        self.pins[str(catalog)] = sha(catalog); self.pins[str(p)] = row['sha256']
        return p

    def report(self, root, name):
        p = self.checked(root, name)
        check(p.stat().st_size < 2**20, 'oversized scalar report')
        return read(p)

    def arrays(self, root, name, keys):
        p = self.checked(root, name)
        with zipfile.ZipFile(p) as z:
            check(len(z.infolist()) <= 12 and sum(x.file_size for x in z.infolist()) <= 512*2**20,
                  'array expansion exceeds budget')
        with np.load(p, allow_pickle=False) as f: return {k:f[k] for k in keys}


def validate_identity(reports, seed, simulator):
    activity = reports['activity']
    if simulator == 'NEST':
        ident = activity['simulation']
        check(activity['parameters_sha256'] == PARAMETERS, 'native parameters differ')
        for k, value in dict(seed=seed, ranks=48, threads=4, duration_ms=100500,
                             dt_ms=.1, nest_version='3.10.0').items():
            check(ident[k] == value, 'native identity differs: '+k)
    else:
        check(simulator == 'Rust' and seed == 1729 and activity['model_sha256'] == RUST_MODEL,
              'unexpected Rust baseline')
    for stage in ['cell', 'correlation', 'series', 'fc', 'lags']:
        ident = reports[stage]['identity']
        if stage == 'cell' and simulator == 'Rust':
            # This retained helper emitted only the model hash. The exact
            # immutable full model, raw audit and catalog bind it to seed1729;
            # do not add a fabricated seed to the historical report.
            check(ident == dict(simulator='Rust',model_sha256=RUST_MODEL), 'legacy Rust cell identity differs')
        else:
            check(ident.get('seed') == seed and ident.get('simulator') == simulator, 'cross-seed/stage report')
        if 'parameters_sha256' in ident: check(ident['parameters_sha256'] == PARAMETERS, 'parameters differ')
        if 'model_sha256' in ident: check(ident['model_sha256'] == RUST_MODEL, 'Rust model differs')
    w = activity['window']; offset = 0 if simulator == 'NEST' else 1
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


def validate_lag_missingness(full, area_names, excluded):
    check(full.shape == (32,32) and len(area_names) == len(set(area_names)) == 32
          and excluded == ['MDP'] and area_names.count('MDP') == 1, 'lag exclusion convention differs')
    mdp = area_names.index('MDP')
    missing = np.zeros((32,32),dtype=bool);missing[mdp,:]=True;missing[:,mdp]=True
    check(np.array_equal(np.isnan(full),missing) and not np.isinf(full).any(),
          'only the excluded MDP row/column may be undefined; do not impute')
    return [i for i in range(32) if i != mdp]


def load_cohort(evidence, t7, *, include_seed1731=True):
    # Read-only preparation may validate the three currently completed cases.
    # run() always uses the full default; summarize() refuses a partial cohort.
    inputs = Inputs(); cohort = {}; artifact = t7/'artifacts'
    compact = artifact/'native-reference-baseline-patterns-v1/data'
    sk = ['area_rates_hz','frequency_hz','power_hz2_per_hz']; fk = ['functional_connectivity','reference_fc']
    lk = ['lag_ms','retained_lag_ms','levels_ms','normalized_levels']
    check(type(include_seed1731) is bool, 'invalid preparation selector')
    selected = SEEDS if include_seed1731 else SEEDS[:2]
    for case, seed in [('rust1729',1729)] + [(f'native{s}',s) for s in selected]:
        old = seed == 1729
        root = evidence/('primary-postrun' if case == 'rust1729' else 'primary-native-postrun/run') if old else evidence/CAMPAIGN/f'seed{seed}/analysis'
        reports = {stage:inputs.report(root/stage, REPORTS[stage]) for stage in ['activity','cell','correlation','series']}
        if old:
            inter = artifact/'mam-primary-interarea-v1'
            stem = 'rust' if case == 'rust1729' else 'native'
            reports.update({stage:inputs.report(inter/(stem+'-'+stage), REPORTS[stage]) for stage in ['fc','lags']})
            for stage in ['series','fc']:
                check(inputs.report(compact,case+'-'+stage+'.json') == reports[stage], 'compact baseline identity differs')
            arrays = inputs.arrays(compact,case+'.npz',sk+fk)
            arrays.update(inputs.arrays(inter/(stem+'-lags'),'lags.npz',lk))
        else:
            source = artifact/f'native-reference{seed}-postrun-v1'
            reports.update({stage:inputs.report(source/stage, REPORTS[stage]) for stage in ['fc','lags']})
            arrays = {}
            for stage, name, keys in [('series','time-series.npz',sk),('fc','fc.npz',fk),('lags','lags.npz',lk)]:
                arrays.update(inputs.arrays(source/stage,name,keys))
        validate_identity(reports,seed,'Rust' if case == 'rust1729' else 'NEST')
        cohort[case] = dict(reports=reports, arrays=arrays)
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
    return cohort, inputs


def summarize(cohort, original):
    check(list(cohort) == ['rust1729','native1729','native1730','native1731'], 'complete ordered cohort required')
    native = [cohort[f'native{s}'] for s in SEEDS]; rust = cohort['rust1729']
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
    for name in values['rust1729']:
        populations.append(dict(name=name,metrics={k:describe([values[f'native{s}'][name][k] for s in SEEDS],values['rust1729'][name][k]) for k in POP_METRICS},
            eligibility={case:counts[case][name] for case in cohort},published={k:norms[name][k] for k in ['rate_hz','lvr','correlation']}))
    areas=[]
    for i,name in enumerate(native[0]['reports']['series']['area_names']):
        summaries={}
        for case,entry in cohort.items():
            rate=entry['arrays']['area_rates_hz'][i];power=entry['arrays']['power_hz2_per_hz'][i];frequency=entry['arrays']['frequency_hz']
            summaries[case]=dict(mean_rate_hz=float(rate.mean()),first_half_mean_hz=float(rate[:50000].mean()),second_half_mean_hz=float(rate[50000:].mean()),
                integrated_psd=float(power.sum()*.9765625),peak_nonzero_frequency_hz=float(frequency[1+np.argmax(power[1:])]))
        areas.append(dict(area=name,metrics={k:describe([summaries[f'native{s}'][k] for s in SEEDS],summaries['rust1729'][k]) for k in summaries['rust1729']}))
    small={};summary_arrays={}
    for name,key in [('fc','functional_connectivity'),('lags','retained_lag_ms'),('hierarchy','levels_ms'),('psd','power_hz2_per_hz')]:
        x=np.stack([r['arrays'][key] for r in native])
        summary_arrays.update({name+'_native_values':x,name+'_native_mean':x.mean(axis=0),name+'_native_sample_std':x.std(axis=0,ddof=1),
            name+'_native_min':x.min(axis=0),name+'_native_max':x.max(axis=0),name+'_rust':rust['arrays'][key]})
        small[name]=dict(shape=list(x.shape[1:]),native_realizations=3,dependent_entries=True)
    summary_arrays['frequency_hz']=native[0]['arrays']['frequency_hz'];summary_arrays['published_fc']=native[0]['arrays']['reference_fc']
    return dict(native_seeds=SEEDS,rust_seed=1729,scientific_acceptance=False,performance_cost_acceptance=False,
        inference='Descriptive sample means, ddof=1 standard deviations and observed ranges; not confidence, prediction or equivalence intervals. No outside-range failure flag or fitted acceptance margin.',
        global_activity_rate_hz=describe([r['reports']['activity']['mean_rate_hz'] for r in native],rust['reports']['activity']['mean_rate_hz']),
        activity_sampling=native[0]['reports']['activity']['sampling'],cell_window=native[0]['reports']['cell']['window'],
        correlation_convention={k:native[0]['reports']['correlation'][k] for k in ['observation_ms','endpoint','bin_ms','selection','calculation']},
        populations=populations,areas=areas,matrices_and_spectra=small,series_area_names=native[0]['reports']['series']['area_names'],
        fc_area_names=native[0]['reports']['fc']['area_names'],hierarchy_area_names=native[0]['reports']['lags']['hierarchy_area_names'],
        excluded_from_lag_hierarchy=['MDP'],excluded_lag_values='The source 32x32 lag matrices retain NaN for the MDP row/column; summaries use only the unchanged finite 31x31 matrices.',
        conventions='Historical LvR denominator remains unresolved; all-cell and eligible-only views are separate. Unavailable correlations are null, never zero. Native realizations are independent; cells, pairs, time bins and Welch segments are not seed replicates. Lowest nonzero PSD grid bin is not proof of a 1 Hz oscillation. Full traces remain in pinned input artifacts.'),summary_arrays


def run(evidence,t7,output):
    state=readiness(evidence,t7)
    if not state['ready']:return dict(**state,summary_written=False)
    check(not output.exists(),'cohort summary already exists')
    check(Path('/Volumes/T7').is_mount() and output.resolve().is_relative_to((t7/'artifacts').resolve())
          and t7.resolve().is_relative_to(Path('/Volumes/T7').resolve()), 'mounted T7 output required')
    disk=os.statvfs(t7)
    check(disk.f_bavail*disk.f_frsize >= 128*2**30+64*2**20, 'T7 reserve')
    started=time.monotonic();cohort,inputs=load_cohort(evidence,t7)
    original=evidence/'mam-original-statistics/original-statistics-v1.json'
    check(sha(original)=='d37e1be7a896e19e60004bb93ab73ca19614077436a894b92b00c1ccff86b15b','published scalar source changed')
    result,arrays=summarize(cohort,read(original));inputs.pins[str(original)]=sha(original)
    for seed in SEEDS[1:]:
        p=evidence/CAMPAIGN/f'seed{seed}/completion.json';inputs.pins[str(p)]=sha(p)
    result.update(schema='b2-mam-full-reference-ensemble-description-v1',protocol_sha256=PROTOCOL_SHA,source_sha256=inputs.pins,
                  implementation_sha256=sha(Path(__file__)),analysis_seconds=time.monotonic()-started)
    payload=json.dumps(result,indent=2,allow_nan=False)+'\n'
    output.mkdir();(output/'comparison.json').write_text(payload)
    np.savez_compressed(output/'ensemble-arrays.npz',**arrays)
    catalog={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir() if p.is_file() and not p.name.startswith('._')}
    (output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    return dict(ready=True,summary_written=True,native_seeds=SEEDS,scientific_acceptance=False,performance_cost_acceptance=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['evidence','t7','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.evidence,a.t7,a.output),indent=2))
