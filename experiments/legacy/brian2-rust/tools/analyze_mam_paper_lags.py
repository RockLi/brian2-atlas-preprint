"""Full Fig. 7 lag matrix/hierarchy under original-code conventions."""
import argparse
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from mam_paper_propagation import lag_matrix,fit_hierarchy,SETTINGS
from analyze_mam_paper_fc import checked,sha
from audit_mam_original_interarea import REV,LABEL


def analyze(a):
    start=time.perf_counter()
    audit_cat=json.loads((a.reference_audit/'catalog.json').read_text())
    audit=json.loads(checked(a.reference_audit,audit_cat,'report.json',2**20).read_text())
    if not all(audit[key] for key in ['published_selected_covariances_reproduced','source_arithmetic_verified','selected_peak_lags_match']):
        raise ValueError('selected published covariance/source verification required')
    if audit['implementation_sha256']['mam_paper_propagation.py']!=sha(Path(__file__).with_name('mam_paper_propagation.py')):
        raise ValueError('propagation implementation changed since reference verification')
    metadata=json.loads((a.matrices/'matrices.json').read_text())
    if metadata['matrix_sha256']!=sha(a.matrices/'matrices.npz') or metadata['matrix_sha256']!='2bf63e2ab2d11a235f2cf95f0eeb1d38e37ffe659ce9439c7dee8b11e161aa55':
        raise ValueError('verified full matrix bundle required for official area/population ordering')
    areas=metadata['area_names'];sources=dict(reference_audit=sha(a.reference_audit/'report.json'),matrices_metadata=sha(a.matrices/'matrices.json'))
    if a.original_rates:
        root=a.original_rates;cat=json.loads((root/'catalog.json').read_text())
        if cat['revision']!=REV or cat['labels']['metastable100']!=LABEL or sha(root/'catalog.json')!=audit['rates_catalog_sha256']:
            raise ValueError('original rate identity mismatch')
        pars=json.loads(checked(root,cat['files'],'metastable100--rate_time_series_full_Parameters.json',2**20).read_text())
        if (pars['t_min'],pars['t_max'],pars['resolution'])!=(500.,100500.,1.) or set(pars['areas'])!=set(areas):
            raise ValueError('original full observation required')
        rates=np.array([np.load(checked(root,cat['files'],'metastable100--rate_time_series_full_'+area+'.npy',2**20),allow_pickle=False) for area in areas])
        if rates.shape!=(32,100000):raise ValueError('complete original 100 s rates required')
        identity=dict(simulator='Published original rates',label=LABEL,revision=REV)
        sources['original_catalog']=sha(root/'catalog.json')
    else:
        if a.fc_audit is None:raise ValueError('previously validated same-series FC evidence required')
        root=a.series;cat=json.loads((root/'catalog.json').read_text())
        report=json.loads(checked(root,cat,'time-series.json',2**20).read_text())
        path=checked(root,cat,'time-series.npz',256*2**20)
        fc_cat=json.loads((a.fc_audit/'catalog.json').read_text());fc=json.loads(checked(a.fc_audit,fc_cat,'fc.json',2**20).read_text())
        if (not fc['validated_rate_input'] or fc['source_sha256']['series_report']!=sha(root/'time-series.json')
                or fc['source_sha256']['series_arrays']!=sha(path) or fc['source_sha256']['matrices_metadata']!=sha(a.matrices/'matrices.json')):
            raise ValueError('FC validation does not bind these complete rate arrays')
        with zipfile.ZipFile(path) as archive:
            if len(archive.infolist())>12 or sum(row.file_size for row in archive.infolist())>512*2**20:
                raise ValueError('rate archive expansion exceeds admission')
        with np.load(a.matrices/'matrices.npz',allow_pickle=False) as data:n=data['neuron_numbers'];indices=data['area_indices']
        with np.load(path,allow_pickle=False) as data:
            counts=data['population_counts'];stored_area_rates=data['area_rates_hz']
        names=['mam_'+name.replace('-','_') for name in metadata['population_names']]
        if len(report['population_names'])!=254 or set(report['population_names'])!=set(names) or set(report['area_names'])!=set(areas):
            raise ValueError('full population/area set differs')
        order=[report['population_names'].index(name) for name in names]
        ordered_counts=counts[order];population_rates=ordered_counts/(n[:,None]*1./1000.)
        rates=np.array([np.average(population_rates[indices==i],axis=0,weights=n[indices==i]) for i in range(32)])
        np.testing.assert_array_equal(rates,stored_area_rates[[report['area_names'].index(area) for area in areas]])
        identity=report['identity'];sources.update(series_report=sha(root/'time-series.json'),series_arrays=sha(path),fc_report=sha(a.fc_audit/'fc.json'))
        del population_rates,ordered_counts,counts,stored_area_rates
    if len(areas)!=32 or len(set(areas))!=32 or not np.isfinite(rates).all() or np.any(rates<0):
        raise ValueError('complete finite nonnegative area rates required')
    lags,diagnostic=lag_matrix(rates,areas)
    retained=[i for i,area in enumerate(areas) if area!='MDP'];retained_areas=[areas[i] for i in retained]
    c=lags[np.ix_(retained,retained)]
    hierarchy=fit_hierarchy(c)
    if not hierarchy['normalization_defined']:raise ValueError('undefined temporal hierarchy')
    order=np.argsort(hierarchy['levels_ms'],kind='stable')
    violations=sum(np.sign(hierarchy['levels_ms'][i]-hierarchy['levels_ms'][j])!=np.sign(c[i,j]) for i in range(len(c)) for j in range(len(c)))/2.
    report=dict(schema='b2-mam-paper-propagation-v1',scientific_equivalence=False,identity=identity,
        observation_seconds=rates.shape[1]/1000.,area_names=areas,excluded_from_hierarchy=['MDP'],
        hierarchy_area_names=retained_areas,levels_ms=hierarchy['levels_ms'].tolist(),normalized_levels=hierarchy['normalized_levels'].tolist(),
        ordered_early_to_late=[retained_areas[i] for i in order],
        hierarchy_residual_norm=hierarchy['residual_norm'],hierarchy_rmse_ms=hierarchy['rmse_ms'],
        source_style_violations_divide_two=violations,
        maximum_reciprocal_lag_residual_ms=float(np.max(np.abs(c+c.T))),
        nonzero_reciprocal_residual_pairs=int(np.count_nonzero(np.triu(np.abs(c+c.T)>1e-12,1))),
        settings=SETTINGS,diagnostic=diagnostic,source_sha256=sources,
        implementation_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('mam_paper_propagation.py')]},
        analysis_seconds=time.perf_counter()-start,
        scope='Available-source temporal-order reanalysis. The selected-reference audit covers four published pair files; this full lag/hierarchy output is newly computed and is not a recovered historical peak matrix. Original-code smoothing, nonzero reciprocal reuse and MDP exclusion are retained. Complete-pair least squares is solved exactly with zero-mean gauge. No causal, BOLD, multi-seed, simulator-equivalence or performance claim.')
    a.output.mkdir(exist_ok=False)
    np.savez_compressed(a.output/'lags.npz',lag_ms=lags,retained_lag_ms=c,levels_ms=hierarchy['levels_ms'],normalized_levels=hierarchy['normalized_levels'])
    (a.output/'lags.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (a.output/'catalog.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in a.output.iterdir()},indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--original-rates',type=Path);group.add_argument('--series',type=Path)
    p.add_argument('--fc-audit',type=Path)
    for name in ['matrices','reference-audit','output']:p.add_argument('--'+name,type=Path,required=True)
    analyze(p.parse_args())
