"""Exploratory NEST variability; observed ranges are not equivalence intervals."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

METRICS = ['mean_rate_hz', 'lvr_eligible_mean', 'lvr_zero_padded_mean',
           'pairwise_corr_mean', 'silent_fraction']
WINDOW_KEYS = ['start_tick', 'end_tick', 'endpoint', 'dt_seconds', 'bin_ticks', 'seconds']


def describe_metric(values, rust_value):
    valid = np.array([v for v in values if v is not None], dtype=float)
    if not np.isfinite(valid).all() or (rust_value is not None and not np.isfinite(rust_value)):
        raise ValueError('nonfinite metric')
    n = len(valid)
    lo, hi = (float(valid.min()), float(valid.max())) if n else (None, None)
    return dict(native_values=values, native_valid_count=n,
                native_mean=float(valid.mean()) if n else None,
                native_sample_std=float(valid.std(ddof=1)) if n >= 2 else None,
                native_min=lo, native_max=hi, rust=rust_value,
                rust_outside_observed_range=(bool(rust_value < lo or rust_value > hi)
                    if n >= 2 and rust_value is not None else None))


def validate_reports(native, rust, protocol):
    if not 2 <= len(native) <= len(protocol['diagnostic_seeds']):
        raise ValueError('requires two or three diagnostic references')
    seeds = [r['simulation']['seed'] for r in native]
    if len(set(seeds)) != len(seeds) or any(s not in protocol['diagnostic_seeds'] for s in seeds):
        raise ValueError('duplicate or unplanned reference seed')
    base = native[0]
    identity = {k: base['simulation'][k] for k in ['ranks','threads','duration_ms','dt_ms','nest_version']}
    if identity['ranks'] != protocol['native_layout']['ranks'] or identity['threads'] != protocol['native_layout']['threads']:
        raise ValueError('reference layout differs from protocol')
    if identity['duration_ms'] != protocol['duration_ms'] or identity['dt_ms'] != protocol['dt_ms']:
        raise ValueError('reference timing differs from protocol')
    for report in native:
        if report['parameters_sha256'] != protocol['parameters_sha256']:
            raise ValueError('reference parameter mismatch')
        if {k: report['simulation'][k] for k in identity} != identity:
            raise ValueError('reference simulator identity mismatch')
    names = [p['name'] for p in base['populations']]
    sizes = [p['neurons'] for p in base['populations']]
    if len(names) != 254 or len(set(names)) != 254 or sum(sizes) != protocol['neurons']:
        raise ValueError('incomplete population universe')
    for report, offset in [(r,0) for r in native] + [(rust,1)]:
        w = report['window']
        if any(w[k] != base['window'][k] for k in WINDOW_KEYS):
            raise ValueError('physical window mismatch')
        if (w['spike_tick_offset'] != offset or w['raw_start_tick']+offset != w['start_tick']
                or w['raw_end_tick']+offset != w['end_tick']):
            raise ValueError('raw-to-physical mapping mismatch')
        if report['sampling'] != base['sampling'] or report['sampling']['seed'] != protocol['analysis_sampling_seed']:
            raise ValueError('sampling mismatch')
        if ([p['name'] for p in report['populations']] != names
                or [p['neurons'] for p in report['populations']] != sizes
                or report['neurons'] != protocol['neurons']
                or [a['area'] for a in report['areas']] != [a['area'] for a in base['areas']]):
            raise ValueError('population/area mismatch')
    if [base['window']['start_tick'],base['window']['end_tick']] != protocol['physical_observation_ticks']:
        raise ValueError('window differs from protocol')
    w = base['window']
    if (w['endpoint'] != protocol['endpoint'] or w['dt_seconds']*1000 != protocol['dt_ms']
            or w['bin_ticks']*protocol['dt_ms'] != protocol['analysis_bin_ms']
            or w['seconds'] != (protocol['duration_ms']-protocol['warmup_ms'])/1000):
        raise ValueError('measurement units differ from protocol')
    return seeds


def compare(native_dirs, rust_dir, protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    native = [json.loads((d/'activity.json').read_text()) for d in native_dirs]
    rust = json.loads((rust_dir/'activity.json').read_text())
    seeds = validate_reports(native, rust, protocol)
    order = np.argsort(seeds)
    native_dirs = [native_dirs[i] for i in order]
    native = [native[i] for i in order]; seeds = sorted(seeds)
    rows = []
    for i, p in enumerate(rust['populations']):
        rows.append(dict(name=p['name'],neurons=p['neurons'],
            metrics={key:describe_metric([r['populations'][i][key] for r in native],p[key]) for key in METRICS},
            native_lvr_eligible_cells=[r['populations'][i]['lvr_eligible_cells'] for r in native],
            rust_lvr_eligible_cells=p['lvr_eligible_cells'],
            native_corr_sample_size=[r['populations'][i]['corr_sample_size'] for r in native],
            rust_corr_sample_size=p['corr_sample_size']))
    areas = [dict(area=a['area'],metrics={key:describe_metric([r['areas'][i][key] for r in native],a[key])
        for key in ['mean_rate_hz','first_half_rate_hz','second_half_rate_hz']}) for i,a in enumerate(rust['areas'])]
    sources = [d/'activity.json' for d in native_dirs+[rust_dir]] + [protocol_path]
    result = dict(schema='b2-mam-reference-variability-v1',scientific_equivalence=False,
        scope='Post-baseline exploratory comparison. Independent realizations; observed range is not a confidence, prediction or equivalence interval. Outside counts are descriptive, not failures. No tolerance fitting, paired-cell or pointwise burst-timing claim.',
        native_seeds=seeds,missing_diagnostic_seeds=[s for s in protocol['diagnostic_seeds'] if s not in seeds],
        physical_window={k:native[0]['window'][k] for k in WINDOW_KEYS},sampling=native[0]['sampling'],
        global_rate=describe_metric([r['mean_rate_hz'] for r in native],rust['mean_rate_hz']),
        populations=rows,areas=areas,
        outside_observed_range_counts={key:sum(p['metrics'][key]['rust_outside_observed_range'] is True for p in rows) for key in METRICS},
        source_sha256={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sources},
        incomplete_metrics='Paper-duration state occupancies, propagation, spectral, synaptic-input and BOLD/FC metrics remain open; two or three references cannot establish reliable equivalence margins.')
    output.mkdir(exist_ok=False)
    (output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot(result,output)
    catalog={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in output.iterdir() if p.is_file()}
    (output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    print(json.dumps(dict(native_seeds=seeds,global_rate=result['global_rate'],outside_counts=result['outside_observed_range_counts'],selected_populations=[p for p in rows if p['name'] in ['mam_PITd_5E','mam_MSTd_5E','mam_MIP_5E']]),indent=2))


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for ax,key,label in zip(axes,['mean_rate_hz','lvr_eligible_mean','pairwise_corr_mean'],['Mean rate (Hz/neuron)','Eligible-cell LvR','Mean pairwise correlation'],strict=True):
        shown=[]
        for j,seed in enumerate(result['native_seeds']):
            pairs=[(p['metrics'][key]['rust'],p['metrics'][key]['native_values'][j]) for p in result['populations']]
            pairs=np.array([(x,y) for x,y in pairs if x is not None and y is not None])
            if key=='mean_rate_hz':pairs=np.maximum(pairs,.001)
            if len(pairs):ax.scatter(pairs[:,0],pairs[:,1],s=12,alpha=.6,label=f'NEST {seed}');shown.extend(pairs.ravel())
        if shown:
            lo,hi=min(shown),max(shown);ax.plot([lo,hi],[lo,hi],color='gray',lw=1,ls='--')
        if key=='mean_rate_hz':ax.set_xscale('log');ax.set_yscale('log')
        ax.set(title=label,xlabel='Rust baseline (seed 1729)',ylabel='Independent native NEST');ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('All 254 populations: exploratory reference variability\nRates below 0.001 Hz shown at floor; no equivalence test')
    fig.savefig(output/'population-metrics.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(15,6),layout='constrained');x=np.arange(len(result['areas']))
    m=[a['metrics']['mean_rate_hz'] for a in result['areas']]
    means=np.array([v['native_mean'] for v in m]);lo=np.array([v['native_min'] for v in m]);hi=np.array([v['native_max'] for v in m])
    ax.errorbar(x-.1,means,yerr=np.array([means-lo,hi-means]),fmt='o',capsize=3,label='NEST mean and observed range',ms=4)
    ax.scatter(x+.1,[v['rust'] for v in m],marker='x',color='#555555',label='Rust baseline')
    ax.set(xticks=x,xticklabels=[a['area'] for a in result['areas']],ylabel='Hz / neuron',yscale='log',title=f"Area means across {len(result['native_seeds'])} NEST realizations\nObserved ranges are not uncertainty or equivalence intervals")
    ax.tick_params(axis='x',labelrotation=60);ax.legend();ax.grid(alpha=.2,axis='y');fig.savefig(output/'area-mean-ranges.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-dir',type=Path,action='append',required=True)
    for name in ['rust-dir','protocol','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();compare(a.native_dir,a.rust_dir,a.protocol,a.output)
