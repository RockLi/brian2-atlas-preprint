"""Inspect original processed arrays and test explicit modern PSD reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from mam_paper_spectrum import spectrum, SETTINGS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def series_contract(array, metadata):
    if array.ndim != 1 or array.dtype.kind != 'f' or not 1 <= len(array) <= 100000 or not np.isfinite(array).all():
        raise ValueError('bounded finite 1D floating array required')
    span = metadata['t_max']-metadata['t_min']
    if span <= 0 or metadata['resolution'] <= 0:
        raise ValueError('invalid declared interval')
    implied = span/len(array)
    return dict(samples=len(array), declared_resolution_ms=metadata['resolution'],
        implied_resolution_ms_from_span_and_length=implied,
        resolution_metadata_consistent=implied == metadata['resolution'],
        declared_interval_ms=[metadata['t_min'], metadata['t_max']],
        mean=float(array.mean()), minimum=float(array.min()), maximum=float(array.max()))


def psd_comparison(rate, original_f, original_p):
    f, p = spectrum(rate)
    if original_f.shape != f.shape or original_p.shape != p.shape or not np.isfinite(original_p).all() or np.any(original_p < 0):
        raise ValueError('invalid published PSD shape or values')
    delta = np.abs(p-original_p)
    # Pre-existing spectral arithmetic fixture tolerance, not fitted to originals.
    close = np.allclose(p, original_p, rtol=5e-13, atol=1e-20)
    return dict(frequency_grid_exact=bool(np.array_equal(f, original_f)),
        reconstruction_within_arithmetic_tolerance=bool(close and np.array_equal(f, original_f)),
        maximum_absolute_error=float(delta.max()),
        maximum_relative_error=float((delta/np.maximum(np.abs(original_p),1e-300)).max()),
        arithmetic_rtol=5e-13, arithmetic_atol=1e-20,
        published_peak_nonzero_hz=float(original_f[1:][np.argmax(original_p[1:])]),
        reconstructed_peak_nonzero_hz=float(f[1:][np.argmax(p[1:])]))


def audit(source, output):
    catalog = json.loads((source/'catalog.json').read_text())
    if catalog['revision'] != '11fa93a4427a0e4e4de307ca7a5455e80265053a' or catalog['failures']:
        raise ValueError('requires complete pinned retrieval')
    if len(catalog['files']) != 99 or sum(x['bytes'] for x in catalog['files'].values()) > 64*2**20:
        raise ValueError('source budget or expected file count differs')
    for name, entry in catalog['files'].items():
        p = source/name
        if Path(name).name != name or entry['bytes'] > 2*2**20 or p.stat().st_size != entry['bytes'] or sha(p) != entry['sha256']:
            raise ValueError('source identity/budget failure')
    def load(case, suffix):
        name=case+'--'+suffix
        if name not in catalog['files']:raise ValueError('uncatalogued input')
        p=source/name
        return json.loads(p.read_text()) if p.suffix=='.json' else np.load(p,allow_pickle=False)
    cases={}
    for case in catalog['labels']:
        metadata=load(case,'rate_time_series_subsample_Parameters.json')
        rate=load(case,'rate_time_series_subsample_V1.npy')
        f=load(case,'power_spectrum_subsample_freq.npy');p=load(case,'power_spectrum_subsample_V1.npy')
        hist=load(case,'rate_histogram_V1.npy');bins=load(case,'rate_histogram_bins.npy')
        if hist.ndim!=1 or hist.shape!=bins.shape or not np.isfinite(hist).all() or not np.isfinite(bins).all() or not np.all(np.diff(bins)>0):
            raise ValueError('invalid original histogram vectors')
        cases[case]=dict(label=catalog['labels'][case],subsample=series_contract(rate,metadata),
            subsample_metadata=metadata,psd=psd_comparison(rate,f,p),
            histogram=dict(vector_length=len(hist),bin_coordinate_min=float(bins.min()),bin_coordinate_max=float(bins.max()),published_value_sum=float(hist.sum()),scope='Original bin-coordinate/value vectors; no assumption that these are edges or unit-normalized density.'))
    full={}
    for case in ['metastable100','ground10']:
        metadata=load(case,'rate_time_series_full_Parameters.json')
        if metadata['resolution']!=1. or len(metadata['areas'])!=32 or len(set(metadata['areas']))!=32:
            raise ValueError('unexpected full-series metadata')
        rows={}
        for area in metadata['areas']:
            rate=load(case,'rate_time_series_full_'+area+'.npy')
            contract=series_contract(rate,metadata)
            if not contract['resolution_metadata_consistent'] or len(rate)%10000:
                raise ValueError('full area grid does not match 1 ms contract')
            contract['consecutive_10s_means_hz']=[float(x.mean()) for x in rate.reshape(-1,10000)]
            rows[area]=contract
        full[case]=dict(metadata=metadata,areas=rows)
    auto={}
    for area in ['V1','V2','FEF']:
        suffix='rate_time_series_auto_kernel_'+area+'.npy'
        auto[area]=dict(arrays_bitwise_identical=catalog['files']['metastable100--'+suffix]['sha256']==catalog['files']['metastable10--'+suffix]['sha256'],cases={})
        for case in ['metastable100','metastable10']:
            auto[area]['cases'][case]=series_contract(load(case,suffix),load(case,'rate_time_series_auto_kernel_Parameters.json'))
    result=dict(schema=1,source_catalog_sha256=sha(source/'catalog.json'),source_revision=catalog['revision'],
        original_source_bytes=catalog['total_bytes'],cases=cases,full_area_series=full,
        auto_kernel_duplicate_audit=auto,reconstruction_settings=SETTINGS,
        reconstruction_rate_assumption='Explicit fs=1000 Hz from pinned modern official PSD wrapper; metadata conflicts are reported, not silently repaired.',
        unresolved_psd_cases=[k for k,v in cases.items() if not v['psd']['reconstruction_within_arithmetic_tolerance']],
        subsample_metadata_conflicts=[k for k,v in cases.items() if not v['subsample']['resolution_metadata_consistent']],
        scientific_acceptance=False,scope='Published simulation references and arithmetic reconstruction audit. Does not reconstruct historical selected cell IDs, establish independent realizations for duplicate arrays, reproduce experimental Chu data, or compare current simulators.')
    output.mkdir(exist_ok=False)
    (output/'original-series.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def plot_reference(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((output/'original-series.json').read_text())
    if report['source_catalog_sha256']!=sha(source/'catalog.json'):
        raise ValueError('reference source changed since audit')
    catalog=json.loads((source/'catalog.json').read_text())
    def load(name):
        if sha(source/name)!=catalog['files'][name]['sha256']:raise ValueError('plot source changed')
        return np.load(source/name,allow_pickle=False)
    names=['original-area-traces.png','original-v1-spectra.png']
    if any((output/n).exists() for n in names):raise FileExistsError('reference plot already exists')
    fig,axes=plt.subplots(3,2,figsize=(12,8),sharex=True,layout='constrained')
    for col,(case,title,color) in enumerate([('ground10',r'$\chi=1$: 10 s observation','#2878b5'),('metastable100',r'$\chi=1.9$: first 10 s of 100 s observation','#9854a3')]):
        axes[0,col].set_title(title)
        for row,area in enumerate(['V1','V2','FEF']):
            rate=load(case+'--rate_time_series_full_'+area+'.npy')[:10000]
            axes[row,col].plot(.5+.005+np.arange(1000)*.01,rate.reshape(-1,10).mean(axis=1),lw=.85,color=color)
            axes[row,col].set_ylabel(area+' rate (spikes/s)')
            axes[row,col].set_xlim(.5,10.5);axes[row,col].set_ylim(bottom=0)
            axes[row,col].grid(alpha=.2)
        axes[-1,col].set_xlabel('Biological time (s)')
    fig.suptitle('Original MAM area activity | 10 ms display averages of published 1 ms arrays')
    fig.savefig(output/names[0],dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    cases=[('ground10',r'$\chi=1$, 10 s','#2878b5'),('metastable10',r'$\chi=1.9$, 10 s','#ec9139'),('metastable100',r'$\chi=1.9$, 100 s','#9854a3'),('high10',r'$\chi=2.5$, 10 s','#32936f')]
    for case,label,color in cases:
        f=load(case+'--power_spectrum_subsample_freq.npy');p=load(case+'--power_spectrum_subsample_V1.npy')
        for ax,upper in zip(axes,[60,500],strict=True):
            selected=(f>0)&(f<=upper)
            ax.semilogy(f[selected],p[selected],label=label,color=color,lw=1.1)
            ax.set_xlim(0,upper);ax.set_xlabel('Frequency (Hz)');ax.set_ylabel('Published power');ax.grid(alpha=.2)
    axes[0].set_title('Detail: 0–60 Hz');axes[1].set_title('Full band: 0–500 Hz')
    axes[1].legend(fontsize=9)
    fig.suptitle('Original MAM V1 subsampled spectra | published simulation references')
    fig.savefig(output/names[1],dpi=160);plt.close(fig)
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [output/'original-series.json',*(output/n for n in names)]}
    (output/'catalog.json').write_text(json.dumps(files,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot',action='store_true')
    a=p.parse_args();r=audit(a.source,a.output)
    if a.plot:plot_reference(a.source,a.output)
    print(json.dumps(dict(unresolved_psd_cases=r['unresolved_psd_cases'],subsample_metadata_conflicts=r['subsample_metadata_conflicts'],cases={k:v['psd'] for k,v in r['cases'].items()}),indent=2))
