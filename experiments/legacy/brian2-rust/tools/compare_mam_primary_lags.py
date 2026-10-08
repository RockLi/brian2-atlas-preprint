"""Matched 100 s propagation comparison for the admitted primary seed.

Read only already analyzed lag artifacts; never recompute the ten short
original windows or claim scientific equivalence from one realization.
"""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from analyze_mam_paper_fc import checked, sha, RUST_MODELS
from audit_mam_original_interarea import LABEL, REV
from compare_mam_paper_lags import lag_stats
from export_mam_interarea_matrices import PARAMETERS_SHA
from mam_paper_propagation import fit_hierarchy, SETTINGS


def require(ok, message):
    if not ok: raise ValueError(message)


def read_artifact(root, case):
    require((root/'catalog.json').stat().st_size <= 2**20, 'oversized lag catalog')
    catalog=json.loads((root/'catalog.json').read_text())
    report_path=checked(root,catalog,'lags.json',2**20)
    array_path=checked(root,catalog,'lags.npz',2**20)
    report=json.loads(report_path.read_text())
    require(report['schema']=='b2-mam-paper-propagation-v1'
            and report['observation_seconds']==100.
            and report['excluded_from_hierarchy']==['MDP'], 'complete 100 s propagation artifact required')
    require(report['settings']==SETTINGS and report['implementation_sha256']['mam_paper_propagation.py']
            ==sha(Path(__file__).with_name('mam_paper_propagation.py')), 'propagation convention/source differs')
    identity=report['identity']
    if case=='original':
        require(identity==dict(simulator='Published original rates',label=LABEL,revision=REV), 'published reference identity differs')
    elif case=='rust':
        require(identity==dict(simulator='Rust',model_sha256=RUST_MODELS[100000],seed=1729), 'Rust primary identity differs')
    elif case=='native':
        require(identity==dict(simulator='NEST',condition='metastable',parameters_sha256=PARAMETERS_SHA,
                seed=1729,ranks=48,threads=4,duration_ms=100500.,dt_ms=.1,nest_version='3.10.0'),
                'NEST primary identity differs')
    else: raise ValueError('unknown propagation comparison case')
    expected={'lag_ms.npy','retained_lag_ms.npy','levels_ms.npy','normalized_levels.npy'}
    with zipfile.ZipFile(array_path) as archive:
        entries=archive.infolist()
        require(len(entries)==len(expected) and {r.filename for r in entries}==expected
                and sum(r.file_size for r in entries)<2**20, 'lag array expansion/coverage exceeds admission')
    with np.load(array_path,allow_pickle=False) as data:
        full,retained,levels,normalized=[data[k] for k in ['lag_ms','retained_lag_ms','levels_ms','normalized_levels']]
    areas=report['area_names'];names=report['hierarchy_area_names']
    require(len(areas)==32 and len(set(areas))==32 and areas.count('MDP')==1
            and names==[a for a in areas if a!='MDP'], 'brain area ordering/coverage differs')
    require(full.shape==(32,32) and retained.shape==(31,31) and levels.shape==normalized.shape==(31,),
            'complete lag/hierarchy arrays required')
    keep=[i for i,a in enumerate(areas) if a!='MDP']
    np.testing.assert_array_equal(retained,full[np.ix_(keep,keep)])
    np.testing.assert_array_equal(levels,np.asarray(report['levels_ms']))
    np.testing.assert_array_equal(normalized,np.asarray(report['normalized_levels']))
    require(np.isfinite(retained).all() and np.isfinite(levels).all() and np.isfinite(normalized).all(),
            'missing lag/hierarchy entries may not be imputed')
    np.testing.assert_array_equal(np.diag(retained),np.zeros(31))
    fitted=fit_hierarchy(retained)
    require(fitted['normalization_defined'], 'undefined hierarchy')
    # Numerical consistency of the existing least-squares summary, not a
    # simulator-equivalence margin. Same tolerance as the retained audit core.
    np.testing.assert_allclose(levels,fitted['levels_ms'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(normalized,fitted['normalized_levels'],rtol=1e-12,atol=1e-12)
    require(report['ordered_early_to_late']==[names[i] for i in np.argsort(levels,kind='stable')],
            'reported temporal order differs from levels')
    return report,retained,levels,normalized,dict(report=sha(report_path),arrays=sha(array_path),catalog=sha(root/'catalog.json'))


def compare(original,rust,native,output):
    require(not output.exists(),'comparison output already exists')
    rows=[read_artifact(root,case) for root,case in [(original,'original'),(rust,'rust'),(native,'native')]]
    reports,curves,levels,normalized,hashes=map(list,zip(*rows))
    for report in reports[1:]:
        for key in ['area_names','hierarchy_area_names','settings']:
            require(report[key]==reports[0][key],'incompatible '+key)
        for key in ['reference_audit','matrices_metadata']:
            require(report['source_sha256'][key]==reports[0]['source_sha256'][key], 'different reference evidence: '+key)
    pairs={'rust_vs_original':(1,0),'native_vs_original':(2,0),'rust_vs_native':(1,2)}
    metrics={name:lag_stats(curves[i],levels[i],curves[j],levels[j]) for name,(i,j) in pairs.items()}
    result=dict(schema='b2-mam-primary-propagation-comparison-v1',scientific_equivalence=False,
        equal_observation_duration=True,observation_seconds=100.,seed=1729,
        hierarchy_area_names=reports[0]['hierarchy_area_names'],metrics=metrics,
        identities={case:report['identity'] for case,report in zip(['original','rust','native'],reports,strict=True)},
        source_sha256=dict(zip(['original','rust','native'],hashes,strict=True)),
        original_short_windows_recomputed=False,acceptance_thresholds_fitted=False,
        implementation_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('compare_mam_paper_lags.py'),
            Path(__file__).with_name('mam_paper_propagation.py')]},
        scope='Descriptive matched-duration comparison: one admitted Rust/NEST realization each and the published '
              'original rate reanalysis, all 100 s after the transient. The 930 directed matrix entries are dependent. '
              'Original full hierarchy is an available-source reanalysis; only four stored covariance pairs were directly '
              'audited. No recomputed short-window ranges, independent-seed confidence intervals, causal conclusion, '
              'scientific equivalence or performance/cost acceptance.')
    # Check serializability/non-finite metrics before creating any output.
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    output.mkdir()
    (output/'comparison.json').write_text(text)
    np.savez_compressed(output/'comparison-arrays.npz',
        **{case+'_'+kind:values[i] for i,case in enumerate(['original','rust','native'])
           for kind,values in [('lags',curves),('levels_ms',levels),('normalized_levels',normalized)]})
    plot(reports,curves,normalized,metrics,output)
    (output/'catalog.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in output.iterdir()},indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


def plot(reports,curves,normalized,metrics,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Original reanalysis','Rust','NEST'];colors=['black','#d95f02','#1f77b4']
    areas=reports[0]['hierarchy_area_names']
    fig=plt.figure(figsize=(16,12),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[1,1.2])
    axes=[fig.add_subplot(grid[0,i]) for i in range(3)]
    for ax,curve,label in zip(axes,curves,labels,strict=True):
        im=ax.imshow(curve,cmap='RdBu_r',vmin=-100,vmax=100,interpolation='nearest')
        ax.set_title(label+' | 100 s')
        ax.set_xticks(range(31),areas,rotation=90,fontsize=6);ax.set_yticks(range(31),areas,fontsize=6)
    fig.colorbar(im,ax=axes,label='Selected lag (ms); relative timing, not causality',shrink=.8)
    ax=fig.add_subplot(grid[1,:2]);order=np.argsort(normalized[0],kind='stable');y=np.arange(31)
    for values,label,color,marker in zip(normalized,labels,colors,['o','x','+'],strict=True):
        ax.scatter(values[order],y,c=color,marker=marker,s=28,label=label+' | 100 s')
    ax.set_yticks(y,[areas[i] for i in order],fontsize=8);ax.invert_yaxis()
    ax.set(xlim=(-.04,1.04),xlabel='Normalized hierarchy (lower = earlier)',title='31-area temporal order; MDP excluded')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.08),ncol=3,fontsize=8);ax.grid(axis='x',alpha=.2)
    ax=fig.add_subplot(grid[1,2]);ax.axis('off')
    lines=[]
    for label,key in [('Rust vs original','rust_vs_original'),('NEST vs original','native_vs_original'),('Rust vs NEST','rust_vs_native')]:
        m=metrics[key]
        lines.append(f'{label}\n  Lag MAE: {m["lag_mae_ms"]:.3f} ms\n  Sign agreement: {m["lag_sign_agreement"]:.1%}\n  Hierarchy Pearson r: {m["hierarchy_pearson"]:.4f}')
    ax.text(0,1,'\n\n'.join(lines)+'\n\nOne realization per simulator.\nNo independent-seed interval.\nNo equivalence threshold.',va='top',fontsize=11)
    fig.suptitle('Multi-Area Cortical Model | Matched 100 s temporal propagation\nDescriptive comparison; scientific equivalence unproven',fontsize=15)
    fig.savefig(output/'propagation-comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['original','rust','native','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();compare(args.original,args.rust,args.native,args.output)
