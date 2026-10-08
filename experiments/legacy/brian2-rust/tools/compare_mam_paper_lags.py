"""Compare temporal order and measure ten-window sensitivity in original data."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from analyze_mam_paper_fc import checked,sha
from mam_paper_propagation import lag_matrix,fit_hierarchy


def lag_stats(c,h,reference_c,reference_h):
    c,h,reference_c,reference_h=[np.asarray(x,dtype=float) for x in [c,h,reference_c,reference_h]]
    if (c.shape!=(31,31) or reference_c.shape!=c.shape or h.shape!=(31,) or reference_h.shape!=h.shape
            or not all(np.isfinite(x).all() for x in [c,h,reference_c,reference_h])):
        raise ValueError('complete finite 31-area comparison required')
    mask=~np.eye(31,dtype=bool);x,y=c[mask],reference_c[mask];d=x-y
    return dict(directed_off_diagonal_entries=930,lag_mae_ms=float(np.abs(d).mean()),
        lag_rmse_ms=float(np.sqrt(np.mean(d**2))),lag_entry_correlation=float(np.corrcoef(x,y)[0,1]),
        lag_sign_agreement=float(np.mean(np.sign(x)==np.sign(y))),
        hierarchy_pearson=float(np.corrcoef(h,reference_h)[0,1]),
        hierarchy_spearman=float(spearmanr(h,reference_h).statistic))


def require_short_window_cases(original_seconds, rust_seconds, native_seconds):
    if (original_seconds, rust_seconds, native_seconds) != (100., 10., 10.):
        raise ValueError("This ten-window diagnostic requires original 100 s and both simulators 10 s; primary-duration comparison needs separate admission")


def compare(a):
    reports=[];curves=[];levels=[];sources={}
    for case,root in [('original',a.original),('rust',a.rust),('native',a.native)]:
        cat=json.loads((root/'catalog.json').read_text());rp=checked(root,cat,'lags.json',2**20)
        report=json.loads(rp.read_text());ap=checked(root,cat,'lags.npz',2**20)
        if report['schema']!='b2-mam-paper-propagation-v1' or report['excluded_from_hierarchy']!=['MDP']:
            raise ValueError('unexpected propagation artifact')
        if reports:
            for key in ['area_names','hierarchy_area_names','settings']:
                if report[key]!=reports[0][key]:raise ValueError('incompatible propagation '+key)
            if report['source_sha256']['reference_audit']!=reports[0]['source_sha256']['reference_audit']:
                raise ValueError('different source verification evidence')
        with np.load(ap,allow_pickle=False) as data:
            c=data['retained_lag_ms'];h=data['levels_ms']
            np.testing.assert_array_equal(h,np.array(report['levels_ms']))
        curves.append(c);levels.append(h);reports.append(report);sources[case]=dict(report=sha(rp),arrays=sha(ap))
    original,rust,native=reports
    if (original['identity']['simulator']!='Published original rates' or original['observation_seconds']!=100.
            or rust['identity']['simulator']!='Rust' or native['identity']['simulator']!='NEST'
            or rust['observation_seconds']!=native['observation_seconds']):raise ValueError('wrong cases or unmatched simulator durations')
    require_short_window_cases(original['observation_seconds'],rust['observation_seconds'],native['observation_seconds'])
    rc=json.loads((a.original_rates/'catalog.json').read_text())
    if sha(a.original_rates/'catalog.json')!=original['source_sha256']['original_catalog']:
        raise ValueError('original full series changed')
    areas=original['area_names'];keep=[i for i,name in enumerate(areas) if name!='MDP']
    rates=np.array([np.load(checked(a.original_rates,rc['files'],'metastable100--rate_time_series_full_'+area+'.npy',2**20),allow_pickle=False) for area in areas])
    if rates.shape!=(32,100000):raise ValueError('complete 100 s original data required')
    pairs={'rust_vs_original':(1,0),'native_vs_original':(2,0),'rust_vs_native':(1,2)}
    metrics={name:lag_stats(curves[i],levels[i],curves[j],levels[j]) for name,(i,j) in pairs.items()}
    blocks=[];block_levels=[];block_lags=[]
    for block in range(10):
        c,diag=lag_matrix(rates[:,block*10000:(block+1)*10000],areas);c=c[np.ix_(keep,keep)]
        result=fit_hierarchy(c)
        if not result['normalization_defined']:raise ValueError('undefined original block hierarchy')
        blocks.append(dict(block=block,start_ms=500+block*10000,end_ms=500+(block+1)*10000,
            **lag_stats(c,result['levels_ms'],curves[0],levels[0])))
        block_levels.append(result['normalized_levels']);block_lags.append(c)
    statistics={key:dict(minimum=float(min(r[key] for r in blocks)),median=float(np.median([r[key] for r in blocks])),maximum=float(max(r[key] for r in blocks))) for key in blocks[0] if key not in ['block','start_ms','end_ms','directed_off_diagonal_entries']}
    report=dict(schema='b2-mam-propagation-comparison-v1',scientific_equivalence=False,
        model_observation_seconds=rust['observation_seconds'],original_observation_seconds=100.,
        equal_observation_duration=(rust['observation_seconds']==100.),hierarchy_area_names=original['hierarchy_area_names'],
        metrics=metrics,original_ten_second_blocks=blocks,original_block_summary=statistics,
        source_sha256=sources,original_rate_catalog_sha256=sha(a.original_rates/'catalog.json'),
        implementation_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('mam_paper_propagation.py')]},
        scope='Descriptive temporal-order comparison and within-original-run duration sensitivity. Ten contiguous blocks are dependent and share one realization; min/max are not confidence intervals, equivalence margins or independent-seed evidence. Full original lag matrix is an available-code reanalysis; only four stored pair covariance files were directly audited. No causal or simulator/scientific-equivalence claim.')
    a.output.mkdir(exist_ok=False)
    np.savez_compressed(a.output/'comparison-arrays.npz',original_lags=curves[0],rust_lags=curves[1],native_lags=curves[2],original_block_lags=np.array(block_lags),original_block_normalized_levels=np.array(block_levels))
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plot(reports,curves,metrics,np.array(block_levels),blocks,a.output)
    (a.output/'catalog.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in a.output.iterdir()},indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='original_ten_second_blocks'},indent=2))


def plot(reports,curves,metrics,block_levels,blocks,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(16,12),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[1,1.3])
    axes=[fig.add_subplot(grid[0,i]) for i in range(3)]
    areas=reports[0]['hierarchy_area_names']
    for ax,c,label,r in zip(axes,curves,['Original reanalysis','Rust','NEST'],reports,strict=True):
        im=ax.imshow(c,cmap='RdBu_r',vmin=-100,vmax=100,interpolation='nearest')
        ax.set_title(f'{label} | {r["observation_seconds"]:g} s')
        ax.set_xticks(range(31),areas,rotation=90,fontsize=6);ax.set_yticks(range(31),areas,fontsize=6)
    fig.colorbar(im,ax=axes,label='Selected lag (ms); relative timing, not causality',shrink=.8)
    ax=fig.add_subplot(grid[1,:2]);order=np.argsort(reports[0]['normalized_levels']);y=np.arange(31)
    lo=block_levels.min(axis=0)[order];hi=block_levels.max(axis=0)[order]
    ax.hlines(y,lo,hi,color='.8',lw=5,label='Original 10 s blocks: min–max (not a CI)')
    for r,label,color,marker in zip(reports,['Original 100 s','Rust 10 s','NEST 10 s'],['black','#d95f02','#1f77b4'],['o','x','+'],strict=True):
        ax.scatter(np.array(r['normalized_levels'])[order],y,c=color,marker=marker,s=26,label=label,zorder=3)
    ax.set_yticks(y,[areas[i] for i in order],fontsize=8);ax.invert_yaxis();ax.set(xlim=(-.04,1.04),xlabel='Normalized hierarchy (lower = earlier)',title='31-area temporal order; MDP excluded as in Fig. 7')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.08),ncol=2,fontsize=8);ax.grid(axis='x',alpha=.2)
    ax=fig.add_subplot(grid[1,2]);x=np.arange(1,11)
    ax.plot(x,[row['hierarchy_pearson'] for row in blocks],'o-',color='.35',label='Original 10 s blocks')
    ax.axhline(metrics['rust_vs_original']['hierarchy_pearson'],color='#d95f02',label='Rust 10 s')
    ax.axhline(metrics['native_vs_original']['hierarchy_pearson'],color='#1f77b4',label='NEST 10 s')
    ax.set(xticks=x,xlabel='Contiguous original block',ylabel='Hierarchy Pearson r vs original 100 s',ylim=(-.05,1.05),title='Short-window sensitivity\nSame realization; not an acceptance test')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Multi-Area Cortical Model | Temporal propagation\nAvailable-source reanalysis; unequal durations; scientific equivalence unproven',fontsize=15)
    fig.savefig(output/'propagation-comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['original','rust','native','original-rates','output']:p.add_argument('--'+name,type=Path,required=True)
    compare(p.parse_args())
