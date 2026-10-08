"""Bounded native NEST spike statistics on the same physical window as Rust.

The raw native records already use physical timestamps. No tick offset is
applied. A stochastic realization is not a paired numerical equivalence test.
"""
import argparse
import hashlib
import importlib.util
import json
from contextlib import ExitStack,closing
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mam_statistics', ROOT/'python/brian2_rust/multi_area_analysis.py')
statistics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(statistics)
DTYPE = np.dtype([('tick','<u4'),('cell','<u4')])


def canonical_population_name(population):
    area, group = population['area'], population['population']
    if population['name'] != f'{area}-{group}':
        raise ValueError('inconsistent native population identity')
    return f'mam_{area}_{group}'


def canonical_population_indices(populations):
    """Match Rust's lexicographic analysis order without changing native IDs."""
    names = [canonical_population_name(p) for p in populations]
    if len(set(names)) != len(names):
        raise ValueError('duplicate population identity')
    return sorted(range(len(populations)), key=names.__getitem__)


def simulation_identity(reports, *, duration_ms=2500):
    """Reject mixed seeds/layouts before treating files as one realization."""
    if len(reports) != 48 or sorted(r['rank'] for r in reports) != list(range(48)):
        raise ValueError('requires exactly 48 distinct native ranks')
    keys = ['seed', 'ranks', 'threads', 'duration_ms', 'dt_ms', 'nest_version']
    identity = {key: reports[0][key] for key in keys}
    if any({key: r[key] for key in keys} != identity for r in reports):
        raise ValueError('mixed native simulation identities')
    if (type(identity['seed']) is not int or identity['seed'] <= 0
            or identity['ranks'] != 48 or identity['threads'] != 4
            or duration_ms not in (2500,10500,100500)
            or identity['duration_ms'] != duration_ms or identity['dt_ms'] != .1):
        raise ValueError('unexpected full native observation contract')
    return identity


def bucket_events(populations, files, output, *, start_tick=5000, end_tick=25000,
                  max_bytes=3072000000, release_file_cache=False):
    """Filter physical [start,end) and partition in fixed-size blocks."""
    if sum(p.stat().st_size for p in files)>max_bytes:
        raise ValueError('native event input budget exceeded')
    if release_file_cache:
        import os
        from mam_correlation_stream import require_cache_release, CACHE_CHUNK
        require_cache_release()
    pending_bytes = 0
    dirty = set()
    def release_outputs():
        for index in sorted(dirty):
            f = handles[index]
            f.flush()
            os.fdatasync(f.fileno())
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        dirty.clear()
    counts=np.array([p['count'] for p in populations],dtype=np.int64)
    if not len(counts) or np.any(counts<=0) or not 0<=start_tick<end_tick:
        raise ValueError('invalid populations or physical window')
    ends=np.cumsum(counts);starts=np.r_[0,ends[:-1]]
    output.mkdir(exist_ok=False)
    paths=[output/f'p{i}.bin' for i in range(len(populations))]
    handles=[];totals=np.zeros(len(paths),dtype=np.int64)
    try:
        handles=[p.open('xb') for p in paths]
        for path in files:
            if path.stat().st_size%DTYPE.itemsize:
                raise ValueError('partial native event record')
            with path.open('rb') as stream:
                while True:
                    offset = stream.tell()
                    a=np.fromfile(stream,dtype=DTYPE,count=131072)
                    if not len(a):break
                    if release_file_cache:
                        os.posix_fadvise(stream.fileno(), offset, stream.tell()-offset, os.POSIX_FADV_DONTNEED)
                    if np.any(a['cell']>=ends[-1]):raise ValueError('native cell ID outside model')
                    a=a[(a['tick']>=start_tick)&(a['tick']<end_tick)]
                    if not len(a):continue
                    groups=np.searchsorted(ends,a['cell'],side='right')
                    order=np.argsort(groups,kind='stable');a=a[order];groups=groups[order]
                    cut=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1,len(groups)]
                    for lo,hi in zip(cut[:-1],cut[1:],strict=True):
                        group=int(groups[lo]);block=a[lo:hi].copy()
                        block['cell']-=starts[group].astype(np.uint32)
                        block.tofile(handles[group]);totals[group]+=hi-lo
                        pending_bytes += block.nbytes
                        if release_file_cache:dirty.add(group)
                    if release_file_cache and pending_bytes >= CACHE_CHUNK:
                        release_outputs();pending_bytes = 0
        if release_file_cache:
            release_outputs()
    finally:
        for f in handles:f.close()
    return paths,totals


def analyze(audit_dir, output, *, end_tick=25000, max_bytes=3072000000, max_population_events=50000000,
            bounded_memory=False,raster_max_points=3000000):
    if end_tick == 1005000 and not bounded_memory:
        raise ValueError('100.5 s activity requires bounded memory')
    if not bounded_memory and raster_max_points != 3000000:
        raise ValueError('custom raster budget requires bounded_memory=True')
    with ExitStack() as stack:
        return _analyze(audit_dir,output,end_tick=end_tick,max_bytes=max_bytes,
            max_population_events=max_population_events,bounded_memory=bounded_memory,
            raster_max_points=raster_max_points,stack=stack)


def _analyze(audit_dir,output,*,end_tick,max_bytes,max_population_events,bounded_memory,raster_max_points,stack):
    if bounded_memory:
        import mam_correlation_stream as streaming
        from mam_streamed_activity import population_activity_streamed
        from mam_raster_budget import RasterBudget,RasterSample
        streaming.require_cache_release()
        budget=RasterBudget(raster_max_points)
    if type(end_tick) is not int or end_tick not in (25000,105000,1005000):
        raise ValueError('requires the declared 2.5, 10.5 or 100.5 second native run')
    if end_tick == 1005000 and (type(max_bytes) is not int or not 1 <= max_bytes <= 128*2**30):
        raise ValueError('100.5 s native raw byte budget must be in 1..128 GiB')
    raster_end = 105000 if end_tick == 1005000 else end_tick
    duration=(end_tick-5000)*.0001
    audit=json.loads((audit_dir/'summary.json').read_text())
    if not audit['passed']:raise ValueError('requires passed complete raw-data audit')
    parameter=list((audit_dir/'node25').glob('*/parameters.json'))
    assert len(parameter)==1
    raw=parameter[0].read_bytes();p=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()==audit['parameters_sha256']
    assert p['total_neurons']==4129924 and p['total_recurrent_synapses']==24126516728
    files=[];source={};rank_meta=[]
    for node in sorted(audit_dir.glob('node*')):
        for f in sorted((node/'runs').glob('*/rank*.events.bin')):
            r=json.loads(f.with_name(f.name.replace('.events.bin','.json')).read_text())
            h=streaming.file_sha(f) if bounded_memory else hashlib.file_digest(f.open('rb'),'sha256').hexdigest()
            assert h==r['event_sha256'] and f.stat().st_size==r['event_bytes']
            assert r['duration_ms']==end_tick//10 and r['parameters_sha256']==audit['parameters_sha256']
            files.append(f);rank_meta.append(r);source[str(f)]=dict(bytes=f.stat().st_size,sha256=h)
    identity = simulation_identity(rank_meta,duration_ms=end_tick//10)
    output.mkdir(exist_ok=False)
    paths,totals=bucket_events(p['populations'],files,output/'population-events',end_tick=end_tick,max_bytes=max_bytes,release_file_cache=bounded_memory)
    rows=[];histograms=[];samples={};raster=[];areas=[];groups=['23E','23I','4E','4I','5E','5I','6E','6I'];seed=20260908
    for i, native_index in enumerate(canonical_population_indices(p['populations'])):
        q, path, count = p['populations'][native_index], paths[native_index], totals[native_index]
        collector=None
        if bounded_memory:
            if q['area'] in ['V1','V2','FEF']:
                collector=RasterSample(q['count'],seed+10000+i,budget,end_tick=raster_end)
            passes=0
            def blocks():
                nonlocal passes
                passes+=1
                source=stack.enter_context(closing(streaming.native_blocks(path,DTYPE)))
                for ticks,ids in source:
                    if collector is not None and passes==1:
                        # Restrict display only; all events reach the statistic core.
                        keep = ticks < raster_end
                        collector.add(ticks[keep],ids[keep])
                    yield ticks,ids
            summary,hist,sample=population_activity_streamed(blocks,q['count'],end_tick=end_tick,
                seed=seed+i,expected_raw=int(count))
        else:
            if count>max_population_events:raise ValueError('population analysis event budget exceeded before allocation')
            events=np.fromfile(path,dtype=DTYPE);assert len(events)==count
            order=np.argsort(events['tick'],kind='stable');events=events[order];del order
            ticks=events['tick'].astype(np.int64);ids=events['cell'].astype(np.int64)
            summary,hist,sample=statistics.population_activity(ticks,ids,q['count'],start_tick=5000,end_tick=end_tick,dt_seconds=.0001,bin_ticks=10,seed=seed+i,sample_size=2000,refractory_ms=2.)
        area,group=q['area'],q['population'];assert q['name']==f'{area}-{group}' and group in groups
        name=canonical_population_name(q)
        if area not in areas:areas.append(area)
        rows.append(dict(population=i,name=name,area=area,group=group,**summary));histograms.append(hist)
        samples.update({f'p{i}_{k}':v for k,v in sample.items()})
        if bounded_memory and collector is not None:
            samples[f'p{i}_raster_ids']=collector.selected
            raster.append(dict(area=area,group=group,**collector.finish()))
        elif not bounded_memory:
            if area in ['V1','V2','FEF']:
                selected=np.sort(np.random.default_rng(seed+10000+i).choice(q['count'],max(1,int(np.ceil(.03*q['count']))),replace=False))
                lookup=np.full(q['count'],-1,dtype=np.int32);lookup[selected]=np.arange(len(selected));local=lookup[ids];keep=local>=0
                raster.append(dict(area=area,group=group,selected=len(selected),times=ticks[keep]*.0001,indices=local[keep]));samples[f'p{i}_raster_ids']=selected
        print(json.dumps(dict(population=i,name=name,mean_rate_hz=summary['mean_rate_hz'],spikes=int(count))),flush=True)
        path.unlink()  # Only this invocation's reconstructible population scratch file.
    (output/'population-events').rmdir()
    histograms=np.array(histograms);assert int(histograms.sum())==int(totals.sum())
    assert int(histograms.sum()) == sum(audit['physical_50ms_bin_counts'][10:])
    area_counts=np.array([histograms[[r['area']==a for r in rows]].sum(axis=0) for a in areas]);area_n=np.array([sum(r['neurons'] for r in rows if r['area']==a) for a in areas]);area_rates=area_counts/area_n[:,None]/.001
    report=dict(schema='b2-native-mam-activity-v1',scientific_equivalence=False,scope=f'One independent native NEST realization and {duration:g}-second observation; no statistical equivalence claim.',
        window=dict(start_tick=5000,end_tick=end_tick,endpoint='[start,end)',dt_seconds=.0001,bin_ticks=10,seconds=duration,spike_tick_offset=0,raw_start_tick=5000,raw_end_tick=end_tick,time_basis='Native NEST physical grid'),
        sampling=dict(seed=seed,maximum_cells_per_population=2000,lvr='Uniform without replacement from all cells; zero-padded and >=3-spike means both reported.',correlation='Uniform without replacement from active cells; zero-variance cells excluded.',raster='Independent uniform 3% of each population in V1, V2, FEF, rounded up.',caveat='These frozen sampling and endpoint conventions are not claimed identical to the historical paper seed or every official helper.'),
        simulation=identity,parameters_sha256=audit['parameters_sha256'],result_files=source,population_count=len(rows),neurons=int(area_n.sum()),observed_spikes=int(histograms.sum()),mean_rate_hz=float(histograms.sum()/area_n.sum()/duration),populations=rows,
        areas=[dict(area=a,neurons=int(n),mean_rate_hz=float(c.sum()/n/duration),first_half_rate_hz=float(c[:len(c)//2].sum()/n/(duration/2)),second_half_rate_hz=float(c[len(c)//2:].sum()/n/(duration/2))) for a,n,c in zip(areas,area_n,area_counts,strict=True)])
    if end_tick == 1005000:
        report['window']['raster_end_tick'] = raster_end
        report['sampling']['raster'] += ' Display excerpt [0.5,10.5) s; all statistics use [0.5,100.5) s.'
    if bounded_memory:
        report['bounded_memory']=True
        report['raster_point_budget']=dict(maximum=budget.maximum,used=budget.used)
        report['implementation_sha256']={p.name:streaming.file_sha(p) for p in [Path(__file__),Path(__file__).with_name('mam_streamed_activity.py'),Path(__file__).with_name('mam_streamed_cell_metrics.py'),Path(__file__).with_name('mam_raster_budget.py'),Path(__file__).with_name('mam_correlation_stream.py'),ROOT/'python/brian2_rust/multi_area_analysis.py']}
    np.savez_compressed(output/'activity-arrays.npz',population_counts=histograms,area_counts=area_counts,area_rates_hz=area_rates,**samples)
    (output/'activity.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plot(rows,areas,groups,area_rates,raster,report['window'],output)
    catalog={f.name:dict(bytes=f.stat().st_size,sha256=streaming.file_sha(f) if bounded_memory else hashlib.file_digest(f.open('rb'),'sha256').hexdigest()) for f in output.iterdir() if f.is_file()}
    (output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['observed_spikes','mean_rate_hz','neurons']},indent=2))


def plot(rows, areas, groups, area_rates, raster, window, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    start = window['start_tick']*window['dt_seconds']
    end = window['end_tick']*window['dt_seconds']
    bin_seconds = window['bin_ticks']*window['dt_seconds']
    rates = np.full((len(groups),len(areas)),np.nan)
    for row in rows:
        rates[groups.index(row['group']),areas.index(row['area'])] = row['mean_rate_hz']
    cmap = plt.colormaps['viridis'].copy();cmap.set_bad('black');cmap.set_under('#dddddd')
    fig, axes = plt.subplots(3,1,figsize=(15,13),layout='constrained')
    shown = np.where(rates==0,.005,rates)
    img = axes[0].imshow(shown,aspect='auto',cmap=cmap,norm=LogNorm(vmin=.01,vmax=max(1,np.nanmax(rates))))
    axes[0].set(yticks=range(len(groups)),yticklabels=groups,xticks=range(len(areas)),xticklabels=areas,
                title=f'Population mean rates: {start:g}–{end:g} s (black: absent population; gray: <0.01 Hz)')
    axes[0].tick_params(axis='x',labelrotation=45)
    fig.colorbar(img,ax=axes[0],label='Hz per neuron')
    img = axes[1].imshow(area_rates,aspect='auto',origin='lower',extent=[start,end,-.5,len(areas)-.5],cmap='magma')
    axes[1].set(yticks=range(len(areas)),yticklabels=areas,xlabel='Biological time (s)',
                title='Area rates, original 1 ms bins')
    axes[1].tick_params(axis='y',labelsize=7)
    fig.colorbar(img,ax=axes[1],label='Hz per neuron')
    for a in ['V1','V2','FEF','MT','MIP']:
        if a in areas:
            smooth = np.convolve(area_rates[areas.index(a)],np.ones(20)/20,mode='valid')
            x = start+(np.arange(len(smooth))+10)*bin_seconds
            axes[2].plot(x,smooth,label=a,lw=1)
    axes[2].set(xlabel='Biological time (s)',ylabel='Hz per neuron',
                title='Selected areas (20 ms boxcar for display; full 1 ms counts archived)')
    axes[2].legend(ncol=5);axes[2].grid(alpha=.2)
    fig.suptitle(f'Native NEST full multi-area model — {end-start:g} s observation; reproduction unproven',fontsize=15)
    fig.savefig(output/'activity-overview.png',dpi=150);plt.close(fig)
    raster_end = window.get('raster_end_tick',window['end_tick'])*window['dt_seconds']
    assert sum(len(r['times']) for r in raster)<=3_000_000,'raster point budget exceeded'
    fig,axes=plt.subplots(3,1,figsize=(14,12),sharex=True,layout='constrained')
    for ax,area in zip(axes,['V1','V2','FEF'],strict=True):
        offset=0;positions=[];labels=[]
        for r in raster:
            if r['area']!=area:
                continue
            ax.scatter(r['times'],r['indices']+offset,s=.12,c='#1f77b4' if r['group'].endswith('E') else '#c43838',linewidths=0)
            positions.append(offset+r['selected']/2);labels.append(r['group']);offset+=r['selected']
        ax.set(title=area,yticks=positions,yticklabels=labels,xlim=(start,raster_end),ylim=(-1,offset),ylabel='Sampled cells')
    axes[-1].set_xlabel('Biological time (s)')
    title = f'Uniform 3% sample per population — E blue / I red; one seed, {raster_end-start:g} s observation'
    if 'raster_end_tick' in window:
        title += f' excerpt; statistics cover {end-start:g} s'
    fig.suptitle(title)
    fig.savefig(output/'spike-rasters.png',dpi=150);plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--end-tick',type=int,choices=[25000,105000,1005000],default=25000)
    parser.add_argument('--max-bytes',type=int,default=3072000000)
    parser.add_argument('--bounded-memory',action='store_true')
    parser.add_argument('--raster-max-points',type=int,default=3000000)
    args=parser.parse_args()
    analyze(args.audit_dir,args.output,end_tick=args.end_tick,max_bytes=args.max_bytes,bounded_memory=args.bounded_memory,raster_max_points=args.raster_max_points)
