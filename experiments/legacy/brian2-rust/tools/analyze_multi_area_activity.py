"""Measure the first post-warmup full-model observation; no equivalence claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from contextlib import ExitStack

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from brian2_rust.results import load_results
from brian2_rust.multi_area_analysis import population_activity


def analyze(model_path, results, output, *, start_tick=5000, end_tick=25000,
            dt_seconds=.0001, bin_ticks=10, seed=20260908, spike_tick_offset=0,
            bounded_memory=False, raster_max_points=3000000):
    if end_tick == 1005000 and (not bounded_memory or spike_tick_offset != 1):
        raise ValueError('100.5 s activity requires bounded memory and the physical spike grid')
    if not bounded_memory and raster_max_points != 3000000:
        raise ValueError('custom raster budget requires bounded_memory=True')
    with ExitStack() as stack:
        return _analyze(model_path,results,output,start_tick=start_tick,end_tick=end_tick,
            dt_seconds=dt_seconds,bin_ticks=bin_ticks,seed=seed,spike_tick_offset=spike_tick_offset,
            bounded_memory=bounded_memory,raster_max_points=raster_max_points,stack=stack)


def _analyze(model_path,results,output,*,start_tick,end_tick,dt_seconds,bin_ticks,
             seed,spike_tick_offset,bounded_memory,raster_max_points,stack):
    if type(spike_tick_offset) is not int or spike_tick_offset not in (0, 1):
        raise ValueError('spike tick offset must be 0 (Brian labels) or 1 (NEST physical grid)')
    if start_tick < spike_tick_offset:
        raise ValueError('observation starts before the mapped recorded grid')
    if bounded_memory:
        import mam_correlation_stream as streaming
        from mam_streamed_activity import population_activity_streamed
        from mam_raster_budget import RasterBudget,RasterSample
        streaming.require_cache_release()
        budget = RasterBudget(raster_max_points)
        if start_tick!=5000 or end_tick not in (25000,105000,1005000) or dt_seconds!=.0001 or bin_ticks!=10:
            raise ValueError('bounded activity CLI requires admitted 2.5/10.5/100.5 s windows')
        if model_path.stat().st_size>=512*2**20 or sum((results/n).stat().st_size for n in ['results.bin','events.bin'])>={25000:8,105000:48,1005000:256}[end_tick]*2**30:
            raise ValueError('bounded activity CLI input byte budget exceeded')
    raw_start, raw_end = start_tick-spike_tick_offset, end_tick-spike_tick_offset
    if output.exists():
        raise FileExistsError(output)
    model = json.loads(model_path.read_text())
    if end_tick == 1005000 and any(p['steps'] != end_tick for p in model['definition']['populations']):
        raise ValueError('100.5 s activity requires an exact model duration')
    raster_end = 105000 if end_tick == 1005000 else end_tick
    data = load_results(model, results, include_times=False, release_file_cache=bounded_memory)
    if bounded_memory:
        release = stack.enter_context(streaming.mapped_cache(data['_dump'],results/'results.bin'))
    import struct
    for p in model['definition']['populations']:
        assert struct.unpack('>d', bytes.fromhex(p['dt']))[0] == dt_seconds
        assert p['steps'] >= end_tick and p['monitor']['window_steps'] >= end_tick-start_tick
        # The independent reader validates any state traces. Statistics below
        # use spike trains; the model's one-cell voltage monitor is preserved.
    output.mkdir(parents=True)
    rows, histograms, samples, raster = [], [], {}, []
    groups = ['23E','23I','4E','4I','5E','5I','6E','6I']
    areas = []
    for number, (definition, population) in enumerate(zip(
            model['definition']['populations'], data['populations'], strict=True)):
        area, group = definition['name'].removeprefix('mam_').rsplit('_', 1)
        assert group in groups
        if area not in areas:
            areas.append(area)
        collector = None
        if bounded_memory:
            if area in ['V1','V2','FEF']:
                collector = RasterSample(definition['count'],seed+10000+number,budget,
                    start_tick=start_tick,end_tick=raster_end,dt_seconds=dt_seconds)
            passes = 0
            def blocks():
                nonlocal passes
                passes += 1
                for ticks,ids in streaming.rust_blocks(population,release):
                    if spike_tick_offset==0: ticks=ticks-1
                    if collector is not None and passes==1:
                        # The statistic core still receives and validates every event.
                        keep = ticks < raster_end
                        collector.add(ticks[keep],ids[keep])
                    yield ticks,ids
            summary,histogram,sample = population_activity_streamed(blocks,definition['count'],
                start_tick=start_tick,end_tick=end_tick,dt_seconds=dt_seconds,bin_ticks=bin_ticks,
                seed=seed+number,sample_size=2000,expected_raw=len(population['indices']))
        else:
            summary, histogram, sample = population_activity(population['spike_ticks'],
                population['indices'], definition['count'], start_tick=raw_start,
                end_tick=raw_end, dt_seconds=dt_seconds, bin_ticks=bin_ticks,
                seed=seed+number, sample_size=2000, refractory_ms=2.0)
        rows.append(dict(population=number,name=definition['name'],area=area,group=group,**summary))
        histograms.append(histogram)
        samples.update({f'p{number}_{k}': v for k,v in sample.items()})
        if bounded_memory and collector is not None:
            samples[f'p{number}_raster_ids']=collector.selected
            raster.append(dict(area=area,group=group,**collector.finish()))
        elif not bounded_memory:
            if area in ['V1','V2','FEF']:
                n = definition['count']
                selected = np.sort(np.random.default_rng(seed+10000+number).choice(
                    n, max(1, int(np.ceil(.03*n))), replace=False))
                samples[f'p{number}_raster_ids'] = selected
                ticks, ids = population['spike_ticks'], population['indices']
                lo, hi = np.searchsorted(ticks, [raw_start,raw_end])
                lookup = np.full(n,-1,dtype=np.int32)
                lookup[selected] = np.arange(len(selected))
                local = lookup[ids[lo:hi]]
                keep = local >= 0
                raster.append(dict(area=area,group=group,selected=len(selected),
                    times=(ticks[lo:hi][keep]+spike_tick_offset)*dt_seconds,indices=local[keep]))
        print(f"{number+1}/254 {area} {group}: {summary['mean_rate_hz']:.4g} Hz",flush=True)
    histograms = np.array(histograms)
    assert int(histograms.sum()) == sum(r['observed_spikes'] for r in rows)
    area_counts = np.array([histograms[[r['area']==a for r in rows]].sum(axis=0) for a in areas])
    area_n = np.array([sum(r['neurons'] for r in rows if r['area']==a) for a in areas])
    area_rates = area_counts / area_n[:,None] / (dt_seconds*bin_ticks)
    duration = (end_tick-start_tick)*dt_seconds
    area_summaries = [dict(area=a,neurons=int(n),mean_rate_hz=float(c.sum()/n/duration),
        first_half_rate_hz=float(c[:len(c)//2].sum()/n/(duration/2)),
        second_half_rate_hz=float(c[len(c)//2:].sum()/n/(duration/2)))
        for a,n,c in zip(areas,area_n,area_counts,strict=True)]
    report = dict(schema='b2-mam-activity-observation-v1',scientific_equivalence=False,
        scope=f'One seed, {duration:g}-second observation; no independent simulator comparison.',
        window=dict(start_tick=start_tick,end_tick=end_tick,endpoint='[start,end)',
                    dt_seconds=dt_seconds,bin_ticks=bin_ticks,seconds=duration,
                    spike_tick_offset=spike_tick_offset,raw_start_tick=raw_start,raw_end_tick=raw_end,
                    time_basis='NEST physical right-edge grid' if spike_tick_offset else 'Brian left-edge spike labels'),
        sampling=dict(seed=seed,maximum_cells_per_population=2000,
            lvr='Uniform without replacement from all cells; zero-padded and >=3-spike means both reported.',
            correlation='Uniform without replacement from active cells; zero-variance cells excluded.',
            raster='Independent uniform 3% of each population in V1, V2, FEF, rounded up.',
            caveat='These frozen sampling and endpoint conventions are not claimed identical to the historical paper seed or every official helper.'),
        source=dict(paper='https://doi.org/10.1371/journal.pcbi.1006359',
            official_commit='0a658be40bef3249cbe452f38809edf7d2f524ba',
            lvr_refractory_ms=2.0),
        model_sha256=streaming.file_sha(model_path) if bounded_memory else hashlib.file_digest(model_path.open('rb'),'sha256').hexdigest(),
        result_sha256={n:(streaming.file_sha(results/n) if bounded_memory else hashlib.file_digest((results/n).open('rb'),'sha256').hexdigest())
                       for n in ['results.bin','events.bin']},
        population_count=len(rows),neurons=int(area_n.sum()),
        observed_spikes=int(histograms.sum()),
        mean_rate_hz=float(histograms.sum()/area_n.sum()/duration),
        populations=rows,areas=area_summaries)
    if end_tick == 1005000:
        report['window']['raster_end_tick'] = raster_end
        report['sampling']['raster'] += ' Display excerpt [0.5,10.5) s; all statistics use [0.5,100.5) s.'
    if bounded_memory:
        report['bounded_memory']=True
        report['raster_point_budget']=dict(maximum=budget.maximum,used=budget.used)
        report['implementation_sha256']={p.name:streaming.file_sha(p) for p in [Path(__file__),Path(__file__).with_name('mam_streamed_activity.py'),Path(__file__).with_name('mam_streamed_cell_metrics.py'),Path(__file__).with_name('mam_raster_budget.py'),Path(__file__).with_name('mam_correlation_stream.py'),Path(__file__).resolve().parents[1]/'python/brian2_rust/results.py']}
    np.savez_compressed(output/'activity-arrays.npz',population_counts=histograms,
                        area_counts=area_counts,area_rates_hz=area_rates,**samples)
    (output/'activity.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plot(rows, areas, groups, area_rates, raster, report['window'], output)
    catalog={p.name:dict(bytes=p.stat().st_size,sha256=streaming.file_sha(p) if bounded_memory else hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
             for p in output.iterdir() if p.is_file()}
    (output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['populations','areas']},indent=2))


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
    fig.suptitle('Full multi-area model — first post-warmup observation; paper reproduction unproven',fontsize=15)
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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['model','results','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--spike-tick-offset',type=int,choices=[0,1],default=0,
                        help='Explicit audited timestamp map: NEST physical tick = Brian tick + 1')
    parser.add_argument('--end-tick',type=int,choices=[25000,105000,1005000],default=25000)
    parser.add_argument('--bounded-memory',action='store_true')
    parser.add_argument('--raster-max-points',type=int,default=3000000)
    args=parser.parse_args()
    analyze(args.model,args.results,args.output,spike_tick_offset=args.spike_tick_offset,end_tick=args.end_tick,bounded_memory=args.bounded_memory,raster_max_points=args.raster_max_points)
