"""Rebin complete raw MAM events into the modern official rate/PSD view.

Keep the frozen diagnostic bins intact and prove their exact reconstruction.
Host memory/time/file admission is enforced by the external service guard.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from contextlib import ExitStack, closing
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from mam_paper_spectrum import spectrum,SETTINGS

DTYPE=np.dtype([('tick','<u4'),('cell','<u4')])


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class Counts:
    def __init__(self, populations=254, end=25000):
        if type(populations) is not int or not 1<=populations<=254 or type(end) is not int or end not in (25000,105000,505000,1005000):
            raise ValueError('full data diagnostic requires 1..254 populations and a declared 2.5/10.5/50.5/100.5 s duration')
        self.end=end;self.nbins=(end-5000)//10;self.shifted=np.zeros((populations,self.nbins),dtype=np.int64)
        self.frozen=np.zeros_like(self.shifted)
        self.lower=np.zeros(populations,dtype=np.int64)
        self.terminal=np.zeros(populations,dtype=np.int64)
        self.raw=0

    def add(self,ticks,groups):
        ticks,groups=np.asarray(ticks),np.asarray(groups)
        if (ticks.ndim!=1 or groups.shape!=ticks.shape or ticks.dtype.kind not in 'iu'
                or groups.dtype.kind not in 'iu' or len(ticks)>131072
                or np.any(ticks<0) or np.any(ticks>self.end)
                or np.any(groups<0) or np.any(groups>=len(self.lower))):
            raise ValueError('invalid bounded physical event block')
        ticks=ticks.astype(np.int64);groups=groups.astype(np.int64);self.raw+=len(ticks)
        for counts,start,end,origin in [(self.shifted,5005,self.end+1,5005),(self.frozen,5000,self.end,5000)]:
            keep=(ticks>=start)&(ticks<end)
            index=groups[keep]*self.nbins+(ticks[keep]-origin)//10
            # Unbuffered indexed accumulation preserves repeated indices without
            # allocating an entire population-by-time histogram for this block.
            np.add.at(counts.reshape(-1), index, 1)
        self.lower+=np.bincount(groups[(ticks>=5000)&(ticks<5005)],minlength=len(self.lower))
        self.terminal+=np.bincount(groups[ticks==self.end],minlength=len(self.lower))


def analyze(a):
    with ExitStack() as stack:
        return _analyze(a, stack)


def _analyze(a, stack):
    bounded = getattr(a, 'bounded_memory', False)
    if bounded:
        import mam_correlation_stream as streaming
        streaming.require_cache_release()
    digest_file = streaming.file_sha if bounded else sha
    start=time.perf_counter();baseline=json.loads((a.baseline/'activity.json').read_text());w=baseline['window']
    end_tick=w['end_tick'];assert end_tick in (25000,105000,1005000)
    if end_tick == 1005000 and not bounded:
        raise ValueError('100 s paper time series requires bounded memory')
    assert (w['start_tick'],w['dt_seconds'],w['bin_ticks'],w['endpoint'])==(5000,.0001,10,'[start,end)')
    rows=baseline['populations'];assert len(rows)==254 and baseline['neurons']==4129924
    names=[r['name'] for r in rows];assert names==sorted(set(names))
    sources={};catalog=json.loads((a.baseline/'catalog.json').read_text())
    for name in ['activity.json','activity-arrays.npz']:
        p=a.baseline/name;assert p.stat().st_size==catalog[name]['bytes'] and digest_file(p)==catalog[name]['sha256'];sources[str(p)]=digest_file(p)
    normalization=json.loads(a.normalization.read_text())
    assert normalization['schema']=='b2-mam-official-analysis-neuron-sizes-v1'
    assert normalization['source_commit']=='0a658be40bef3249cbe452f38809edf7d2f524ba'
    assert normalization['parameters_sha256']=='ee1a642c94b9d6e808627c54f78162e7f5d87f4140ba56496c5dfee3ed900d3e'
    nr=normalization['populations'];assert [p['name'] for p in nr]==names
    assert [p['simulated_neurons'] for p in nr]==[p['neurons'] for p in rows]
    generated=a.normalization.parent/normalization['generated_data_name']
    assert digest_file(generated)==normalization['generated_data_sha256']=='8c66bb68d55cf2bff222c67716952a2b6260576d6ba0a3650cc150af49f7c2cb'
    ndata=json.loads(generated.read_text())['neuron_numbers']
    for p,row in zip(nr,rows,strict=True):
        value=p['official_normalization_neurons']
        assert np.isfinite(value) and value==ndata[row['area']][row['group']] and int(value)==row['neurons']
    sources[str(a.normalization)]=digest_file(a.normalization);sources[str(generated)]=digest_file(generated)
    bins=Counts(end=end_tick);identity={}
    if a.native_audit:
        from analyze_native_mam_activity import simulation_identity
        assert not a.model and not a.results and w['spike_tick_offset']==0
        audit_path=a.native_audit/'summary.json';audit=json.loads(audit_path.read_text());assert audit['passed']
        paths=list((a.native_audit/'node25').glob('*/parameters.json'));assert len(paths)==1
        assert digest_file(paths[0])==audit['parameters_sha256']==baseline['parameters_sha256'];params=json.loads(paths[0].read_text());pops=params['populations']
        assert params['total_neurons']==4129924 and params['total_recurrent_synapses']==24126516728
        mapping=np.array([names.index('mam_'+p['name'].replace('-','_')) for p in pops]);ends=np.cumsum([p['count'] for p in pops])
        for p,index in zip(pops,mapping):assert rows[index]['neurons']==p['count']
        files=sorted(a.native_audit.glob('node*/runs/*/rank*.events.bin'));assert len(files)==48 and sum(f.stat().st_size for f in files)<=(128*2**30 if end_tick==1005000 else 3072000000*end_tick//25000)
        reports=[]
        for f in files:
            r=json.loads(f.with_name(f.name.replace('.events.bin','.json')).read_text());reports.append(r)
            assert r['parameters_sha256']==baseline['parameters_sha256'] and f.stat().st_size==r['event_bytes']==r['local_spikes']*8
            assert digest_file(f)==r['event_sha256']==baseline['result_files'][str(f)]['sha256'];sources[str(f)]=r['event_sha256']
            def blocks():
                if bounded:
                    yield from streaming.native_blocks(f, DTYPE)
                else:
                    with f.open('rb') as stream:
                        while len(block:=np.fromfile(stream,dtype=DTYPE,count=131072)):
                            yield block['tick'],block['cell']
            with closing(blocks()) as events:
                for ticks,cells in events:
                    assert np.all(cells<4129924) and np.all((cells.astype(np.uint64)+1)%48==r['rank'])
                    bins.add(ticks,mapping[np.searchsorted(ends,cells,side='right')])
        assert bins.raw==audit['spikes'] and int(bins.terminal.sum())==audit['terminal_tick_events']
        assert simulation_identity(reports,duration_ms=end_tick//10)==baseline['simulation']
        identity=dict(simulator='NEST',condition=params['state'],parameters_sha256=baseline['parameters_sha256'],**baseline['simulation'])
        sources[str(paths[0])]=digest_file(paths[0]);sources[str(audit_path)]=digest_file(audit_path)
    else:
        from brian2_rust.results import load_results
        assert a.model and a.results and w['spike_tick_offset']==1
        assert a.model.stat().st_size<512*2**20 and digest_file(a.model)==baseline['model_sha256'];sources[str(a.model)]=digest_file(a.model)
        assert sum((a.results/n).stat().st_size for n in ['results.bin','events.bin'])<{25000:8,105000:48,1005000:256}[end_tick]*2**30
        for n,h in baseline['result_sha256'].items():assert digest_file(a.results/n)==h;sources[str(a.results/n)]=h
        model=json.loads(a.model.read_text())
        assert all(p['steps']==end_tick for p in model['definition']['populations'])
        data=load_results(model,a.results,include_times=False,release_file_cache=bounded)
        if bounded:
            release=stack.enter_context(streaming.mapped_cache(data['_dump'],a.results/'results.bin'))
        assert [p['name'] for p in model['definition']['populations']]==names
        for i,(p,v) in enumerate(zip(model['definition']['populations'],data['populations'],strict=True)):
            assert p['count']==rows[i]['neurons'] and p['steps']==end_tick
            if bounded:
                for ticks,_ in streaming.rust_blocks(v,release):
                    bins.add(ticks,np.full(len(ticks),i,dtype=np.int64))
            else:
                for offset in range(0,len(v['spike_ticks']),131072):
                    ticks=v['spike_ticks'][offset:offset+131072].astype(np.int64)+1
                    bins.add(ticks,np.full(len(ticks),i,dtype=np.int64))
        assert bins.raw==data['metadata']['spike_count']
        identity=dict(simulator='Rust',model_sha256=baseline['model_sha256'],seed=model['instance']['rng_seed'])
    with np.load(a.baseline/'activity-arrays.npz',allow_pickle=False) as arrays:
        np.testing.assert_array_equal(bins.frozen,arrays['population_counts'])
    np.testing.assert_array_equal(bins.shifted.sum(axis=1),bins.frozen.sum(axis=1)-bins.lower+bins.terminal)
    neurons=np.array([r['official_normalization_neurons'] for r in nr]);rates=bins.shifted/(neurons[:,None]*1./1000.)
    areas=[r['area'] for r in baseline['areas']];assert len(areas)==32
    area_rates=np.array([np.average(rates[[r['area']==area for r in rows]],axis=0,weights=neurons[[r['area']==area for r in rows]]) for area in areas])
    powers=[]
    for rate in area_rates:
        freq,power=spectrum(rate);powers.append(power)
    powers=np.array(powers);assert powers.shape==(32,513) and np.isfinite(powers).all() and np.all(powers>=0)
    a.output.mkdir(exist_ok=False)
    np.savez_compressed(a.output/'time-series.npz',population_counts=bins.shifted,population_rates_hz=rates,area_rates_hz=area_rates,frequency_hz=freq,power_hz2_per_hz=powers,lower_half_ms_counts=bins.lower,terminal_counts=bins.terminal)
    result=dict(schema='b2-mam-modern-paper-time-series-v1',scientific_equivalence=False,identity=identity,
        physical_tick_ms=.1,wrapper_window_ms=f'(500,{end_tick//10}]',helper_histogram_range_ms=[500.5,end_tick/10+.5],bin_ms=1,
        frozen_histogram_exact=True,endpoint_count_identity_exact=True,raw_events=bins.raw,frozen_events=int(bins.frozen.sum()),
        shifted_events=int(bins.shifted.sum()),excluded_lower_half_ms_events=int(bins.lower.sum()),included_terminal_events=int(bins.terminal.sum()),
        changed_population_bin_entries=int(np.count_nonzero(bins.shifted!=bins.frozen)),total_population_bin_entries=int(bins.frozen.size),
        maximum_bin_count_change=int(np.max(np.abs(bins.shifted-bins.frozen))),
        population_names=names,area_names=areas,area_weighting='np.average with unrounded official M.N weights in canonical layer order',
        normalization='Unrounded official M.N, matching the modern wrapper; integer simulated counts remain in frozen diagnostics',
        unrounded_population_sum=float(neurons.sum()),actual_simulated_neurons=4129924,
        welch=SETTINGS,welch_segments=1+(bins.nbins-1024)//24,frequency_step_hz=float(freq[1]-freq[0]),
        areas=[dict(area=area,mean_rate_hz=float(rate.mean()),peak_nonzero_frequency_hz=float(freq[1+np.argmax(power[1:])]),peak_nonzero_power=float(power[1:].max())) for area,rate,power in zip(areas,area_rates,powers,strict=True)],
        source_sha256=sources,implementation_sha256={p.name:digest_file(p) for p in [Path(__file__),Path(__file__).with_name('mam_paper_spectrum.py')]},
        analysis_seconds=time.perf_counter()-start,
        scope=f'Separate available-modern-source full-rate/PSD convention on a {bins.nbins/1000:g} s diagnostic. Historical environment, long-duration patterns and scientific equivalence remain unproven; frequency bins and overlapping segments are not independent samples.')
    if bounded:
        result['bounded_memory']=True
        result['implementation_sha256'].update({p.name:digest_file(p) for p in [Path(__file__).with_name('mam_correlation_stream.py'),Path(__file__).with_name('analyze_native_mam_activity.py'),ROOT/'python/brian2_rust/results.py']})
    (a.output/'time-series.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(areas,freq,powers,a.output,observation_seconds=bins.nbins/1000)
    catalog={p.name:dict(bytes=p.stat().st_size,sha256=digest_file(p)) for p in a.output.iterdir() if p.is_file()};(a.output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','population_names','areas']},indent=2))


def plot(areas,freq,power,output,*,observation_seconds=2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,9),layout='constrained')
    for area in ['V1','V2','FEF','MT','MIP']:
        i=areas.index(area);axes[0].semilogy(freq[1:],np.maximum(power[i,1:],1e-20),label=area,lw=1)
    axes[0].set(xlim=(0,150),xlabel='Frequency (Hz)',ylabel='PSD ((Hz/neuron)^2 / Hz)',title='Selected areas; periodic Hann, 1024 ms segments');axes[0].legend(ncol=5);axes[0].grid(alpha=.2)
    im=axes[1].imshow(np.log10(np.maximum(power[:,1:155],1e-20)),aspect='auto',origin='lower',extent=[freq[1],freq[154],-.5,31.5],cmap='magma')
    segments=1+(int(observation_seconds*1000)-1024)//24
    axes[1].set(yticks=range(32),yticklabels=areas,xlabel='Frequency (Hz)',title=f'Log10 PSD; {observation_seconds:g} s view, {segments} strongly overlapping segments');axes[1].tick_params(axis='y',labelsize=7);fig.colorbar(im,ax=axes[1],label='log10 PSD')
    fig.suptitle('Full MAM diagnostic — modern official rate/PSD convention; paper reproduction unproven')
    fig.savefig(output/'spectrum.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['baseline','output','normalization']:p.add_argument('--'+n,type=Path,required=True)
    for n in ['native-audit','model','results']:p.add_argument('--'+n,type=Path)
    p.add_argument('--bounded-memory',action='store_true',help='Release Linux file cache while streaming raw events; required for 100 s.')
    analyze(p.parse_args())
