"""One bounded, explicitly sampled V1 diagnostic on completed primary data.

This does not modify the full-reference six-stage protocol or claim historical
neuron recovery. Run only under the separately admitted Linux resource guard.
"""
import argparse
from contextlib import ExitStack,closing
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import welch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from mam_paper_spectrum import spectrum,SETTINGS

BASE=Path('/data/brick2/brian2-mpi-region-20260907')
RUST_LABEL='full32-n1-k1-spool-primary-v1-100500ms'
NATIVE_LABEL='nest-mam-primary-v1-metastable-seed1729-100500ms'
END=1005000
SAMPLE_SEED=20260910
SAMPLERS=['wrapper_first_eligible','paper_random_eligible']


def check(ok,message):
    if not ok:raise ValueError(message)


class SubsampleCounts:
    def __init__(self,sizes,samples,official_sizes):
        self.sizes=np.asarray(sizes,dtype=np.int64);self.ends=np.cumsum(self.sizes)
        self.samples=np.asarray(samples,dtype=np.int64);self.official=np.asarray(official_sizes,dtype=float)
        check(self.sizes.shape==self.samples.shape==self.official.shape==(8,) and np.all(self.samples>0)
              and np.all(self.sizes>=self.samples) and np.all(self.official>0),'invalid V1 population sizes')
        check(self.samples.sum()==140,'requires exactly 140 selected cells')
        self.counts=np.zeros(self.ends[-1],dtype=np.int64)
        self.frozen=np.zeros((8,100000),dtype=np.int64)
        self.raw=0;self.second_raw=0

    def validate(self,ticks,cells):
        ticks=np.asarray(ticks);cells=np.asarray(cells)
        check(ticks.shape==cells.shape and ticks.ndim==1 and ticks.dtype.kind in 'iu' and cells.dtype.kind in 'iu'
              and len(ticks)<=131072 and np.all(ticks>=0) and np.all(ticks<=END)
              and np.all(cells>=0) and np.all(cells<self.ends[-1]),'invalid bounded V1 events')
        return ticks.astype(np.int64),cells.astype(np.int64)

    def first(self,ticks,cells):
        ticks,cells=self.validate(ticks,cells);self.raw+=len(ticks)
        strict=(ticks>5000)&(ticks<=END);np.add.at(self.counts,cells[strict],1)
        keep=(ticks>=5000)&(ticks<END);groups=np.searchsorted(self.ends,cells[keep],side='right')
        np.add.at(self.frozen,(groups,(ticks[keep]-5000)//10),1)

    def select(self):
        rng=np.random.Generator(np.random.PCG64(SAMPLE_SEED))
        self.selected=np.zeros((2,len(self.counts)),dtype=bool);self.selection=[]
        for i,(start,end,n) in enumerate(zip(np.r_[0,self.ends[:-1]],self.ends,self.samples,strict=True)):
            counts=self.counts[start:end]
            # Exactly100 s: paper excludes rates<.56 Hz; modern wrapper uses>.56.
            candidates=[np.flatnonzero(counts>56),np.flatnonzero(counts>=56)]
            check(all(len(c)>=n for c in candidates),'insufficient eligible cells; no replacement or downsampling')
            ids=[candidates[0][:n],np.sort(rng.choice(candidates[1],size=int(n),replace=False))]
            row=dict(population_index=i,samples=int(n),exact_cutoff_cells=int(np.count_nonzero(counts==56)))
            for j,name in enumerate(SAMPLERS):
                self.selected[j,start+ids[j]]=True
                row[name]=dict(eligible_cells=len(candidates[j]),population_local_ids=ids[j].tolist(),
                               strict_window_spikes=self.counts[start+ids[j]].tolist())
            self.selection.append(row)
        self.second_counts=np.zeros_like(self.selected,dtype=np.int64)
        self.bins=np.zeros((2,8,100000),dtype=np.int64)
        self.lower=np.zeros((2,8),dtype=np.int64)
        check(np.all(self.selected.sum(axis=1)==140),'sample size differs')

    def second(self,ticks,cells):
        ticks,cells=self.validate(ticks,cells);self.second_raw+=len(ticks)
        for j in range(2):
            keep=self.selected[j,cells]&(ticks>5000)&(ticks<=END)
            t=ticks[keep];c=cells[keep];groups=np.searchsorted(self.ends,c,side='right')
            np.add.at(self.second_counts[j],c,1)
            low=t<5005;np.add.at(self.lower[j],groups[low],1)
            np.add.at(self.bins[j],(groups[~low],(t[~low]-5005)//10),1)

    def finish(self):
        check(self.raw==self.second_raw,'V1 source event count changed between passes')
        for j in range(2):
            np.testing.assert_array_equal(self.second_counts[j,self.selected[j]],self.counts[self.selected[j]])
            check(self.bins[j].sum()+self.lower[j].sum()==self.second_counts[j].sum(),'sampled endpoint count identity differs')
        return self.bins


def run(a):
    import mam_correlation_stream as stream
    stream.require_cache_release();started=time.monotonic()
    digest=stream.file_sha
    protocol=json.loads(a.protocol.read_text());check(protocol['schema']=='b2-mam-primary-v1-subsample-protocol-v1','protocol identity')
    check(protocol['sampling_seed']==SAMPLE_SEED and protocol['samplers']==SAMPLERS and protocol['duration_ms']==100500,'sampling protocol differs')
    identity=protocol['inputs'][a.simulator];baseline_dir=BASE/identity['baseline_directory']/'activity'
    sources={};signatures={}
    def verified(path,expected):
        check(path.is_file() and not path.is_symlink() and path.stat().st_size<=128*2**30,'invalid bounded input')
        check(digest(path)==expected,'input hash changed: '+str(path))
        sources[str(path)]=expected;stat=path.stat();signatures[str(path)]=(stat.st_size,stat.st_mtime_ns)
        check(sum(size for size,_ in signatures.values())<=protocol['resources']['max_input_bytes_hashed'],'input hash budget exceeded')
    verified(baseline_dir/'activity.json',identity['activity_sha256'])
    verified(baseline_dir/'catalog.json',identity['catalog_sha256'])
    baseline=json.loads((baseline_dir/'activity.json').read_text());catalog=json.loads((baseline_dir/'catalog.json').read_text())
    check(baseline['window']['start_tick']==5000 and baseline['window']['end_tick']==END
          and baseline['window']['seconds']==100. and baseline['neurons']==4129924,'full primary observation required')
    verified(baseline_dir/'activity-arrays.npz',catalog['activity-arrays.npz']['sha256'])
    verified(a.normalization,protocol['normalization_sha256']);norm=json.loads(a.normalization.read_text())
    names=[p['name'] for p in baseline['populations']];indices=[i for i,n in enumerate(names) if n.startswith('mam_V1_')]
    rows=[baseline['populations'][i] for i in indices];nr={p['name']:p for p in norm['populations']}
    official=np.array([nr[p['name']]['official_normalization_neurons'] for p in rows]);sizes=[p['neurons'] for p in rows]
    samples=np.round(140*official/official.sum()).astype(int)
    check(samples.tolist()==protocol['sample_counts'] and [r['name'] for r in rows]==protocol['population_names'],'sample allocation differs')
    counter=SubsampleCounts(sizes,samples,official)
    with ExitStack() as stack:
        if a.simulator=='rust':
            from brian2_rust.results import load_results
            model_path=BASE/RUST_LABEL/'model.json';result_dir=BASE/'primary-host-v1/runs'/RUST_LABEL
            verified(model_path,baseline['model_sha256']);model=json.loads(model_path.read_text())
            check(model['instance']['rng_seed']==1729 and baseline['window']['spike_tick_offset']==1,'Rust primary identity')
            for name,h in baseline['result_sha256'].items():verified(result_dir/name,h)
            data=load_results(model,result_dir,include_times=False,release_file_cache=True)
            check([p['name'] for p in model['definition']['populations']]==names,'Rust population order')
            release=stack.enter_context(stream.mapped_cache(data['_dump'],result_dir/'results.bin'))
            def blocks():
                for j,i in enumerate(indices):
                    offset=int(np.r_[0,counter.ends[:-1]][j])
                    with closing(stream.rust_blocks(data['populations'][i],release)) as events:
                        for ticks,cells in events:yield ticks,cells.astype(np.int64)+offset
            simulation=dict(simulator='Rust',seed=1729,model_sha256=baseline['model_sha256'])
        else:
            from analyze_native_mam_activity import simulation_identity,canonical_population_name
            raw=BASE/(NATIVE_LABEL+'-audit');summary=raw/'summary.json'
            verified(summary,protocol['native_raw_summary_sha256']);audit=json.loads(summary.read_text())
            check(audit['passed'] and audit['duration_ms']==100500,'native full raw audit')
            parameters=list((raw/'node25').glob('*/parameters.json'));check(len(parameters)==1,'native parameter input')
            verified(parameters[0],baseline['parameters_sha256']);params=json.loads(parameters[0].read_text())
            populations=params['populations'];native_indices=[i for i,p in enumerate(populations) if p['area']=='V1']
            check(native_indices==list(range(native_indices[0],native_indices[0]+8)),'V1 must be contiguous in native IDs')
            check([canonical_population_name(populations[i]) for i in native_indices]==[r['name'] for r in rows],'native V1 order')
            start=sum(p['count'] for p in populations[:native_indices[0]]);end=start+sum(sizes)
            paths=sorted(raw.glob('node*/runs/*/rank*.events.bin'));check(len(paths)==48,'native rank coverage')
            reports=[]
            for path in paths:
                report=json.loads(path.with_name(path.name.replace('.events.bin','.json')).read_text());reports.append(report)
                check(path.stat().st_size==report['event_bytes']==baseline['result_files'][str(path)]['bytes'],'native raw size')
                check(report['event_sha256']==baseline['result_files'][str(path)]['sha256'],'native event identity')
                verified(path,report['event_sha256'])
            simulation=dict(simulator='NEST',**simulation_identity(reports,duration_ms=100500))
            check(simulation['seed']==1729 and baseline['simulation']=={k:v for k,v in simulation.items() if k!='simulator'},'native primary identity')
            def blocks():
                dtype=np.dtype([('tick','<u4'),('cell','<u4')])
                for path in paths:
                    with closing(stream.native_blocks(path,dtype)) as events:
                        for ticks,cells in events:
                            keep=(cells>=start)&(cells<end)
                            if np.any(keep):yield ticks[keep],cells[keep].astype(np.int64)-start
        for ticks,cells in blocks():counter.first(ticks,cells)
        with np.load(baseline_dir/'activity-arrays.npz',allow_pickle=False) as arrays:
            np.testing.assert_array_equal(counter.frozen,arrays['population_counts'][indices])
        counter.select()
        for ticks,cells in blocks():counter.second(ticks,cells)
        bins=counter.finish()
    for name,expected in signatures.items():
        stat=Path(name).stat();check((stat.st_size,stat.st_mtime_ns)==expected,'source changed during analysis')
    output_arrays=dict(population_bin_counts=bins,selected_lower_half_ms_counts=counter.lower)
    metrics=[]
    for j,sampler in enumerate(SAMPLERS):
        pop_rates=bins[j]/(samples[:,None]/1000.)
        for weight_name,weights in [('sample_count',samples),('original_population',official)]:
            rate=np.average(pop_rates,axis=0,weights=weights);frequency,hann=spectrum(rate)
            _,boxcar=welch(rate-rate.mean(),**{**SETTINGS,'window':'boxcar'})
            prefix=sampler+'__'+weight_name
            output_arrays[prefix+'__rates_hz']=rate
            for window,power in [('hann',hann),('boxcar',boxcar)]:
                output_arrays[prefix+'__'+window+'__psd']=power
                metrics.append(dict(sampler=sampler,weighting=weight_name,window=window,mean_hz=float(rate.mean()),
                    nonzero_peak_hz=float(frequency[1+np.argmax(power[1:])]),
                    low_frequency_power_0_to_3_hz=float(np.trapezoid(power[frequency<=3],frequency[frequency<=3])),
                    full_band_power=float(np.trapezoid(power,frequency))))
    output_arrays['frequency_hz']=frequency
    a.output.mkdir(exist_ok=False);np.savez_compressed(a.output/'subsample.npz',**output_arrays)
    report=dict(schema='b2-mam-primary-v1-subsample-v1',simulation=simulation,protocol_sha256=digest(a.protocol),
        raw_v1_events=counter.raw,frozen_full_population_histograms_exact=True,selected_endpoint_identity_exact=True,
        sample_seed=SAMPLE_SEED,random_generator='numpy.Generator(PCG64)',numpy_version=np.__version__,
        selection_window_ms='(500,100500]',histogram_window_ms='[500.5,100500.5]',population_names=[r['name'] for r in rows],
        selection=counter.selection,sample_counts=samples.tolist(),metrics=metrics,
        source_sha256=sources,hashed_input_bytes=sum(size for size,_ in signatures.values()),elapsed_seconds=time.monotonic()-started,
        implementation_sha256={p.name:digest(p) for p in [Path(__file__),ROOT/'tools/mam_paper_spectrum.py',ROOT/'tools/mam_correlation_stream.py']},
        historical_neuron_ids_recovered=False,scientific_acceptance=False,performance_cost_acceptance=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    catalog={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in a.output.iterdir() if p.is_file()}
    (a.output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n');print(json.dumps(dict(output=str(a.output),elapsed_seconds=report['elapsed_seconds'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--simulator',choices=['rust','native'],required=True)
    for name in ['protocol','normalization','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
