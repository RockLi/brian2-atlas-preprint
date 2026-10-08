"""Freeze a NEST-realized scaled V1/V2 graph and replay identical inputs.

This is a deterministic cross-simulator validation fixture, not a claim of
full-model reproduction. External and omitted-area input follows the adapter's
homogeneous Poisson replacement, frozen into unit-multiplicity input lanes.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

DT=.1
STEPS=1000
SEED=20260908


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare(parameters,out):
    import nest
    p=json.loads(parameters.read_text())
    assert p['source_commit']=='0a658be40bef3249cbe452f38809edf7d2f524ba'
    assert p['N_scaling']==p['K_scaling']==.01
    selected={i:q for i,q in enumerate(p['populations']) if q['area'] in ['V1','V2']}
    n=sum(q['count'] for q in selected.values());assert n<=5000
    projections=[q for q in p['projections'] if q['source'] in selected and q['target'] in selected]
    assert sum(q['count'] for q in projections)<=200000
    nest.ResetKernel();nest.SetKernelStatus(dict(resolution=DT,local_num_threads=1,rng_seed=SEED))
    cell=p['params']['neuron_params']['single_neuron_dict'];cells=nest.Create('iaf_psc_exp',n,params=cell)
    offsets={};position=0;populations=[]
    for i,q in selected.items():
        offsets[i]=(position,position+q['count']);populations.append(dict(q,start=position,end=position+q['count']));position+=q['count']
    for q in projections:
        a,z=offsets[q['source']];c,d=offsets[q['target']]
        weight=nest.math.redraw(nest.random.normal(mean=q['weight_mean_pA'],std=q['weight_sd_pA']),
                               min=0. if q['excitatory'] else -np.inf,max=np.inf if q['excitatory'] else 0.)
        delay=nest.math.redraw(nest.random.normal(mean=q['delay_mean_ms'],std=q['delay_sd_ms']),min=DT,max=np.inf)
        nest.Connect(cells[a:z],cells[c:d],dict(rule='fixed_total_number',N=q['count']),
                     dict(weight=weight,delay=delay))
    connections=nest.GetConnections(source=cells,target=cells).get(['source','target','weight','delay'])
    first=int(cells.tolist()[0]);source=np.asarray(connections['source'],dtype=np.int64)-first
    target=np.asarray(connections['target'],dtype=np.int64)-first
    weight=np.asarray(connections['weight']);delay=np.rint(np.asarray(connections['delay'])/DT).astype(np.int64)
    assert len(source)==sum(q['count'] for q in projections)
    order=np.argsort(source,kind='stable');source,target,weight,delay=[a[order] for a in [source,target,weight,delay]]
    rng=np.random.default_rng(SEED)
    init=p['params']['neuron_params'];voltage=rng.normal(init['V0_mean'],init['V0_sd'],n)
    dc=np.concatenate([np.full(q['count'],q['dc_pA']) for q in selected.values()])
    input_target=[];input_weight=[];input_delay=[];event_source=[];event_tick=[];components=[];lane_count=0
    for i,q in selected.items():
        inputs=defaultdict(float);inputs[(q['external_weight_pA'],10)]+=q['external_indegree']*10.
        for projection in p['projections']:
            if projection['target']==i and projection['source'] not in selected:
                inputs[(projection['weight_mean_pA'],int(np.floor(projection['delay_mean_ms']/DT+.5)))]+=projection['indegree']*10.
        start,end=offsets[i]
        for (w,d),rate in sorted(inputs.items()):
            if rate<=0:continue
            # NEST poisson_generator start=0 first emits at physical tick 2.
            counts=rng.poisson(rate*DT*.001,size=(STEPS-2,end-start))
            maximum=int(counts.max());assert maximum<=32
            components.append(dict(population=q['name'],weight_pa=w,delay_ticks=d,rate_hz=rate,lanes=maximum))
            for k in range(maximum):
                assert lane_count+(end-start)<=100000
                tick,local=np.nonzero(counts>k)
                event_source.append(local+lane_count);event_tick.append(tick+2)
                input_target.extend(range(start,end));input_weight.extend([w]*(end-start));input_delay.extend([d]*(end-start))
                lane_count+=end-start
    event_source=np.concatenate(event_source);event_tick=np.concatenate(event_tick)
    assert len(event_source)<=2000000
    order=np.lexsort((event_source,event_tick));event_source,event_tick=event_source[order],event_tick[order]
    np.savez(out/'fixture.npz',source=source,target=target,weight_pa=weight,delay_ticks=delay,
        voltage_mv=voltage,dc_pa=dc,input_target=np.array(input_target,dtype=np.int64),
        input_weight_pa=np.array(input_weight),input_delay_ticks=np.array(input_delay,dtype=np.int64),
        event_source=event_source,event_tick=event_tick)
    pairs=source*n+target
    report=dict(scope='NEST-realized fixed graph and frozen Poisson inputs for the scaled V1/V2 adapter; not paper-level activity reproduction.',
        parameters_sha256=digest(parameters),fixture_sha256=digest(out/'fixture.npz'),nest_version=nest.__version__,
        seed=SEED,dt_ms=DT,steps=STEPS,N_scaling=.01,K_scaling=.01,cell=cell,populations=populations,
        neurons=n,edges=len(source),projections=len(projections),autapses=int(np.sum(source==target)),
        duplicate_pairs=len(pairs)-len(np.unique(pairs)),input_lanes=lane_count,input_events=len(event_source),
        input_components=components,weight_range_pa=[float(weight.min()),float(weight.max())],
        delay_range_ticks=[int(delay.min()),int(delay.max())])
    (out/'fixture.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['populations','input_components']},indent=2))


def load(fixture):
    meta=json.loads((fixture/'fixture.json').read_text());assert digest(fixture/'fixture.npz')==meta['fixture_sha256']
    return meta,dict(np.load(fixture/'fixture.npz'))


def nest_run(fixture,out):
    import nest
    m,a=load(fixture);n=m['neurons']
    nest.ResetKernel();nest.SetKernelStatus(dict(resolution=DT,local_num_threads=1,rng_seed=SEED))
    cells=nest.Create('iaf_psc_exp',n,params=m['cell']);cells.set(V_m=a['voltage_mv'],I_e=a['dc_pa'])
    gids=np.asarray(cells.tolist(),dtype=np.int64)
    nest.Connect(gids[a['source']],gids[a['target']],'one_to_one',dict(weight=a['weight_pa'],delay=a['delay_ticks']*DT))
    inputs=nest.Create('spike_generator',m['input_lanes']);input_gids=np.asarray(inputs.tolist(),dtype=np.int64)
    order=np.argsort(a['event_source'],kind='stable');ids=a['event_source'][order];ticks=a['event_tick'][order]
    offsets=np.searchsorted(ids,np.arange(m['input_lanes']+1))
    nest.SetStatus(inputs,[dict(spike_times=ticks[offsets[i]:offsets[i+1]]*DT) for i in range(m['input_lanes'])])
    nest.Connect(input_gids,gids[a['input_target']],'one_to_one',dict(weight=a['input_weight_pa'],delay=a['input_delay_ticks']*DT))
    meter=nest.Create('multimeter',params=dict(interval=DT,record_from=['V_m','I_syn_ex','I_syn_in']))
    recorder=nest.Create('spike_recorder');nest.Connect(meter,cells,syn_spec=dict(delay=DT));nest.Connect(cells,recorder,syn_spec=dict(delay=DT))
    nest.Simulate(STEPS*DT)
    events=meter.get('events');spikes=recorder.get('events');times=np.asarray(events['times'])
    ids=np.asarray(events['senders'])-gids[0];order=np.lexsort((ids,times))
    np.savez(out/'trace.npz',times_ms=times[order].reshape(-1,n)[:,0],
        voltage_mv=np.asarray(events['V_m'])[order].reshape(-1,n),
        current_pa=(np.asarray(events['I_syn_ex'])+events['I_syn_in'])[order].reshape(-1,n),
        spike_tick=np.rint(np.asarray(spikes['times'])/DT).astype(np.int64),spike_cell=np.asarray(spikes['senders'])-gids[0],
        final_voltage_mv=cells.get('V_m'))
    return dict(simulator='NEST',version=nest.__version__,fixture_sha256=m['fixture_sha256'])


def brian_run(fixture,out,runner,ranks):
    import brian2 as b
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
    import brian2_rust
    from brian2_rust.multi_area_semantics import nest_grid_refractory_ms
    from brian2_rust.export import lower_network
    from brian2_rust.distributed import write_mpi_project,compile_mpi_project,run_mpi_project
    from brian2_rust.results import load_results
    m,a=load(fixture);n=m['neurons']
    if ranks:b.set_device('rust_standalone',runner=runner)
    else:b.set_device('runtime');b.prefs.codegen.target='numpy'
    clock=b.Clock(dt=DT*b.ms)
    cells=b.NeuronGroup(n,'dv/dt=(-65*mV-v)/(10*ms)+(current+dc)/(250*pF):volt (unless refractory)\ndcurrent/dt=-current/(.5*ms):amp\ndc:amp (constant)',
        threshold='v>=-50*mV',reset='v=-65*mV',refractory=nest_grid_refractory_ms(2.,DT)*b.ms,clock=clock,name='frozen_cells')
    cells.v=a['voltage_mv']*b.mV;cells.dc=a['dc_pa']*b.pA
    inputs=b.SpikeGeneratorGroup(m['input_lanes'],a['event_source'],(a['event_tick']-1)*DT*b.ms,clock=clock,name='frozen_inputs')
    recurrent=b.Synapses(cells,cells,'w:amp (constant)',on_pre='current_post+=w',clock=clock,name='frozen_recurrent')
    recurrent.connect(i=a['source'],j=a['target']);recurrent.w=a['weight_pa']*b.pA;recurrent.delay=a['delay_ticks']*DT*b.ms
    external=b.Synapses(inputs,cells,'w:amp (constant)',on_pre='current_post+=w',clock=clock,name='frozen_external')
    external.connect(i=np.arange(m['input_lanes']),j=a['input_target']);external.w=a['input_weight_pa']*b.pA;external.delay=a['input_delay_ticks']*DT*b.ms
    meter=b.StateMonitor(cells,['v','current'],record=True,when='start',name='frozen_meter');spikes=b.SpikeMonitor(cells,name='frozen_spikes')
    net=b.Network(cells,inputs,recurrent,external,meter,spikes)
    if not ranks:
        net.run(STEPS*DT*b.ms)
        np.savez(out/'trace.npz',times_ms=np.asarray(meter.t/b.ms),voltage_mv=np.asarray(meter.v/b.mV).T,
            current_pa=np.asarray(meter.current/b.pA).T,spike_tick=np.rint(np.asarray(spikes.t/b.ms)/DT).astype(np.int64)+1,
            spike_cell=np.asarray(spikes.i),final_voltage_mv=np.asarray(cells.v/b.mV))
        return dict(simulator='Brian2 numpy',version=b.__version__,fixture_sha256=m['fixture_sha256'])
    model=lower_network(net,STEPS*DT*b.ms,rng_seed=SEED);(out/'model.json').write_text(json.dumps(model)+'\n')
    subprocess.run([str(runner),str(out/'model.json'),str(out/'reference')],check=True,timeout=90)
    write_mpi_project(model,out/'mpi',ranks=ranks,runner=runner);compile_mpi_project(out/'mpi',opt_level=1,panic_strategy='abort')
    report=run_mpi_project(out/'mpi',out/'result',timeout=90)
    for name in ['results.bin','events.bin']:assert digest(out/'reference'/name)==digest(out/'result'/name)
    data=load_results(model,out/'result');population=next(p for d,p in zip(model['definition']['populations'],data['populations'],strict=True) if d['name']=='frozen_cells')
    np.savez(out/'trace.npz',times_ms=population['times']*1000,voltage_mv=population['trace']['v']*1000,
        current_pa=population['trace']['current']*1e12,spike_tick=population['spike_ticks']+1,
        spike_cell=population['indices'],final_voltage_mv=population['states']['v']*1000)
    return dict(simulator='Rust MPI',ranks=ranks,fixture_sha256=m['fixture_sha256'],internal_reference_exact=True,plan_sha256=report['plan_sha256'])


def compare(fixture,out):
    m,_=load(fixture);data={name:dict(np.load(out/name/'trace.npz')) for name in ['nest','brian','mpi2','mpi4']}
    end=min(int(np.rint(d['times_ms'][-1]/DT)) for d in data.values());rows=[]
    def spikes(d):
        mask=d['spike_tick']<=end;order=np.lexsort((d['spike_cell'][mask],d['spike_tick'][mask]))
        return np.column_stack([d['spike_tick'][mask],d['spike_cell'][mask]])[order]
    expected=spikes(data['nest'])
    for name in ['brian','mpi2','mpi4']:
        d=data[name];nticks=np.rint(data['nest']['times_ms']/DT).astype(int);mask=nticks<=end;nticks=nticks[mask]
        ticks=np.rint(d['times_ms']/DT).astype(int);assert np.all(np.isin(nticks,ticks));positions=np.searchsorted(ticks,nticks)
        voltage=np.abs(d['voltage_mv'][positions]-data['nest']['voltage_mv'][mask]);current=np.abs(d['current_pa'][positions]-data['nest']['current_pa'][mask])
        observed=spikes(d);match=np.array_equal(expected,observed)
        row=dict(simulator=name,spikes=len(observed),spike_ticks_match=match,max_voltage_error_mv=float(voltage.max()),
            max_current_error_pa=float(current.max()),max_final_voltage_error_mv=float(np.max(np.abs(d['final_voltage_mv']-data['nest']['final_voltage_mv']))))
        row['passed']=match and row['max_voltage_error_mv']<1e-8 and row['max_current_error_pa']<1e-7 and row['max_final_voltage_error_mv']<1e-8
        rows.append(row)
    result=dict(fixture_sha256=m['fixture_sha256'],neurons=m['neurons'],edges=m['edges'],nest_spikes=len(expected),
        comparison_end_ms=end*DT,rows=rows,passed=all(r['passed'] for r in rows),
        limits=dict(voltage_mv=1e-8,current_pa=1e-7,spike_ticks='exact after fixed right-edge mapping'),
        scope='Deterministic frozen scaled V1/V2 graph replay; not procedural graph RNG equivalence or full-model statistical reproduction.')
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit('Frozen-network comparison failed; raw traces retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','nest','brian','mpi2','mpi4','compare'])
    parser.add_argument('--fixture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parameters',type=Path);parser.add_argument('--runner',type=Path)
    args=parser.parse_args()
    if args.mode=='compare':compare(args.fixture,args.output)
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        if args.mode=='prepare':prepare(args.parameters,args.output)
        else:
            info=nest_run(args.fixture,args.output) if args.mode=='nest' else brian_run(args.fixture,args.output,args.runner,int(args.mode[-1]) if args.mode.startswith('mpi') else 0)
            (args.output/'provenance.json').write_text(json.dumps(info,indent=2)+'\n')
