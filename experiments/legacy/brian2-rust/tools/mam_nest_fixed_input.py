"""Fixed-input MAM cell semantic probe; no network equivalence claim.

NEST records physical right-edge timestamps. Brian start-slot state samples
already label physical time, while threshold spikes label the integration
step's left edge. The comparison maps only Brian spikes to (tick+1)*dt;
it never searches for a best-fitting lag.
Both the raw timestamps and mapped comparison are retained.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

DT_MS = .1
DURATION_MS = 100.
INITIAL_MV = [-60., -65., -65., -65., -49., -65.]
DC_PA = [0., 250., 600., 0., 0., 450.]
EX_TIMES_MS = [1., 2., 3., 8., 12., 20., 21., 22., 50.]
IN_TIMES_MS = [4., 9., 23., 51.]
# Source 0 = excitatory input train; source 1 = inhibitory input train.
EDGES = [(0,3,100.,.1),(1,3,-80.,.3),(0,5,500.,1.),(1,5,-300.,1.5)]
# Cross-rank excitation/inhibition, a duplicate contact and an autapse.
RECURRENT_EDGES = [(2,3,8000.,.1),(3,2,-4000.,.3),(5,0,1200.,1.5),
                   (0,5,-800.,.1),(2,2,150.,.2),(2,3,2000.,.1)]
CELL = dict(E_L=-65.,V_th=-50.,V_reset=-65.,C_m=250.,tau_m=10.,
            tau_syn_ex=.5,tau_syn_in=.5,t_ref=2.)


def nest_run(output, *, recurrent=False):
    import nest
    nest.ResetKernel()
    nest.SetKernelStatus(dict(resolution=DT_MS,local_num_threads=1,rng_seed=1729))
    cells=nest.Create('iaf_psc_exp',len(INITIAL_MV),params=CELL)
    cells.set(V_m=INITIAL_MV,I_e=DC_PA)
    inputs=[nest.Create('spike_generator',params=dict(spike_times=times))
            for times in [EX_TIMES_MS,IN_TIMES_MS]]
    for source,target,weight,delay in EDGES:
        nest.Connect(inputs[source],cells[target:target+1],
                     syn_spec=dict(weight=weight,delay=delay))
    if recurrent:
        for source,target,weight,delay in RECURRENT_EDGES:
            nest.Connect(cells[source:source+1],cells[target:target+1],
                         syn_spec=dict(weight=weight,delay=delay))
    meter=nest.Create('multimeter',params=dict(interval=DT_MS,
        record_from=['V_m','I_syn_ex','I_syn_in']))
    recorder=nest.Create('spike_recorder')
    nest.Connect(meter,cells);nest.Connect(cells,recorder)
    nest.Simulate(DURATION_MS)
    events=meter.get('events');spikes=recorder.get('events')
    first=int(cells.tolist()[0]);ids=np.asarray(events['senders'])-first
    times=np.asarray(events['times']);order=np.lexsort((ids,times))
    assert len(times)%len(cells)==0
    np.savez(output/'trace.npz',times_ms=times[order].reshape(-1,len(cells))[:,0],
        voltage_mv=np.asarray(events['V_m'])[order].reshape(-1,len(cells)),
        current_pa=(np.asarray(events['I_syn_ex'])+events['I_syn_in'])[order].reshape(-1,len(cells)),
        spike_times_ms=spikes['times'],spike_cells=np.asarray(spikes['senders'])-first,
        final_voltage_mv=cells.get('V_m'))
    return dict(simulator='NEST',version=nest.__version__,cell=CELL,
        recordables=nest.GetDefaults('iaf_psc_exp')['recordables'],recurrent=recurrent)


def brian_network(b, *, nest_grid=False, recurrent=False):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
    from brian2_rust.multi_area_semantics import nest_grid_refractory_ms
    clock=b.Clock(dt=DT_MS*b.ms)
    cells=b.NeuronGroup(len(INITIAL_MV),
        'dv/dt=(-65*mV-v)/(10*ms)+(current+dc)/(250*pF):volt (unless refractory)\n'
        'dcurrent/dt=-current/(.5*ms):amp\ndc:amp (constant)',
        threshold='v>=-50*mV',reset='v=-65*mV',refractory=(nest_grid_refractory_ms(CELL['t_ref'],DT_MS) if nest_grid else CELL['t_ref'])*b.ms,
        clock=clock,name='probe_cells')
    cells.v=np.array(INITIAL_MV)*b.mV;cells.dc=np.array(DC_PA)*b.pA
    events=sorted([(t,0) for t in EX_TIMES_MS]+[(t,1) for t in IN_TIMES_MS])
    inputs=b.SpikeGeneratorGroup(2,[i for t,i in events],
        (np.array([t for t,i in events])-(DT_MS if nest_grid else 0))*b.ms,clock=clock,name='probe_inputs')
    syn=b.Synapses(inputs,cells,'w:amp (constant)',on_pre='current_post+=w',
        clock=clock,name='probe_synapses')
    syn.connect(i=[s for s,t,w,d in EDGES],j=[t for s,t,w,d in EDGES])
    syn.w=np.array([w for s,t,w,d in EDGES])*b.pA
    syn.delay=np.array([d for s,t,w,d in EDGES])*b.ms
    extra=[]
    if recurrent:
        rec=b.Synapses(cells,cells,'w:amp (constant)',on_pre='current_post+=w',
                       clock=clock,name='probe_recurrent')
        rec.connect(i=[s for s,t,w,d in RECURRENT_EDGES],j=[t for s,t,w,d in RECURRENT_EDGES])
        rec.w=np.array([w for s,t,w,d in RECURRENT_EDGES])*b.pA
        rec.delay=np.array([d for s,t,w,d in RECURRENT_EDGES])*b.ms
        extra.append(rec)
    meter=b.StateMonitor(cells,['v','current'],record=True,when='start',name='probe_meter')
    spikes=b.SpikeMonitor(cells,name='probe_spikes')
    return b.Network(cells,inputs,syn,meter,spikes,*extra),cells,meter,spikes


def brian_run(output, *, runner=None, mpi=False, nest_grid=False, recurrent=False):
    import brian2 as b
    if not mpi:
        b.set_device('runtime');b.prefs.codegen.target='numpy'
        net,cells,meter,spikes=brian_network(b,nest_grid=nest_grid,recurrent=recurrent);net.run(DURATION_MS*b.ms)
        np.savez(output/'trace.npz',times_ms=np.asarray(meter.t/b.ms),
            voltage_mv=np.asarray(meter.v/b.mV).T,current_pa=np.asarray(meter.current/b.pA).T,
            spike_times_ms=np.asarray(spikes.t/b.ms),spike_cells=np.asarray(spikes.i),
            final_voltage_mv=np.asarray(cells.v/b.mV))
        return dict(simulator='Brian2 numpy',version=b.__version__,nest_grid=nest_grid,recurrent=recurrent)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
    import brian2_rust
    from brian2_rust.export import lower_network
    from brian2_rust.distributed import write_mpi_project,compile_mpi_project,run_mpi_project
    from brian2_rust.results import load_results
    import subprocess
    b.set_device('rust_standalone',runner=runner)
    net,_,_,_=brian_network(b,nest_grid=nest_grid,recurrent=recurrent);model=lower_network(net,DURATION_MS*b.ms,rng_seed=1729)
    (output/'model.json').write_text(json.dumps(model)+'\n')
    subprocess.run([str(runner),str(output/'model.json'),str(output/'reference')],check=True,timeout=30)
    write_mpi_project(model,output/'mpi',ranks=2,runner=runner)
    compile_mpi_project(output/'mpi',opt_level=1,panic_strategy='abort')
    report=run_mpi_project(output/'mpi',output/'result',timeout=30)
    for name in ['results.bin','events.bin']:
        assert (output/'reference'/name).read_bytes()==(output/'result'/name).read_bytes()
    result=load_results(model,output/'result')
    p=next(p for d,p in zip(model['definition']['populations'],result['populations'],strict=True)
           if d['name']=='probe_cells')
    np.savez(output/'trace.npz',times_ms=p['times']*1000,voltage_mv=p['trace']['v']*1000,
        current_pa=p['trace']['current']*1e12,spike_times_ms=p['spike_ticks']*DT_MS,
        spike_cells=p['indices'],final_voltage_mv=p['states']['v']*1000)
    return dict(simulator='Rust MPI',brian_version=b.__version__,ranks=report['ranks'],
        internal_reference_exact_bytes=True,plan_sha256=report['plan_sha256'],nest_grid=nest_grid,recurrent=recurrent)


def compare(root, *, require_match=False):
    data={name:dict(np.load(root/name/'trace.npz')) for name in ['nest','brian','mpi']}
    n=data['nest'];rows=[]
    for name in ['brian','mpi']:
        b=data[name];ticks=np.rint(n['times_ms']/DT_MS).astype(int)
        bticks=np.rint(b['times_ms']/DT_MS).astype(int)
        assert len(set(bticks))==len(bticks) and np.all(np.isin(ticks,bticks))
        positions=np.searchsorted(bticks,ticks)
        for cell in range(len(INITIAL_MV)):
            raw=b['spike_times_ms'][b['spike_cells']==cell]
            mapped=np.rint(raw/DT_MS).astype(int)+1
            nest_ticks=np.rint(n['spike_times_ms'][n['spike_cells']==cell]/DT_MS).astype(int)
            rows.append(dict(simulator=name,cell=cell,
                initial_mv=INITIAL_MV[cell],dc_pa=DC_PA[cell],
                max_voltage_difference_mv=float(np.max(np.abs(b['voltage_mv'][positions,cell]-n['voltage_mv'][:,cell]))),
                max_current_difference_pa=float(np.max(np.abs(b['current_pa'][positions,cell]-n['current_pa'][:,cell]))),
                final_voltage_difference_mv=float(abs(b['final_voltage_mv'][cell]-n['final_voltage_mv'][cell])),
                nest_spike_ticks=nest_ticks.tolist(),brian_raw_spike_times_ms=raw.tolist(),
                right_edge_spike_ticks=mapped.tolist(),spike_ticks_match=bool(np.array_equal(mapped,nest_ticks))))
    # The external comparison is diagnostic, and must preserve disagreements.
    result=dict(scope='Six fixed-input cells, optionally with explicit recurrent edges. Not MAM network or stochastic equivalence.',duration_ms=DURATION_MS,
        provenance={name:json.loads((root/name/'provenance.json').read_text()) for name in data},
        timestamp_mapping='State samples compare at physical time; Brian threshold spike tick + 1 -> physical right edge. Optional nest_grid translates supplied physical generator times to left-edge labels and adds one clock tick to Brian refractory duration; no lag search.',
        cell=CELL,initial_mv=INITIAL_MV,dc_pa=DC_PA,edges=EDGES,
        recurrent_edges=RECURRENT_EDGES if json.loads((root/'nest/provenance.json').read_text()).get('recurrent') else [],
        rows=rows,all_mapped_spikes_match=all(r['spike_ticks_match'] for r in rows),
        brian_mpi_max_voltage_difference_mv=float(np.max(np.abs(data['brian']['voltage_mv']-data['mpi']['voltage_mv']))))
    # Absolute round-off bounds for this deterministic fixture, not fitted
    # biological/statistical equivalence margins.
    result['numerical_limits']=dict(voltage_mv=1e-9,current_pa=1e-8)
    result['deterministic_match']=result['all_mapped_spikes_match'] and all(
        r['max_voltage_difference_mv']<1e-9 and r['max_current_difference_pa']<1e-8 and r['final_voltage_difference_mv']<1e-9 for r in rows)
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if require_match and not result['deterministic_match']:
        raise SystemExit('Fixed-input comparison failed; raw traces retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['nest','brian','mpi','compare'])
    parser.add_argument('output',type=Path)
    parser.add_argument('--runner',type=Path)
    parser.add_argument('--nest-grid',action='store_true',help='Explicit NEST grid scheduling translation; retain raw Brian default as the diagnostic control')
    parser.add_argument('--recurrent',action='store_true')
    parser.add_argument('--require-match',action='store_true')
    parser.add_argument('--duration-ms',type=float,default=DURATION_MS)
    args=parser.parse_args()
    if not 1 <= args.duration_ms <= 2500 or not np.isclose(args.duration_ms/DT_MS,round(args.duration_ms/DT_MS),rtol=0,atol=1e-9):
        parser.error('duration must be grid-aligned and between 1 and 2500 ms')
    DURATION_MS=args.duration_ms
    if args.mode=='compare':
        compare(args.output,require_match=args.require_match)
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        info=nest_run(args.output,recurrent=args.recurrent) if args.mode=='nest' else brian_run(args.output,
            runner=args.runner,mpi=args.mode=='mpi',nest_grid=args.nest_grid,recurrent=args.recurrent)
        (args.output/'provenance.json').write_text(json.dumps(info,indent=2)+'\n')
