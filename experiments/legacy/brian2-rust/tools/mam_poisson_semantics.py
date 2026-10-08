"""Bounded NEST/Brian/MPI audit of MAM superposed Poisson current input.

Recover integer arrivals from exponential current traces. Compare each
simulator with the analytical Poisson law, never identical cross-simulator RNG
streams. Limits and seed schedule are fixed before collecting observations.
"""
import argparse
import json
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

DT_MS=.1
DURATION_MS=100.
TAU_MS=.5
WEIGHT_PA=73.24
DELAY_TICKS=10
LAMBDAS=[.2,2.,20.]
CELLS_PER_RATE=128
SEEDS=[1729,1730,1731]


def nest_run(out,seed):
    import nest
    nest.ResetKernel()
    nest.SetKernelStatus(dict(resolution=DT_MS,local_num_threads=1,rng_seed=seed))
    cells=nest.Create('iaf_psc_exp',CELLS_PER_RATE*len(LAMBDAS),params=dict(
        E_L=-65.,V_m=-65.,V_th=-50.,V_reset=-65.,C_m=250.,tau_m=10.,
        tau_syn_ex=TAU_MS,tau_syn_in=TAU_MS,t_ref=2.))
    for index,lam in enumerate(LAMBDAS):
        generator=nest.Create('poisson_generator',params=dict(rate=lam/(DT_MS*.001)))
        # Match the official model: default static delay, one generator per
        # population with a distinct train for each target.
        nest.Connect(generator,cells[index*CELLS_PER_RATE:(index+1)*CELLS_PER_RATE],
                     syn_spec=dict(weight=WEIGHT_PA))
    meter=nest.Create('multimeter',params=dict(interval=DT_MS,record_from=['I_syn_ex']))
    nest.Connect(meter,cells);nest.Simulate(DURATION_MS)
    events=meter.get('events');times=np.asarray(events['times'])
    ids=np.asarray(events['senders'])-int(cells.tolist()[0]);order=np.lexsort((ids,times))
    np.savez(out/'trace.npz',times_ms=times[order].reshape(-1,len(cells))[:,0],
             current_pa=np.asarray(events['I_syn_ex'])[order].reshape(-1,len(cells)))
    return dict(simulator='NEST',version=nest.__version__,seed=seed,
                default_static_delay_ms=nest.GetDefaults('static_synapse')['delay'])


def brian_run(out,seed,runner,mpi,nest_poisson_start=False):
    import brian2 as b
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
    import brian2_rust
    from brian2_rust.multi_area_semantics import nest_grid_refractory_ms,nest_poisson_gate_tick
    from brian2_rust.export import lower_network
    from brian2_rust.distributed import write_mpi_project,compile_mpi_project,run_mpi_project
    from brian2_rust.results import load_results
    if mpi:b.set_device('rust_standalone',runner=runner)
    else:b.set_device('runtime');b.prefs.codegen.target='numpy';b.seed(seed)
    clock=b.Clock(dt=DT_MS*b.ms)
    cells=b.NeuronGroup(CELLS_PER_RATE*len(LAMBDAS),
        'dv/dt=(-65*mV-v)/(10*ms)+current/(250*pF):volt (unless refractory)\n'
        'dcurrent/dt=-current/(.5*ms):amp\nlam:1 (constant)',
        threshold='v>=-50*mV',reset='v=-65*mV',
        refractory=nest_grid_refractory_ms(2.,DT_MS)*b.ms,clock=clock,name='poisson_cells')
    cells.v=-65*b.mV;cells.lam=np.repeat(LAMBDAS,CELLS_PER_RATE)
    gate=nest_poisson_gate_tick(DELAY_TICKS) if nest_poisson_start else DELAY_TICKS
    cells.run_regularly(f'current += int(timestep(t, dt)>={gate})*({WEIGHT_PA!r}*pA)*poisson(lam)',when='synapses')
    meter=b.StateMonitor(cells,'current',record=True,when='start',name='poisson_meter')
    net=b.Network(cells,meter)
    if not mpi:
        net.run(DURATION_MS*b.ms)
        np.savez(out/'trace.npz',times_ms=np.asarray(meter.t/b.ms),current_pa=np.asarray(meter.current/b.pA).T)
        return dict(simulator='Brian2 numpy',version=b.__version__,seed=seed,nest_poisson_start=nest_poisson_start)
    model=lower_network(net,DURATION_MS*b.ms,rng_seed=seed)
    (out/'model.json').write_text(json.dumps(model)+'\n')
    subprocess.run([str(runner),str(out/'model.json'),str(out/'reference')],check=True,timeout=30)
    write_mpi_project(model,out/'mpi',ranks=2,runner=runner)
    compile_mpi_project(out/'mpi',opt_level=1,panic_strategy='abort')
    report=run_mpi_project(out/'mpi',out/'result',timeout=30)
    for name in ['results.bin','events.bin']:
        assert (out/'reference'/name).read_bytes()==(out/'result'/name).read_bytes()
    population=load_results(model,out/'result')['populations'][0]
    np.savez(out/'trace.npz',times_ms=population['times']*1000,
             current_pa=population['trace']['current']*1e12)
    return dict(simulator='Rust MPI',brian_version=b.__version__,seed=seed,
                internal_reference_exact_bytes=True,ranks=report['ranks'],nest_poisson_start=nest_poisson_start)


def analyze(out):
    rows=[]
    for seed in SEEDS:
        last_ticks={name:int(np.rint(np.load(out/str(seed)/name/'trace.npz')['times_ms'][-1]/DT_MS)) for name in ['nest','brian','mpi']}
        common_end=min(last_ticks.values())
        assert 900<=common_end<=999
        for simulator in ['nest','brian','mpi']:
            directory=out/str(seed)/simulator
            data=np.load(directory/'trace.npz');times=np.rint(data['times_ms']/DT_MS).astype(int)
            currents=data['current_pa'];assert currents.shape==(len(times),CELLS_PER_RATE*len(LAMBDAS))
            assert np.isfinite(currents).all() and np.array_equal(times,np.arange(times[0],times[-1]+1))
            if times[0]==1:
                times=np.insert(times,0,0);currents=np.vstack([np.zeros(currents.shape[1]),currents])
            assert times[0]==0
            currents=currents[times<=common_end];times=times[times<=common_end]
            raw=(currents[1:]-math.exp(-DT_MS/TAU_MS)*currents[:-1])/WEIGHT_PA
            counts=np.rint(raw).astype(np.int64)
            residual=float(np.max(np.abs(raw-counts)))
            assert residual<1e-8 and np.all(counts>=0)
            for index,lam in enumerate(LAMBDAS):
                all_counts=counts[:,index*CELLS_PER_RATE:(index+1)*CELLS_PER_RATE]
                active_ticks=times[1:][np.any(all_counts>0,axis=1)]
                first=int(active_ticks[0])
                # Preserve first-arrival disagreement as a failed check, then
                # still report the predeclared probability diagnostics.
                sample=all_counts[DELAY_TICKS+1:];n=sample.size
                mean=float(sample.mean());variance=float(sample.var(ddof=1))
                mean_z=(mean-lam)/math.sqrt(lam/n)
                variance_z=(variance-lam)/math.sqrt((lam+2*lam*lam)/(n-1))
                centered=sample-lam
                cross_z=float(np.mean(centered[:,:-1]*centered[:,1:]))*math.sqrt(sample.shape[0]*(sample.shape[1]-1))/lam
                temporal_z=float(np.mean(centered[:-1]*centered[1:]))*math.sqrt((sample.shape[0]-1)*sample.shape[1])/lam
                probabilities=[math.exp(-lam),lam*math.exp(-lam),1-(1+lam)*math.exp(-lam)]
                measured=[float(np.mean(sample==0)),float(np.mean(sample==1)),float(np.mean(sample>=2))]
                pmf_z=[(observed-p)/math.sqrt(p*(1-p)/n) for observed,p in zip(measured,probabilities,strict=True)]
                zscores=[mean_z,variance_z,cross_z,temporal_z,*pmf_z]
                rows.append(dict(seed=seed,simulator=simulator,lambda_per_bin=lam,samples=n,
                    original_last_tick=last_ticks[simulator],common_last_tick=common_end,
                    first_arrival_tick=first,integer_reconstruction_error=residual,mean=mean,variance=variance,
                    mean_z=mean_z,variance_z=variance_z,adjacent_target_covariance_z=cross_z,
                    lag_one_covariance_z=temporal_z,pmf_0_1_ge2=measured,pmf_z=pmf_z,
                    passed=first==DELAY_TICKS+2 and not np.any(all_counts[:DELAY_TICKS+1]) and all(abs(z)<6 for z in zscores)))
    result=dict(scope='Poisson arrival-bin and marginal/independence smoke audit, not whole-model equivalence or an exhaustive RNG test.',
        dt_ms=DT_MS,duration_ms=DURATION_MS,delay_ticks=DELAY_TICKS,seeds=SEEDS,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cells_per_rate=CELLS_PER_RATE,lambdas=LAMBDAS,
        acceptance='Absolute standardized discrepancies <6 for mean, variance, PMF(0,1,>=2), adjacent-target and lag-one covariance; integer residual <1e-8. Source-audited first arrival physical tick 12 and post-start statistics tick 12 onward. The v1/v2 expected tick 11 was contradicted and remains archived; numerical/6-sigma limits are unchanged.',
        rows=rows,passed=all(r['passed'] for r in rows))
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit('Poisson audit failed; all raw traces retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['nest','brian','mpi','analyze']);parser.add_argument('output',type=Path)
    parser.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0]);parser.add_argument('--runner',type=Path)
    parser.add_argument('--nest-poisson-start',action='store_true')
    args=parser.parse_args()
    if args.mode=='analyze':analyze(args.output)
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        info=nest_run(args.output,args.seed) if args.mode=='nest' else brian_run(args.output,args.seed,args.runner,args.mode=='mpi',args.nest_poisson_start)
        (args.output/'provenance.json').write_text(json.dumps(info,indent=2)+'\n')
