"""Same-cubin geometry sweep; no automatic policy promotion from tuning cases."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import numpy as np
from cuda_cooperative_compare import CASES


def candidate_grids(sms, active, lanes):
    proposals=dict(b1=1,b4=4,b16=16,b32=32,sm=sms,two_sm=2*sms,
                   max_useful=min(sms*active,max(1,(lanes+127)//128)))
    if any(not 1<=n<=sms*active for n in proposals.values()):
        raise ValueError('Declared geometry sweep is unsupported by this kernel/device')
    return proposals


def compare(output):
    from gpu_stdp_compare import brian_run,oracle,checks,configuration
    from gpu_stdp_precompiled import native_arrays,forbid_compilation,artifact_hashes
    from brian2_rust.cuda import CudaExecutor
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='b2-geometry-comparison-v0',cases=[],passed=False,repeats=5,seed=1729,
        scope='one compiled cooperative executor and unchanged resident addresses for all block counts; separate default executor; full reset-to-result timings; grid configuration and validation outside measured interval')
    (output/'declared-protocol.json').write_text(json.dumps(dict(cases=CASES,steps=1024,repeats=5,warmups=1,seed=1729,
        blocks=['1','4','16','32','SM count','2*SM count','min(ceil(max stage lanes/128), residency limit)'],
        numeric_contract='explicit-f32-v1',promote_policy=False),indent=2)+'\n')
    def save(): (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for name,n,degree,drive,delay,post in CASES:
            directory=output/name;directory.mkdir()
            opts=dict(drive=drive,delay_span=delay,post_delay=post,topology_kind='random-fixed-outdegree',topology_seed=42)
            row=dict(name=name,configuration=configuration(n,degree,1024,**opts),samples=[],passed=False)
            report['cases'].append(row);save()
            refs={p:oracle(n,degree,1024,dtype=dtype,**opts) for p,dtype in [('f32',np.float32),('f64',np.float64)]}
            context={};(directory/'cpu').mkdir()
            cpu,_=brian_run('cpu-f32',n,degree,1024,directory/'cpu',prepared=context,**opts)
            refs['compiled_f32']=cpu
            for label,values in refs.items():np.savez_compressed(directory/('reference-'+label+'.npz'),**values)
            if not checks(cpu,refs['f32'])['passed']:raise RuntimeError('CPU f32 fails independent reference')
            model=context['control'].model;(directory/'model.json').write_text(json.dumps(model)+'\n')
            executors={}
            try:
                for mode in ('auto','cooperative'):
                    executors[mode]=CudaExecutor(model,directory/mode,numeric_mode='float32',event_delivery='sparse',dag_execution=mode)
                initial=executors['cooperative'].run();initial_values=native_arrays(initial)
                np.savez_compressed(directory/'initial-cooperative.npz',**initial_values)
                assert checks(initial_values,refs['f32'])['passed'] and checks(initial_values,cpu)['passed']
                runtime=executors['cooperative']._cooperative_dag
                grids=candidate_grids(runtime.sms,runtime.active_blocks_per_sm,max(d.lanes for d in executors['cooperative'].plan.dispatches))
                row['grids']=grids;row['initial_runtime']=initial['cuda_runtime'];save()
                function=runtime.function;addresses=tuple(a.data.ptr for a in runtime.resident.gpu)
                row['resident_addresses_sha256']=hashlib.sha256(np.asarray(addresses,np.uint64).tobytes()).hexdigest()
                before={mode:artifact_hashes(directory/mode) for mode in executors}
                modes=['auto',*grids]
                def run(mode,round_index,position):
                    binding=None
                    if mode!='auto':binding=runtime.configure_grid(grids[mode])
                    started=time.perf_counter()
                    raw=executors['auto' if mode=='auto' else 'cooperative'].run();values=native_arrays(raw)
                    elapsed=time.perf_counter()-started
                    gates={p:checks(values,ref) for p,ref in refs.items()}
                    file=f'{mode}-{round_index}.npz';np.savez_compressed(directory/file,**values)
                    sample=dict(mode=mode,round=round_index,position=position,wall_seconds=elapsed,
                        timings=raw['timings'],runtime=raw['cuda_runtime'],host_storage=raw.get('host_storage'),gates=gates,
                        result=file,sha256=hashlib.sha256((directory/file).read_bytes()).hexdigest())
                    row['samples'].append(sample);save()
                    if not (gates['f32']['passed'] and gates['compiled_f32']['passed']):raise RuntimeError('Full f32 gate failed')
                    if any(not np.array_equal(values[k],cpu[k]) for k in values):raise RuntimeError('Compiled f32 bitwise gate failed')
                    if mode!='auto':
                        if raw['cuda_runtime']['dag_execution']['launch_binding']!=binding:raise RuntimeError('Observed grid changed')
                        if runtime.function is not function or tuple(a.data.ptr for a in runtime.resident.gpu)!=addresses:
                            raise RuntimeError('Cooperative program or buffer identity changed')
                for position,mode in enumerate(modes):
                    with forbid_compilation():run(mode,-2,position)
                rng=random.Random(1729)
                for i in range(-1,5):
                    order=list(modes);rng.shuffle(order)
                    for position,mode in enumerate(order):
                        with forbid_compilation():run(mode,i,position)
                    print(name,'round',i,'complete',flush=True)
                after={mode:artifact_hashes(directory/mode) for mode in executors}
                if before!=after:raise RuntimeError('Compiled binaries changed during replay')
                row['compiled']=after
                row['summary']={}
                for mode in modes:
                    samples=[s for s in row['samples'] if s['mode']==mode and s['round']>=0]
                    row['summary'][mode]={}
                    for metric in ('wall_seconds','gpu_seconds','command_seconds'):
                        values=[s['wall_seconds'] if metric=='wall_seconds' else s['timings'][0][metric] for s in samples]
                        row['summary'][mode][metric]=dict(median_ms=statistics.median(values)*1000,min_ms=min(values)*1000,max_ms=max(values)*1000)
                row['passed']=True;save()
            finally:
                for ex in executors.values():ex.close()
        report['passed']=True
    finally:save()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(compare(args.output),indent=2))
