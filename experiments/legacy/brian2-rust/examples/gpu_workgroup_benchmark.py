"""Paired full-result replay: standard DAG versus one persistent workgroup."""
import argparse,json,random,statistics,time
from pathlib import Path
import numpy as np

CASES=(dict(name='small',neurons=64,degree=8,steps=128,drive=.125,delay_span=4,post_delay=2),
       dict(name='low-drive',neurons=1024,degree=8,steps=256,drive=.0625,delay_span=8,post_delay=3),
       dict(name='active',neurons=512,degree=32,steps=128,drive=.125,delay_span=4,post_delay=2))


def benchmark(backend,output,repeats=7):
    import brian2 as b
    import brian2_rust
    from gpu_stdp_compare import build_brian,oracle,checks,activity,DT,configuration
    from gpu_stdp_precompiled import native_arrays,write_result,forbid_compilation
    from brian2_rust.cuda import CudaExecutor
    from brian2_rust.metal import MetalExecutor
    from brian2_rust.export import lower_network
    from brian2_rust.protocol import canonical_bytes
    from brian2_rust.results import load_results
    cls=MetalExecutor if backend=='metal' else CudaExecutor
    output.mkdir(parents=True,exist_ok=False);reports=[]
    for case in CASES:
        name,n,k,t=(case[x] for x in ('name','neurons','degree','steps'));opts={x:case[x] for x in ('drive','delay_span','post_delay')}
        folder=output/name;folder.mkdir();b.get_device().reinit()
        b.set_device('rust_standalone',engine='reference',directory=folder/'ref')
        net,*_=build_brian(n,k,**opts);model=lower_network(net,t*DT*b.second)
        (folder/'model.json').write_text(json.dumps(model,indent=2)+'\n')
        expected=oracle(n,k,t,**opts);executors={};rows=[];snapshots={};gpu_reference=None
        try:
            for mode in ('standard','workgroup'):
                ex=cls(model,folder/mode,numeric_mode='float32',event_delivery='sparse');executors[mode]=ex
                raw=ex.run(dag_execution='workgroup' if mode=='workgroup' else None);actual=native_arrays(raw)
                gate=checks(actual,expected);assert gate['passed'],(name,mode,gate)
                if gpu_reference is None:gpu_reference=actual
                else:
                    for field in actual:assert actual[field].dtype==gpu_reference[field].dtype and actual[field].tobytes()==gpu_reference[field].tobytes(),(name,field)
                snapshots[mode+'-bootstrap']=write_result(folder,mode+'-bootstrap',actual)
                # Prime the standard graph/cache and compile the workgroup before timing.
                raw=ex.run(dag_execution='workgroup' if mode=='workgroup' else None)
                actual=native_arrays(raw);assert checks(actual,expected)['passed']
                snapshots[mode+'-warmup']=write_result(folder,mode+'-warmup',actual)
            rng=random.Random(1729)
            for trial in range(repeats):
                modes=['standard','workgroup'];rng.shuffle(modes)
                for order,mode in enumerate(modes):
                    with forbid_compilation():
                        start=time.perf_counter();raw=executors[mode].run(dag_execution='workgroup' if mode=='workgroup' else None);actual=native_arrays(raw);wall=time.perf_counter()-start
                    gate=checks(actual,expected);assert gate['passed'],(name,mode,gate)
                    for field in actual:assert actual[field].dtype==gpu_reference[field].dtype and actual[field].tobytes()==gpu_reference[field].tobytes(),(name,mode,field)
                    label=f'{mode}-{trial}';snapshots[label]=write_result(folder,label,actual)
                    runtime=raw['metal_runtime'] if backend=='metal' else raw['cuda_runtime']['dag_execution']
                    if mode=='workgroup':assert runtime['workgroups']==1 and runtime['dispatches']==1 and runtime['compiled_program_reused']
                    rows.append(dict(mode=mode,trial=trial,order=order,wall_seconds=wall,run_seconds=raw['run_seconds'],timings=raw['timings'],runtime=runtime,gate=gate,snapshot=label))
            report=dict(case=case,configuration=configuration(n,k,t,**opts),activity=activity(n,k,t,expected,**opts),passed=True,
                samples=rows,snapshots=snapshots,medians={mode:statistics.median(s['wall_seconds'] for s in rows if s['mode']==mode) for mode in ('standard','workgroup')})
            for a in report['snapshots'].values():a['path']=Path(a['path']).name
            (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
        finally:
            for ex in executors.values():ex.close()
    report=dict(schema='b2-workgroup-replay-v0',backend=backend,passed=True,repeats=repeats,cases=reports,
        scope='Separate warmed executors, one simulation at a time, random paired order. Complete host reset, packing/upload, simulation and full results inside wall; compilation and serialization outside. All fields pass unchanged f64 gates and are byte-identical to standard native GPU. Workgroup is explicit, single-SM/threadgroup and bounded; no universal performance claim.')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('metal','cuda'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();benchmark(a.backend,a.output)
