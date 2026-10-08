"""Measure exact-input decision reuse through complete GPU transport activations."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np

from gpu_autotune_benchmark import CASES, save_observables


def benchmark(output, backend, *, profile_preparation=False):
    from brian2_rust.gpu_autotune import observable_fingerprint
    from brian2_rust.gpu_tuning_cache import TuningCache
    from brian2_rust.gpu_buffer_transfer import execute_device
    from brian2_rust.results import load_results
    from brian2_rust.cuda import CudaExecutor
    from gpu_stdp_compare import brian_run, oracle, checks, configuration, activity
    from gpu_stdp_precompiled import native_arrays
    root=Path(__file__).resolve().parents[1]
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='b2-gpu-tuning-cache-benchmark-v1',backend=backend,passed=False,cases=[],
        scope='identical complete model; fresh executor, input hashing, plan validation, compiler-context checks, execution, full-result hashing, result write/read and cache publication; Brian lowering and model export excluded')
    def save():(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for name,n,degree,drive,delay,post in CASES:
            directory=output/name;directory.mkdir()
            opts=dict(drive=drive,delay_span=delay,post_delay=post,topology_kind='random-fixed-outdegree',topology_seed=42)
            prepared={};cpu_dir=directory/'cpu';cpu_dir.mkdir()
            cpu,_=brian_run('cpu-f32',n,degree,1024,cpu_dir,prepared=prepared,**opts)
            model=prepared['control'].model
            control=CudaExecutor._cpu_control(prepared['control'],512*1024**2,1)
            refs={p:oracle(n,degree,1024,dtype=d,**opts) for p,d in [('f32',np.float32),('f64',np.float64)]}
            assert checks(cpu,refs['f32'])['passed']
            activities=activity(n,degree,1024,refs['f32'],**opts)
            events=sum(activities[k] for k in ('delivered_pre_events','delivered_post_events'))
            (directory/'model.json').write_text(json.dumps(model)+'\n')
            for p,values in refs.items():np.savez_compressed(directory/('reference-'+p+'.npz'),**values)
            device=SimpleNamespace(build_options=dict(engine=backend,event_delivery='sparse',
                gpu_autotune=True,gpu_autotune_cache=True,gpu_compile_reuse=True,gpu_buffer_reuse=True),
                _gpu_executor=None,_gpu_tuning_cache=TuningCache(),last_gpu_tuning=None)
            row=dict(name=name,configuration=configuration(n,degree,1024,**opts),expected_events=events,
                     activations=[],passed=False)
            report['cases'].append(row);save()
            try:
                for i in range(5 if profile_preparation else 4):
                    path=directory/('activation-'+str(i));path.mkdir()
                    profiler=None
                    if profile_preparation and i==4:
                        import cProfile
                        profiler=cProfile.Profile();profiler.enable()
                    start=time.perf_counter()
                    pending=execute_device(device,model,path,root/'target/release/b2-runner')
                    actual=load_results(model,path/'rust')
                    device._gpu_tuning_cache.publish(*pending)
                    wall=time.perf_counter()-start
                    if profiler is not None:
                        profiler.disable()
                        import pstats
                        stats=pstats.Stats(profiler)
                        entries=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],
                            calls=v[1],self_seconds=v[2],cumulative_seconds=v[3]) for k,v in stats.stats.items()]
                        (path/'preparation-profile.json').write_text(json.dumps(dict(
                            scope='cProfile instrumentation of complete transport activation; do not use as a timing sample',
                            total_seconds=stats.total_tt,functions=sorted(entries,key=lambda x:-x['cumulative_seconds'])),indent=2)+'\n')
                    tuning=device.last_gpu_tuning
                    assert tuning['cache']['status']==('miss' if i==0 else 'hit')
                    # The transport adds seconds arrays; compare the complete raw
                    # semantic contract separately using a fresh retained replay.
                    raw=device._gpu_executor.run()
                    control['numeric_profile']=raw['numeric_profile']
                    assert control.get('rng_profile')==raw.get('rng_profile')
                    expected_hash=observable_fingerprint(control)
                    assert observable_fingerprint(raw)==expected_hash==tuning['reference_observable_sha256']
                    save_observables(path/'raw-observables',raw)
                    save_observables(path/'transport-observables',actual)
                    if i==0:save_observables(directory/'control-observables',control)
                    values=native_arrays(actual)
                    gates={p:checks(values,ref) for p,ref in refs.items()}
                    assert gates['f32']['passed']
                    assert sum(s['events'] for s in actual['synapses'])==events
                    np.savez_compressed(path/'snapshot.npz',**values)
                    row['activations'].append(dict(index=i,wall_seconds=wall,tuning=tuning,gates=gates,
                        compilation=actual['metadata'][backend+'_runtime']['compilation'],
                        buffer_reuse=actual['metadata'][backend+'_runtime']['activation_buffer_reuse'],
                        observable_sha256=expected_hash))
                    if profile_preparation:row['activations'][-1]['profiled']=profiler is not None
                    save()
                assert len(device._gpu_tuning_cache)==1
                row['passed']=True;save()
            finally:
                if device._gpu_executor is not None:device._gpu_executor.close()
            print(name,[(a['tuning']['cache']['status'],a['wall_seconds']) for a in row['activations']],flush=True)
        report['passed']=True
    finally:save()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('metal','cuda'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile-preparation',action='store_true',help='Add a separately labelled instrumented cache hit after the timing samples')
    args=parser.parse_args();benchmark(args.output,args.backend,profile_preparation=args.profile_preparation)
