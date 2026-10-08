"""Prospectively declared cooperative versus default CUDA replay experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import numpy as np

CASES=(('quiet',4096,8,1/64,16,16),('low',4096,8,17/512,16,16),('dense',1024,128,1/16,8,3))


def compare(output):
    from gpu_stdp_compare import brian_run,oracle,checks,configuration
    from gpu_stdp_precompiled import native_arrays,forbid_compilation,artifact_hashes
    from brian2_rust.cuda import CudaExecutor
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='b2-cooperative-comparison-v0',cases=[],passed=False,repeats=5,seed=1729,
        scope='separate retained executors for default and cooperative; bootstrap compilation excluded; full fresh-reset to result timings; no external backend ranking')
    (output/'declared-protocol.json').write_text(json.dumps(dict(cases=CASES,steps=1024,repeats=5,warmups=1,seed=1729,numeric_contract='explicit-f32-v1'),indent=2)+'\n')
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
            model=context['control'].model
            (directory/'model.json').write_text(json.dumps(model)+'\n')
            executors={}
            try:
                for mode in ('auto','cooperative'):
                    ex=CudaExecutor(model,directory/mode,numeric_mode='float32',event_delivery='sparse',dag_execution=mode)
                    executors[mode]=ex
                def run(mode,round_index,position):
                    started=time.perf_counter()
                    raw=executors[mode].run();values=native_arrays(raw)
                    elapsed=time.perf_counter()-started
                    gates={p:checks(values,ref) for p,ref in refs.items()}
                    file=f'{mode}-{round_index}.npz';np.savez_compressed(directory/file,**values)
                    sample=dict(mode=mode,round=round_index,position=position,wall_seconds=elapsed,
                        timings=raw['timings'],runtime=raw['cuda_runtime'],host_storage=raw.get('host_storage'),gates=gates,
                        result=file,sha256=hashlib.sha256((directory/file).read_bytes()).hexdigest())
                    row['samples'].append(sample);save()
                    if not (gates['f32']['passed'] and gates['compiled_f32']['passed']):raise RuntimeError('Full f32 gate failed')
                    if any(not np.array_equal(values[k],cpu[k]) for k in values):raise RuntimeError('Compiled f32 bitwise gate failed')
                    if mode=='cooperative' and raw['cuda_runtime']['dag_execution']['workgroups']<=1:
                        raise RuntimeError('Multi-block execution was not observed')
                for mode in executors:run(mode,-2,0)
                before={mode:artifact_hashes(directory/mode) for mode in executors}
                rng=random.Random(1729)
                for i in range(-1,5):
                    modes=list(executors);rng.shuffle(modes)
                    for position,mode in enumerate(modes):
                        with forbid_compilation():run(mode,i,position)
                    print(name,'round',i,'complete',flush=True)
                after={mode:artifact_hashes(directory/mode) for mode in executors}
                if before!=after:raise RuntimeError('Compiled binaries changed during replay')
                row['compiled']=after
                row['summary']={mode:{'median_ms':statistics.median(values)*1000,'min_ms':min(values)*1000,'max_ms':max(values)*1000}
                    for mode in executors for values in [[s['wall_seconds'] for s in row['samples'] if s['mode']==mode and s['round']>=0]]}
                row['passed']=True;save()
            finally:
                for ex in executors.values():ex.close()
        report['passed']=True
    finally:save()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(compare(args.output),indent=2))
