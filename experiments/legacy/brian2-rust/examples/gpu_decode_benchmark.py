"""Paired event-decoder ablation with unchanged GPU kernels and full results."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import numpy as np


def snapshot(result,directory,label):
    arrays={}
    def encode(value,path):
        if isinstance(value,np.ndarray):
            arrays[path]=value
            return dict(array=path,dtype=value.dtype.str,shape=list(value.shape),
                        sha256=hashlib.sha256(value.tobytes()).hexdigest())
        if isinstance(value,dict):return {k:encode(v,path+'/'+k) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [encode(v,path+'/'+str(i)) for i,v in enumerate(value)]
        return value.item() if isinstance(value,np.generic) else value
    tree=encode({k:result[k] for k in ('populations','synapses')},'result')
    path=directory/(label+'.npz');np.savez_compressed(path,**arrays)
    return dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),tree=tree)


@contextmanager
def decoder(policy,counters):
    import brian2_rust.metal_dag as dag
    import brian2_rust.metal_monitors as monitors
    from brian2_rust.metal_event_layout import event_coordinates
    old=dag.event_coordinates,monitors.event_coordinates
    function=np.nonzero if policy=='legacy' else event_coordinates
    def measured(flags):
        started=time.perf_counter();result=function(flags)
        counters['seconds']+=time.perf_counter()-started;counters['calls']+=1
        return result
    dag.event_coordinates=monitors.event_coordinates=measured
    try:yield
    finally:dag.event_coordinates,monitors.event_coordinates=old


def benchmark(backend,neurons,steps,degree,repeats,output):
    from cuda_dag_benchmark import workload,result_digest
    from gpu_recurrent import configuration
    from brian2_rust.cuda import CudaExecutor
    from brian2_rust.metal import MetalExecutor,build_metal_plan
    from brian2_rust.metal_dag import run_dag
    output.mkdir(parents=True,exist_ok=False)
    model=workload(neurons,steps,degree,output/'model')
    encoded=json.dumps(model,sort_keys=True,indent=2)+'\n'
    (output/'model.json').write_text(encoded)
    started=time.perf_counter()
    if backend=='cpu-f32':
        directory=output/'cpu';directory.mkdir()
        ex=SimpleNamespace(model=model,plan=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse'),
            directory=directory,compile_seconds=0,device_name='CPU f32 control')
        run=lambda:run_dag(ex,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=1)
    else:
        cls=CudaExecutor if backend=='cuda' else MetalExecutor
        ex=cls(model,output/'executor',numeric_mode='float32',event_delivery='sparse')
        run=ex.run
    report=dict(schema='b2-event-decode-ablation-v0',backend=backend,configuration=configuration(neurons,steps,degree),
        model_sha256=hashlib.sha256(encoded.encode()).hexdigest(),plan_sha256=ex.plan.sha256,
        setup_seconds=time.perf_counter()-started,repeats=repeats,samples=[],snapshots={},
        scope='same prepared executor, kernels, transfer and complete output; only host event-coordinate decoder changes; wall excludes hashing and NPZ serialization',
        compiler_source_sha256={k.entry:hashlib.sha256(k.source.encode()).hexdigest() for k in ex.plan.kernels})
    expected=None
    def measure(policy,round_index,order):
        nonlocal expected
        counters=dict(seconds=0.,calls=0)
        with decoder(policy,counters):
            started=time.perf_counter();result=run();elapsed=time.perf_counter()-started
        digest=result_digest(result)
        if expected is None:expected=digest
        if digest!=expected:raise AssertionError('Decoder change altered complete population/synapse results')
        if policy not in report['snapshots']:report['snapshots'][policy]=snapshot(result,output,policy)
        row=dict(policy=policy,round=round_index,order=order,wall_seconds=elapsed,
            decoder_seconds=counters['seconds'],decoder_calls=counters['calls'],result_sha256=digest,
            timings=result['timings'],runtime=result.get('cuda_runtime'))
        report['samples'].append(row)
    try:
        measure('legacy',-2,0)  # first run (allocation/preparation), excluded
        for order,policy in enumerate(('legacy','boolean')):measure(policy,-1,order)
        rng=np.random.default_rng(1729)
        for round_index in range(repeats):
            for order,policy in enumerate(rng.permutation(['legacy','boolean'])):measure(str(policy),round_index,order)
    finally:
        if backend!='cpu-f32':ex.close()
    report['summary']={}
    for policy in ('legacy','boolean'):
        rows=[r for r in report['samples'] if r['policy']==policy and r['round']>=0]
        report['summary'][policy]={k:dict(samples=[r[k] for r in rows],median=statistics.median(r[k] for r in rows),
            min=min(r[k] for r in rows),max=max(r[k] for r in rows)) for k in ('wall_seconds','decoder_seconds')}
    report.update(passed=True,result_sha256=expected)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('cuda','metal','cpu-f32'),default='cuda')
    p.add_argument('--neurons',type=int,default=4096);p.add_argument('--steps',type=int,default=2048)
    p.add_argument('--degree',type=int,default=32);p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from gpu_recurrent import validate_size
    if not 8<=a.neurons<=8192 or not 1<=a.steps<=4096 or not 1<=a.repeats<=20:p.error('bounded benchmark size exceeded')
    try:validate_size(a.neurons,a.degree)
    except ValueError as error:p.error(str(error))
    r=benchmark(a.backend,a.neurons,a.steps,a.degree,a.repeats,a.output)
    print(json.dumps(r['summary'],indent=2))


if __name__=='__main__':main()
