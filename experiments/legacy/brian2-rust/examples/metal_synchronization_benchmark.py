"""One-queue paired serial/tracked versus explicit-barrier Metal replay."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from gpu_summed_benchmark import workload
from gpu_decode_benchmark import snapshot
from cuda_dag_benchmark import result_digest
from brian2_rust.metal import MetalExecutor


def benchmark(neurons,steps,repeats,output):
    output.mkdir(parents=True,exist_ok=False)
    model=workload(neurons,64,steps,True,output/'model')
    encoded=json.dumps(model,sort_keys=True,indent=2)+'\n'
    (output/'model.json').write_text(encoded)
    report=dict(schema='b2-metal-synchronization-v0',neurons=neurons,steps=steps,degree=64,
        repeats=repeats,model_sha256=hashlib.sha256(encoded.encode()).hexdigest(),samples=[],snapshots={},
        scope='Same executor, pipelines, queue, plan and direct buffer policy; only explicit per-dispatch barriers change. Wall includes reset, dispatch, synchronization and full host results; excludes compilation, hashing and NPZ serialization.')
    expected=None
    with MetalExecutor(model,output/'metal',numeric_mode='float32',event_delivery='sparse',dag_execution='direct') as ex:
        report.update(device=ex.device_name,plan_sha256=ex.plan.sha256,compile_seconds=ex.compile_seconds)
        dispatches=sum(ex.plan.logical.clocks[d.clock].steps for d in ex.plan.dispatches)
        def measure(policy,round_index,order):
            nonlocal expected
            start=time.perf_counter();result=ex.run(dag_synchronization=policy);wall=time.perf_counter()-start
            digest=result_digest(result)
            if expected is None:expected=digest
            assert digest==expected,'synchronization policy changed complete result'
            runtime=result['metal_runtime']
            assert runtime['dispatches']==dispatches
            assert runtime['explicit_barriers']==(dispatches if policy=='explicit' else 0)
            assert runtime['dag_execution']=='direct'
            assert runtime['allocated_bytes']==runtime['uploaded_bytes']==runtime['readback_bytes']==runtime['total_buffer_bytes']
            if policy not in report['snapshots']:report['snapshots'][policy]=snapshot(result,output,policy)
            report['samples'].append(dict(policy=policy,round=round_index,order=order,
                wall_seconds=wall,timings=result['timings'][0],runtime=runtime,result_sha256=digest))
        for index in (-2,-1):
            for order,policy in enumerate(('explicit','tracked')):measure(policy,index,order)
        # Balanced first/second positions within an even number of pairs.
        orderings=[('explicit','tracked'),('tracked','explicit')]*(repeats//2)
        np.random.default_rng(1729).shuffle(orderings)
        for index,ordering in enumerate(orderings):
            for order,policy in enumerate(ordering):measure(policy,index,order)
    report['summary']={}
    for policy in ('explicit','tracked'):
        rows=[r for r in report['samples'] if r['policy']==policy and r['round']>=0]
        values={'wall_seconds':[r['wall_seconds'] for r in rows]}
        values.update({k:[r['timings'][k] for r in rows] for k in rows[0]['timings']})
        report['summary'][policy]={k:dict(median=statistics.median(v),min=min(v),max=max(v)) for k,v in values.items()}
    report.update(passed=True,result_sha256=expected,
        wall_speedup=report['summary']['explicit']['wall_seconds']['median']/report['summary']['tracked']['wall_seconds']['median'])
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--neurons',type=int,default=1024);p.add_argument('--steps',type=int,default=128)
    p.add_argument('--repeats',type=int,default=10);a=p.parse_args()
    if not 65<=a.neurons<=4096 or not 1<=a.steps<=512 or not 4<=a.repeats<=20 or a.repeats%2:
        p.error('bounded sizes and an even number of pairs within 4..20 are required')
    r=benchmark(a.neurons,a.steps,a.repeats,a.output)
    print(json.dumps(dict(summary=r['summary'],wall_speedup=r['wall_speedup']),indent=2))


if __name__=='__main__':main()
