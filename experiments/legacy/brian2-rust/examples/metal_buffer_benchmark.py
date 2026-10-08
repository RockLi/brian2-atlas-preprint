"""Paired direct/resident Metal DAG replay with complete result verification."""
import argparse
from contextlib import ExitStack
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
    report=dict(schema='b2-metal-buffer-replay-v0',neurons=neurons,steps=steps,degree=64,repeats=repeats,
        model_sha256=hashlib.sha256(encoded.encode()).hexdigest(),samples=[],snapshots={},setup={},
        scope='Two retained executors of the same plan; direct reallocates/transfers every buffer, resident reuses buffers and resets/reads only writable buffers. Wall includes reset, synchronization and full host output; excludes compilation, hashing and NPZ serialization. Each call replays the original snapshot, not Device continuation.')
    expected=None
    with ExitStack() as stack:
        executors={}
        for mode in ('direct','resident'):
            start=time.perf_counter()
            ex=stack.enter_context(MetalExecutor(model,output/mode,numeric_mode='float32',event_delivery='sparse',dag_execution=mode))
            executors[mode]=ex
            report['setup'][mode]=dict(wall_seconds=time.perf_counter()-start,compile_seconds=ex.compile_seconds,device=ex.device_name,plan_sha256=ex.plan.sha256)
        assert executors['direct'].plan.sha256==executors['resident'].plan.sha256
        def measure(mode,round_index,order):
            nonlocal expected
            start=time.perf_counter();result=executors[mode].run();wall=time.perf_counter()-start
            digest=result_digest(result)
            if expected is None:expected=digest
            assert digest==expected,'buffer policy changed complete result'
            if mode not in report['snapshots']:report['snapshots'][mode]=snapshot(result,output,mode)
            report['samples'].append(dict(mode=mode,round=round_index,order=order,wall_seconds=wall,
                timings=result['timings'][0],runtime=result['metal_runtime'],result_sha256=digest))
        for round_index in (-2,-1):
            for order,mode in enumerate(('direct','resident')):measure(mode,round_index,order)
        rng=np.random.default_rng(1729)
        for round_index in range(repeats):
            for order,mode in enumerate(rng.permutation(['direct','resident'])):measure(str(mode),round_index,order)
    report['summary']={}
    for mode in ('direct','resident'):
        rows=[r for r in report['samples'] if r['mode']==mode and r['round']>=0]
        keys={'wall_seconds':[r['wall_seconds'] for r in rows]}
        keys.update({k:[r['timings'][k] for r in rows] for k in rows[0]['timings']})
        report['summary'][mode]={k:dict(median=statistics.median(v),min=min(v),max=max(v)) for k,v in keys.items()}
    report.update(passed=True,result_sha256=expected,
        wall_speedup=report['summary']['direct']['wall_seconds']['median']/report['summary']['resident']['wall_seconds']['median'])
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--neurons',type=int,default=1024);p.add_argument('--steps',type=int,default=128)
    p.add_argument('--repeats',type=int,default=7);a=p.parse_args()
    if not 65<=a.neurons<=4096 or not 1<=a.steps<=512 or not 3<=a.repeats<=15:p.error('bounded benchmark size exceeded')
    r=benchmark(a.neurons,a.steps,a.repeats,a.output)
    print(json.dumps(dict(summary=r['summary'],wall_speedup=r['wall_speedup']),indent=2))


if __name__=='__main__':main()
