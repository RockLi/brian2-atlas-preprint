"""Cold-activation ablation of direct, resident, full and chunked CUDA graphs.

Five fresh executors per mode/route use the same precompiled cubins and model.
Mode order is seeded and interleaved. First result and second replay are timed
separately; every output must match the direct f32 reference bit for bit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np


def benchmark(model,directory,trials):
    from brian2_rust.cuda import CudaExecutor
    from cuda_dag_benchmark import measure
    rows=[];expected=None;bootstrap=[]
    modes=('direct','resident','graph','chunked','auto')
    for route in ('scan','sparse'):
        artifacts=directory/route
        started=time.perf_counter()
        with CudaExecutor(model,artifacts,numeric_mode='float32',event_delivery=route,dag_execution='direct') as executor:
            setup=time.perf_counter()-started
            result,sample=measure(executor,expected)
            if expected is None:expected=sample['result_sha256']
            bootstrap.append(dict(route=route,setup_seconds=setup,compile_seconds=executor.compile_seconds,sample=sample))
            runtime=result['cuda_runtime'];plan_sha=executor.plan.sha256
        group={mode:dict(route=route,mode=mode,plan_sha256=plan_sha,samples=[]) for mode in modes}
        rng=np.random.default_rng(1729)
        for trial in range(trials):
            for order,mode in enumerate(rng.permutation(modes)):
                started=time.perf_counter()
                with CudaExecutor(model,artifacts,numeric_mode='float32',event_delivery=route,dag_execution=str(mode)) as executor:
                    setup_seconds=time.perf_counter()-started
                    _,cold=measure(executor,expected)
                    _,warm=measure(executor,expected)
                    assert executor.plan.sha256==plan_sha
                    group[mode]['samples'].append(dict(trial=trial,order=order,setup_seconds=setup_seconds,
                        compile_seconds=executor.compile_seconds,setup_plus_first_result_seconds=setup_seconds+cold['wall_seconds'],
                        cold=cold,second_replay=warm))
        for row in group.values():
            row['median']=dict(cold_wall_seconds=statistics.median(s['cold']['wall_seconds'] for s in row['samples']),
                setup_plus_first_result_seconds=statistics.median(s['setup_plus_first_result_seconds'] for s in row['samples']),
                cold_gpu_seconds=statistics.median(s['cold']['gpu_seconds'] for s in row['samples']),
                second_replay_wall_seconds=statistics.median(s['second_replay']['wall_seconds'] for s in row['samples']),
                graph_build_seconds=statistics.median(s['cold']['execution']['graph_build_seconds'] for s in row['samples']))
            rows.append(row)
    return dict(rows=rows,bootstrap=bootstrap,result_sha256=expected,cuda_runtime=runtime,
                correctness='all fresh-executor and second-replay results bitwise equal across routes/modes')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--neurons',type=int,default=512)
    parser.add_argument('--steps',type=int,default=512)
    parser.add_argument('--degree',type=int,default=32)
    parser.add_argument('--trials',type=int,default=5)
    args=parser.parse_args()
    if not 3<=args.trials<=9 or not 1<=args.steps<=4096 or not 8<=args.neurons<=8192:parser.error('bounded benchmark size exceeded')
    args.output.mkdir(parents=True,exist_ok=False)
    from cuda_dag_benchmark import workload
    from gpu_recurrent import configuration
    model=workload(args.neurons,args.steps,args.degree,args.output/'model')
    source=json.dumps(model,sort_keys=True,indent=2)+'\n';(args.output/'model.json').write_text(source)
    report=dict(schema='b2-cuda-chunk-ablation-v0',configuration=configuration(args.neurons,args.steps,args.degree),
                trials=args.trials,model_sha256=hashlib.sha256(source.encode()).hexdigest(),
                timing_scope='same model/cubins, fresh executors; cold wall includes storage preparation, input reset, graph construction/upload, dispatch, synchronized readback and result decoding',
                setup_scope='validation, plan derivation, source/cubin lookup and module loading; first NVCC compilation is the separate bootstrap',
                limitations=['model/frontend construction and executor close are outside measured scopes',
                             'warmed CUDA context and memory pool; single-activation executor cold, not process cold',
                             'same deterministic f32 snapshot, not reference-f64 or competing-backend performance'],
                **benchmark(model,args.output/'executors',args.trials))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{key:row[key] for key in ('route','mode','median')} for row in report['rows']],indent=2))


if __name__=='__main__':main()
