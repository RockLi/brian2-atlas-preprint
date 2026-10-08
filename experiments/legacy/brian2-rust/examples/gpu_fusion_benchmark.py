"""Paired previous/source/full GPU fusion with exact complete-result gates."""
import argparse
from contextlib import contextmanager,ExitStack
import hashlib,json,statistics,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

POLICIES=('previous','source','full')


@contextmanager
def planning_policy(policy):
    from brian2_rust import metal_dag
    from brian2_rust.gpu_dispatch_fusion import fuse_dispatches
    if policy not in POLICIES:raise ValueError('Unknown fusion policy')
    original=metal_dag.fuse_dispatches
    metal_dag.fuse_dispatches=(lambda m,l,k,d:(k,d)) if policy=='previous' else (
        lambda m,l,k,d:fuse_dispatches(m,l,k,d,target_delivery=policy=='full'))
    try:yield
    finally:metal_dag.fuse_dispatches=original


def benchmark(backend,neurons,steps,degree,repeats,output):
    from cuda_dag_benchmark import workload,result_digest
    from gpu_decode_benchmark import snapshot
    from gpu_recurrent import configuration
    from brian2_rust.cuda import CudaExecutor
    from brian2_rust.metal import MetalExecutor,build_metal_plan
    from brian2_rust.metal_dag import run_dag
    output.mkdir(parents=True,exist_ok=False)
    model=workload(neurons,steps,degree,output/'model')
    encoded=json.dumps(model,sort_keys=True,indent=2)+'\n';(output/'model.json').write_text(encoded)
    report=dict(schema='b2-dispatch-fusion-ablation-v0',backend=backend,configuration=configuration(neurons,steps,degree),
        model_sha256=hashlib.sha256(encoded.encode()).hexdigest(),repeats=repeats,policies={},samples=[],snapshots={},
        scope='same model and full result contract; prior plan versus source-only and source+target fusion; warm wall includes reset/dispatch/readback/decoding, excludes model/compilation/hash/NPZ work')
    expected=None
    with ExitStack() as stack:
        runners={}
        def measure(policy,round_index,order):
            nonlocal expected
            started=time.perf_counter();result=runners[policy]();wall=time.perf_counter()-started
            digest=result_digest(result)
            if expected is None:expected=digest
            if digest!=expected:raise AssertionError('Fusion changed complete population/synapse results')
            if policy not in report['snapshots']:report['snapshots'][policy]=snapshot(result,output,policy)
            report['samples'].append(dict(policy=policy,round=round_index,order=order,wall_seconds=wall,
                result_sha256=digest,timings=result['timings'],runtime=result.get('cuda_runtime'),device=result['device']))
        for order,policy in enumerate(POLICIES):
            started=time.perf_counter()
            with planning_policy(policy):
                if backend=='cpu-f32':
                    directory=output/policy;directory.mkdir()
                    ex=SimpleNamespace(model=model,plan=build_metal_plan(model,numeric_mode='float32',event_delivery='sparse'),
                        directory=directory,compile_seconds=0,device_name='CPU f32 control')
                    runners[policy]=lambda ex=ex:run_dag(ex,max_buffer_bytes=512*1024**2,compute='cpu-f32',workers=1)
                else:
                    cls=CudaExecutor if backend=='cuda' else MetalExecutor
                    ex=stack.enter_context(cls(model,output/policy,numeric_mode='float32',event_delivery='sparse'))
                    runners[policy]=ex.run
            preparation=time.perf_counter()-started
            plan=ex.plan.to_json();(output/(policy+'-plan.json')).write_text(plan)
            report['policies'][policy]=dict(setup_seconds=preparation,plan_sha256=ex.plan.sha256,
                plan_file_sha256=hashlib.sha256(plan.encode()).hexdigest(),dispatch_count=len(ex.plan.dispatches),
                launch_count=sum(ex.plan.logical.clocks[d.clock].steps for d in ex.plan.dispatches if d.lanes),
                kernel_attributes=[k.attributes for k in ex.kernels] if backend=='cuda' else None)
            measure(policy,-2,order);measure(policy,-1,order)
        rng=np.random.default_rng(1729)
        for index in range(repeats):
            for order,policy in enumerate(rng.permutation(POLICIES)):measure(str(policy),index,order)
    report['summary']={};report['paired']={}
    for policy in POLICIES:
        values=[r['wall_seconds'] for r in report['samples'] if r['policy']==policy and r['round']>=0]
        report['summary'][policy]=dict(samples=values,median=statistics.median(values),min=min(values),max=max(values))
        if policy!='previous':
            pairs=[]
            for index in range(repeats):
                r={r['policy']:r for r in report['samples'] if r['round']==index}
                a=r['previous']['wall_seconds'];b=r[policy]['wall_seconds']
                pairs.append(dict(round=index,delta_seconds=b-a,reduction_fraction=1-b/a))
            report['paired'][policy]=dict(samples=pairs,median_delta_seconds=statistics.median(p['delta_seconds'] for p in pairs),
                median_reduction_fraction=statistics.median(p['reduction_fraction'] for p in pairs),improved_pairs=sum(p['delta_seconds']<0 for p in pairs))
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
    print(json.dumps(dict(summary=r['summary'],paired=r['paired']),indent=2))


if __name__=='__main__':main()
