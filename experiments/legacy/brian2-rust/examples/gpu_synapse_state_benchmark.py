"""Paired canonical/edge-parallel synapse-state replay with full-result gates."""
import argparse,hashlib,json,statistics,time
from contextlib import contextmanager,ExitStack
from pathlib import Path
import numpy as np
POLICIES=('canonical','parallel')


@contextmanager
def policy(name):
    from brian2_rust import metal_synapses
    old=metal_synapses.independent_state_update
    if name=='canonical':metal_synapses.independent_state_update=lambda syn,code:False
    elif name!='parallel':raise ValueError(name)
    try:yield
    finally:metal_synapses.independent_state_update=old


def workload(neurons,degree,steps,mixed,directory):
    import brian2 as b
    import brian2_rust
    from brian2_rust.export import lower_network
    b.get_device().reinit();dt=b.second/1024
    b.set_device('rust_standalone',engine='reference',directory=directory)
    pop=b.NeuronGroup(neurons,'v:1',threshold='v>0.5',reset='',dt=dt,name='population');pop.v=1 if mixed else 0
    equations='da/dt=(-a+b/4)/tau:1 (clock-driven)\ndb/dt=-b/tau:1 (clock-driven)\nz=a+b:1 (constant over dt)'
    options=dict(on_pre='a+=0.0078125',on_post='b+=0.00390625') if mixed else dict(on_pre='a+=0.0078125')
    syn=b.Synapses(pop,pop,equations,method='euler',namespace={'tau':32*dt},clock=pop.clock,name='plastic',**options)
    sources=np.repeat(np.arange(neurons),degree);targets=(sources+np.tile(np.arange(degree),neurons)+1)%neurons
    syn.connect(i=sources,j=targets);syn.a=.5;syn.b=.25
    if mixed:syn.pre.delay=(np.arange(len(sources))%4)*dt;syn.post.delay=2*dt
    return lower_network(b.Network(pop,syn,b.SpikeMonitor(pop)),steps*dt)


def benchmark(backend,neurons,degree,steps,mixed,repeats,output):
    from brian2_rust.metal import MetalExecutor
    from brian2_rust.cuda import CudaExecutor
    from cuda_dag_benchmark import result_digest
    from gpu_decode_benchmark import snapshot
    output.mkdir(parents=True,exist_ok=False);model=workload(neurons,degree,steps,mixed,output/'model')
    encoded=json.dumps(model,sort_keys=True,indent=2)+'\n';(output/'model.json').write_text(encoded)
    r=dict(schema='b2-synapse-state-ablation-v0',backend=backend,configuration=dict(neurons=neurons,degree=degree,edges=neurons*degree,steps=steps,mixed=mixed,dt_seconds=1/1024,numeric_mode='float32'),repeats=repeats,
        model_sha256=hashlib.sha256(encoded.encode()).hexdigest(),policies={},snapshots={},samples=[],scope='same-model reset-to-full-result wall; excludes model generation, compilation, hashing and NPZ writing')
    expected=None
    with ExitStack() as stack:
        executors={}
        for name in POLICIES:
            started=time.perf_counter()
            with policy(name):executors[name]=stack.enter_context((CudaExecutor if backend=='cuda' else MetalExecutor)(model,output/name,numeric_mode='float32',event_delivery='sparse'))
            ex=executors[name];p=ex.plan.to_json();(output/(name+'-plan.json')).write_text(p)
            r['policies'][name]=dict(setup_seconds=time.perf_counter()-started,plan_sha256=ex.plan.sha256,plan_file_sha256=hashlib.sha256(p.encode()).hexdigest(),dispatches=[dict(role=d.role,lanes=d.lanes) for d in ex.plan.dispatches])
        def measure(name,index,order):
            nonlocal expected
            t=time.perf_counter();result=executors[name].run();wall=time.perf_counter()-t;digest=result_digest(result)
            if expected is None:expected=digest
            assert digest==expected,'parallel update changed complete results'
            if name not in r['snapshots']:r['snapshots'][name]=snapshot(result,output,name)
            r['samples'].append(dict(policy=name,round=index,order=order,wall_seconds=wall,result_sha256=digest,timings=result['timings'],runtime=result.get('cuda_runtime'),device=result['device']))
        for index in [-2,-1]:
            for order,name in enumerate(POLICIES):measure(name,index,order)
        rng=np.random.default_rng(1729)
        for index in range(repeats):
            for order,name in enumerate(rng.permutation(POLICIES)):measure(str(name),index,order)
    r['summary']={}
    for name in POLICIES:
        values=[a['wall_seconds'] for a in r['samples'] if a['policy']==name and a['round']>=0]
        r['summary'][name]=dict(samples=values,median=statistics.median(values),min=min(values),max=max(values))
    pairs=[]
    for index in range(repeats):
        rows={a['policy']:a for a in r['samples'] if a['round']==index};a=rows['canonical']['wall_seconds'];b=rows['parallel']['wall_seconds']
        pairs.append(dict(round=index,delta_seconds=b-a,reduction_fraction=1-b/a))
    r['paired']=dict(samples=pairs,median_delta_seconds=statistics.median(a['delta_seconds'] for a in pairs),median_reduction_fraction=statistics.median(a['reduction_fraction'] for a in pairs),improved_pairs=sum(a['delta_seconds']<0 for a in pairs))
    r.update(passed=True,result_sha256=expected);(output/'report.json').write_text(json.dumps(r,indent=2)+'\n');return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=['metal','cuda'],default='cuda');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--neurons',type=int,default=256);p.add_argument('--degree',type=int,default=128);p.add_argument('--steps',type=int,default=128);p.add_argument('--repeats',type=int,default=7);p.add_argument('--mixed',action='store_true');a=p.parse_args()
    if not (1<=a.neurons<=512 and 1<=a.degree<=128 and 1<=a.steps<=512 and 1<=a.repeats<=10):p.error('bounded workload exceeded')
    r=benchmark(a.backend,a.neurons,a.degree,a.steps,a.mixed,a.repeats,a.output);print(json.dumps(dict(summary=r['summary'],paired=r['paired']),indent=2))
if __name__=='__main__':main()
