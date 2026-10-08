"""Paired canonical/endpoint-owned summed reductions in a mutable network."""
import argparse,json
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import gpu_synapse_state_benchmark as common
POLICIES=common.POLICIES


@contextmanager
def policy(name):
    from brian2_rust import metal_synapses
    old=metal_synapses.independent_summed
    if name=='canonical':metal_synapses.independent_summed=lambda model,syn,code:False
    elif name!='parallel':raise ValueError(name)
    try:yield
    finally:metal_synapses.independent_summed=old


def workload(neurons,degree,steps,mixed,directory):
    import brian2 as b
    import brian2_rust
    from brian2_rust.export import lower_network
    b.get_device().reinit();dt=b.second/1024
    b.set_device('rust_standalone',engine='reference',directory=directory);b.seed(1729)
    pop=b.NeuronGroup(neurons,'dv/dt=(incoming-v)/tau:1\nincoming:1\noutgoing:1',threshold='v>1',reset='v=0',
        method='euler',namespace={'tau':32*dt},dt=dt,name='population')
    if mixed:pop.v=np.arange(neurons)%16/16;pop.run_regularly('v+=0.125',name='drive')
    syn=b.Synapses(pop,pop,'w:1\noutgoing_pre=w:1 (summed)\nincoming_post=w*v_pre/64:1 (summed)',
        on_pre='w=clip(w+0.0078125,0,0.5)',on_post='w=clip(w-0.00390625,0,0.5)',clock=pop.clock,name='plastic')
    sources=np.repeat(np.arange(neurons),degree);targets=(sources+np.tile(np.arange(degree),neurons)+1)%neurons
    syn.connect(i=sources,j=targets);syn.w=.25
    syn.pre.delay=(np.arange(len(sources))%4)*dt;syn.post.delay=2*dt
    monitor=b.StateMonitor(pop,['v','incoming','outgoing'],record=[0,neurons-1])
    return lower_network(b.Network(pop,syn,b.SpikeMonitor(pop),monitor),steps*dt)


def benchmark(backend,neurons,degree,steps,mixed,repeats,output):
    previous=common.policy,common.workload;common.policy=policy;common.workload=workload
    try:r=common.benchmark(backend,neurons,degree,steps,mixed,repeats,output)
    finally:common.policy,common.workload=previous
    r.update(schema='b2-summed-ablation-v0',ablation='Only summed reduction ownership changes; pre/post pathways stay per edge in both policies.',
        workload='Recurrent Euler neurons, source and target summed variables, mutable delayed pre/post weights and full spike/state readback.')
    (output/'report.json').write_text(json.dumps(r,indent=2)+'\n');return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=['metal','cuda'],default='cuda');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--neurons',type=int,default=1024);p.add_argument('--degree',type=int,default=64);p.add_argument('--steps',type=int,default=128)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--mixed',action='store_true');a=p.parse_args()
    if not (2<=a.neurons<=2048 and 1<=a.degree<min(a.neurons,129) and 1<=a.steps<=512 and 1<=a.repeats<=10):p.error('bounded workload exceeded')
    r=benchmark(a.backend,a.neurons,a.degree,a.steps,a.mixed,a.repeats,a.output);print(json.dumps(dict(summary=r['summary'],paired=r['paired']),indent=2))
if __name__=='__main__':main()
