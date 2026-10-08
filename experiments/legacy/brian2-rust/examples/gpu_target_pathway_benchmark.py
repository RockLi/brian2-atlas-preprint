"""Paired canonical/target-owned recurrent LIF with delayed STDP transmission."""
import argparse,json
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import gpu_synapse_state_benchmark as common
POLICIES=common.POLICIES


@contextmanager
def policy(name):
    from brian2_rust import metal_synapses
    old=metal_synapses.target_owned_pathway
    if name=='canonical':metal_synapses.target_owned_pathway=lambda model,syn,code:False
    elif name!='parallel':raise ValueError(name)
    try:yield
    finally:metal_synapses.target_owned_pathway=old


def workload(neurons,degree,steps,mixed,directory):
    import brian2 as b
    import brian2_rust
    from brian2_rust.export import lower_network
    b.get_device().reinit();dt=b.second/1024
    b.set_device('rust_standalone',engine='reference',directory=directory);b.seed(1729)
    pop=b.NeuronGroup(neurons,'dv/dt=-v/tau:1',threshold='v>1',reset='v=0',method='euler',
        namespace={'tau':32*dt},dt=dt,name='population')
    if mixed:pop.v=np.arange(neurons)%16/16;pop.run_regularly('v+=0.125')
    syn=b.Synapses(pop,pop,'dApre/dt=-Apre/taupre:1 (event-driven)\ndApost/dt=-Apost/taupost:1 (event-driven)\nw:1',
        on_pre='v_post+=w/16; Apre+=0.0078125; w=clip(w+Apost,0,0.5)',
        on_post='Apost-=0.00390625; w=clip(w+Apre,0,0.5)',
        namespace={'taupre':16*dt,'taupost':32*dt},clock=pop.clock,name='plastic')
    sources=np.repeat(np.arange(neurons),degree);targets=(sources+np.tile(np.arange(degree),neurons)+1)%neurons
    syn.connect(i=sources,j=targets);syn.w=.25
    syn.pre.delay=(np.arange(len(sources))%4)*dt;syn.post.delay=2*dt
    return lower_network(b.Network(pop,syn,b.SpikeMonitor(pop),b.StateMonitor(pop,'v',record=[0,neurons-1])),steps*dt)


def benchmark(backend,neurons,degree,steps,mixed,repeats,output):
    previous=common.policy,common.workload;common.policy=policy;common.workload=workload
    try:r=common.benchmark(backend,neurons,degree,steps,mixed,repeats,output)
    finally:common.policy,common.workload=previous
    r.update(schema='b2-target-pathway-ablation-v0',ablation='Only target-owned pre delivery changes; post plasticity remains per edge in both policies.',
        workload='Recurrent Euler LIF, heterogeneous pre/uniform post delays, event-driven pair traces, mutable weights and target voltage transmission.')
    (output/'report.json').write_text(json.dumps(r,indent=2)+'\n');return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=['metal','cuda'],default='cuda');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--neurons',type=int,default=256);p.add_argument('--degree',type=int,default=128);p.add_argument('--steps',type=int,default=128)
    p.add_argument('--repeats',type=int,default=7);p.add_argument('--mixed',action='store_true');a=p.parse_args()
    if not (2<=a.neurons<=512 and 1<=a.degree<=128 and 1<=a.steps<=512 and 1<=a.repeats<=10):p.error('bounded workload exceeded')
    r=benchmark(a.backend,a.neurons,a.degree,a.steps,a.mixed,a.repeats,a.output);print(json.dumps(dict(summary=r['summary'],paired=r['paired']),indent=2))
if __name__=='__main__':main()
