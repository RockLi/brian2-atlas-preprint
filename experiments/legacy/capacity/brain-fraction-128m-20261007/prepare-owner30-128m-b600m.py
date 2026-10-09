"""Explicit connected synthetic E/I capacity workload, not human-brain anatomy."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--neurons',type=int,required=True)
    p.add_argument('--ranks',type=int,default=32)
    p.add_argument('--degree',type=int,default=1000)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--compile',action='store_true')
    a=p.parse_args()
    assert 24000<=a.neurons<=128_000_000 and a.neurons%20==0
    assert a.degree==1000 and 1<=a.steps<=1000 and a.ranks==30
    os.environ.update(B2_MAX_NEURONS=str(a.neurons),B2_MAX_INITIAL_VALUES='600000000',B2_MAX_IR_BYTES=str(8*2**30))
    sys.path.insert(0,str(a.source/'python'))
    import brian2 as b
    import numpy as np
    import brian2_rust
    from brian2_rust.export import lower_network
    from brian2_rust.encoded_array import packed_export
    from brian2_rust.protocol import _canonical_chunks
    from brian2_rust.distributed import write_mpi_project,compile_mpi_project
    started=time.monotonic()
    a.output.mkdir(exist_ok=False,parents=True)
    b.set_device('rust_standalone',runner=a.source/'target/release/b2-runner')
    clock=b.Clock(dt=.1*b.ms)
    excitatory=a.neurons*4//5;inhibitory=a.neurons-excitatory
    sizes=[excitatory//24+(i<excitatory%24) for i in range(24)]+[inhibitory//6+(i<inhibitory%6) for i in range(6)]
    numerators=[x*y*a.degree for x in sizes for y in sizes]
    edge_counts=[x//a.neurons for x in numerators]
    remainder=a.neurons*a.degree-sum(edge_counts)
    for i in sorted(range(900),key=lambda i:(-(numerators[i]%a.neurons),i))[:remainder]:edge_counts[i]+=1
    assert sum(sizes)==a.neurons and sum(edge_counts)==a.neurons*a.degree
    groups=[];objects=[]
    for k,n in enumerate(sizes):
        g=b.NeuronGroup(n,'dv/dt=(drive-v+current)/(20*ms):1 (unless refractory)\ndcurrent/dt=-current/(5*ms):1',
                       threshold='v>1',reset='v=0',refractory=2*b.ms,method='euler',clock=clock,
                       namespace={'drive':1.05},name=f'capacity_{k}')
        g.v=(np.arange(n,dtype=np.int64)%1000)*.00095
        groups.append(g);objects += [g,b.SpikeMonitor(g),b.StateMonitor(g,['v','current'],record=[0,n-1])]
    for s,sg in enumerate(groups):
        for t,tg in enumerate(groups):
            count=edge_counts[s*30+t]
            assert count<=2**31-1
            syn=b.Synapses(sg,tg,'w:1 (constant)',on_pre='current_post += w',clock=clock,name=f'capacity_projection_{s}_{t}')
            lo,hi=(.0027,.0033) if s<24 else (-.0132,-.0108)
            brian2_rust.connect_fixed_total(syn,count,seed=2026100700+s*30+t,
                initializers={'w':brian2_rust.Uniform(lo,hi)},
                delay_initializer=brian2_rust.ClippedNormal(1.5*b.ms,.25*b.ms,minimum=.1*b.ms,maximum=3*b.ms))
            objects.append(syn)
    with packed_export():model=lower_network(b.Network(*objects),a.steps*clock.dt,rng_seed=20261007)
    with (a.output/'model.json').open('wb') as f:
        for chunk in _canonical_chunks(model):f.write(chunk)
    print(json.dumps({'event':'model_exported','neurons':a.neurons,'edges':sum(edge_counts),'seconds':time.monotonic()-started}),flush=True)
    plan=write_mpi_project(model,a.output/'mpi',ranks=a.ranks,runner=a.source/'target/release/b2-runner',
        population_owners=tuple(range(30)),compact_populations=True,compact_queue_indices=True,compact_spike_history=True,compact_spike_output=True)
    report={'schema':'atlas-brain-fraction-preparation-v1','neurons':a.neurons,'connections':sum(edge_counts),'mean_indegree':a.degree,
        'populations':30,'projections':900,'population_owners':list(range(30)),'excitatory_populations':24,'population_sizes':sizes,'seed':20261007,'dt_ms':.1,'duration_ms':a.steps*.1,'precision':'reference-f64',
        'ranks':a.ranks,'neuron_count_fraction_of_86B_percent':a.neurons/86e9*100,
        'scientific_scope':'Connected synthetic fixed-total random E/I capacity workload; not human brain anatomy, not reproduction of a published brain model.',
        'connection_semantics':'Uniform random source and target with replacement; multapses and autapses permitted; expected mean indegree 1000.',
        'state_observation':'Two neurons per population; complete spike histories retained.',
        'layers':model['protocol']['layers'],'plan_sha256':plan.sha256,'prepare_seconds':time.monotonic()-started,
        'versions':{'python':sys.version,'brian2':b.__version__,'numpy':np.__version__},'compiled':False}
    if a.compile:
        compile_started=time.monotonic()
        compile_mpi_project(a.output/'mpi',mpicc='/atlas-home/0003/workspace/brian2-mpi-primary-20260909/mpi/bin/mpicc',
            rustc='/atlas-home/0003/workspace/brian2-lk-20260906/.cargo/bin/rustc',opt_level=3,panic_strategy='abort')
        report.update(compiled=True,compile_seconds=time.monotonic()-compile_started)
    report['wall_seconds']=time.monotonic()-started
    report['files']={str(f.relative_to(a.output)):{'bytes':f.stat().st_size,'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest()}
        for f in a.output.rglob('*') if f.is_file()}
    (a.output/'prepared.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
