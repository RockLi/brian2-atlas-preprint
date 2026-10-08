"""Paired canonical/per-edge delayed pre/post delivery; state updates stay parallel."""
import argparse,json
from contextlib import contextmanager
from pathlib import Path
import gpu_synapse_state_benchmark as common

POLICIES=common.POLICIES


@contextmanager
def policy(name):
    from brian2_rust import metal_synapses
    old=metal_synapses.independent_pathway
    if name=='canonical':metal_synapses.independent_pathway=lambda syn,code:False
    elif name!='parallel':raise ValueError(name)
    try:yield
    finally:metal_synapses.independent_pathway=old


def benchmark(backend,neurons,degree,steps,mixed,repeats,output):
    old=common.policy;common.policy=policy
    try:r=common.benchmark(backend,neurons,degree,steps,mixed,repeats,output)
    finally:common.policy=old
    r.update(schema='b2-edge-pathway-ablation-v0',
        ablation='Only independent pre/post pathway policy changes; both policies retain parallel synapse state updates.')
    (output/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['metal','cuda'],default='cuda');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--neurons',type=int,default=256);p.add_argument('--degree',type=int,default=128)
    p.add_argument('--steps',type=int,default=128);p.add_argument('--repeats',type=int,default=7);p.add_argument('--mixed',action='store_true')
    a=p.parse_args()
    if not (1<=a.neurons<=512 and 1<=a.degree<=128 and 1<=a.steps<=512 and 1<=a.repeats<=10):p.error('bounded workload exceeded')
    r=benchmark(a.backend,a.neurons,a.degree,a.steps,a.mixed,a.repeats,a.output)
    print(json.dumps(dict(summary=r['summary'],paired=r['paired']),indent=2))


if __name__=='__main__':main()
