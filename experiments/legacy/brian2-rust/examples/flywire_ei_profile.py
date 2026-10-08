"""Profile existing EI native artifacts and quantify target-owner work balance."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from flywire_benchmark import replay
from flywire_device import read_snapshot
from brian2_rust.binary_topology import inspect_csr,csr_arrays


def owners_for(degrees,workers):
    boundaries=[0];target=prefix=0;total=int(degrees.sum())
    for worker in range(1,workers):
        desired=total*worker//workers
        while target<len(degrees) and prefix+int(degrees[target])<=desired:
            prefix+=int(degrees[target]);target+=1
        if target<len(degrees) and desired-prefix>prefix+int(degrees[target])-desired:
            prefix+=int(degrees[target]);target+=1
        boundaries.append(target)
    boundaries.append(len(degrees))
    return np.repeat(np.arange(workers),np.diff(boundaries))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False);a.build=a.build.resolve()
    with np.load(a.build/'snapshot.npz') as d:expected={k:d[k] for k in d.files}
    info=inspect_csr(a.graph/'connectome.b2csr');offsets,targets,_=csr_arrays(info)
    degree=np.zeros(info['target_count'],dtype=np.int64);activity=np.zeros(len(degree))
    for start in range(0,len(targets),65536):
        end=min(start+65536,len(targets));target=targets[start:end].astype(np.int64)
        source=np.searchsorted(offsets,np.arange(start,end),side='right')-1
        degree+=np.bincount(target,minlength=len(degree))
        activity+=np.bincount(target,weights=expected['spike_count'][source],minlength=len(degree))
    # Background's full-population projection joins the shared owner map.
    # The 68-edge sensory projection is below the parallel-route threshold.
    combined=degree+1
    balance={}
    for workers in (1,4,8):
        equal=np.minimum(np.arange(len(degree))*workers//len(degree),workers-1)
        balanced=owners_for(combined,workers)
        row={}
        for name,owners in [('equal_neuron_count',equal),('existing_degree_balanced',balanced)]:
            fixed=np.bincount(owners,weights=combined,minlength=workers)
            events=np.bincount(owners,weights=activity,minlength=workers)
            row[name]={'static_edge_counts':fixed.tolist(),'static_max_over_mean':float(fixed.max()/fixed.mean()),
                       'recurrent_emitted_edge_events':events.tolist(),'event_max_over_mean':float(events.max()/events.mean())}
        balance[str(workers)]=row
    os.environ['B2_AOT_PROFILE_PHASES']='1'
    phases={}
    for workers in (1,4,8):
        result=a.output/f't{workers}'
        replay('aot',a.build,result,workers,expected,reader=read_snapshot)
        phases[str(workers)]=json.loads((result/'summary.json').read_text())
    report={'partition_balance':balance,'profiled_runs':phases,
            'hubs_over_2000_incoming_edges':int(np.count_nonzero(degree>2000)),
            'maximum_indegree':int(degree.max()),
            'scope':'Recurrent emitted edge-event estimates from source spike counts, not weights; includes events scheduled beyond the final step. Profile timings include instrumentation and are excluded from benchmark medians. Existing partitioning was already degree-balanced.'}
    (a.output/'profile.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v['existing_degree_balanced']['event_max_over_mean'] for k,v in balance.items()}))


if __name__=='__main__':main()
