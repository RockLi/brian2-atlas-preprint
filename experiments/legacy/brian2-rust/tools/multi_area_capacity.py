"""Necessary array bounds and deterministic region placement; not job admission.

Queue occupancy, Python/LLVM, monitors, MPI buffers and allocator overhead are
excluded. A small bound cannot authorize a full run; measured guards still apply.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def estimate(model,ranks):
    if type(ranks) is not int or not 1<=ranks<=256:raise ValueError('invalid rank count')
    pops=model['populations'];areas=defaultdict(lambda:{'neurons':0,'edges':0,'offset_bytes':0,'max_projection':0})
    for p in pops:areas[p['area']]['neurons']+=p['count']
    for q in model['projections']:
        area=areas[pops[q['target']]['area']]
        area['edges']+=q['count'];area['offset_bytes']+=(pops[q['source']]['count']+1)*8
        area['max_projection']=max(area['max_projection'],q['count'])
    mapping={};loads=[0]*ranks;rows=[dict(neurons=0,edges=0,offset_bytes=0,max_projection=0) for _ in range(ranks)]
    # Greedy largest incoming-edge area first; no claim of optimal event balance.
    for area in sorted(areas,key=lambda a:(-areas[a]['edges'],a)):
        rank=min(range(ranks),key=lambda r:(loads[r],r));mapping[area]=rank
        for key in ['neurons','edges','offset_bytes']:rows[rank][key]+=areas[area][key]
        rows[rank]['max_projection']=max(rows[rank]['max_projection'],areas[area]['max_projection'])
        loads[rank]+=areas[area]['edges']
    for row in rows:
        # target u32 + weight f64 + delay usize. IDs freed after initialization.
        row['steady_edge_and_csr_bytes']=20*row['edges']+row['offset_bytes']
        row['conservative_build_array_allowance_bytes']=row['steady_edge_and_csr_bytes']+28*row['max_projection']
    return {'schema':'b2-mam-capacity-lower-bound-v1','ranks':ranks,'area_to_rank':mapping,'rank_bounds':rows,
            'old_replicated_csr_offset_bytes_per_rank':sum(a['offset_bytes'] for a in areas.values()),
            'scope':'Necessary array bounds only, not admission or measured memory. Excludes queues, monitors, population initialization, compiler and runtime overhead.',
            'default_procedural_edge_budget_satisfied':max(loads)<=10000000}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parameters',type=Path,required=True)
    p.add_argument('--ranks',type=int,nargs='+',default=[2,4,8]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();model=json.loads(a.parameters.read_text())
    a.output.write_text(json.dumps([estimate(model,r) for r in a.ranks],indent=2)+'\n')
