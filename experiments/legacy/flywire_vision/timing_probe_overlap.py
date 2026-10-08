"""Post-hoc anatomical convergence audit; no stimulus or neural-model changes."""
import argparse
from pathlib import Path
import numpy as np
from brian2_rust.binary_topology import inspect_csr,csr_arrays
from .motion_refinement import read,sha,load_npz
from .run_experiment import save


def audit(root):
    p=read(root/'protocol.json');artifact=Path(p['artifact']);channels=read(artifact/'channels.json');model=read(artifact/'model.json')
    topology=next(s['topology'] for s in model['instance']['synapses'] if s['topology']['kind']=='binary_csr')
    if sha(Path(topology['path']))!=topology['sha256']:raise ValueError('graph changed')
    offsets,targets,values=csr_arrays(inspect_csr(topology['path']));mapping=load_npz(root/'mapping.npz');rows=[]
    for site in p['sites']:
        wanted=set(mapping['cells'][mapping['family']==(site['family']=='Tm1')]);sets=[]
        for patch in ('a','b'):
            hits=set()
            for i in site[patch]:
                cell=channels[i]['index'];a,b=int(offsets[cell]),int(offsets[cell+1])
                hits.update(int(v) for v,w in zip(targets[a:b],values[0,a:b]) if w>0 and int(v) in wanted)
            sets.append(hits)
        rows.append({'site':site['id'],'family':site['family'],'axis':site['axis'],
            'a_direct_targets':len(sets[0]),'b_direct_targets':len(sets[1]),'shared_direct_targets':len(sets[0]&sets[1]),
            'shared_target_indices':sorted(sets[0]&sets[1])})
    return {'scope':'post-hoc topology audit motivated by weak early count interaction; direct positive contacts into recorded same-family T4/T5 only, not complete functional receptive fields or all multi-hop paths; no fitting or new native run',
        'protocol_sha256':sha(root/'protocol.json'),'mapping_sha256':sha(root/'mapping.npz'),'channels_sha256':sha(artifact/'channels.json'),
        'model_sha256':sha(artifact/'model.json'),'graph_sha256':topology['sha256'],'source_sha256':sha(Path(__file__)),'sites':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    result=audit(args.root);path=args.root/'overlap-audit.json'
    if args.verify:
        if read(path)!=result:raise ValueError('overlap audit changed')
        save(args.root/'overlap-audit-verification.json',{'all_passed':True,'audit_sha256':sha(path)});print({'all_passed':True},flush=True)
    else:
        if path.exists():raise ValueError('audit exists')
        save(path,result);print(result['sites'],flush=True)
