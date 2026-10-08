"""Supplementary frozen input-decoder time and count audit, never model selection."""
import argparse
from pathlib import Path
import numpy as np
from .motion_refinement import read,sha,load_npz
from .generalization_study import input_grids
from .generalization_readout import transform
from .generalization_data import digest
from .motion_readout import predict
from .pilot import DIRECTIONS
from .direction_study import evaluate
from .causal_motion import paired_interval
from .run_experiment import save


def audit(root):
    p=read(root/'protocol.json');rows=read(root/'rows.json');f=load_npz(root/'features.npz');arrays=load_npz(root/'transform.npz');model=load_npz(root/'input.npz')
    if read(root/'report.json')['status']!='complete':raise ValueError('complete study required')
    result={'scope':'supplementary after selection; one fixed group-level time permutation shared by all directions/phases, seed 916; no fitting or native run',
        'features_sha256':sha(root/'features.npz'),'model_sha256':sha(root/'input.npz'),'protocol_sha256':sha(root/'protocol.json'),
        'source_sha256':sha(Path(__file__)),'speeds':{}}
    for speed in (2,3,5):
        indices=[i for i,r in enumerate(rows) if r['condition']==f'speed_{speed}'];rr=[rows[i] for i in indices];encoded=f['encoded'][indices]
        lookup={(r['group'],*r['phase_index'],r['kind']):i for i,r in enumerate(rr)}
        pairs=[(i,lookup[r['group'],*r['phase_index'],{'right':'left','up':'down'}[r['kind']]]) for i,r in enumerate(rr) if r['kind'] in ('right','up')]
        totals=encoded.sum(1);equal=sum(np.array_equal(totals[a],totals[b]) for a,b in pairs)
        changed=[]
        for counts,row in zip(encoded,rr):
            order=np.arange(60);order[10:50]=np.random.default_rng(np.random.SeedSequence([row['group'],916])).permutation(order[10:50])
            moved=counts[order];np.testing.assert_array_equal(moved.sum(0),counts.sum(0))
            changed.append(transform(input_grids(moved[None],arrays,p['grid_recipe']),p['input_choice']['recipe'])[0])
        y=np.array([DIRECTIONS.index(r['kind']) for r in rr]);original=predict(model,f['input'][indices]).argmax(1);shuffled=predict(model,np.array(changed)).argmax(1)
        result['speeds'][str(speed)]={'opposite_pairs':len(pairs),'identical_total_count_pairs':int(equal),
            'total_count_accuracy_upper_bound':1-.5*equal/len(pairs),'total_counts_sha256':digest(totals),
            'time_shuffle':evaluate(shuffled,y,rr),'original_minus_shuffle':paired_interval(original==y,shuffled==y,rr)}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    result=audit(args.root);path=args.root/'input-audit.json'
    if args.verify:
        if read(path)!=result:raise ValueError('input audit mismatch')
        save(args.root/'input-audit-verification.json',{'all_passed':True,'audit_sha256':sha(path)})
        print({'all_passed':True},flush=True)
    else:
        if path.exists():raise ValueError('audit already exists')
        save(path,result);print({k:{'equal_pairs':v['identical_total_count_pairs'],'shuffle_accuracy':v['time_shuffle']['accuracy']} for k,v in result['speeds'].items()},flush=True)
