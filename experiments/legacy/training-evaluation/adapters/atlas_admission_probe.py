"""Admission-only probe: materializes reproducible E1 arrays, never executes a step."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from atlas_adapter import admission, load_api, _canonical_identity
from oracle import fixture


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);args=parser.parse_args();base=args.base.resolve()
    output=base/'evidence/atlas-e1-admission.json'
    if output.exists():raise FileExistsError('preserve prior evidence')
    api=load_api(base/'snapshot/brian2-rust')
    report=dict(schema='atlas-e1-admission-v1',scope='initial request only; no training or benchmark run',
        fixture_generator_sha256=hashlib.sha256((base/'adapters/oracle.py').read_bytes()).hexdigest(),
        numpy_version=np.__version__,probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=[])
    for name,sizes,batch,ticks in [('E1-small',[128,128,10],16,128),('E1-large',[512,1024,1024,20],32,128)]:
        case=fixture(seed=11,sizes=sizes,B=batch,T=ticks)
        plan=api.lif_training_plan(sizes,beta=.95,threshold=1.,learning_rate=.001,seed=1,max_tape_bytes=1024**3)
        state=dict(weights=case['weights'],first_moment=[[0.]*len(w) for w in case['weights']],second_moment=[[0.]*len(w) for w in case['weights']],step=0,rng=1)
        result=dict(case_id=name,seed=11,sizes=sizes,batch=batch,ticks=ticks,
            arrays_sha256={k:hashlib.sha256(np.asarray(v,dtype='<f8').tobytes()).hexdigest() for k,v in [('inputs',case['inputs']),*[(f'weights_{j}',w) for j,w in enumerate(case['weights'])]]},
            admission=admission(plan,state,case['inputs'],case['labels']))
        report['results'].append(result)
        print(json.dumps(dict(case=name,admission=result['admission'])),flush=True)
        del case,plan,state
    # Numeric lower bounds only; dataset T and actual count arrays are unavailable.
    report['shd']=dict(status='unqualified',reason='dataset max event time, actual count arrays and masks not frozen',
        B=32,I=700,hidden=256,classes=20,
        input_only_lower_bound_float_json='32*(T*(4*700+2)+2)+1',
        input_only_lower_bound_integer_json='32*(T*(2*700+2)+2)+1',
        float_input_only_rejected_from_T=(67108864-65)//(32*2802)+1,
        integer_input_only_rejected_from_T=(67108864-65)//(32*1402)+1,
        note='integer count JSON is lossless and accepted by Vec<f64>; neither bound alone is an actual SHD result')
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
