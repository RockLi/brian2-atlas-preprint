"""Verify new speed/background stimuli and frozen-readout stress results."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from .motion_stress import stimulus, phase_for, CONDITIONS
from .motion_refinement import read, sha, load_npz, trial_features
from .motion_resolution import temporal_counts
from .refinement import encode_variant
from .simulation import SimulationConfig
from .pilot import DIRECTIONS,scores
from .direction_study import evaluate
from .causal_motion import paired_interval
from .run_experiment import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args()
    root=args.study/'robustness';p=read(root/'protocol.json');r=read(root/'report.json');parent=read(args.study/'protocol.json');rows=read(root/'rows.json')
    f=load_npz(root/'features.npz');arrays=load_npz(args.study/'transform.npz');model=load_npz(args.study/'neural_motion-readout.npz')
    checks={'complete':r['status']=='complete','protocol':sha(root/'protocol.json')==r['protocol_sha256'],
            'parent_protocol':sha(args.study/'protocol.json')==p['parent_protocol_sha256'],'selection':sha(args.study/'selection.json')==p['selection_sha256'],
            'weights':sha(args.study/'neural_motion-readout.npz')==p['readout_sha256'],
            'source':sha(Path(__file__).with_name('motion_stress.py'))==p['source_sha256'],
            'data':sha(root/'features.npz')==r['features_sha256'] and sha(root/'rows.json')==r['rows_sha256'],
            'row_count':len(rows)==160 and r['native_runs']==128,'shared_groups':p['test_groups']==parent['test_groups'],
            'actual_base':sha(root/'execution/base.bin')==read(root/'execution/identity.json')['base_sha256']==read(args.study/'intact/identity.json')['base_sha256'],
            'binary':sha(args.artifact/'compile/native/b2-native')==read(root/'execution/identity.json')['binary_sha256']}
    with np.load(args.study/'features.npz') as pf:base_n=pf['neural_counts'][:128];base_i=pf['input_counts'][:128]
    base_rows=read(args.study/'rows.json')[:128];lookup={(row['group'],*row['phase_index'],row['kind']):i for i,row in enumerate(base_rows)}
    cfg=SimulationConfig(**parent['config']);channels=read(args.artifact/'channels.json')
    endpoints={};checks['movies']=True;checks['inputs']=True;checks['features']=True;checks['baseline_exact']=True
    for j,row in enumerate(rows):
        frames=stimulus(row['kind'],row['group'],row['condition']);h=hashlib.sha256(frames[0].tobytes()).hexdigest()
        endpoints.setdefault((row['condition'],row['group']),[]).append(h)
        checks['movies'] &= np.array_equal(frames[0],frames[-1]) and hashlib.sha256(frames.tobytes()).hexdigest()==row['movie_sha256']
        ii,tt,_=encode_variant(frames,channels,cfg,parent['input_mode'])
        checks['inputs'] &= len(ii)==row['external_spikes'] and hashlib.sha256(np.stack([tt,ii],1).astype('<i8').tobytes()).hexdigest()==row['input_sha256'] and np.array_equal(temporal_counts(ii,tt,np.arange(len(channels)),len(channels)),f['input_counts'][j])
        xx=trial_features(f['neural_counts'][j],f['input_counts'][j],arrays,parent['chosen']['recipe'])['neural_motion']
        checks['features'] &= np.allclose(xx,f['neural_motion'][j],rtol=1e-12,atol=1e-12)
        if row['condition']=='baseline':
            idx=lookup[row['group'],*phase_for(row['kind']),row['kind']]
            checks['baseline_exact'] &= np.array_equal(f['neural_counts'][j],base_n[idx]) and np.array_equal(f['input_counts'][j],base_i[idx]) and row['events_sha256']==base_rows[idx]['events_sha256']
    checks['identical_direction_endpoints']=len(endpoints)==40 and all(len(h)==4 and len(set(h))==1 for h in endpoints.values())
    labels=np.array([DIRECTIONS.index(row['kind']) for row in rows]);pred=scores(model,f['neural_motion']).argmax(1);correct=pred==labels;stats={}
    for i,c in enumerate(CONDITIONS):
        a,b=32*i,32*(i+1);rr=rows[a:b];stats[c]=all(row['condition']==c for row in rr) and evaluate(pred[a:b],labels[a:b],rr)==r['conditions'][c]
        if i:stats[c] &= paired_interval(correct[a:b],correct[:32],rows[:32])==r['paired_vs_baseline'][c]
    result={'all_passed':bool(all(checks.values()) and all(stats.values())),'report_sha256':sha(root/'report.json'),'checks':{k:bool(v) for k,v in checks.items()},'statistics':{k:bool(v) for k,v in stats.items()}}
    save(root/'verification.json',result);print(result,flush=True)
    if not result['all_passed']:raise ValueError('stress verification failed')


if __name__=='__main__':main()
