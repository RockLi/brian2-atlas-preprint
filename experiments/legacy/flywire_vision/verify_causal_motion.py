"""Verify endpoint balance, frozen readouts and paired causal-motion results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .causal_motion import motion_features, paired_interval, CONDITIONS
from .motion_challenge import endpoint_hashes, movie
from .refinement import spatial_projection, linear_scores
from .pilot import DIRECTIONS, scores, fit
from .sparse_refinement import fit_sparse
from .direction_study import evaluate
from .run_experiment import save


def trial_movie(row, travel):
    frames=movie(row['kind'],row['group'],travel,tuple(row['phase_index']))
    if row['condition']=='static_first':return np.repeat(frames[:1],40,axis=0)
    if row['condition']=='scrambled':
        i,j=row['phase_index']
        order=np.random.default_rng(np.random.SeedSequence([row['group'],i,j,551])).permutation(np.arange(1,39))
        return frames[np.r_[0,order,39]].copy()
    return frames


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    root=parser.parse_args().directory
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    p,r,selection,rows=[read(root/(n+'.json')) for n in ('protocol','report','selection','rows')]
    parent=Path(p['parent_study']);parent_p=read(parent/'protocol.json');identities=read(root/'identities.json')
    checks={'complete':r['status']=='complete','protocol_hash':sha(root/'protocol.json')==r['protocol_sha256']==selection['protocol_sha256'],
            'selection_hash':sha(root/'selection.json')==r['selection_sha256'],'selection_before_test':selection['test_clips_executed']==0,
            'feature_hash':sha(root/'features.npz')==r['features_sha256'],'parent_report_hash':sha(parent/'report.json')==p['parent_report_sha256'],
            'sources_unchanged':all(sha(Path(__file__).with_name(n))==h for n,h in p['sources_sha256'].items()),
            'spatial_map_hash':sha(root/'spatial-map.npz')==p['spatial_map_sha256'],
            'splits_disjoint':len(set(p['fit_groups']+p['validation_groups']+p['test_groups']))==20,
            'row_count':len(rows)==832==r['logical_rows'],
            'no_input_readout_overlap':not set(p['readout_indices'])&set(p['input_indices']),
            'matched_mask_count':len(set(p['matched_cut_indices']))==len(p['input_indices'])==1535,
            'matched_excludes_input_and_readout':not set(p['matched_cut_indices'])&set(p['input_indices']+p['readout_indices']),
            'all_conditions':tuple(r['conditions'])==CONDITIONS,
            'static_reset_and_intact_rescue':r['static_reset_exact'] and r['intact_rescue_exact']}
    for name,identity in identities.items():
        recorded=read(root/name/'identity.json')
        checks[name+'_actual_base']=sha(root/name/'base.bin')==identity['base_sha256']==recorded['base_sha256']
        checks[name+'_binary']=identity['binary_sha256']==recorded['binary_sha256']==parent_p['binary_sha256']
    checks['intact_parent_base']=identities['intact']['base_sha256']==parent_p['base_sha256']
    checks['endpoint_balance']=True
    for seed in p['fit_groups']+p['validation_groups']+p['test_groups']:
        for by_kind in endpoint_hashes(seed,p['travel']).values():
            checks['endpoint_balance'] &= all(v==list(by_kind.values())[0] for v in by_kind.values())
    lookup={(row['split'],row['condition'],row['group'],*row['phase_index'],row['kind']):i for i,row in enumerate(rows)}
    checks['unique_logical_rows']=len(lookup)==len(rows)
    checks['all_full_orbits']=all(sum(row['split']==split and row['group']==group and row['condition']==condition for row in rows)==16
        for split in ('fit','validation','test') for group in p[split+'_groups'] for condition in (CONDITIONS if split=='test' else ('intact',)))
    checks['all_movie_hashes_and_closed_endpoints']=True
    for row in rows:
        frames=trial_movie(row,p['travel'])
        checks['all_movie_hashes_and_closed_endpoints'] &= hashlib.sha256(frames.tobytes()).hexdigest()==row['movie_sha256'] and np.array_equal(frames[0],frames[-1])
    checks['causal_inputs_identical']=all(row['input_sha256']==rows[lookup['test','intact',row['group'],*row['phase_index'],row['kind']]]['input_sha256']
        for row in rows if row['condition'] in ('cut_input','matched_cut'))
    models={}
    for name,expected in selection['readouts'].items():
        path=root/(name+'-readout.npz')
        checks[name+'_weights_locked']=sha(path)==expected['readout_sha256'] and path.stat().st_mtime_ns<=(root/'selection.json').stat().st_mtime_ns
        with np.load(path,allow_pickle=False) as m:models[name]=dict(m)
    with np.load(root/'spatial-map.npz',allow_pickle=False) as m:mapping=dict(m)
    projection=spatial_projection(mapping)
    with np.load(root/'blank-features.npz',allow_pickle=False) as b:blank=dict(b)
    statistics={}
    with np.load(root/'features.npz',allow_pickle=False) as f:
        y=np.asarray([DIRECTIONS.index(row['kind']) for row in rows])
        checks['label_order']=np.array_equal(y,f['labels'])
        shuffled=y[:128].reshape(8,16).copy();rng=np.random.default_rng(111926)
        for group in shuffled:rng.shuffle(group)
        for name,model in models.items():
            x=f['neural_motion' if name=='labels_shuffled' else name][:128]
            info=selection['readouts'][name]
            trained=(fit_sparse(x,y[:128],info['cells'],'raw',info['alpha']) if name=='neural_linear' else
                     fit(x,shuffled.ravel() if name=='labels_shuffled' else y[:128],info['alpha']))
            checks[name+'_fit_only_model_recomputed']=all(np.array_equal(trained[k],model[k]) for k in trained)
        checks['motion_features_recomputed']=all(np.allclose(motion_features(raw,blank['intact'],projection),actual,rtol=1e-12,atol=1e-12) for raw,actual in zip(f['neural_linear'],f['neural_motion']))
        checks['cut_downstream_equals_cut_blank']=all(np.array_equal(f['neural_linear'][i],blank['cut_input']) for i,row in enumerate(rows) if row['condition']=='cut_input')
        static={}
        for i,row in enumerate(rows):
            if row['condition']=='static_first':
                key=row['movie_sha256']
                if key in static:
                    checks['static_identical_images_identical_features']=checks.get('static_identical_images_identical_features',True) and all(np.array_equal(f[n][i],f[n][static[key]]) for n in ('neural_linear','neural_motion','input_motion'))
                else:static[key]=i
        for condition,result in r['conditions'].items():
            mask=np.asarray([row['split']=='test' and row['condition']==condition for row in rows]);selected_rows=[row for row,m in zip(rows,mask) if m]
            statistics[condition]={}
            for name,m in models.items():
                x=f['neural_motion' if name=='labels_shuffled' else name][mask]
                pred=(linear_scores(m,x) if name=='neural_linear' else scores(m,x)).argmax(1)
                statistics[condition][name]=evaluate(pred,y[mask],selected_rows)==result[name]
                if condition=='static_first':checks[name+'_static_exact_chance']=result[name]['accuracy']==.25
        testrows=[row for row in rows if row['split']=='test' and row['condition']=='intact'];truth=np.array([DIRECTIONS.index(row['kind']) for row in testrows])
        for name in models:
            correct={c:np.asarray(r['conditions'][c][name]['predictions'])==truth for c in CONDITIONS}
            comparisons={'dynamic_minus_static':('intact','static_first'),'intact_minus_target_cut':('intact','cut_input'),
                         'intact_minus_matched_cut':('intact','matched_cut'),'matched_minus_target_cut':('matched_cut','cut_input'),
                         'dynamic_minus_scrambled':('intact','scrambled')}
            checks[name+'_paired_statistics']=all(paired_interval(correct[a],correct[b],testrows)==r['paired'][name][key] for key,(a,b) in comparisons.items())
    passed=all(checks.values()) and all(all(v.values()) for v in statistics.values())
    save(root/'verification.json',{'all_passed':bool(passed),'checks':{k:bool(v) for k,v in checks.items()},'statistics':statistics})
    print(json.dumps({'all_passed':bool(passed),'checks':{k:bool(v) for k,v in checks.items()}}))
    if not passed:raise ValueError('causal motion verification failed')


if __name__=='__main__':main()
