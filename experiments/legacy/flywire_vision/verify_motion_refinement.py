"""Independent regeneration of the selected motion model and new-orbit report."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .motion_refinement import read, sha, load_npz, coarse, trial_features, model_scores, transformed_frames, shuffle_readout_time, MODELS, CONDITIONS
from .motion_readout import cell_weights, transform, fit_readout
from .motion_challenge import movie, endpoint_hashes, parameters
from .causal_motion import paired_interval, motion_features
from .refinement import spatial_projection, encode_variant
from .simulation import SimulationConfig
from .pilot import DIRECTIONS, fit, scores
from .sparse_refinement import fit_sparse
from .direction_study import evaluate
from .run_experiment import save


def verify(root, artifact):
    p=read(root/'protocol.json');r=read(root/'report.json');s=read(root/'selection.json');rows=read(root/'rows.json')
    parent=Path(p['parent']);old=read(parent/'protocol.json');arrays=load_npz(root/'transform.npz')
    checks={'complete':r['status']=='complete','protocol_hash':sha(root/'protocol.json')==r['protocol_sha256']==s['protocol_sha256'],
            'selection_hash':sha(root/'selection.json')==r['selection_sha256'],'selected_before_test':s['test_clips_executed']==0,
            'parent_hash':sha(parent/'protocol.json')==p['parent_protocol_sha256'] and sha(parent/'report.json')==p['parent_report_sha256'],
            'feature_hash':sha(root/'features.npz')==r['features_sha256'],'transform_hash':sha(root/'transform.npz')==p['transform_sha256'],
            'source_hashes':all(sha(Path(__file__).with_name(n))==h for n,h in p['source_sha256'].items()),
            'new_test_groups':not set(p['test_groups'])&set(old['fit_groups']+old['validation_groups']+old['test_groups']),
            'row_count':len(rows)==768==r['logical_rows'],'conditions':tuple(r['conditions'])==CONDITIONS,
            'reset_and_restore':r['static_reset_exact'] and r['intact_rescue_exact'],
            'no_input_readout_overlap':not set(p['input_indices'])&set(p['readout_indices'])}
    parent_map=load_npz(parent/'spatial-map.npz')
    checks['original_audited_map']=sha(parent/'spatial-map.npz')==old['spatial_map_sha256'] and all(np.array_equal(arrays[k],v) for k,v in parent_map.items())
    checks['original_cell_sets']=all(p[k]==old[k] for k in ('input_indices','readout_indices','matched_cut_indices')) and np.array_equal(arrays['cells'],p['readout_indices'])
    checks['polarity_metadata']=all(row['polarity']==('bright' if parameters(row['group'])['polarity']>0 else 'dark') for row in rows)
    identities=read(parent/'identities.json')
    for n in ('intact','cut_input','matched_cut'):
        identity=read(root/n/'identity.json')
        checks[n+'_actual_base']=sha(root/n/'base.bin')==identity['base_sha256']==identities[n]['base_sha256']
        checks[n+'_binary']=sha(artifact/'compile/native/b2-native')==identity['binary_sha256']==identities[n]['binary_sha256']
    checks['development_records_unchanged']=all(sha(Path(d)/'report.json')==v['report_sha256'] and sha(Path(d)/'features.npz')==v['features_sha256'] and sha(Path(d)/'blank.npz')==v['blank_sha256'] and all(sha(Path(d)/name)==h for name,h in v['searches'].items()) for d,v in p['development_records'].items())
    models={n:load_npz(root/(n+'-readout.npz')) for n in MODELS}
    checks['selection_before_native_bases']=(root/'selection.json').stat().st_mtime_ns<min((root/n/'base.bin').stat().st_mtime_ns for n in ('intact','cut_input','matched_cut'))
    expected=max(s['candidates'],key=lambda v:(v['accuracy'],v['recipe']['separation']=='family',v['recipe']['width'],not v['reverse'],v['alpha'],v['input_mode']=='contrast'))
    checks['declared_selection_rule']=expected==p['chosen'] and len(s['candidates'])==s['candidate_count']
    checks['locked_readouts']=all(sha(root/(n+'-readout.npz'))==s['readouts'][n] for n in MODELS)
    checks['old_readouts_exact']=all(sha(root/(n+'-readout.npz'))==sha(parent/(oldname+'-readout.npz')) for n,oldname in [('previous_motion','neural_motion')])
    dev=Path(p['chosen']['directory']);data=load_npz(dev/'features.npz');devrows=read(dev/'rows.json');labels=np.array([DIRECTIONS.index(row['kind']) for row in devrows])
    linear=fit_sparse(coarse(data['neural'][:128]),labels[:128],s['linear_choice']['cells'],'raw',s['linear_choice']['alpha'])
    checks['linear_model_fit_only']=all(np.array_equal(v,models['neural_linear'][k]) for k,v in linear.items())
    checks['fixed_development_blank']=np.array_equal(arrays['blank'],load_npz(dev/'blank.npz')['neural'])
    recipe=p['chosen']['recipe'];weights=cell_weights(data['neural'][:128],arrays['blank'],recipe)
    checks['weights_fit_only']=np.array_equal(weights,arrays['weights'])
    x=transform(data['neural'],arrays['blank'],arrays,weights,recipe)
    m=fit_readout(x[:128],labels[:128],p['chosen']['alpha'],p['chosen']['reverse'])
    checks['selected_model_recomputed']=all(np.array_equal(v,models['neural_motion'][k]) for k,v in m.items())
    checks['selected_validation_recomputed']=evaluate(scores(m,x[128:]).argmax(1),labels[128:],devrows[128:])==s['validation']
    shuffled=labels[:128].reshape(8,16).copy();rng=np.random.default_rng(120926)
    for group in shuffled:rng.shuffle(group)
    m=fit_readout(x[:128],shuffled.ravel(),p['chosen']['alpha'],p['chosen']['reverse'])
    checks['shuffled_model_recomputed']=all(np.array_equal(v,models['labels_shuffled'][k]) for k,v in m.items())
    ip=spatial_projection({'xy':arrays['input_xy'],'valid':np.ones(len(p['input_indices']),dtype=bool)})
    xx=np.stack([motion_features(v,np.zeros_like(v),ip) for v in coarse(data['encoded'][:128])])
    m=fit(xx,labels[:128],s['input_alpha'])
    checks['input_model_fit_only']=all(np.array_equal(v,models['input_motion'][k]) for k,v in m.items())
    checks['endpoint_multisets']=all(all(all(v==list(d.values())[0] for v in d.values()) for d in endpoint_hashes(seed,2.).values()) for seed in p['test_groups'])
    lookup={(row['condition'],row['group'],*row['phase_index'],row['kind']):i for i,row in enumerate(rows)}
    checks['unique_rows']=len(lookup)==len(rows)
    checks['full_orbits']=all(sum(row['condition']==c and row['group']==g for row in rows)==16 for c in CONDITIONS for g in p['test_groups'])
    channels=read(artifact/'channels.json');cfg=SimulationConfig(**p['config']);blanks=load_npz(root/'blank-features.npz')
    f=load_npz(root/'features.npz');truth=np.array([DIRECTIONS.index(row['kind']) for row in rows])
    checks['labels_match']=np.array_equal(truth,f['labels'])
    if (root/'input-count-audit.json').exists():
        audit=read(root/'input-count-audit.json');counts=f['input_counts'][:128];totals=counts.sum(1)
        pairs=[(i,lookup['intact',row['group'],*row['phase_index'],{'right':'left','up':'down'}[row['kind']]]) for i,row in enumerate(rows[:128]) if row['kind'] in ('right','up')]
        checks['supplementary_input_histograms']=hashlib.sha256(counts.tobytes()).hexdigest()==audit['input_counts_prefix_sha256'] and len(pairs)==audit['pairs']==64 and sum(np.array_equal(totals[a],totals[b]) for a,b in pairs)==audit['identical_pairs']==64

    checks['features_regenerated']=True;checks['movies_and_inputs']=True;checks['ablation_inputs_identical']=True;checks['cut_equals_blank']=True
    static={}
    for i,row in enumerate(rows):
        frames=transformed_frames(row,movie(row['kind'],row['group'],2.,tuple(row['phase_index'])),row['condition'])
        checks['movies_and_inputs'] &= np.array_equal(frames[0],frames[-1]) and hashlib.sha256(frames.tobytes()).hexdigest()==row['movie_sha256']
        # Regenerate every visual schedule, not just its motion feature.
        ii,tt,_=encode_variant(frames,channels,cfg,row['input_mode'])
        checks['movies_and_inputs'] &= hashlib.sha256(np.stack([tt,ii],1).astype('<i8').tobytes()).hexdigest()==row['input_sha256'] and len(ii)==row['external_spikes']
        from .motion_resolution import temporal_counts
        checks['movies_and_inputs'] &= np.array_equal(temporal_counts(ii,tt,np.arange(len(channels)),len(channels)),f['input_counts'][i])
        computed=trial_features(f['neural_counts'][i],f['input_counts'][i],arrays,recipe)
        checks['features_regenerated'] &= all(np.allclose(computed[n],f[n][i],rtol=1e-12,atol=1e-12) for n in computed)
        if row['condition'] in ('cut_input','matched_cut'):
            intact=rows[lookup['intact',row['group'],*row['phase_index'],row['kind']]]
            checks['ablation_inputs_identical'] &= row['input_sha256']==intact['input_sha256']
        if row['condition']=='cut_input':checks['cut_equals_blank'] &= np.array_equal(f['neural_counts'][i],blanks['cut_input'])
        if row['condition']=='static_first':
            if row['movie_sha256'] in static:checks['static_same_response']=checks.get('static_same_response',True) and np.array_equal(f['neural_counts'][i],f['neural_counts'][static[row['movie_sha256']]])
            else:static[row['movie_sha256']]=i
    stats={}
    for c in CONDITIONS:
        mask=np.array([row['condition']==c for row in rows]);rr=[row for row in rows if row['condition']==c];stats[c]={}
        for name,m in models.items():
            xx=f['neural_motion' if name=='labels_shuffled' else name][mask];pred=model_scores(name,m,xx).argmax(1)
            stats[c][name]=evaluate(pred,truth[mask],rr)==r['conditions'][c][name]
            if c=='static_first':checks[name+'_static_chance']=r['conditions'][c][name]['accuracy']==.25
    rr=[row for row in rows if row['condition']=='intact'];y=np.array([DIRECTIONS.index(row['kind']) for row in rr])
    for n in MODELS:
        correct={c:np.array(r['conditions'][c][n]['predictions'])==y for c in CONDITIONS}
        checks[n+'_paired']=all(paired_interval(correct[a],correct[b],rr)==r['paired'][n][key] for key,a,b in (
            ('dynamic_minus_static','intact','static_first'),('dynamic_minus_scrambled','intact','scrambled'),
            ('intact_minus_target_cut','intact','cut_input'),('matched_minus_target_cut','matched_cut','cut_input'),
            ('intact_minus_matched_cut','intact','matched_cut')))
    ti=np.stack([shuffle_readout_time(f['neural_counts'][i],arrays,recipe,row['group'],row['phase_index']) for i,row in enumerate(rows) if row['condition']=='intact'])
    checks['readout_shuffle_features']=sha(root/'readout-time-shuffle.npz')==r['readout_time_shuffle_sha256'] and np.array_equal(ti,load_npz(root/'readout-time-shuffle.npz')['features'])
    tp=scores(models['neural_motion'],ti).argmax(1)
    checks['readout_shuffle_statistics']=evaluate(tp,y,rr)==r['readout_time_shuffle'] and paired_interval(np.array(r['conditions']['intact']['neural_motion']['predictions'])==y,tp==y,rr)==r['paired_time_shuffle']
    checks['paired_gain']=paired_interval(np.array(r['conditions']['intact']['neural_motion']['predictions'])==y,np.array(r['conditions']['reference']['previous_motion']['predictions'])==y,rr)==r['paired_gain_vs_p6']
    result={'all_passed':bool(all(checks.values()) and all(all(v.values()) for v in stats.values())),'checks':{k:bool(v) for k,v in checks.items()},'statistics':stats}
    save(root/'verification.json',result);print(json.dumps(result),flush=True)
    if not result['all_passed']:raise ValueError('motion refinement verification failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args();verify(args.directory,args.artifact)
