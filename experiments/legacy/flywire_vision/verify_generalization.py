"""Independent reconstruction of held-speed selection, native inputs and test statistics."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from .motion_refinement import read,sha,load_npz
from .motion_readout import fit_readout,predict
from .multispeed_readout import bank_features
from .multispeed_study import grids_for
from .generalization_readout import from_pairs,candidates,rank
from .generalization_study import development_arrays,features_for,CONDITIONS,MODELS,REFERENCE
from .generalization_data import generalization_movie,digest
from .motion_stress import phase_for
from .motion_challenge import parameters
from .motion_resolution import temporal_counts
from .refinement import encode_variant
from .simulation import SimulationConfig
from .pilot import DIRECTIONS
from .direction_study import evaluate
from .causal_motion import paired_interval
from .run_experiment import save


def verify(root,artifact):
    p,s,r=[read(root/(n+'.json')) for n in ('protocol','selection','report')]
    rows=read(root/'rows.json');f=load_npz(root/'features.npz');arrays=load_npz(root/'transform.npz')
    parent=Path(p['parent']);neural=Path(p['neural_parent']);p8=read(parent/'protocol.json');pp=read(neural/'protocol.json')
    checks={'complete':r['status']=='complete','protocol':sha(root/'protocol.json')==r['protocol_sha256']==s['protocol_sha256'],
        'selection':sha(root/'selection.json')==r['selection_sha256'] and s['test_clips_executed']==0,
        'parent':sha(parent/'protocol.json')==p['parent_protocol_sha256'] and sha(parent/'report.json')==p['parent_report_sha256'] and sha(neural/'protocol.json')==p['neural_protocol_sha256'],
        'sources':all(sha(Path(__file__).with_name(n))==h for n,h in p['source_sha256'].items()),
        'data':sha(root/'features.npz')==r['features_sha256'] and sha(root/'rows.json')==r['rows_sha256'],
        'transform':sha(root/'transform.npz')==p['transform_sha256'],
        'new_groups':not set(p['test_groups'])&set(p8['test_groups']+pp['test_groups']+pp['fit_groups']+pp['validation_groups']),
        'unseen_speeds':not {3,5}&set(p['speeds_development']),
        'rows':len(rows)==576 and len({(v['condition'],v['group'],*v['phase_index'],v['kind']) for v in rows})==576,
        'polarity':all(v['polarity']==('bright' if parameters(v['group'])['polarity']>0 else 'dark') for v in rows)}
    checks['original_mapping_blank']=all(np.array_equal(v,arrays[k]) for k,v in load_npz(neural/'transform.npz').items())
    channels=read(artifact/'channels.json');cfg=SimulationConfig(**pp['config'])
    checks['input_mapping']=np.array_equal(arrays['input_family'],[0 if c['cell_type']=='Mi1' else 1 for c in channels])
    models={n:load_npz(root/(n+'.npz')) for n in MODELS}
    checks['models']=all(sha(root/(n+'.npz'))==p['readouts'][n] for n in MODELS)
    checks['p8_exact']=all(np.array_equal(v,models['p8'][k]) for k,v in load_npz(parent/'multispeed.npz').items())
    for name in ('intact','cut_input','matched_cut'):
        identity=read(root/name/'identity.json');old=read(neural/name/'identity.json')
        checks[name+'_base']=sha(root/name/'base.bin')==identity['base_sha256']==old['base_sha256']
        checks[name+'_binary']=sha(artifact/'compile/native/b2-native')==identity['binary_sha256']==old['binary_sha256']
        # Hardlinked bases retain old timestamps. The newly created identity
        # record marks construction of each new checked runner after freezing.
        checks[name+'_locked_before_test']=(root/'selection.json').stat().st_mtime_ns<(root/name/'identity.json').stat().st_mtime_ns
    checks['development_hashes']=all(all(sha(Path(d['directory'])/(n+'.json'))==d[n+'_sha256'] for n in ('report','rows','protocol')) and sha(Path(d['directory'])/'features.npz')==d['features_sha256'] for d in p['development'])
    checks['development_inputs']=True
    for speed,d in zip((1,2,4),p['development']):
        directory=Path(d['directory']);dd=load_npz(directory/'features.npz')
        for idx,row in enumerate(read(directory/'rows.json')):
            frames=generalization_movie(row['kind'],row['group'],row['phase_index'],speed)
            ii,tt,_=encode_variant(frames,channels,cfg,pp['input_mode'])
            checks['development_inputs'] &= digest(np.stack([tt,ii],1).astype('<i8'))==row['input_sha256'] and np.array_equal(dd['encoded'][idx],temporal_counts(ii,tt,np.arange(len(channels)),len(channels)))
    devrows,npairs,ipairs=development_arrays(p,arrays);labels=np.array([DIRECTIONS.index(v['kind']) for v in devrows]);fitmask=np.array([v['split']=='fit' for v in devrows])
    checks['development_split']=len(devrows)==576 and fitmask.sum()==384 and all(set(v['group'] for v in devrows if v['split']==split)==set(p[key]) for split,key in (('fit','fit_groups'),('validation','validation_groups')))
    regenerated={};checks['fold_exclusion']=True
    for name,pairs in (('neural',npairs),('input',ipairs)):
        options=[]
        for recipe in candidates():
            x=from_pairs(pairs,recipe)
            for alpha in (.01,.1,1.,10.,100.):
                acc=[]
                for held in (1,2,4):
                    tm=np.array([v['split']=='fit' and v['speed']!=held for v in devrows]);vm=np.array([v['split']=='validation' and v['speed']==held for v in devrows])
                    checks['fold_exclusion'] &= tm.sum()==256 and vm.sum()==64 and not set(v['group'] for v,k in zip(devrows,tm) if k)&set(v['group'] for v,k in zip(devrows,vm) if k)
                    m=fit_readout(x[tm],labels[tm],alpha,True);acc.append(float(np.mean(predict(m,x[vm]).argmax(1)==labels[vm])))
                options.append({'recipe':recipe,'alpha':alpha,'held_speed_accuracy':acc,'features':x.shape[1]})
        regenerated[name]=options
    checks['all_candidates']=regenerated['neural']==s['candidates'] and regenerated['input']==s['input_candidates']
    checks['selection_rule']=max(regenerated['neural'],key=rank)==p['chosen'] and max([v for v in regenerated['neural'] if v['recipe']==REFERENCE],key=rank)==p['fixed_retrained'] and max(regenerated['input'],key=rank)==p['input_choice']
    for name,choice,pairs in (('generalized',p['chosen'],npairs),('fixed_retrained',p['fixed_retrained'],npairs),('input',p['input_choice'],ipairs),('p8',{'recipe':REFERENCE},npairs)):
        x=from_pairs(pairs,choice['recipe'])
        if name!='p8':
            m=fit_readout(x[fitmask],labels[fitmask],choice['alpha'],True)
            checks[name+'_fit_only']=all(np.array_equal(v,models[name][k]) for k,v in m.items())
        checks[name+'_validation']=True
        for speed in (1,2,4):
            vm=np.array([v['split']=='validation' and v['speed']==speed for v in devrows]);rr=[v for v,k in zip(devrows,vm) if k]
            checks[name+'_validation'] &= evaluate(predict(models[name],x[vm]).argmax(1),labels[vm],rr)==s['validation'][name][str(speed)]
    print({'development_verified':all(checks.values())},flush=True)
    lookup={(v['condition'],v['group'],*v['phase_index'],v['kind']):i for i,v in enumerate(rows)};blanks=load_npz(root/'blanks.npz');endpoints={};static={}
    checks.update({k:True for k in ('movies','inputs','features','shuffle','cut_blank','lesion_same_input','static_same','original_p8_features','original_p8_predictions')})
    for idx,row in enumerate(rows):
        c=row['condition'];frames=generalization_movie(row['kind'],row['group'],row['phase_index'],row['speed'])
        if c=='static_first':frames=np.repeat(frames[:1],40,axis=0)
        checks['movies'] &= digest(frames)==row['movie_sha256'] and np.array_equal(frames[0],frames[-1])
        endpoints.setdefault((c,row['group']),{k:Counter() for k in DIRECTIONS})[row['kind']][digest(frames[0])]+=1
        ii,tt,_=encode_variant(frames,channels,cfg,pp['input_mode'])
        checks['inputs'] &= digest(np.stack([tt,ii],1).astype('<i8'))==row['input_sha256'] and len(ii)==row['external_spikes'] and np.array_equal(f['encoded'][idx],temporal_counts(ii,tt,np.arange(len(channels)),len(channels)))
        ff=features_for(f['neural'][idx],f['encoded'][idx],arrays,p)
        checks['features'] &= all(np.allclose(ff[n],f[n][idx],atol=1e-12,rtol=1e-12) for n in MODELS)
        old=bank_features(grids_for(f['neural'][idx:idx+1],arrays,p['grid_recipe']),p8['chosen']['recipe'])
        checks['original_p8_features'] &= np.allclose(old[0],f['p8'][idx],atol=1e-12,rtol=1e-12)
        if c.startswith('speed_'):checks['original_p8_predictions'] &= int(predict(models['p8'],old).argmax(1)[0])==int(predict(models['p8'],f['p8'][idx:idx+1]).argmax(1)[0])
        order=np.arange(60);order[10:50]=np.random.default_rng(np.random.SeedSequence([row['group'],*row['phase_index'],915])).permutation(order[10:50])
        checks['shuffle'] &= np.allclose(features_for(f['neural'][idx],f['encoded'][idx],arrays,p,order)['generalized'],f['shuffled'][idx],atol=1e-12,rtol=1e-12)
        if c in ('cut_speed_3','matched_speed_3'):
            ref=lookup['speed_3',row['group'],*row['phase_index'],row['kind']]
            checks['lesion_same_input'] &= rows[ref]['input_sha256']==row['input_sha256'] and np.array_equal(f['input'][ref],f['input'][idx]) and tuple(row['phase_index'])==phase_for(row['kind'])
        if c=='cut_speed_3':checks['cut_blank'] &= np.array_equal(f['neural'][idx],blanks['cut_input'])
        if c=='static_first':
            h=row['movie_sha256']
            if h in static:checks['static_same'] &= np.array_equal(f['neural'][idx],f['neural'][static[h]])
            else:static[h]=idx
    checks['endpoints']=len(endpoints)==48 and all(all(v==next(iter(d.values())) for v in d.values()) for d in endpoints.values())
    checks['complete_groups']=all(sum(v['condition']==c and v['group']==g for v in rows)==(4 if c in ('cut_speed_3','matched_speed_3') else 16) for c in CONDITIONS for g in p['test_groups'])
    restored=read(root/'restoration.json');checks['reset']=all(rows[0][k]==restored[k] for k in ('events_sha256','states_sha256','input_sha256')) and r['reset_exact']
    checks['native_count']=r['native_runs']==sum(not v['reused'] for v in rows)+4
    truth=np.array([DIRECTIONS.index(v['kind']) for v in rows]);pred={n:predict(m,f[n]).argmax(1) for n,m in models.items()};stats={}
    for c in CONDITIONS:
        mask=np.array([v['condition']==c for v in rows]);rr=[v for v,k in zip(rows,mask) if k]
        stats[c]=all(evaluate(y[mask],truth[mask],rr)==r['conditions'][c][n] for n,y in pred.items()) and paired_interval(pred['generalized'][mask]==truth[mask],pred['p8'][mask]==truth[mask],rr)==r['paired_vs_p8'][c]
        if c=='static_first':stats[c] &= all(v['accuracy']==.25 for v in r['conditions'][c].values())
        if c=='cut_speed_3':stats[c] &= all(r['conditions'][c][n]['accuracy']==.25 for n in MODELS if n!='input')
    mask=np.array([v['condition'] in ('speed_3','speed_5') for v in rows]);rr=[v for v,k in zip(rows,mask) if k]
    stats['primary']=all(evaluate(y[mask],truth[mask],rr)==r['primary'][n] for n,y in pred.items()) and paired_interval(pred['generalized'][mask]==truth[mask],pred['p8'][mask]==truth[mask],rr)==r['primary_gain']
    subset=np.array([v['condition']=='speed_3' and tuple(v['phase_index'])==phase_for(v['kind']) for v in rows]);rr=[v for v,k in zip(rows,subset) if k]
    stats['subset']=all(evaluate(y[subset],truth[subset],rr)==r['lesion_subset_baseline'][n] for n,y in pred.items())
    for c in ('cut_speed_3','matched_speed_3'):
        mask=np.array([v['condition']==c for v in rows]);stats[c+'_effect']=paired_interval(pred['generalized'][subset]==truth[subset],pred['generalized'][mask]==truth[mask],rr)==r['lesion_effects'][c]
    for speed in (2,3,5):
        mask=np.array([v['condition']==f'speed_{speed}' for v in rows]);rr=[v for v,k in zip(rows,mask) if k];yp=predict(models['generalized'],f['shuffled'][mask]).argmax(1)
        stats[f'shuffle_{speed}']=evaluate(yp,truth[mask],rr)==r['time_shuffle'][str(speed)]['result'] and paired_interval(pred['generalized'][mask]==truth[mask],yp==truth[mask],rr)==r['time_shuffle'][str(speed)]['intact_minus_shuffle']
    result={'all_passed':bool(all(checks.values()) and all(stats.values())),'report_sha256':sha(root/'report.json'),'checks':{k:bool(v) for k,v in checks.items()},'statistics':{k:bool(v) for k,v in stats.items()}}
    save(root/'verification.json',result);print(result,flush=True)
    if not result['all_passed']:raise ValueError('generalization verification failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--artifact',type=Path,required=True)
    args=parser.parse_args();verify(args.root,args.artifact)
