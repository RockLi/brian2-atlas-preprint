"""Frozen new-orbit evaluation of motion readout and visual frontend refinements."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .motion_readout import cell_weights, transform, fit_readout, predict
from .motion_resolution import temporal_counts
from .motion_challenge import balanced_orbit_seeds, iter_orbit, endpoint_hashes
from .causal_motion import motion_features, paired_interval, intervene
from .refinement import encode_variant, spatial_projection, linear_scores
from .pilot import DIRECTIONS, fit, scores
from .sparse_refinement import fit_sparse
from .simulation import SimulationConfig
from .direction_study import evaluate
from .run_experiment import save

CONDITIONS=('intact','cut_input','matched_cut','static_first','scrambled','reference')
MODELS=('neural_linear','previous_motion','neural_motion','input_motion','labels_shuffled')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def coarse(counts):
    return np.asarray(counts,dtype=np.float64)[...,10:50,:].reshape(*np.shape(counts)[:-2],8,5,np.shape(counts)[-1]).sum(-2).reshape(*np.shape(counts)[:-2],-1)


def select(args):
    args.output.mkdir(parents=True,exist_ok=False)
    parent=read(args.parent/'protocol.json');options=[]
    for directory in args.development:
        report=read(directory/'report.json')
        if report['status']!='complete' or sha(directory/'features.npz')!=report['features_sha256']:raise ValueError('development data changed')
        for search in sorted(directory.glob('development-search*.json')):
            candidate_rows=read(search)
            if len(candidate_rows)!=96:raise ValueError('development search incomplete')
            for result in candidate_rows:
                options.append({**result,'directory':str(directory),'input_mode':read(directory/'protocol.json').get('input_mode','contrast')})
    # Explicit tie break: smaller anatomical feature set, coarser bins, fewer
    # transformations, stronger regularization; never inspect held-out scores.
    chosen=max(options,key=lambda r:(r['accuracy'],r['recipe']['separation']=='family',r['recipe']['width'],not r['reverse'],r['alpha'],r['input_mode']=='contrast'))
    directory=Path(chosen['directory']);data=load_npz(directory/'features.npz');blank=load_npz(directory/'blank.npz')['neural']
    rows=read(directory/'rows.json');labels=np.array([DIRECTIONS.index(r['kind']) for r in rows])
    if len(rows)!=192 or any(r['split']!=('fit' if i<128 else 'validation') for i,r in enumerate(rows)):raise ValueError('invalid development split')
    mapping=load_npz(args.parent/'spatial-map.npz');groups=read(args.artifact/'groups.json')
    mapping['subtype']=np.concatenate([np.full(len(groups[f'T{n}{a}']),idx) for idx,(n,a) in enumerate((n,a) for n in [4,5] for a in 'abcd')]+[np.full(len(groups['LC4']),8)])
    weights=cell_weights(data['neural'][:128],blank,chosen['recipe'])
    features=transform(data['neural'],blank,mapping,weights,chosen['recipe'])
    model=fit_readout(features[:128],labels[:128],chosen['alpha'],chosen['reverse'])
    validation=evaluate(predict(model,features[128:]).argmax(1),labels[128:],rows[128:])
    if validation['accuracy']!=chosen['accuracy']:raise ValueError('selected validation not reproduced')
    shuffled=labels[:128].reshape(8,16).copy();rng=np.random.default_rng(120926)
    for group in shuffled:rng.shuffle(group)
    control=fit_readout(features[:128],shuffled.ravel(),chosen['alpha'],chosen['reverse'])
    projection=spatial_projection({'xy':mapping['input_xy'],'valid':np.ones(len(parent['input_indices']),dtype=bool)})
    input_features=np.stack([motion_features(x,np.zeros_like(x),projection) for x in coarse(data['encoded'])])
    candidates=[]
    for alpha in (.01,.1,1.,10.):
        m=fit(input_features[:128],labels[:128],alpha)
        pred=scores(m,input_features[128:]).argmax(1)
        candidates.append((float(np.mean(pred==labels[128:])),alpha,m))
    _,input_alpha,input_model=max(candidates,key=lambda v:v[:2])
    for name,m in (('neural_motion',model),('labels_shuffled',control),('input_motion',input_model)):
        np.savez_compressed(args.output/(name+'-readout.npz'),**m)
    linear_candidates=[];raw=coarse(data['neural'])
    for count in (32,128,512):
        for alpha in (.01,.1,1.):
            m=fit_sparse(raw[:128],labels[:128],count,'raw',alpha)
            acc=float(np.mean(linear_scores(m,raw[128:]).argmax(1)==labels[128:]))
            linear_candidates.append((acc,-count,alpha,m))
    linear_accuracy,negative_count,linear_alpha,linear_model=max(linear_candidates,key=lambda v:v[:3])
    np.savez_compressed(args.output/'neural_linear-readout.npz',**linear_model)
    shutil.copy2(args.parent/'neural_motion-readout.npz',args.output/'previous_motion-readout.npz')
    np.savez_compressed(args.output/'transform.npz',blank=blank,weights=weights,**mapping)
    protocol={'schema':'flywire-motion-refinement-v1','parent':str(args.parent),'parent_protocol_sha256':sha(args.parent/'protocol.json'),
              'parent_report_sha256':sha(args.parent/'report.json'),'development':[str(p) for p in args.development],
              'development_records':{str(d):{'report_sha256':sha(d/'report.json'),'features_sha256':sha(d/'features.npz'),'blank_sha256':sha(d/'blank.npz'),'searches':{q.name:sha(q) for q in sorted(d.glob('development-search*.json'))}} for d in args.development},
              'chosen':chosen,'input_mode':chosen['input_mode'],'test_groups':balanced_orbit_seeds(1120000,8),
              'fit_groups':parent['fit_groups'],'validation_groups':parent['validation_groups'],
              'travel':2.,'conditions':CONDITIONS,'models':MODELS,'config':parent['config'],
              'readout_indices':parent['readout_indices'],'input_indices':parent['input_indices'],
              'matched_cut_indices':parent['matched_cut_indices'],'transform_sha256':sha(args.output/'transform.npz'),
              'source_sha256':{n:sha(Path(__file__).with_name(n)) for n in ('motion_refinement.py','motion_readout.py','motion_resolution.py','motion_frontend.py','motion_search.py','motion_challenge.py','causal_motion.py','refinement.py','simulation.py','pilot.py','sparse_refinement.py','direction_study.py')},
              'scope':'same full-cycle task and neural model; external readout and optional causal frontend selected only on development; old P6 test not used',
              'paired_reference':'reference condition uses original contrast input on the same new orbits; previous_motion is the original frozen P6 motion decoder',
              'static_reuse':'identical static movies reuse exactly identical frozen inputs/results; reference reuses intact only when selected input is contrast'}
    if set(protocol['test_groups'])&set(parent['fit_groups']+parent['validation_groups']+parent['test_groups']):raise ValueError('test group overlap')
    save(args.output/'protocol.json',protocol)
    selection={'test_clips_executed':0,'protocol_sha256':sha(args.output/'protocol.json'),'validation':validation,'input_alpha':input_alpha,'linear_choice':{'cells':-negative_count,'alpha':linear_alpha,'validation_accuracy':linear_accuracy},
               'readouts':{n:sha(args.output/(n+'-readout.npz')) for n in MODELS},'candidate_count':len(options),'candidates':options}
    save(args.output/'selection.json',selection)
    print(json.dumps({'chosen':chosen,'validation':validation,'candidates':len(options)}),flush=True)


def transformed_frames(metadata, frames, condition):
    if condition=='static_first':return np.repeat(frames[:1],40,axis=0)
    if condition=='scrambled':
        i,j=metadata['phase_index'];order=np.random.default_rng(np.random.SeedSequence([metadata['group'],i,j,712])).permutation(np.arange(1,39))
        return frames[np.r_[0,order,39]].copy()
    return frames


def trial_features(neural, encoded, arrays, recipe):
    raw=coarse(neural);ip=spatial_projection({'xy':arrays['input_xy'],'valid':np.ones(encoded.shape[1],dtype=bool)})
    return {'neural_linear':raw,'previous_motion':motion_features(raw,coarse(arrays['blank']),spatial_projection(arrays)),
            'neural_motion':transform(neural[None],arrays['blank'],arrays,arrays['weights'],recipe)[0],
            'input_motion':motion_features(coarse(encoded),np.zeros(encoded.shape[1]*8),ip)}


def shuffle_readout_time(neural, arrays, recipe, group, phase):
    """Feature-only temporal ablation of actual recorded counts.

    Reorder counts and the corresponding blank together within the selected
    window, preserving every cell's count histogram and avoiding a new input
    schedule or a spurious baseline mismatch. This is not a native trial.
    """
    order=np.arange(60)
    start,stop=int(recipe['start']),int(recipe['stop'])
    order[start:stop]=np.random.default_rng(np.random.SeedSequence([group,*phase,713])).permutation(order[start:stop])
    return transform(neural[None,order],arrays['blank'][order],arrays,arrays['weights'],recipe)[0]


def model_scores(name,model,features):
    return linear_scores(model,features) if name=='neural_linear' else scores(model,features)


def run(args):
    p=read(args.output/'protocol.json');selection=read(args.output/'selection.json')
    if selection['protocol_sha256']!=sha(args.output/'protocol.json') or any(sha(Path(__file__).with_name(n))!=h for n,h in p['source_sha256'].items()):raise ValueError('locked protocol or sources changed')
    from flywire_mnist.backends.frozen_cpu import FrozenCPU
    model=read(args.artifact/'model.json');channels=read(args.artifact/'channels.json');cfg=SimulationConfig(**p['config'])
    ni=next(i for i,pop in enumerate(model['definition']['populations']) if pop['name']=='flywire_neurons')
    neurons=model['definition']['populations'][ni]['count'];cells=np.array(p['readout_indices'])
    identities=read(args.parent/'identities.json');runners={}
    for name,mask in (('intact',None),('cut_input',p['input_indices']),('matched_cut',p['matched_cut_indices'])):
        template=model if mask is None else intervene(model,mask)
        runners[name]=FrozenCPU(template,args.artifact/'compile/native',args.output/name,population='visual_input',threads=4)
        if runners[name].base_hash!=identities[name]['base_sha256'] or runners[name].binary_hash!=identities[name]['binary_sha256']:raise ValueError('frozen neural model changed')
    arrays=load_npz(args.output/'transform.npz')
    if sha(args.output/'transform.npz')!=p['transform_sha256']:raise ValueError('transform changed')
    models={name:load_npz(args.output/(name+'-readout.npz')) for name in MODELS}
    rows=[];features={n:[] for n in MODELS if n!='labels_shuffled'};all_neural=[];all_input=[];native=0;start=time.perf_counter();cache={};reset_check=None
    def simulate(frames,name,key,mode):
        nonlocal native
        i,t,_=encode_variant(frames,channels,cfg,mode);result=runners[name].run(i,t,key);pop=result['populations'][ni]
        neural=temporal_counts(pop['indices'],pop['spike_ticks'],cells,neurons)
        encoded=temporal_counts(i,t,np.arange(len(channels)),len(channels))
        record={'input_sha256':hashlib.sha256(np.stack([t,i],1).astype('<i8').tobytes()).hexdigest(),
                'events_sha256':hashlib.sha256(np.stack([pop['spike_ticks'],pop['indices']],1).astype('<i8').tobytes()).hexdigest(),
                'states_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in pop['states'].items()},
                'external_spikes':len(i),'native_key':name+'/'+key,'input_mode':mode}
        del result,pop
        shutil.rmtree(runners[name].directory/key);(runners[name].directory/(key+'.spikes')).unlink();native+=1
        return neural,encoded,record
    blank=np.full((40,48,48),.5,dtype=np.float32)
    blanks={n:simulate(blank,n,'blank',p['input_mode']) for n in runners}
    np.testing.assert_array_equal(blanks['intact'][0],arrays['blank'])
    np.savez_compressed(args.output/'blank-features.npz',**{n:v[0] for n,v in blanks.items()})
    save(args.output/'blank-events.json',{n:v[2] for n,v in blanks.items()})
    audits=[]
    for seed in p['test_groups']:
        audit=endpoint_hashes(seed,p['travel'])
        if not all(all(v==list(d.values())[0] for v in d.values()) for d in audit.values()):raise ValueError('endpoint balance failed')
        audits.append({'group':seed,'endpoint_multisets_equal':True})
    save(args.output/'endpoint-audit.json',audits)
    for condition in CONDITIONS:
        for seed in p['test_groups']:
            for metadata,frames in iter_orbit(seed,p['travel']):
                frames=transformed_frames(metadata,frames,condition)
                if not np.array_equal(frames[0],frames[-1]):raise ValueError('cycle not closed')
                mh=hashlib.sha256(frames.tobytes()).hexdigest();mode='contrast' if condition=='reference' else p['input_mode']
                name=condition if condition in runners else 'intact';key=f'{condition}-{seed}-{metadata["phase_index"][0]}-{metadata["phase_index"][1]}-{metadata["kind"]}'
                cached_key=(mode,mh)
                reused=name=='intact' and cached_key in cache
                if reused:
                    neural,encoded,record=cache[cached_key]
                    if condition=='static_first' and reset_check is None:
                        nn,ee,rr=simulate(frames,name,key+'-reset',mode)
                        reset_check=np.array_equal(nn,neural) and np.array_equal(ee,encoded) and all(rr[k]==record[k] for k in ('events_sha256','states_sha256','input_sha256'))
                        if not reset_check:raise ValueError('static reset failed')
                else:
                    neural,encoded,record=simulate(frames,name,key,mode)
                    if name=='intact':cache[cached_key]=(neural,encoded,record)
                ff=trial_features(neural,encoded,arrays,p['chosen']['recipe'])
                for n in features:features[n].append(ff[n])
                all_neural.append(neural);all_input.append(encoded)
                rows.append({**metadata,**record,'split':'test','condition':condition,'movie_sha256':mh,'reused_identical':reused})
            save(args.output/'rows.json',rows)
            np.savez_compressed(args.output/'features.npz',**{n:np.array(v) for n,v in features.items()},neural_counts=np.array(all_neural),input_counts=np.array(all_input),labels=[DIRECTIONS.index(r['kind']) for r in rows])
            print(json.dumps({'condition':condition,'rows':len(rows),'native_runs':native,'seconds':time.perf_counter()-start}),flush=True)
    reference=rows[0];frames=next(fr for md,fr in iter_orbit(reference['group'],p['travel']) if md['phase_index']==reference['phase_index'] and md['kind']==reference['kind'])
    _,_,rescue=simulate(frames,'intact','restoration',p['input_mode'])
    rescue_ok=all(rescue[k]==reference[k] for k in ('input_sha256','events_sha256','states_sha256'))
    if not rescue_ok:raise ValueError('restoration mismatch')
    results={};truth=np.array([DIRECTIONS.index(r['kind']) for r in rows]);features={n:np.asarray(v) for n,v in features.items()}
    for c in CONDITIONS:
        mask=np.array([r['condition']==c for r in rows]);rr=[r for r in rows if r['condition']==c];results[c]={}
        for name,m in models.items():
            x=features['neural_motion' if name=='labels_shuffled' else name][mask]
            results[c][name]=evaluate(model_scores(name,m,x).argmax(1),truth[mask],rr)
    paired={};rr=[r for r in rows if r['condition']=='intact'];y=np.array([DIRECTIONS.index(r['kind']) for r in rr])
    for name in MODELS:
        correct={c:np.array(results[c][name]['predictions'])==y for c in CONDITIONS}
        paired[name]={key:paired_interval(correct[a],correct[b],rr) for key,a,b in (
            ('dynamic_minus_static','intact','static_first'),('dynamic_minus_scrambled','intact','scrambled'),
            ('intact_minus_target_cut','intact','cut_input'),('matched_minus_target_cut','matched_cut','cut_input'),
            ('intact_minus_matched_cut','intact','matched_cut'))}
    gain=paired_interval(np.array(results['intact']['neural_motion']['predictions'])==y,np.array(results['reference']['previous_motion']['predictions'])==y,rr)
    time_features=np.stack([shuffle_readout_time(all_neural[i],arrays,p['chosen']['recipe'],row['group'],row['phase_index']) for i,row in enumerate(rows) if row['condition']=='intact'])
    time_pred=scores(models['neural_motion'],time_features).argmax(1)
    time_score=evaluate(time_pred,y,rr)
    time_difference=paired_interval(np.array(results['intact']['neural_motion']['predictions'])==y,time_pred==y,rr)
    np.savez_compressed(args.output/'readout-time-shuffle.npz',features=time_features)

    if any(sha(args.output/(n+'-readout.npz'))!=h for n,h in selection['readouts'].items()):raise ValueError('weights changed')
    report={'schema':p['schema'],'status':'complete','protocol_sha256':sha(args.output/'protocol.json'),'selection_sha256':sha(args.output/'selection.json'),
            'features_sha256':sha(args.output/'features.npz'),'conditions':results,'paired':paired,'paired_gain_vs_p6':gain,'readout_time_shuffle':time_score,'paired_time_shuffle':time_difference,
            'readout_time_shuffle_sha256':sha(args.output/'readout-time-shuffle.npz'),
            'test_groups':8,'test_clips_per_condition':128,'fit_clips':128,'validation_clips':64,'native_runs':native,'logical_rows':len(rows),
            'endpoint_balance_verified':True,'static_reset_exact':bool(reset_check),'intact_rescue_exact':bool(rescue_ok),
            'selected':p['chosen'],'seconds':time.perf_counter()-start,
            'limits':['external motion computation; no intrinsic connectome or topology advantage claim','fixed full-cycle speed/background; 8 independent groups and one inherited matched mask','validation search is exploratory; fresh test never used for selection','scrambled movies change frame-difference pulse counts; separate feature-only time shuffle preserves recorded per-cell count histograms','reference uses original contrast input; prior decoder is frozen P6']}
    save(args.output/'report.json',report)
    print(json.dumps({'complete':True,'results':{c:{n:v['accuracy'] for n,v in result.items()} for c,result in results.items()},'paired_gain':gain}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['select','test'])
    for n in ('artifact','parent','output'):parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--development',nargs='+',type=Path,default=[]);args=parser.parse_args()
    (select if args.stage=='select' else run)(args)


if __name__=='__main__':main()
