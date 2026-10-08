"""Held-speed development selection followed by frozen interpolation/extrapolation tests."""
import argparse
from pathlib import Path
import time
import numpy as np
from .motion_refinement import read,sha,load_npz
from .motion_readout import pooled_grids,fit_readout,predict
from .multispeed_study import grids_for
from .generalization_readout import feature_pairs,from_pairs,transform,candidates,rank
from .generalization_data import generalization_movie,digest,SharedRecorder
from .motion_challenge import balanced_orbit_seeds,iter_orbit
from .motion_stress import phase_for
from .causal_motion import intervene,paired_interval
from .pilot import DIRECTIONS
from .direction_study import evaluate
from .run_experiment import save

CONDITIONS=('speed_2','speed_3','speed_5','static_first','cut_speed_3','matched_speed_3')
MODELS=('p8','fixed_retrained','generalized','input')
SOURCES=('generalization_study.py','generalization_readout.py','generalization_data.py','multispeed_readout.py',
         'multispeed_data.py','multispeed_study.py','motion_readout.py','motion_challenge.py','motion_stress.py',
         'motion_refinement.py','motion_resolution.py','refinement.py','simulation.py','pilot.py','direction_study.py','causal_motion.py')
REFERENCE={'band':'p8','pool':False}


def input_grids(encoded,arrays,recipe):
    mapping={'xy':arrays['input_xy'],'valid':np.ones(encoded.shape[-1],dtype=bool),'family':arrays['input_family']}
    return pooled_grids(encoded,np.zeros(encoded.shape[1:]),mapping,np.ones(encoded.shape[-1]),recipe)


def development_arrays(protocol,arrays):
    rows=[];neural=[];inputs=[]
    for speed,record in zip((1,2,4),protocol['development']):
        directory=Path(record['directory']);dd=load_npz(directory/'features.npz');rr=read(directory/'rows.json')
        if len(rr)!=192 or any(r['split']!=('fit' if i<128 else 'validation') for i,r in enumerate(rr)):raise ValueError('development only')
        neural.append(grids_for(dd['neural'],arrays,protocol['grid_recipe']))
        inputs.append(input_grids(dd['encoded'],arrays,protocol['grid_recipe']))
        rows.extend([{**r,'speed':speed} for r in rr])
    return rows,feature_pairs(np.concatenate(neural)),feature_pairs(np.concatenate(inputs))


def search(pairs,rows):
    y=np.array([DIRECTIONS.index(r['kind']) for r in rows]);options=[]
    for recipe in candidates():
        x=from_pairs(pairs,recipe)
        for alpha in (.01,.1,1.,10.,100.):
            folds=[]
            for held in (1,2,4):
                train=np.array([r['split']=='fit' and r['speed']!=held for r in rows])
                val=np.array([r['split']=='validation' and r['speed']==held for r in rows])
                model=fit_readout(x[train],y[train],alpha,True)
                folds.append(float(np.mean(predict(model,x[val]).argmax(1)==y[val])))
            options.append({'recipe':recipe,'alpha':alpha,'held_speed_accuracy':folds,'features':x.shape[1]})
    return options


def select(args):
    args.output.mkdir(parents=True,exist_ok=False)
    p8=read(args.parent/'protocol.json');parent=Path(p8['parent']);p7=read(parent/'protocol.json')
    if p8['chosen']['recipe']!={'mode':'sum','speeds':[1,2]}:raise ValueError('requires frozen P8 summed bank')
    arrays=load_npz(parent/'transform.npz');channels=read(args.artifact/'channels.json')
    if any(c['cell_type'] not in ('Mi1','Tm1') for c in channels):raise ValueError('unknown input family')
    arrays['input_family']=np.array([0 if c['cell_type']=='Mi1' else 1 for c in channels])
    records=[]
    for speed,directory in zip((1,2,4),args.development):
        if read(directory/'protocol.json').get('speed',1)!=speed:raise ValueError('development speed order differs')
        report=read(directory/'report.json')
        if report['status']!='complete' or sha(directory/'features.npz')!=report['features_sha256']:raise ValueError('development changed')
        records.append({'directory':str(directory),**{n+'_sha256':sha(directory/(n+'.json')) for n in ('report','rows','protocol')},'features_sha256':sha(directory/'features.npz')})
    p={'development':records,'grid_recipe':p7['chosen']['recipe']}
    rows,npairs,ipairs=development_arrays(p,arrays)
    options=search(npairs,rows);input_options=search(ipairs,rows)
    chosen=max(options,key=rank);fixed=max([v for v in options if v['recipe']==REFERENCE],key=rank);ichosen=max(input_options,key=rank)
    y=np.array([DIRECTIONS.index(r['kind']) for r in rows]);train=np.array([r['split']=='fit' for r in rows]);models={};validation={}
    for name,choice,pairs in (('generalized',chosen,npairs),('fixed_retrained',fixed,npairs),('input',ichosen,ipairs),('p8',{'recipe':REFERENCE},npairs)):
        x=from_pairs(pairs,choice['recipe'])
        model=load_npz(args.parent/'multispeed.npz') if name=='p8' else fit_readout(x[train],y[train],choice['alpha'],True)
        models[name]=model;np.savez_compressed(args.output/(name+'.npz'),**model)
        validation[name]={}
        for speed in (1,2,4):
            mask=np.array([r['split']=='validation' and r['speed']==speed for r in rows]);rr=[r for r,keep in zip(rows,mask) if keep]
            validation[name][str(speed)]=evaluate(predict(model,x[mask]).argmax(1),y[mask],rr)
    np.savez_compressed(args.output/'transform.npz',**arrays)
    groups=balanced_orbit_seeds(1320000,8)
    if set(groups)&set(p8['test_groups']+p7['test_groups']+p7['fit_groups']+p7['validation_groups']):raise ValueError('new group overlap')
    protocol={**p,'schema':'flywire-speed-generalization-v1','parent':str(args.parent),'neural_parent':str(parent),
        'parent_protocol_sha256':sha(args.parent/'protocol.json'),'parent_report_sha256':sha(args.parent/'report.json'),
        'neural_protocol_sha256':sha(parent/'protocol.json'),'chosen':chosen,'fixed_retrained':fixed,'input_choice':ichosen,
        'fit_groups':p7['fit_groups'],'validation_groups':p7['validation_groups'],'test_groups':groups,
        'speeds_development':[1,2,4],'speeds_test':[2,3,5],'conditions':CONDITIONS,'models':MODELS,
        'selection_rule':'worst held-speed validation, mean, fewer features, stronger regularization; then refit all development fit speeds',
        'primary':'equal-weight unseen speeds 3 and 5 on new complete orbits; speed 2 is retained-speed reference',
        'scope':'128 clips/speed/static, eight 2x2-phase orbits; 32 shared-start clips per actual lesion at speed 3',
        'readouts':{n:sha(args.output/(n+'.npz')) for n in MODELS},'transform_sha256':sha(args.output/'transform.npz'),
        'source_sha256':{n:sha(Path(__file__).with_name(n)) for n in SOURCES}}
    save(args.output/'protocol.json',protocol)
    save(args.output/'selection.json',{'protocol_sha256':sha(args.output/'protocol.json'),'test_clips_executed':0,
        'candidates':options,'input_candidates':input_options,'validation':validation})
    print({'chosen':chosen,'fixed':fixed,'input_choice':ichosen,'joint_validation':{n:{s:v['accuracy'] for s,v in d.items()} for n,d in validation.items()}},flush=True)


def features_for(neural,encoded,arrays,p,order=None):
    grids=grids_for(neural[None],arrays,p['grid_recipe'],order);pairs=feature_pairs(grids)
    reference=from_pairs(pairs,REFERENCE)[0]
    return {'p8':reference,'fixed_retrained':reference,'generalized':from_pairs(pairs,p['chosen']['recipe'])[0],
            'input':transform(input_grids(encoded[None],arrays,p['grid_recipe']),p['input_choice']['recipe'])[0]}


def statistics(rows,features,models,shuffled):
    truth=np.array([DIRECTIONS.index(r['kind']) for r in rows]);pred={n:predict(m,features[n]).argmax(1) for n,m in models.items()}
    results={};gains={}
    for c in CONDITIONS:
        mask=np.array([r['condition']==c for r in rows]);rr=[r for r,k in zip(rows,mask) if k]
        results[c]={n:evaluate(y[mask],truth[mask],rr) for n,y in pred.items()}
        gains[c]=paired_interval(pred['generalized'][mask]==truth[mask],pred['p8'][mask]==truth[mask],rr)
    mask=np.array([r['condition'] in ('speed_3','speed_5') for r in rows]);rr=[r for r,k in zip(rows,mask) if k]
    primary={n:evaluate(y[mask],truth[mask],rr) for n,y in pred.items()}
    gain=paired_interval(pred['generalized'][mask]==truth[mask],pred['p8'][mask]==truth[mask],rr)
    subset=np.array([r['condition']=='speed_3' and tuple(r['phase_index'])==phase_for(r['kind']) for r in rows]);rr=[r for r,k in zip(rows,subset) if k]
    baseline={n:evaluate(y[subset],truth[subset],rr) for n,y in pred.items()};effects={}
    for c in ('cut_speed_3','matched_speed_3'):
        m=np.array([r['condition']==c for r in rows]);effects[c]=paired_interval(pred['generalized'][subset]==truth[subset],pred['generalized'][m]==truth[m],rr)
    timed={}
    for speed in (2,3,5):
        m=np.array([r['condition']==f'speed_{speed}' for r in rows]);rr=[r for r,k in zip(rows,m) if k]
        yp=predict(models['generalized'],shuffled[m]).argmax(1)
        timed[str(speed)]={'result':evaluate(yp,truth[m],rr),'intact_minus_shuffle':paired_interval(pred['generalized'][m]==truth[m],yp==truth[m],rr)}
    return {'conditions':results,'paired_vs_p8':gains,'primary':primary,'primary_gain':gain,'lesion_subset_baseline':baseline,'lesion_effects':effects,'time_shuffle':timed}


def run(args):
    root = args.output; p = read(root/'protocol.json'); parent = Path(p['neural_parent'])
    if read(root/'selection.json')['protocol_sha256'] != sha(root/'protocol.json'):
        raise ValueError('protocol changed')
    if any(sha(Path(__file__).with_name(n)) != h for n, h in p['source_sha256'].items()):
        raise ValueError('frozen sources changed')
    if any(sha(root/(n+'.npz')) != h for n, h in p['readouts'].items()) or sha(root/'transform.npz') != p['transform_sha256']:
        raise ValueError('frozen transforms changed')
    arrays = load_npz(root/'transform.npz'); models = {n: load_npz(root/(n+'.npz')) for n in MODELS}
    pp = read(parent/'protocol.json'); template = read(args.artifact/'model.json')
    runners = {'intact': SharedRecorder(args.artifact, root/'intact', parent)}
    for name, mask in (('cut_input', pp['input_indices']), ('matched_cut', pp['matched_cut_indices'])):
        runners[name] = SharedRecorder(args.artifact, root/name, parent, intervene(template, mask), name)
    blank = np.full((40,48,48), .5, dtype=np.float32)
    blanks = {n: r.run(blank, 'blank') for n, r in runners.items()}
    np.testing.assert_array_equal(blanks['intact'][0], arrays['blank'])
    np.savez_compressed(root/'blanks.npz', **{n: v[0] for n, v in blanks.items()})
    save(root/'blank-events.json', {n: v[2] for n, v in blanks.items()})
    neural, inputs, rows, shuffled = [], [], [], []
    features = {n: [] for n in MODELS}; cache = {}; audits = []; start = time.perf_counter()
    for condition in CONDITIONS:
        speed = int(condition[-1]) if condition.startswith('speed_') else 3
        runner_name = {'cut_speed_3': 'cut_input', 'matched_speed_3': 'matched_cut'}.get(condition, 'intact')
        for group in p['test_groups']:
            endpoints = {k: [] for k in DIRECTIONS}
            for metadata, _ in iter_orbit(group, 2.):
                if runner_name != 'intact' and tuple(metadata['phase_index']) != phase_for(metadata['kind']):continue
                frames = generalization_movie(metadata['kind'], group, metadata['phase_index'], speed)
                if condition == 'static_first':frames = np.repeat(frames[:1], 40, axis=0)
                if not np.array_equal(frames[0], frames[-1]):raise ValueError('cycle not closed')
                endpoints[metadata['kind']].append(digest(frames[0]))
                key = f'{condition}-{group}-{metadata["phase_index"][0]}-{metadata["phase_index"][1]}-{metadata["kind"]}'
                cache_key = (runner_name, digest(frames))
                reused = cache_key in cache
                if reused:nn, ee, rr = cache[cache_key]
                else:
                    nn, ee, rr = runners[runner_name].run(frames, key)
                    cache[cache_key] = (nn, ee, rr)
                if runner_name == 'cut_input':np.testing.assert_array_equal(nn, blanks['cut_input'][0])
                ff = features_for(nn, ee, arrays, p)
                for n in MODELS:features[n].append(ff[n])
                order = np.arange(60)
                order[10:50] = np.random.default_rng(np.random.SeedSequence([group, *metadata['phase_index'], 915])).permutation(order[10:50])
                shuffled.append(features_for(nn, ee, arrays, p, order)['generalized'])
                neural.append(nn); inputs.append(ee)
                rows.append({**metadata, **rr, 'condition': condition, 'speed': speed, 'reused': reused, 'runner': runner_name})
            if any(sorted(v) != sorted(endpoints['right']) for v in endpoints.values()):raise ValueError('endpoint multiset differs')
            audits.append({'condition': condition, 'group': group, 'endpoint_equal': True})
            save(root/'rows.json', rows)
            np.savez_compressed(root/'features.npz', neural=np.array(neural), encoded=np.array(inputs),
                                shuffled=np.array(shuffled), **{n: np.array(v) for n, v in features.items()})
            print({'condition': condition, 'rows': len(rows), 'native_runs': sum(r.runs for r in runners.values()), 'seconds': time.perf_counter()-start}, flush=True)
    # Actual repeat/restoration, including all network events and state hashes.
    first = rows[0]
    _, _, restored = runners['intact'].run(generalization_movie(first['kind'], first['group'], first['phase_index'], 2), 'restoration')
    reset = all(first[k] == restored[k] for k in ('events_sha256', 'states_sha256', 'input_sha256'))
    if not reset:raise ValueError('restoration failed')
    save(root/'restoration.json', restored)
    save(root/'endpoint-audit.json', audits)
    summary = statistics(rows, {n: np.array(v) for n, v in features.items()}, models, np.array(shuffled))
    report = {'schema': p['schema'], 'status': 'complete', 'protocol_sha256': sha(root/'protocol.json'),
              'selection_sha256': sha(root/'selection.json'), 'features_sha256': sha(root/'features.npz'),
              'rows_sha256': sha(root/'rows.json'), 'native_runs': sum(r.runs for r in runners.values()),
              'reset_exact': reset, 'target_cut_equals_blank': True, 'seconds': time.perf_counter()-start, **summary}
    save(root/'report.json', report)
    print({'primary': {n: r['accuracy'] for n, r in report['primary'].items()}, 'gain': report['primary_gain'],
           'speeds': {c: {n: r['accuracy'] for n, r in d.items()} for c, d in report['conditions'].items()}}, flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    sp=sub.add_parser('select')
    for name in ('parent','output','artifact'):sp.add_argument('--'+name,type=Path,required=True)
    sp.add_argument('--development',type=Path,nargs=3,required=True)
    rp=sub.add_parser('test')
    for name in ('output','artifact'):rp.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();(select if args.stage=='select' else run)(args)
