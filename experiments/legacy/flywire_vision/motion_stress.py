"""Frozen P7 speed/background stress probe with identical endpoints per group."""
import argparse
import hashlib
import io
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
from scipy.ndimage import gaussian_filter
from .motion_refinement import read, sha, load_npz, trial_features
from .motion_readout import predict
from .motion_resolution import temporal_counts
from .motion_challenge import movie, centers, render
from .refinement import encode_variant
from .simulation import SimulationConfig
from .pilot import DIRECTIONS
from .direction_study import evaluate
from .causal_motion import paired_interval
from .run_experiment import save

CONDITIONS=('baseline','speed_2x','gray_040','gray_060','texture')


def phase_for(kind):
    # One start image shared by all four directions; each row already exists
    # in P7's complete orbit and thus requires no new baseline simulation.
    return (1,0) if kind in ('right','left') else (0,1)


def stimulus(kind,group,condition):
    phase=phase_for(kind)
    if condition=='speed_2x':
        if kind in ('left','down'):
            return stimulus({'left':'right','down':'up'}[kind],group,condition)[::-1].copy()
        start=centers(kind,group,2.,phase)[0]
        positions=np.repeat(start[None],40,axis=0)
        axis,sign=(0,1) if kind=='right' else (1,-1)
        positions[:,axis]+=sign*4*np.linspace(0,1,40)
        positions=(positions+1)%2-1;positions[0]=positions[-1]=start
        return render(positions,group)
    frames=movie(kind,group,2.,phase)
    if condition=='baseline':return frames
    if condition in ('gray_040','gray_060'):
        return np.clip(frames+(-.1 if condition=='gray_040' else .1),0,1).astype(np.float32)
    if condition=='texture':
        texture=gaussian_filter(np.random.default_rng(np.random.SeedSequence([group,771])).normal(size=(48,48)),3,mode='wrap')
        texture=.05*texture/max(float(texture.std()),1e-12)
        return np.clip(frames+np.clip(texture,-.1,.1),0,1).astype(np.float32)
    raise ValueError('unknown stress condition')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args()
    from flywire_mnist.backends.frozen_cpu import FrozenCPU
    root=args.study/'robustness';root.mkdir(exist_ok=False)
    p=read(args.study/'protocol.json');s=read(args.study/'selection.json')
    arrays=load_npz(args.study/'transform.npz');model=load_npz(args.study/'neural_motion-readout.npz')
    if sha(args.study/'neural_motion-readout.npz')!=s['readouts']['neural_motion']:raise ValueError('model changed')
    protocol={'schema':'flywire-motion-stress-v1','parent_protocol_sha256':sha(args.study/'protocol.json'),'selection_sha256':sha(args.study/'selection.json'),
              'readout_sha256':s['readouts']['neural_motion'],'source_sha256':sha(Path(__file__)),'test_groups':p['test_groups'],'conditions':CONDITIONS,
              'scope':'supplementary frozen-model stress test; 32 clips/8 shared groups per condition; no new model selection; baseline is a matched subset, not the full 128-clip score',
              'speed':'two cycles instead of one in the same 400 ms; same endpoint and starting image',
              'background':'absolute gray offsets +/-0.1 or fixed smoothed texture, with clipping; neural background and decoder blank unchanged'}
    save(root/'protocol.json',protocol)
    # Snapshot only the already-completed intact prefix while the main run may
    # still be writing later conditions. Never use partial checkpoints.
    for attempt in range(10):
        try:
            with np.load(io.BytesIO((args.study/'features.npz').read_bytes())) as f:old_neural=f['neural_counts'][:128];old_input=f['input_counts'][:128]
            break
        except (EOFError,zipfile.BadZipFile):time.sleep(1)
    else:raise RuntimeError('intact checkpoint unavailable')
    old_rows=read(args.study/'rows.json')[:128]
    if len(old_rows)!=128 or any(r['condition']!='intact' for r in old_rows):raise ValueError('intact prefix not complete')
    lookup={(r['group'],*r['phase_index'],r['kind']):i for i,r in enumerate(old_rows)}
    template=read(args.artifact/'model.json');channels=read(args.artifact/'channels.json');cfg=SimulationConfig(**p['config'])
    ni=next(i for i,pop in enumerate(template['definition']['populations']) if pop['name']=='flywire_neurons')
    neurons=template['definition']['populations'][ni]['count'];cells=np.array(p['readout_indices'])
    runner=FrozenCPU(template,args.artifact/'compile/native',root/'execution',population='visual_input',threads=4)
    identity=read(args.study/'intact/identity.json')
    if runner.base_hash!=identity['base_sha256'] or runner.binary_hash!=identity['binary_sha256']:raise ValueError('stress model differs')
    rows=[];features=[];neural=[];encoded=[];started=time.perf_counter();runs=0
    for condition in CONDITIONS:
        for group in p['test_groups']:
            endpoint=[]
            for kind in DIRECTIONS:
                frames=stimulus(kind,group,condition);endpoint.append(hashlib.sha256(frames[0].tobytes()).hexdigest())
                if not np.array_equal(frames[0],frames[-1]):raise ValueError('stress cycle not closed')
                i,t,_=encode_variant(frames,channels,cfg,p['input_mode']);idx=lookup[group,*phase_for(kind),kind]
                row={k:old_rows[idx][k] for k in ('group','kind','phase_index','polarity')}
                if condition=='baseline':
                    nn,ee=old_neural[idx],old_input[idx];details={k:old_rows[idx][k] for k in ('events_sha256','states_sha256','native_key')}
                    if hashlib.sha256(frames.tobytes()).hexdigest()!=old_rows[idx]['movie_sha256']:raise ValueError('baseline movie differs')
                else:
                    key=f'{condition}-{group}-{kind}';result=runner.run(i,t,key);pop=result['populations'][ni]
                    nn=temporal_counts(pop['indices'],pop['spike_ticks'],cells,neurons);ee=temporal_counts(i,t,np.arange(len(channels)),len(channels))
                    details={'events_sha256':hashlib.sha256(np.stack([pop['spike_ticks'],pop['indices']],1).astype('<i8').tobytes()).hexdigest(),
                        'states_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in pop['states'].items()},'native_key':key}
                    del result,pop;shutil.rmtree(runner.directory/key);(runner.directory/(key+'.spikes')).unlink();runs+=1
                features.append(trial_features(nn,ee,arrays,p['chosen']['recipe'])['neural_motion']);neural.append(nn);encoded.append(ee)
                rows.append({**row,**details,'condition':condition,'reused_baseline':condition=='baseline','movie_sha256':hashlib.sha256(frames.tobytes()).hexdigest(),
                             'input_sha256':hashlib.sha256(np.stack([t,i],1).astype('<i8').tobytes()).hexdigest(),'external_spikes':len(i)})
            if len(set(endpoint))!=1:raise ValueError('directions have different starting images')
        print({'condition':condition,'native_runs':runs,'seconds':time.perf_counter()-started},flush=True)
    save(root/'rows.json',rows);np.savez_compressed(root/'features.npz',neural_counts=np.array(neural),input_counts=np.array(encoded),neural_motion=np.array(features))
    labels=np.array([DIRECTIONS.index(r['kind']) for r in rows]);pred=predict(model,np.array(features)).argmax(1);results={};paired={}
    baseline_rows=rows[:32];correct=(pred==labels)
    for j,c in enumerate(CONDITIONS):
        a,b=j*32,(j+1)*32;results[c]=evaluate(pred[a:b],labels[a:b],rows[a:b])
        if j:paired[c]=paired_interval(correct[a:b],correct[:32],baseline_rows)
    report={'schema':protocol['schema'],'status':'complete','protocol_sha256':sha(root/'protocol.json'),'features_sha256':sha(root/'features.npz'),
            'rows_sha256':sha(root/'rows.json'),'native_runs':runs,'clips_per_condition':32,'groups':8,'conditions':results,'paired_vs_baseline':paired,'seconds':time.perf_counter()-started}
    save(root/'report.json',report);print({c:v['accuracy'] for c,v in results.items()},flush=True)


if __name__=='__main__':main()
