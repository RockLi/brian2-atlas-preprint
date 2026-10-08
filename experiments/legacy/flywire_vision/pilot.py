"""Small grouped direction-decoding pilot; validation is not a held-out test."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .simulation import SimulationConfig, encode_movie, READOUT_TYPES
from .stimuli import movie
from .run_experiment import save

DIRECTIONS=("right","left","up","down")


def event_features(indices,ticks,cells,neurons):
    indices,ticks=np.asarray(indices),np.asarray(ticks)
    lookup=np.full(neurons,-1,dtype=int);lookup[cells]=np.arange(len(cells))
    mask=(ticks>=1000)&(ticks<5000)&(lookup[indices]>=0)
    counts=np.zeros((8,len(cells)),dtype=np.float32)
    np.add.at(counts,((ticks[mask]-1000)//500,lookup[indices[mask]]),1)
    return counts.ravel()


def fit(x,y,alpha):
    mean=x.mean(0,dtype=np.float64)
    scale=x.std(0,dtype=np.float64);scale[scale<1e-8]=1
    z=((x-mean)/scale).astype(np.float32)
    gram=(z@z.T).astype(np.float64)/z.shape[1]
    weights=np.linalg.solve(gram+alpha*np.eye(len(x)),np.eye(4)[y])
    return dict(mean=mean,scale=scale,train=z,weights=weights,alpha=np.array(alpha))


def scores(model,x):
    z=((np.asarray(x)-model['mean'])/model['scale']).astype(np.float32)
    return (z@model['train'].T)/model['train'].shape[1]@model['weights']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from flywire_mnist.backends.frozen_cpu import FrozenCPU
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.artifact/'manifest.json').read_text())
    cfg=SimulationConfig(**manifest['config'])
    if (cfg.frames,cfg.warmup_ms,cfg.tail_ms,cfg.dt_ms,cfg.frame_ms)!=(40,100.,100.,.1,10.):raise ValueError('pilot requires P1 timing')
    model=json.loads((args.artifact/'model.json').read_text())
    channels=json.loads((args.artifact/'channels.json').read_text())
    groups=json.loads((args.artifact/'groups.json').read_text())
    cells=np.concatenate([groups[k] for k in READOUT_TYPES])
    ni=next(i for i,pop in enumerate(model['definition']['populations']) if pop['name']=='flywire_neurons')
    runner=FrozenCPU(model,args.artifact/'compile/native',args.output/'execution',population='visual_input',threads=manifest['threads'])
    # All transformed versions of a nuisance trajectory stay together.
    protocol={'scope':'development pilot, 16 fit groups / 8 validation groups, no test; alpha selected on validation',
              'fit_groups':list(range(1000,1016)),'validation_groups':list(range(2000,2008)),
              'directions':DIRECTIONS,'alphas':[.01,.1,1.,10.], 'config':manifest['config'],
              'base_sha256':runner.base_hash,'binary_sha256':runner.binary_hash,'readout_indices':cells.tolist(),
              'model_definition':manifest['definition'],'feature':'8 x 50 ms spike counts, fit-only standardization; linear ridge',
              'sources_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('pilot.py','simulation.py','stimuli.py')}}
    save(args.output/'protocol.json',protocol)
    rows=[];features={name:[] for name in ('neural','encoded_input','pixels')};labels=[];start=time.perf_counter()
    for group in protocol['fit_groups']+protocol['validation_groups']:
        for label,kind in enumerate(DIRECTIONS):
            frames=movie(kind,group)
            i,t,_=encode_movie(frames,channels,cfg)
            key=f'g{group}-{kind}'
            result=runner.run(i,t,key);pop=result['populations'][ni]
            features['neural'].append(event_features(pop['indices'],pop['spike_ticks'],cells,manifest['neurons']))
            features['encoded_input'].append(event_features(i,t,np.arange(len(channels)),len(channels)))
            features['pixels'].append(frames.reshape(8,5,48,48).mean(1).ravel())
            labels.append(label)
            rows.append({'group':group,'kind':kind,'split':'fit' if group in protocol['fit_groups'] else 'validation',
                         'input_sha256':hashlib.sha256(np.stack([t,i],1).astype('<i8').tobytes()).hexdigest(),
                         'events_sha256':hashlib.sha256(np.stack([pop['spike_ticks'],pop['indices']],1).astype('<i8').tobytes()).hexdigest(),
                         'seconds':result['wall_seconds']})
            # Results were created only by this pilot. Retain features + source/event hashes.
            del pop,result
            shutil.rmtree(args.output/'execution'/key)
            (args.output/'execution'/(key+'.spikes')).unlink()
        print(json.dumps({'completed_groups':len(rows)//4,'total_groups':24,'seconds':time.perf_counter()-start}),flush=True)
    y=np.asarray(labels);nfit=64;reports={}
    np.savez_compressed(args.output/'features.npz',**{k:np.asarray(v) for k,v in features.items()},labels=y)
    for name,values in features.items():
        x=np.asarray(values,dtype=np.float32);candidates=[]
        for alpha in protocol['alphas']:
            readout=fit(x[:nfit],y[:nfit],alpha)
            predicted=scores(readout,x[nfit:]).argmax(1)
            candidates.append((float(np.mean(predicted==y[nfit:])),alpha,readout,predicted))
        accuracy,alpha,readout,predicted=max(candidates,key=lambda v:(v[0],v[1]))
        np.savez_compressed(args.output/(name+'-readout.npz'),**readout)
        confusion=np.zeros((4,4),dtype=int);np.add.at(confusion,(y[nfit:],predicted),1)
        reports[name]={'development_validation_accuracy':accuracy,'selected_alpha':alpha,'confusion':confusion.tolist(),
                       'predictions':predicted.tolist(),'features':x.shape[1],
                       'readout_sha256':hashlib.sha256((args.output/(name+'-readout.npz')).read_bytes()).hexdigest()}
    save(args.output/'rows.json',rows)
    report={'status':'complete','fit_clips':64,'validation_clips':32,'independent_validation_groups':8,
            'official_test_used':False,'scope':protocol['scope'],'readouts':reports,'seconds':time.perf_counter()-start}
    save(args.output/'report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
