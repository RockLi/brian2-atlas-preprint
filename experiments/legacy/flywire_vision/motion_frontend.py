"""Development-only comparison of causal ON/OFF frame differences."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .refinement import encode_variant
from .simulation import SimulationConfig
from .motion_challenge import movie
from .motion_resolution import temporal_counts
from .run_experiment import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('artifact','parent','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    from flywire_mnist.backends.frozen_cpu import FrozenCPU
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    p=read(args.parent/'protocol.json')
    rows=[r for r in read(args.parent/'rows.json') if r['split'] in ('fit','validation')]
    if len(rows)!=192:raise ValueError('development rows only')
    model=read(args.artifact/'model.json');channels=read(args.artifact/'channels.json');cfg=SimulationConfig(**p['config'])
    ni=next(i for i,pop in enumerate(model['definition']['populations']) if pop['name']=='flywire_neurons')
    neurons=model['definition']['populations'][ni]['count'];cells=np.array(p['readout_indices'])
    args.output.mkdir(parents=True,exist_ok=False)
    protocol={'schema':'flywire-motion-frontend-v1','parent':str(args.parent),'parent_protocol_sha256':sha(args.parent/'protocol.json'),
              'input_mode':'temporal_difference_x4','source_sha256':sha(Path(__file__)),
              'fit_groups':p['fit_groups'],'validation_groups':p['validation_groups'],
              'scope':'development-only; fixed gain 32, same neural model; frame-to-frame causal contrast drives both polarity routes'}
    save(args.output/'protocol.json',protocol)
    runner=FrozenCPU(model,args.artifact/'compile/native',args.output/'execution',population='visual_input',threads=4)
    identity=read(args.parent/'identities.json')['intact']
    if runner.base_hash!=identity['base_sha256'] or runner.binary_hash!=identity['binary_sha256']:raise ValueError('neural model changed')
    records=[];neural=[];inputs=[];start=time.perf_counter()
    for idx,row in enumerate([None]+rows):
        frames=np.full((40,48,48),.5,dtype=np.float32) if row is None else movie(row['kind'],row['group'],p['travel'],tuple(row['phase_index']))
        i,t,_=encode_variant(frames,channels,cfg,protocol['input_mode'])
        key='blank' if row is None else f'development-{idx-1}'
        result=runner.run(i,t,key);pop=result['populations'][ni]
        counts=temporal_counts(pop['indices'],pop['spike_ticks'],cells,neurons)
        encoded=temporal_counts(i,t,np.arange(len(channels)),len(channels))
        if row is None:np.savez_compressed(args.output/'blank.npz',neural=counts,encoded=encoded)
        else:
            neural.append(counts);inputs.append(encoded)
            records.append({**row,'input_mode':protocol['input_mode'],
                'input_sha256':hashlib.sha256(np.stack([t,i],1).astype('<i8').tobytes()).hexdigest(),
                'events_sha256':hashlib.sha256(np.stack([pop['spike_ticks'],pop['indices']],1).astype('<i8').tobytes()).hexdigest(),
                'states_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in pop['states'].items()},'external_spikes':len(i)})
        del result,pop
        shutil.rmtree(runner.directory/key);(runner.directory/(key+'.spikes')).unlink()
        if idx and idx%16==0:
            np.savez_compressed(args.output/'features.npz',neural=np.asarray(neural),encoded=np.asarray(inputs));save(args.output/'rows.json',records)
            print(json.dumps({'completed':idx,'total':192,'seconds':time.perf_counter()-start}),flush=True)
    save(args.output/'report.json',{'status':'complete','native_runs':193,'test_used':False,
        'protocol_sha256':sha(args.output/'protocol.json'),'features_sha256':sha(args.output/'features.npz'),'seconds':time.perf_counter()-start})


if __name__=='__main__':main()
