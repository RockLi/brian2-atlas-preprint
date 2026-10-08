"""Fresh supplementary replay: full non-input event and state equality under cut."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .causal_motion import intervene
from .simulation import SimulationConfig
from .refinement import encode_variant
from .motion_challenge import movie
from .run_experiment import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True);parser.add_argument('--study',type=Path,required=True)
    args=parser.parse_args();read=lambda p:json.loads(p.read_text())
    from flywire_mnist.backends.frozen_cpu import FrozenCPU
    p=read(args.study/'protocol.json');cfg=SimulationConfig(**p['config'])
    channels=read(args.artifact/'channels.json');template=read(args.artifact/'model.json')
    model=intervene(template,p['input_indices']);ni=next(i for i,pop in enumerate(model['definition']['populations']) if pop['name']=='flywire_neurons')
    root=args.study/'path-verification';root.mkdir(exist_ok=False)
    runner=FrozenCPU(model,args.artifact/'compile/native',root/'execution',population='visual_input',threads=4)
    identity=read(args.study/'identities.json')['cut_input']
    if runner.base_hash!=identity['base_sha256'] or runner.binary_hash!=identity['binary_sha256']:raise ValueError('cut model changed')
    cellmask=np.ones(template['definition']['populations'][ni]['count'],dtype=bool);cellmask[p['input_indices']]=False
    seed=p['test_groups'][0];records={}
    for kind in ('blank','right','down'):
        frames=np.full((40,48,48),.5,dtype=np.float32) if kind=='blank' else movie(kind,seed,p['travel'],(0,0))
        i,t,_=encode_variant(frames,channels,cfg,p['input_mode'])
        result=runner.run(i,t,kind);pop=result['populations'][ni];mask=cellmask[pop['indices']]
        records[kind]={'external_spikes':len(i),'all_noninput_events':int(mask.sum()),
                       'all_noninput_events_sha256':hashlib.sha256(np.stack([pop['spike_ticks'][mask],pop['indices'][mask]],1).astype('<i8').tobytes()).hexdigest(),
                       'all_noninput_final_states_sha256':{k:hashlib.sha256(v[cellmask].tobytes()).hexdigest() for k,v in pop['states'].items()},
                       'input_sha256':hashlib.sha256(np.stack([t,i],1).astype('<i8').tobytes()).hexdigest()}
        del result,pop
    checks={k:all(records[k][field]==records['blank'][field] for field in ('all_noninput_events_sha256','all_noninput_final_states_sha256')) for k in ('right','down')}
    checks['nonzero_visual_drive']=all(records[k]['external_spikes']>0 for k in ('right','down'))
    rows=read(args.study/'rows.json')
    checks['original_input_schedules_reproduced']=all(records[k]['input_sha256']==next(row['input_sha256'] for row in rows if row['split']=='test' and row['condition']=='cut_input' and row['group']==seed and row['phase_index']==[0,0] and row['kind']==k) for k in ('right','down'))
    save(root/'report.json',{'all_passed':all(checks.values()),'checks':checks,'records':records,'additional_native_runs':3,
                           'scope':'all 137720 non-input cells, full 600 ms event times and final states; two new replays plus blank; raw outputs retained',
                           'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({'all_passed':all(checks.values()),'checks':checks}))
    if not all(checks.values()):raise ValueError('full non-input path isolation failed')


if __name__=='__main__':main()
