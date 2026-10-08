"""Real FlyWire induced-subgraph / full-brain CPU conditioning experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np

from spiking import ROOT, NEURON_EQUATIONS, plastic, setup
from audit_connectome import CSR
sys.path.insert(0,str(ROOT/'examples'))
from flywire_device import poisson_schedule

SEEDS=[11,23,47,83,131]
CONDITIONS=['paired','frozen','teaching_off','shuffled_reward','reversal']


def schedule(seed,condition,protocol):
    rng=np.random.default_rng(seed+10000)
    reward_rng=np.random.default_rng(seed+20000)
    trials=[]; stages=[]; cursor=100.
    blocks=([('pilot',2,False,False)] if protocol=='pilot' else
            [('validation',2,True,False)] if protocol=='validation' else
            [('pre',4,False,False),('acquisition',12,True,False),
             ('post',4,False,True),('second',12,True,False),('final',4,False,True)])
    for name,count,training,rest in blocks:
        if rest:cursor+=500
        labels=np.tile([0,1],count//2);rng.shuffle(labels)
        rewards=(labels==(1 if name=='second' and condition=='reversal' else 0)).astype(int)
        if condition=='shuffled_reward' and training:rewards=reward_rng.permutation(rewards)
        for label,reward in zip(labels,rewards):
            trials.append(dict(block=name,start_ms=cursor,cue=int(label),training=training,
                               reward=int(reward) if training and condition!='teaching_off' else 0))
            cursor+=500
        stages.append(dict(name=name,end_ms=cursor,training=training))
    ticks=int(round(cursor/.1))
    teaching=np.zeros(ticks);learning=np.zeros(ticks)
    for trial in trials:
        if not trial['training']:continue
        start=int(round(trial['start_ms']/.1))
        if condition!='frozen':learning[start+500:start+3000]=1
        if trial['reward']:teaching[start+2000:start+3000]=1
    return trials,stages,teaching,learning,cursor


def model(prepared,output,backend,scope,seed,condition,protocol,amplitude,threads,unsplit=False):
    import brian2 as b
    import brian2_rust as rust
    setup(backend,output/'project',threads)
    manifest=json.loads((prepared/'manifest.json').read_text())
    for name,digest in manifest['hashes'].items():
        if CSR['file_hash'](prepared/name)!=digest:raise ValueError(f'prepared file changed: {name}')
    groups=np.load(prepared/'groups.npz',allow_pickle=False)
    edge=np.load(prepared/'plastic.npz',allow_pickle=False)
    sub=np.load(prepared/'subgraph.npz',allow_pickle=False)
    if scope=='full':
        nodes=np.arange(manifest['neurons'],dtype=np.int32);mbon=groups['mbon'];kc=groups['kc']
        inputs=[groups['ORN_DM1'],groups['ORN_DM2']]
        pre,post,contacts=edge['source_index'],edge['target_index'],edge['signed_contacts']
    else:
        nodes=sub['nodes'];mbon=sub['mbon'];kc=np.flatnonzero(~np.isin(np.arange(len(nodes)),mbon))
        order=np.random.default_rng(783).permutation(kc)
        count=max(1,int(.16*len(kc)));inputs=[order[:count],order[count:2*count]]
        pre,post,contacts=sub['plastic_source'],sub['plastic_target'],sub['plastic_contacts']
    n=len(nodes);trials,stages,teach,learn,duration=schedule(seed,condition,protocol)
    g=b.NeuronGroup(n,NEURON_EQUATIONS,threshold='v>-45*mV',reset='v=-52*mV',
                    refractory=2.2*b.ms,dt=.1*b.ms,method='euler',name='flywire_neurons')
    g.v=(-52+np.random.default_rng(seed).uniform(-.8,.8,n))*b.mV
    g.ge=300*.005*3.5/52;g.gi=0
    static=b.Synapses(g,g,'signed_contacts : 1 (constant)',
                      on_pre='ge_post += (.275/52)*signed_contacts*int(signed_contacts>0); gi_post -= (4*.275/52)*signed_contacts*int(signed_contacts<0)',
                      clock=g.clock,delay=1.8*b.ms,name='static_recurrent')
    if scope=='full':
        path=Path(manifest['source']['source_graph'])/'connectome.b2csr' if unsplit else prepared/'static.b2csr'
        if backend in ('aot','reference'):rust.connect_binary_csr(static,path,parameters={'signed_contacts':0})
        else:
            offsets,targets,values=CSR['csr_arrays'](CSR['inspect_csr'](path))
            static.connect(i=np.repeat(np.arange(n,dtype=np.int32),np.diff(offsets).astype(np.int64)),j=np.asarray(targets,dtype=np.int32))
            static.signed_contacts=values[0]
    else:
        static.connect(i=sub['source'],j=sub['target']);static.signed_contacts=sub['contacts']
    # Every protocol transition is on a 50 ms boundary. Encode this exact
    # piecewise-constant function compactly; the integration dt remains .1 ms.
    stride=500
    for values in (teach,learn):
        np.testing.assert_array_equal(np.repeat(values[::stride],stride)[:len(values)],values)
    teaching=b.TimedArray(teach[::stride],dt=50*b.ms,name='teaching')
    learning=b.TimedArray(learn[::stride],dt=50*b.ms,name='learning')
    syn=plastic(g,g,pre,post,contacts*0 if unsplit else contacts,teaching,learning)
    channels=min(n,512)
    bg_i,bg_tick=poisson_schedule(channels,300,.1,duration,seed+30000)
    bg_source=b.SpikeGeneratorGroup(channels,bg_i,bg_tick*.1*b.ms,clock=g.clock,sorted=True,name='background')
    mapping=np.random.default_rng(seed+40000).permutation(n)%channels
    bg=b.Synapses(bg_source,g,on_pre='ge_post += 3.5/52',clock=g.clock,name='background_projection')
    bg.connect(i=mapping,j=np.arange(n))
    all_input=np.r_[inputs[0],inputs[1]]
    stim_ids=[];stim_ticks=[]
    for trial_id,trial in enumerate(trials):
        label=trial['cue'];begin=trial['start_ms']+50
        source_i,tick=poisson_schedule(len(inputs[label]),80,.1,duration,seed+50000+trial_id,
                                       start=begin,end=begin+200)
        stim_ids.append(source_i+(len(inputs[0]) if label else 0));stim_ticks.append(tick)
    stim_ids=np.concatenate(stim_ids);stim_ticks=np.concatenate(stim_ticks)
    order=np.lexsort((stim_ids,stim_ticks));stim_ids=stim_ids[order];stim_ticks=stim_ticks[order]
    stim=b.SpikeGeneratorGroup(len(all_input),stim_ids,stim_ticks*.1*b.ms,clock=g.clock,sorted=True,name='stimulus')
    stim_syn=b.Synapses(stim,g,'amplitude : 1 (constant)',on_pre='ge_post += amplitude',clock=g.clock,name='stimulus_projection')
    stim_syn.connect(i=np.arange(len(all_input)),j=all_input)
    stim_syn.amplitude=(amplitude if scope=='full' else 3.5)/52
    records=np.unique(np.r_[mbon,kc[:2],all_input[:2]])
    monitor=b.StateMonitor(g,['v','ge','gi'],record=records,name='neuron_trace')
    spikes=b.SpikeMonitor(g,name='spikes')
    net=b.Network(g,static,syn,bg_source,bg,stim,stim_syn,monitor,spikes)
    arrays=dict(background_ids=bg_i,background_ticks=bg_tick,mapping=mapping,
                stimulus_ids=stim_ids,stimulus_ticks=stim_ticks,stimulus_targets=all_input,
                teaching=teach,learning=learn)
    input_hash=hashlib.sha256()
    for key,value in sorted(arrays.items()):input_hash.update(key.encode()+np.asarray(value).tobytes())
    np.savez_compressed(output/'input.npz',**arrays)
    metadata=dict(scope=scope,backend=backend,threads=threads,seed=seed,condition=condition,protocol=protocol,
                  amplitude_mv=amplitude,duration_ms=duration,neurons=n,plastic_edges=len(pre),
                  control_table_dt_ms=50.,control_table_dense_equivalence=True,
                  static_edges=(manifest['static_edges']+len(pre)*int(unsplit)) if scope=='full' else len(sub['source']),
                  unsplit=unsplit,trials=trials,stages=stages,mbon_indices=mbon.tolist(),kc_indices=kc.tolist(),
                  stimulus_groups=[i.tolist() for i in inputs],recorded_indices=records.tolist(),
                  input_sha256=input_hash.hexdigest(),prepared_manifest_sha256=CSR['file_hash'](prepared/'manifest.json'),
                  rule_sha256=CSR['file_hash'](Path(__file__).with_name('spiking.py')),
                  experiment_sha256=CSR['file_hash'](Path(__file__)),
                  protocol_sha256=CSR['file_hash'](Path(__file__).with_name('PROTOCOL.md')),
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
                  brian2_version=b.__version__,numpy_version=np.__version__,python=sys.version)
    (output/'configuration.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return net,g,syn,spikes,monitor,metadata


def run(args):
    import brian2 as b
    started=time.perf_counter();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    net,g,syn,spikes,monitor,metadata=model(args.prepared.resolve(),output,args.backend,args.scope,args.seed,
                                          args.condition,args.protocol,args.amplitude,args.threads,args.unsplit)
    if args.backend=='aot':
        original=b.get_device()._invoke
        def measured(command,**kwargs):
            if Path(command[0]).name=='b2-native':
                metrics=Path(command[-1]).parent/'native_metrics.json'
                command=[sys.executable,str(Path(__file__).with_name('native_measure.py')),str(metrics),*map(str,command)]
            return original(command,**kwargs)
        b.get_device()._invoke=measured
    stages=[];first=0
    checkpoint_key={k:metadata[k] for k in ('scope','seed','condition','protocol','amplitude_mv','input_sha256','prepared_manifest_sha256','rule_sha256','protocol_sha256','experiment_sha256')}
    if args.resume:
        saved=json.loads(args.resume.with_suffix('.json').read_text())
        if checkpoint_key!=saved['identity']:raise ValueError('checkpoint experiment identity mismatch')
        net.restore('acquisition',filename=args.resume,restore_random_state=True)
        stages=saved['stages'];first=len(stages)
    if args.single_run or args.backend=='cpp':
        net.run(metadata['duration_ms']*b.ms)
        if args.backend=='cpp':b.device.build(directory=str(output/'project'),compile=True,run=True,with_output=False)
    else:
        for stage in metadata['stages'][first:]:
            before=np.array(syn.gain[:])
            stage_start=float(net.t/b.ms)
            net.run((stage['end_ms']-stage_start)*b.ms)
            gain,eligibility=np.array(syn.gain[:]),np.array(syn.eligibility[:])
            if not stage['training']:np.testing.assert_array_equal(gain,before)
            assert np.isfinite(gain).all() and np.all((gain>=0)&(gain<=1))
            assert np.isfinite(eligibility).all() and np.all((eligibility>=0)&(eligibility<=1))
            np.savez(output/f"{stage['name']}-plastic.npz",gain=gain,eligibility=eligibility)
            entry={**stage,'start_ms':stage_start,'weights_frozen_exact':not stage['training'],
                   'gain_min':float(gain.min()),'gain_mean':float(gain.mean()),'gain_max':float(gain.max())}
            if args.backend=='aot':
                folder=b.get_device().last_run_directory
                entry['artifact_directory']=str(folder)
                entry['native']=json.loads((folder/'native_metrics.json').read_text())
                entry['timings']=json.loads((folder/'rust/summary.json').read_text())['timings']
                entry['build_timings']=dict(b.get_device().last_build_timings)
            stages.append(entry)
            if args.save_after_acquisition and stage['name']=='acquisition':
                checkpoint=output/'acquisition.checkpoint'
                net.store('acquisition',filename=checkpoint)
                checkpoint.with_suffix('.json').write_text(json.dumps({'identity':checkpoint_key,'stages':stages},indent=2)+'\n')
                (output/'partial.json').write_text(json.dumps({'checkpoint':str(checkpoint),'stages':stages},indent=2)+'\n')
                return
    snap=dict(v=np.array(g.v[:]),ge=np.array(g.ge[:]),gi=np.array(g.gi[:]),
              gain=np.array(syn.gain[:]),eligibility=np.array(syn.eligibility[:]),
              spike_i=np.array(spikes.i[:]),spike_t=np.array(spikes.t[:]),counts=np.array(spikes.count[:]),
              neuron_trace=np.array([monitor.v,monitor.ge,monitor.gi]),trace_t=np.array(monitor.t[:]),
              lastspike=np.array(g.lastspike[:]),not_refractory=np.array(g.not_refractory[:]))
    for field in ('v','ge','gi','gain','eligibility','neuron_trace'):assert np.isfinite(snap[field]).all(),field
    assert np.all(snap['ge']>=0) and np.all(snap['gi']>=0)
    np.savez(output/'snapshot.npz',**snap)
    trial_metrics=[]
    spike_ticks=np.rint(snap['spike_t']/.0001).astype(np.int64)
    for trial in metadata['trials']:
        start=int(round((trial['start_ms']+50)/.1));end=start+2500
        selected=snap['spike_i'][(spike_ticks>=start)&(spike_ticks<end)]
        item={**trial}
        for name,group in [('mbon',metadata['mbon_indices']),('kc',metadata['kc_indices']),('input',metadata['stimulus_groups'][trial['cue']])]:
            count=int(np.isin(selected,group).sum());item[name+'_spikes']=count;item[name+'_hz']=count/(len(group)*.25)
        trial_metrics.append(item)
    contrasts={}
    for stage in metadata['stages']:
        samples=[t for t in trial_metrics if t['block']==stage['name']]
        rates=[float(np.mean([t['mbon_hz'] for t in samples if t['cue']==cue])) for cue in (0,1)]
        contrasts[stage['name']]={'A_hz':rates[0],'B_hz':rates[1],'B_minus_A_hz':rates[1]-rates[0]}
    report=dict(stages=stages,trials=trial_metrics,contrasts=contrasts,
                final_gain_min=float(snap['gain'].min()),final_gain_mean=float(snap['gain'].mean()),
                changed_edges=int(np.count_nonzero(snap['gain']!=1)),spikes=len(spike_ticks),
                end_to_end_seconds=time.perf_counter()-started,
                frontend_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
                finite_states=True,nonnegative_conductances=True,
                final_voltage_min_mv=float(snap['v'].min()*1000),final_voltage_max_mv=float(snap['v'].max()*1000))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('stages','trials')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['numpy','cpp','aot','reference'],default='aot')
    p.add_argument('--scope',choices=['subgraph','full'],default='subgraph')
    p.add_argument('--seed',type=int,default=783);p.add_argument('--condition',choices=CONDITIONS,default='paired')
    p.add_argument('--protocol',choices=['formal','validation','pilot'],default='formal')
    p.add_argument('--amplitude',type=float,default=40);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--single-run',action='store_true');p.add_argument('--unsplit',action='store_true')
    p.add_argument('--save-after-acquisition',action='store_true');p.add_argument('--resume',type=Path)
    args=p.parse_args()
    if args.backend in ('numpy','cpp','reference') and args.threads!=1:p.error('select --threads 1 for reference backends')
    if args.unsplit and (args.scope!='full' or args.condition!='frozen'):p.error('unsplit is a frozen full-graph control')
    if (args.save_after_acquisition or args.resume) and (args.protocol!='formal' or args.single_run or args.backend not in ('aot','reference')):p.error('checkpoint requires segmented native formal protocol')
    run(args)
