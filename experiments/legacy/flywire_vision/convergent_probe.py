"""Anatomy-selected convergent pulse pairs with passive voltage/conductance traces."""
import argparse
import copy
import itertools
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from brian2_rust.binary_topology import inspect_csr, csr_arrays
from brian2_rust.protocol import attach_protocol
from brian2_rust.schedule import build_schedule
from brian2_rust.native import write_project
from flywire_mnist.backends.frozen_cpu import FrozenCPU
from .motion_refinement import read, sha, load_npz
from .multispeed_data import digest
from .timing_probe import schedule
from .causal_motion import intervene
from .run_experiment import save


def select_sites(artifact, parent):
    channels=read(artifact/'channels.json');model=read(artifact/'model.json');m=load_npz(parent/'transform.npz')
    xy=np.array([[np.sqrt(3)/2*(r['q']-17-r['p']+19)/22,-(r['p']-19+r['q']-17)/44] for r in channels])
    topology=next(s['topology'] for s in model['instance']['synapses'] if s['topology']['kind']=='binary_csr')
    if sha(Path(topology['path']))!=topology['sha256']:raise ValueError('CSR changed')
    offsets,targets,values=csr_arrays(inspect_csr(topology['path']));incoming={int(c):[] for c in m['cells']}
    for i,ch in enumerate(channels):
        a,b=int(offsets[ch['index']]),int(offsets[ch['index']+1])
        for target,w in zip(targets[a:b],values[0,a:b]):
            if int(target) in incoming and w>0:incoming[int(target)].append((i,float(w)))
    sites=[]
    for sub in range(8):
        family='Mi1' if sub<4 else 'Tm1';center=np.array(((-.3,-.3),(.3,-.3),(-.3,.3),(.3,.3))[sub%4])
        slots=np.flatnonzero((m['subtype']==sub)&m['valid'])
        slots=sorted(slots,key=lambda i:(float(np.linalg.norm(m['xy'][i]-center)),int(m['cells'][i])))
        chosen=None
        for slot in slots:
            if np.linalg.norm(m['xy'][slot]-center)>.25:break
            candidates=[(i,w) for i,w in incoming[int(m['cells'][slot])] if channels[i]['cell_type']==family]
            pairs=[]
            for (a,wa),(b,wb) in itertools.combinations(candidates,2):
                distance=float(np.linalg.norm(xy[a]-xy[b]))
                if .02<=distance<=.08:
                    pairs.append(((-min(wa,wb),abs(wa-wb),distance,a,b),a,b,wa,wb))
            if not pairs:continue
            _,a,b,wa,wb=min(pairs)
            if tuple(xy[a])>tuple(xy[b]):a,b,wa,wb=b,a,wb,wa
            chosen={'id':sub,'subtype':('T4' if sub<4 else 'T5')+'abcd'[sub%4], 'family':family,
                'target':int(m['cells'][slot]),'target_slot':int(slot),'target_xy':m['xy'][slot].tolist(),
                'requested_center':center.tolist(),'a':[a],'b':[b],'a_cell':channels[a]['index'],'b_cell':channels[b]['index'],
                'a_xy':xy[a].tolist(),'b_xy':xy[b].tolist(),'contacts':[wa,wb], 'distance':float(np.linalg.norm(xy[a]-xy[b]))}
            break
        if chosen is None:raise ValueError('no anatomically eligible pair')
        sites.append(chosen)
    return sites


def monitored_model(template, cells):
    model=copy.deepcopy(template);ni=next(i for i,p in enumerate(model['definition']['populations']) if p['name']=='flywire_neurons')
    pop=model['definition']['populations'][ni];variables=['v','ge','gi']
    pop['state_monitors']=[{'name':'convergent_state','variables':variables,'record':cells,'clock':pop['clock']}]
    pop['monitor']={**pop['monitor'],'variables':variables,'record':cells}
    model['definition']['schedule']=build_schedule(model['definition'],model['instance'],['start','groups','thresholds','synapses','resets','end'])
    attach_protocol(model)
    return model,ni


def design(sites):
    rows=[]
    for condition in ('intact','cut_input'):
        rows.append({'condition':condition,'site':-1,'kind':'blank','delay':0})
        for s in sites:
            specs=[('A0',0),('B0',0),('Ad',200),('Bd',200),('AB',0),('AB',200),('BA',200)]
            rows.extend({'condition':condition,'site':s['id'],'kind':k,'delay':d} for k,d in specs)
    return rows


def summarize(root):
    p=read(root/'protocol.json');rows=read(root/'rows.json');data={(r['condition'],r['site'],r['kind'],r['delay']):load_npz(root/'trials'/(r['key']+'.npz')) for r in rows};records=[]
    for condition in ('intact','cut_input'):
        blank=data[condition,-1,'blank',0]
        for site in p['sites']:
            s=site['id'];col=p['monitor_cells'].index(site['target']);get=lambda k,d:data[condition,s,k,d]
            for end in (2020,2320,2820,3820):
                # Matched 20/50/100/200 ms after final pulse, start at first pulse.
                sl=slice(1520,end)
                ab=get('AB',200)['v'][sl,col]*1000;ba=get('BA',200)['v'][sl,col]*1000
                j=ab-ba-get('A0',0)['v'][sl,col]*1000-get('Bd',200)['v'][sl,col]*1000+get('B0',0)['v'][sl,col]*1000+get('Ad',200)['v'][sl,col]*1000
                z=blank['v'][sl,col]*1000
                counts=lambda k,d:int(np.sum((get(k,d)['indices']==site['target'])&(get(k,d)['ticks']>=1520)&(get(k,d)['ticks']<end)))
                nab,nba=counts('AB',200),counts('BA',200)
                records.append({'condition':condition,'site':s,'subtype':site['subtype'],'after_last_pulse_ms':(end-1820)/10,
                    'voltage_order_mean_mv':float(np.mean(ab-ba)),'voltage_order_peak_abs_mv':float(np.max(abs(ab-ba))),
                    'voltage_interaction_mean_mv':float(np.mean(j)),'voltage_interaction_peak_abs_mv':float(np.max(abs(j))),
                    'voltage_response_peak_abs_mv':float(max(np.max(abs(ab-z)),np.max(abs(ba-z)))),
                    'ab_spikes':nab,'ba_spikes':nba,'spike_order':nab-nba,
                    'spike_interaction':nab-nba-counts('A0',0)-counts('Bd',200)+counts('B0',0)+counts('Ad',200)})
    return records


def run(root,artifact,parent,p10):
    root.mkdir(parents=True,exist_ok=False);(root/'trials').mkdir();sites=select_sites(artifact,parent)
    cells=sorted({c for s in sites for c in (s['target'],s['a_cell'],s['b_cell'])})
    template=read(artifact/'model.json');model,ni=monitored_model(template,cells)
    p={'schema':'flywire-convergent-probe-v1','artifact':str(artifact),'parent':str(parent),'p10':str(p10),'sites':sites,'monitor_cells':cells,
       'source_sha256':{n:sha(Path(__file__).with_name(n)) for n in ('convergent_probe.py','timing_probe.py','simulation.py','causal_motion.py')},
       'artifact_model_sha256':sha(artifact/'model.json'),'parent_protocol_sha256':sha(parent/'protocol.json'),
       'selection':'nearest valid target to one fixed quadrant center per subtype; same-family positive direct contacts; input pair distance 0.02..0.08; maximize minimum contacts, then minimize imbalance, distance, indices; no response-based selection',
       'input':'one channel per packet, two pulses +2/+12 ms; first onset 150 ms, AB/BA onset interval 20 ms; four external pulses in either order',
       'primary':'signed target voltage order-interaction mean over 152..202 ms, with peak absolute effect and spike counts; all targets and windows reported, no accuracy or significance claim',
       'limits':['one target per subtype, one background realization','one input per packet versus four in P10; geometry and input count both change','voltage traces include threshold/reset effects; not pure subthreshold physiology','single-control interaction and full-input cut do not establish topology-specific direction computation'],
       'planned_native_runs':116,'monitor_when':'start of each 0.1 ms tick; v in V and ge/gi dimensionless','observation_ends_ms':[202,232,282,382]}
    save(root/'protocol.json',p);save(root/'design.json',design(sites));save(root/'monitor-definition.json',model['definition'])
    started=time.perf_counter();native=root/'native';source,instance,manifest=write_project(model,native)
    if sha(instance)!=read(parent/'intact'/'identity.json')['base_sha256']:raise ValueError('monitor changed instance')
    instance.unlink();os.link(parent/'intact'/'base.bin',instance)
    subprocess.run(['rustc','--edition=2021','-C','opt-level=3','-C','codegen-units=1','-C','panic=abort',str(source),'-o',str(native/'b2-native')],check=True,capture_output=True)
    rows=[];runs=0;first=None
    for condition in ('intact','cut_input'):
        current=model if condition=='intact' else intervene(model,read(parent/'protocol.json')['input_indices'])
        runner=FrozenCPU(current,native,root/condition,population='visual_input',threads=4)
        if runner.base_hash!=read(parent/condition/'identity.json')['base_sha256']:raise ValueError('base changed')
        runner.base.unlink();os.link(parent/condition/'base.bin',runner.base)
        def record(ii,tt,key):
            result=runner.run(ii,tt,key);pop=result['populations'][ni]
            output=root/'trials'/(key+'.npz')
            # Preserve selected target AND stimulated-cell event streams, plus full-network hashes.
            keep=np.isin(pop['indices'],cells)
            np.savez_compressed(output,**{k:np.asarray(v) for k,v in pop['trace'].items()},indices=pop['indices'][keep],ticks=pop['spike_ticks'][keep],input_indices=ii,input_ticks=tt)
            rec={'events_sha256':digest(np.stack([pop['spike_ticks'],pop['indices']],1).astype('<i8')),'states_sha256':{k:digest(v) for k,v in pop['states'].items()},'data_sha256':sha(output)}
            del result,pop
            shutil.rmtree(runner.directory/key);(runner.directory/(key+'.spikes')).unlink()
            return rec
        if condition=='intact':
            oldrows=read(p10/'rows.json');old=next(r for r in oldrows if r['condition']=='intact' and r['kind']=='AB' and r['delay']==200)
            olddata=load_npz(p10/'trials'/(old['key']+'.npz'));rec=record(olddata['input_indices'],olddata['input_ticks'],'monitor_control');runs+=1
            assert all(rec[k]==old[k] for k in ('events_sha256','states_sha256')),'passive monitor changed dynamics'
            save(root/'monitor-control.json',rec);print({'monitor_passive_exact':True},flush=True)
        for row in [r for r in design(sites) if r['condition']==condition]:
            key=f'r{len(rows):03d}';ii,tt=schedule(sites[max(row['site'],0)],row['kind'],row['delay']);rec=record(ii,tt,key);runs+=1
            rows.append({**row,**rec,'key':key});save(root/'rows.json',rows)
            if first is None and row['kind']=='AB' and row['delay']==200:first=(ii,tt,rec)
            if runs%10==0:print({'native_runs':runs,'planned':116,'seconds':time.perf_counter()-started},flush=True)
        if condition=='intact':intact=runner
    runner=intact;ii,tt,old=first;rec=record(ii,tt,'restoration');runs+=1
    assert all(rec[k]==old[k] for k in ('events_sha256','states_sha256'))
    save(root/'restoration.json',rec)
    # 114 designed trials + monitor-control + restoration = 116.
    report={'status':'complete','native_runs':runs,'protocol_sha256':sha(root/'protocol.json'),'rows_sha256':sha(root/'rows.json'),'seconds':time.perf_counter()-started,'per_site':summarize(root)}
    save(root/'report.json',report);print({'native_runs':runs,'complete':True,'seconds':report['seconds']},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();[ap.add_argument('--'+n,type=Path,required=True) for n in ('output','artifact','parent','p10')];a=ap.parse_args();run(a.output,a.artifact,a.parent,a.p10)
