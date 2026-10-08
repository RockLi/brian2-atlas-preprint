"""Count-balanced two-site pulse-order probe of the unchanged full FlyWire model.

This is an intervention on artificial visual-input channels, not a new video
classifier or a claim about biological direction tuning. No readout is fitted.
"""
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
from .motion_refinement import read, sha, load_npz
from .generalization_data import SharedRecorder
from .causal_motion import intervene
from .simulation import READOUT_TYPES
from .multispeed_data import digest
from .run_experiment import save

ONSET_TICK = 1500
OFFSETS = (20, 120)  # Two identical spikes per channel, at +2 and +12 ms.
DELAYS = (200, 500, 1000)  # Onset-to-onset: 20, 50, 100 ms.


def choose_sites(channels):
    xy=np.array([[np.sqrt(3)/2*(r['q']-17-r['p']+19)/22,
                  -(r['p']-19+r['q']-17)/44] for r in channels])
    sites=[]
    for family in ('Mi1','Tm1'):
        available=np.array([i for i,r in enumerate(channels) if r['cell_type']==family])
        for k,center in enumerate(((-.3,-.3),(.3,-.3),(-.3,.3),(.3,.3))):
            axis=(0,1,1,0)[k]; shift=np.eye(2)[axis]*.1
            targets=[np.array(center)-shift,np.array(center)+shift];chosen=[]
            for target in targets:
                candidates=np.array([i for i in available if i not in sum(chosen,[])])
                order=np.argsort(np.sum((xy[candidates]-target)**2,1),kind='stable')[:4]
                slots=candidates[order]
                if np.max(np.linalg.norm(xy[slots]-target,axis=1))>.12:raise ValueError('sparse patch')
                chosen.append(slots.tolist())
            sites.append({'id':len(sites),'family':family,'axis':'x' if axis==0 else 'y','center':list(center),
                'a':chosen[0],'b':chosen[1],'a_xy':xy[chosen[0]].tolist(),'b_xy':xy[chosen[1]].tolist()})
    return sites


def schedule(site, kind, delay):
    parts={'AB':(('a',0),('b',delay)),'BA':(('b',0),('a',delay)),
           'A0':(('a',0),),'B0':(('b',0),),'Ad':(('a',delay),),'Bd':(('b',delay),),'blank':()}
    if kind not in parts or delay not in (0,*DELAYS):raise ValueError('invalid probe')
    events=[(ONSET_TICK+shift+offset,i) for patch,shift in parts[kind] for i in site[patch] for offset in OFFSETS]
    events.sort()
    if len(events)!=len(set(events)):raise ValueError('overlapping pulse packets')
    return (np.array([i for t,i in events],dtype=np.int64),np.array([t for t,i in events],dtype=np.int64))


def design(sites):
    rows=[]
    for condition in ('intact','cut_input','matched_cut'):
        rows.append({'condition':condition,'site':-1,'kind':'blank','delay':0})
        for site in sites:
            if condition=='intact':
                specs=[('A0',0),('B0',0),('AB',0)]
                specs += [(kind,delay) for delay in DELAYS for kind in ('Ad','Bd','AB','BA')]
            else:specs=[('AB',DELAYS[0]),('BA',DELAYS[0])]
            rows.extend({'condition':condition,'site':site['id'],'kind':kind,'delay':delay} for kind,delay in specs)
    return rows


def event_counts(indices,ticks,cells,subtype):
    lookup={int(cell):i for i,cell in enumerate(cells)}
    keep=np.array([int(i) in lookup for i in indices]);ticks=np.asarray(ticks)[keep]
    slots=np.array([lookup[int(i)] for i in np.asarray(indices)[keep]],dtype=np.int32)
    totals=np.bincount(slots,minlength=len(cells)).astype(np.int32)
    trace=np.zeros((600,len(READOUT_TYPES)),dtype=np.int32)
    np.add.at(trace,(ticks//10,subtype[slots]),1)
    return totals,trace,slots,ticks.astype(np.int32)


def interaction(ab,ba,a0,b0,ad,bd):
    """Signed order interaction after removing time-matched single responses."""
    return np.asarray(ab,dtype=np.int64)-ba-a0-bd+b0+ad


def metrics(root,p,rows):
    mapping=load_npz(root/'mapping.npz');lookup={(r['condition'],r['site'],r['kind'],r['delay']):r for r in rows}
    cache={r['key']:load_npz(root/'trials'/(r['key']+'.npz'))['totals'].astype(np.int64) for r in rows}
    def total(c,s,k,d):return cache[lookup[c,s,k,d]['key']]
    records=[];cuts=[]
    for site in p['sites']:
        s=site['id'];family=0 if site['family']=='Mi1' else 1
        mask=(mapping['family']==family)&mapping['valid']&(np.linalg.norm(mapping['xy']-site['center'],axis=1)<=.25)
        if not mask.any():raise ValueError('empty local readout')
        blank=total('intact',-1,'blank',0);a0=total('intact',s,'A0',0);b0=total('intact',s,'B0',0)
        for d in DELAYS:
            ab=total('intact',s,'AB',d);ba=total('intact',s,'BA',d);ad=total('intact',s,'Ad',d);bd=total('intact',s,'Bd',d)
            j=interaction(ab,ba,a0,b0,ad,bd);direct=ab-ba
            records.append({'site':s,'family':site['family'],'axis':site['axis'],'onset_interval_ms':d/10,'local_cells':int(mask.sum()),
                'local_response_l1':float((abs(ab-blank)[mask].sum()+abs(ba-blank)[mask].sum())/2),
                'local_order_l1':int(abs(direct[mask]).sum()),'local_interaction_l1':int(abs(j[mask]).sum()),
                'local_order_changed_cells':int(np.count_nonzero(direct[mask])),
                'local_interaction_cells':int(np.count_nonzero(j[mask])),
                'all_readout_order_l1':int(abs(direct).sum()),'all_readout_interaction_l1':int(abs(j).sum()),
                'subtype_signed_order':[int(direct[mapping['subtype']==k].sum()) for k in range(9)],
                'subtype_signed_interaction':[int(j[mapping['subtype']==k].sum()) for k in range(9)]})
        for condition in ('intact','cut_input','matched_cut'):
            v=total(condition,s,'AB',DELAYS[0])-total(condition,s,'BA',DELAYS[0])
            cuts.append({'site':s,'condition':condition,'local_order_l1':int(abs(v[mask]).sum()),'all_readout_order_l1':int(abs(v).sum())})
    summary={str(d/10):{key:float(np.mean([r[key] for r in records if r['onset_interval_ms']==d/10])) for key in
             ('local_response_l1','local_order_l1','local_interaction_l1','local_order_changed_cells','local_interaction_cells')} for d in DELAYS}
    return {'per_site':records,'interval_means':summary,'cut_per_site':cuts,
        'cut_means':{c:float(np.mean([r['local_order_l1'] for r in cuts if r['condition']==c])) for c in ('intact','cut_input','matched_cut')}}


class ProbeRecorder(SharedRecorder):
    def run_probe(self,indices,ticks,key,subtype,output):
        result=self.runner.run(indices,ticks,key);pop=result['populations'][self.ni]
        totals,trace,slots,st=event_counts(pop['indices'],pop['spike_ticks'],self.cells,subtype)
        np.savez_compressed(output,totals=totals,trace=trace,slots=slots,ticks=st,input_indices=indices,input_ticks=ticks)
        record={'input_sha256':digest(np.stack([ticks,indices],1).astype('<i8')),
            'events_sha256':digest(np.stack([pop['spike_ticks'],pop['indices']],1).astype('<i8')),
            'states_sha256':{k:digest(v) for k,v in pop['states'].items()},'external_spikes':len(indices),'data_sha256':sha(output)}
        del result,pop
        shutil.rmtree(self.runner.directory/key);(self.runner.directory/(key+'.spikes')).unlink();self.runs+=1
        return record


def run(root,artifact,parent):
    root.mkdir(parents=True,exist_ok=False);(root/'trials').mkdir()
    pp=read(parent/'protocol.json');channels=read(artifact/'channels.json');groups=read(artifact/'groups.json');mapping=load_npz(parent/'transform.npz')
    subtype=np.concatenate([np.full(len(groups[name]),k) for k,name in enumerate(READOUT_TYPES)])
    cells=np.concatenate([groups[name] for name in READOUT_TYPES]);np.testing.assert_array_equal(cells,pp['readout_indices'])
    np.testing.assert_array_equal(subtype,mapping['subtype'])
    np.savez_compressed(root/'mapping.npz',cells=cells,subtype=subtype,family=mapping['family'],xy=mapping['xy'],valid=mapping['valid'])
    p={'schema':'flywire-timing-probe-v1','parent':str(parent),'artifact':str(artifact),'parent_protocol_sha256':sha(parent/'protocol.json'),
       'mapping_sha256':sha(root/'mapping.npz'),'channels_sha256':sha(artifact/'channels.json'),'groups_sha256':sha(artifact/'groups.json'),
       'sites':choose_sites(channels),'dt_ms':.1,'duration_ms':600,'onset_ms':150,'packet_offsets_ms':[2,12],
       'onset_intervals_ms':[0,20,50,100],'pulse_channels_per_patch':4,'spikes_per_channel':2,'source_sha256':
       {n:sha(Path(__file__).with_name(n)) for n in ('timing_probe.py','generalization_data.py','multispeed_data.py','causal_motion.py','simulation.py')},
       'scope':'predeclared exploratory two-site intervention; direct artificial Mi1/Tm1 input packets bypass frame-difference encoder; no training, no speed or direction accuracy; unchanged P7 neural model and inherited lesion masks',
       'primary':'local same-family absolute count order-interaction at 20 ms onset interval after time-matched single controls; descriptive, not a classification score or a statistical topology advantage',
       'limitations':['8 fixed sites, one background and initial-state seed, deterministic repeats are not independent biological samples',
           'physical temporal dynamics are uniform model assumptions; no timing-parameter or total-duration sweep yet',
           'local masks use the inherited anatomical centroid mapping, not functional receptive-field calibration',
           'lesion controls measure raw order differences only; time-matched single controls run on intact graph',
           'one inherited approximate global random output mask, not matched to the eight stimulated channels',
           'simultaneous AB/BA are the identical schedule by construction, not an independent biological null']}
    save(root/'protocol.json',p);planned=design(p['sites']);save(root/'design.json',planned)
    template=read(artifact/'model.json');rows=[];start=time.perf_counter();first=None;native=0
    for condition in ('intact','cut_input','matched_cut'):
        model=template if condition=='intact' else intervene(template,pp['input_indices'] if condition=='cut_input' else pp['matched_cut_indices'])
        recorder=ProbeRecorder(artifact,root/condition,parent,model,condition)
        for row in [r for r in planned if r['condition']==condition]:
            key=f'r{len(rows):03d}';site=p['sites'][max(row['site'],0)];ii,tt=schedule(site,row['kind'],row['delay'])
            record=recorder.run_probe(ii,tt,key,subtype,root/'trials'/(key+'.npz'))
            rows.append({**row,**record,'key':key});native+=1
            if first is None and row['kind']=='AB' and row['delay']==DELAYS[0]:first=(ii,tt,record)
            save(root/'rows.json',rows)
            if native%10==0:print({'native_runs':native,'planned':len(planned)+1,'seconds':time.perf_counter()-start},flush=True)
        if condition=='intact':intact=recorder
        del model
    # Restore after both lesion runners; compare the complete first AB stimulus event/state stream.
    ii,tt,record=first;restored=intact.run_probe(ii,tt,'restoration',subtype,root/'restoration.npz');native+=1
    exact=all(record[k]==restored[k] for k in ('input_sha256','events_sha256','states_sha256'))
    save(root/'restoration.json',restored)
    report={'schema':p['schema'],'status':'complete','native_runs':native,'protocol_sha256':sha(root/'protocol.json'),
        'rows_sha256':sha(root/'rows.json'),'design_sha256':sha(root/'design.json'),'reset_exact':exact,'seconds':time.perf_counter()-start,**metrics(root,p,rows)}
    save(root/'report.json',report)
    print({k:report[k] for k in ('native_runs','reset_exact','interval_means','cut_means')},flush=True)
    if not exact:raise ValueError('restoration mismatch')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','artifact','parent'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();run(a.output,a.artifact,a.parent)
