"""Response-blind anatomy and passive monitoring for the bounded P13 repetition."""
import itertools
import re
from pathlib import Path
import numpy as np
from brian2_rust.binary_topology import inspect_csr,csr_arrays
from .motion_refinement import read,sha,load_npz

CENTERS=((-0.3,0.),(0.3,0.),(0.,-0.3),(0.,0.3))
SPECS=(('blank',0),('A0',0),('B0',0),('Ad',200),('Bd',200),('AB',0),('AB',200),('BA',200))
BACKGROUNDS=(785,786,787,788)
MONITOR_PATTERN=r'p1_samples_([012])\.push\(p1_state_([301])\[(\d+)\]\);'


def selection(artifact,parent,oldsites):
    channels=read(artifact/'channels.json');model=read(artifact/'model.json');m=load_npz(parent/'transform.npz')
    xy=np.array([[np.sqrt(3)/2*(r['q']-17-r['p']+19)/22,-(r['p']-19+r['q']-17)/44] for r in channels])
    top=model['instance']['synapses'][1]['topology']
    if sha(Path(top['path']))!=top['sha256']:raise ValueError('CSR changed')
    offsets,targets,weights=csr_arrays(inspect_csr(top['path']));incoming={int(c):[] for c in m['cells']}
    for i,ch in enumerate(channels):
        a,b=map(int,(offsets[ch['index']],offsets[ch['index']+1]))
        for target,w in zip(targets[a:b],weights[0,a:b]):
            if int(target) in incoming and w>0:incoming[int(target)].append((i,float(w)))
    excluded={s[k] for s in oldsites for k in ('target','a_cell','b_cell')};oldxy=np.array([s['target_xy'] for s in oldsites]);used=set(excluded);sites=[];candidate_log=[]
    for sub in range(8):
        family='Mi1' if sub<4 else 'Tm1';center=np.array(CENTERS[sub%4]);slots=np.flatnonzero((m['subtype']==sub)&m['valid']);slots=sorted(slots,key=lambda i:(float(np.linalg.norm(m['xy'][i]-center)),int(m['cells'][i])))
        chosen=None;considered=[]
        for slot in slots:
            cell=int(m['cells'][slot]);dist=float(np.linalg.norm(m['xy'][slot]-center));separation=float(np.min(np.linalg.norm(oldxy-m['xy'][slot],axis=1)))
            if dist>.25:break
            if cell in used or separation<.15:continue
            candidates=[(i,w) for i,w in incoming[cell] if channels[i]['cell_type']==family and channels[i]['index'] not in used];pairs=[]
            for (a,wa),(b,wb) in itertools.combinations(candidates,2):
                distance=float(np.linalg.norm(xy[a]-xy[b]))
                if .02<=distance<=.08:pairs.append(((-min(wa,wb),abs(wa-wb),distance,a,b),a,b,wa,wb))
            considered.append({'target':cell,'center_distance':dist,'old_target_separation':separation,'eligible_pairs':len(pairs)})
            if not pairs:continue
            _,a,b,wa,wb=min(pairs)
            if tuple(xy[a])>tuple(xy[b]):a,b,wa,wb=b,a,wb,wa
            delta=xy[b]-xy[a]
            chosen={'id':sub,'subtype':('T4' if sub<4 else 'T5')+'abcd'[sub%4],'family':family,'target':cell,'target_slot':int(slot),'target_xy':m['xy'][slot].tolist(),'requested_center':center.tolist(),'old_target_separation':separation,
                'a':[a],'b':[b],'a_cell':channels[a]['index'],'b_cell':channels[b]['index'],'a_xy':xy[a].tolist(),'b_xy':xy[b].tolist(),'contacts':[wa,wb],'distance':float(np.linalg.norm(delta)),'signed_displacement':delta.tolist(),'axis_radians':float(np.arctan2(delta[1],delta[0]))}
            used.update((cell,channels[a]['index'],channels[b]['index']));break
        if chosen is None:raise ValueError(f'no eligible new target for subtype {sub}; do not change rules after responses')
        sites.append(chosen);candidate_log.append({'subtype':sub,'considered':considered})
    return sites,candidate_log


def change_monitor_only(source,oldcells,newcells):
    if len(oldcells)!=24 or len(newcells)!=24 or len(set(newcells))!=24:raise ValueError('need 24 distinct monitored cells')
    matches=re.findall(MONITOR_PATTERN,source);expected=[(str(k),str(state),str(cell)) for cell in oldcells for k,state in ((0,3),(1,0),(2,1))]
    if matches!=expected:raise ValueError('unexpected frozen monitor statements')
    mapping=dict(zip(oldcells,newcells));new=re.sub(MONITOR_PATTERN,lambda m:f'p1_samples_{m[1]}.push(p1_state_{m[2]}[{mapping[int(m[3])]}]);',source)
    if re.sub(MONITOR_PATTERN,'MONITOR',source)!=re.sub(MONITOR_PATTERN,'MONITOR',new):raise ValueError('non-monitor native source changed')
    return new


def subset(data,cells,selected):
    columns=[data['cells'].tolist().index(c) for c in cells];mask=np.isin(data['indices'],cells)
    return {**{k:v.copy() for k,v in data.items() if k not in ('v','ge','gi','cells','indices','ticks','audit')},**{k:data[k][:,columns].copy() for k in ('v','ge','gi')},
        'cells':np.array(cells,dtype=np.int64),'indices':data['indices'][mask].copy(),'ticks':data['ticks'][mask].copy(),'audit':data['audit'][np.isin(data['audit'][:,1],selected)].copy()}
