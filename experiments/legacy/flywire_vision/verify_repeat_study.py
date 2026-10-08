"""Independent P13 raw-archive, anatomy, event-delivery and metric reconstruction."""
import argparse
import hashlib
import itertools
import json
import re
import struct
from pathlib import Path
import numpy as np
from brian2_rust.binary_topology import inspect_csr,csr_arrays
from .verify_background_clamp import delivery_checks
from .motion_refinement import read,sha,load_npz
from .multispeed_data import digest
from .run_experiment import save

SPECS=[('blank',0),('A0',0),('B0',0),('Ad',200),('Bd',200),('AB',0),('AB',200),('BA',200)]


def verify(root):
    p=read(root/'protocol.json');rows=read(root/'rows.json');report=read(root/'report.json');budget=read(root/'budget.json');oldroot=Path('brian2-rust/validation/flywire-vision-convergent-probe-v1');old=read(oldroot/'protocol.json');p12=Path('brian2-rust/validation/flywire-vision-event-clamp-v1');p12b=Path(p['p12b']);model=read(Path(p['artifact'])/'model.json');top=model['instance']['synapses'][1]['topology'];offsets,targets,weights=csr_arrays(inspect_csr(top['path']));checks={};writes=0;steps=0;max_error=0.;used=set()
    def check(k,value):checks[k]=checks.get(k,True) and bool(value)
    def same(a,b,trace=True):return all(a[k]==b[k] for k in (('events_sha256','states_sha256','trace_sha256') if trace else ('events_sha256','states_sha256')))
    def load(directory,row):
        d=load_npz(directory/(row['key']+'.npz'))
        if row['blank_key']:
            b=load_npz(directory/(row['blank_key']+'.npz'))
            for k in ('v','ge','gi'):d[k]=np.bitwise_xor(d[k],b[k].view(np.uint64)).view(np.float64)
        check('lossless_archives',sha(directory/(row['key']+'.npz'))==row['archive_sha256'] and {k:digest(v) for k,v in d.items()}==row['array_sha256'] and {k:list(v.shape) for k,v in d.items()}==row['array_shapes']);return d
    def pairs(d,prefix=''):return [(int(t),int(i)) for t,i in zip(d[prefix+'ticks'],d[prefix+'indices'])]
    def spike_bytes(ev):return b'B2SPIK01'+struct.pack('<Q',len(ev))+np.array([t for t,i in ev],dtype='<u8').tobytes()+np.array([i for t,i in ev],dtype='<u8').tobytes()
    def audit(d,selected,mode):
        nonlocal writes
        ev=pairs(d,'outgoing_') if mode==1 else [(t,i) for t,i in pairs(d) if i in selected]
        for k,val in delivery_checks(d['audit'],ev,offsets,targets,weights).items():check(k,val)
        writes+=len(d['audit'])
    def attempt(row,d):
        n=row['attempt'];used.add(n);a=budget['attempts'][n-1];work=root/'executor/work'/a['key'];bg=row['background'];cfg=b'P12CFG01'+struct.pack('<QQ',row['mode'],len(row['selected']))+np.array(row['selected'],dtype='<u8').tobytes()
        check('attempt_links',a['key']==(row.get('gate_alias') or row['key']) and same(a['result'],row) and a['result']['audit_sha256']==digest(d['audit']) and a['mode']==row['mode'] and a['selected']==row['selected'])
        check('attempt_sidecars',sha(work/'visual.bin')==a['visual_sha256'] and sha(work/'outgoing.bin')==a['outgoing_sha256'] and (work/'visual.bin').read_bytes()==spike_bytes(pairs(d,'visual_')) and (work/'outgoing.bin').read_bytes()==spike_bytes(pairs(d,'outgoing_')) and (work/'config.bin').read_bytes()==cfg)
        check('attempt_model_background_identity',a['binary_sha256']==p['native_identity']['binary_sha256'] and a['base_sha256']==read(Path(p['parent'])/'intact/identity.json')['base_sha256'] and a['background_sha256']==(p['background_generator_sha256'][str(bg)] if bg!=783 else None))
    check('complete_budget',report['status']=='complete' and len(rows)==256 and len(budget['attempts'])==report['native_attempts']==p['planned_native']['total']==265 and budget['limit']==p['hard_limit']==300 and all(a['status']=='completed' for a in budget['attempts']) and [a['number'] for a in budget['attempts']]==list(range(1,266)) and len({a['key'] for a in budget['attempts']})==265)
    check('frozen_before_runs',all(a['time_unix']>p['frozen_at_unix'] for a in budget['attempts']))
    check('producer_sources',all(sha(Path(__file__).with_name(n))==h for n,h in p['sources'].items()))
    check('all_prior_evidence_unchanged',all(sha(Path(path))==h for path,h in p['prior_evidence'].items()))
    check('prior_frozen_sources',all(sha(Path(__file__).with_name(n))==h for n,h in read(p12b/'source-hashes.json').items()))
    native=root/'executor/native';check('native_binary_source_identity',sha(native/'main.rs')==p['native_identity']['source_sha256'] and sha(native/'b2-native')==p['native_identity']['binary_sha256'] and read(native/'identity.json')==p['native_identity'])
    pattern=r'p1_samples_([012])\.push\(p1_state_([301])\[(\d+)\]\);';before=(p12/'validation-v1/native/main.rs').read_text();after=(native/'main.rs').read_text();expected=[(str(k),str(state),str(c)) for c in p['monitor_cells'] for k,state in ((0,3),(1,0),(2,1))]
    check('native_changes_only_passive_monitors',re.sub(pattern,'MONITOR',before)==re.sub(pattern,'MONITOR',after) and re.findall(pattern,after)==expected and p['native_identity']['parent_source_sha256']==sha(p12/'validation-v1/native/main.rs'))
    check('fixed_model_graph',sha(Path(top['path']))==top['sha256'] and sha(Path(p['parent'])/'intact/base.bin')==read(Path(p['parent'])/'intact/identity.json')['base_sha256'] and p['backgrounds']==[785,786,787,788] and p['primary_window_ticks']==[1520,2020])
    check('report_links',report['protocol_sha256']==sha(root/'protocol.json') and report['rows_sha256']==sha(root/'rows.json') and report['postchecks_sha256']==sha(root/'postchecks.json') and p['monitor_definition_sha256']==sha(root/'monitor-definition.json') and p['selection_sha256']==sha(root/'selection.json'))
    # Independently rank all anatomical candidates without reading any trial response.
    channels=read(Path(p['artifact'])/'channels.json');m=load_npz(Path(p['parent'])/'transform.npz');xy=np.array([[np.sqrt(3)/2*(c['q']-17-c['p']+19)/22,-(c['p']-19+c['q']-17)/44] for c in channels]);incoming={int(c):[] for c in m['cells']}
    for ch,cell in enumerate(channels):
        for edge in range(int(offsets[cell['index']]),int(offsets[cell['index']+1])):
            target=int(targets[edge]);w=float(weights[0,edge])
            if target in incoming and w>0:incoming[target].append((ch,w))
    excluded={s[k] for s in old['sites'] for k in ('target','a_cell','b_cell')};chosen_cells=set(excluded);oldxy=np.array([s['target_xy'] for s in old['sites']]);centers=[[-.3,0],[.3,0],[0,-.3],[0,.3]]
    for sid,site in enumerate(p['sites']):
        center=np.array(centers[sid%4]);family='Mi1' if sid<4 else 'Tm1';options=[]
        for slot in np.flatnonzero((m['subtype']==sid)&m['valid']):
            cell=int(m['cells'][slot]);dist=float(np.linalg.norm(m['xy'][slot]-center));separation=float(np.min(np.linalg.norm(oldxy-m['xy'][slot],axis=1)))
            if cell in chosen_cells or dist>.25 or separation<.15:continue
            inputs=[(ch,w) for ch,w in incoming[cell] if channels[ch]['cell_type']==family and channels[ch]['index'] not in chosen_cells];candidate_pairs=[]
            for (a,wa),(b,wb) in itertools.combinations(inputs,2):
                d=float(np.linalg.norm(xy[a]-xy[b]))
                if .02<=d<=.08:candidate_pairs.append(((-min(wa,wb),abs(wa-wb),d,a,b),a,b,wa,wb))
            if candidate_pairs:options.append((dist,cell,int(slot),min(candidate_pairs)))
        dist,cell,slot,(_,a,b,wa,wb)=min(options)
        if tuple(xy[a])>tuple(xy[b]):a,b,wa,wb=b,a,wb,wa
        check('anatomy_selection_recomputed',site['id']==sid and site['subtype']==('T4' if sid<4 else 'T5')+'abcd'[sid%4] and site['family']==family and np.isclose(site['distance'],np.linalg.norm(xy[b]-xy[a])) and np.isclose(site['old_target_separation'],np.min(np.linalg.norm(oldxy-m['xy'][slot],axis=1))) and site['target']==cell and site['target_slot']==slot and site['a']==[a] and site['b']==[b] and site['a_cell']==channels[a]['index'] and site['b_cell']==channels[b]['index'] and site['contacts']==[wa,wb] and site['requested_center']==center.tolist() and site['target_xy']==m['xy'][slot].tolist() and site['a_xy']==xy[a].tolist() and site['b_xy']==xy[b].tolist() and np.array_equal(site['signed_displacement'],xy[b]-xy[a]) and np.isclose(site['axis_radians'],np.arctan2(*(xy[b]-xy[a])[::-1])))
        chosen_cells.update((cell,site['a_cell'],site['b_cell']))
    newcells={s[k] for s in p['sites'] for k in ('target','a_cell','b_cell')};check('new_distinct_targets_and_sources',len(newcells)==24 and not newcells&excluded and sorted(newcells)==p['monitor_cells'])
    graph=load_npz(root/'outgoing-graph.npz');edges=[]
    for c in sorted({s[k] for s in p['sites'] for k in ('a_cell','b_cell')}):edges.extend((c,e,int(targets[e]),float(weights[0,e])) for e in range(int(offsets[c]),int(offsets[c+1])))
    check('graph_subset_archive',sha(root/'outgoing-graph.npz')==p['graph_sha256'] and all(np.array_equal(graph[k],np.array([x[j] for x in edges])) for j,k in enumerate(('source','edge','target','contacts'))))
    for bg in p['backgrounds']:
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'flywire-mnist-v1/{bg}/background/0'.encode()).digest()[:16],'little'));ev=[]
        for ch in range(512):
            tick=-1
            while True:
                tick+=int(rng.geometric(.03))
                if tick>=6000:break
                ev.append((tick,ch))
        ev.sort();path=root/'backgrounds'/f'generator-{bg}.bin';check('independent_background_generators',path.read_bytes()==spike_bytes(ev) and sha(path)==p['background_generator_sha256'][str(bg)])
    monitor=read(root/'monitor-control.json');d=load(root/'baselines/trials',monitor);audit(d,monitor['selected'],0);attempt(monitor,d);ref=next(r for r in read(p12/'formal-v1/rows.json') if r['key']==monitor['reference_key']);check('passive_full_graph_gate',monitor['mode']==0 and monitor['background']==783 and monitor['selected']==sorted(newcells-{s['target'] for s in p['sites']}) and monitor['exact'] and same(monitor,ref,False) and monitor['attempt']==1)
    baselines={};base_rows=read(root/'baselines/rows.json')
    check('four_new_ordinary_baselines',[r['background'] for r in base_rows]==p['backgrounds'] and [r['attempt'] for r in base_rows]==[2,3,4,5])
    for row in base_rows:
        d=load(root/'baselines/trials',row);baselines[row['background']]=(d,row);audit(d,row['selected'],0);attempt(row,d);check('ordinary_baseline_identity',row['mode']==0 and row['selected']==sorted(newcells-{s['target'] for s in p['sites']}) and d['cells'].tolist()==p['monitor_cells'])
    freeze=read(root/'formal-freeze.json');check('formal_freeze_before_stimulation',freeze['protocol_sha256']==sha(root/'protocol.json') and freeze['background_records_sha256']==sha(root/'background-records.json') and freeze['design_sha256']==sha(root/'design.json') and freeze['baseline_rows_sha256']==sha(root/'baselines/rows.json') and all(a['time_unix']<freeze['time_unix'] for a in budget['attempts'][:5]) and all(a['time_unix']>freeze['time_unix'] for a in budget['attempts'][5:]))
    bgmap={};shifts={}
    for record in read(root/'background-records.json'):
        bg=record['background'];sid=record['site'];s=p['sites'][sid];pair=[s['a_cell'],s['b_cell']];d,_=baselines[bg];ev=[(t,i) for t,i in pairs(d) if i in pair];stored=load_npz(root/'backgrounds'/(record['key']+'.npz'));check('blank_derived_background_records',ev==pairs(stored) and len(ev)==record['event_count'] and record['early_support']==any(t+18<2020 for t,i in ev) and sha(root/'backgrounds'/(record['key']+'.npz'))==record['sha256']);possible=[]
        for shift in range(101):
            combined=sorted(ev+[(t+shift,i) for i in pair for t in (1520,1620,1720,1820)])
            if len(combined)==len(set(combined)) and all(all(b-a>=22 for a,b in zip(ts,ts[1:])) for ts in ([t for t,i in combined if i==source] for source in pair)):possible.append(shift)
        check('minimal_order_independent_conflict_shift',bool(possible) and record['shift']==possible[0]);bgmap[bg,sid]=ev;shifts[bg,sid]=record['shift']
    gates=read(root/'gates/rows.json');gate_data={}
    for row in gates:
        d=load(root/'gates/trials',row);bg=row['background'];sid=row['site'];s=p['sites'][sid];ordinary,br=baselines[bg];cells=[s['target'],s['a_cell'],s['b_cell']];cols=[ordinary['cells'].tolist().index(c) for c in cells];mask=np.isin(ordinary['indices'],cells);a=ordinary['audit'][np.isin(ordinary['audit'][:,1],row['selected'])]
        check('all_32_blank_replays_exact',row['exact'] and same(row,br) and all(np.array_equal(d[k],ordinary[k][:,cols]) for k in ('v','ge','gi')) and np.array_equal(d['indices'],ordinary['indices'][mask]) and np.array_equal(d['ticks'],ordinary['ticks'][mask]) and np.array_equal(d['audit'],a) and pairs(d,'outgoing_')==bgmap[bg,sid] and pairs(d,'visual_')==[])
        audit(d,row['selected'],1);attempt(row,d);gate_data[bg,sid]=(d,row)
    check('replay_gates_before_all_formal_stimuli',len(gates)==32 and {r['attempt'] for r in gates}==set(range(6,38)) and read(root/'gates/verification.json')['all_passed'] and read(root/'gates/verification.json')['rows_sha256']==sha(root/'gates/rows.json'))
    expected_design=[{'background':bg,'site':sid,'kind':k,'delay':delay,'shift':shifts[bg,sid],'key':f'b{bg}-s{sid}-{k}-{delay}'} for bg in p['backgrounds'] for sid in range(8) for k,delay in SPECS];check('complete_literal_design',expected_design==read(root/'design.json')==[{k:r[k] for k in expected_design[0]} for r in rows]);by={};loaded={}
    for row in rows:
        d=load(root/'trials',row);bg=row['background'];sid=row['site'];s=p['sites'][sid];kind=row['kind'];delay=row['delay'];shift=shifts[bg,sid];by[bg,sid,kind,delay]=d;loaded[row['key']]=d;parts={'blank':[],'A0':[('a',0)],'B0':[('b',0)],'Ad':[('a',delay)],'Bd':[('b',delay)],'AB':[('a',0),('b',delay)],'BA':[('b',0),('a',delay)]}[kind]
        vis=sorted((1520+shift+offset+t,s[part][0]) for part,offset in parts for t in (0,100));stim=sorted((1520+shift+offset+t,s[part+'_cell']) for part,offset in parts for t in (0,100));merged=sorted(bgmap[bg,sid]+stim)
        check('literal_stimuli_and_fixed_background',vis==pairs(d,'visual_') and stim==pairs(d,'stimulus_') and bgmap[bg,sid]==pairs(d,'background_') and merged==pairs(d,'outgoing_') and d['cells'].tolist()==[s['target'],s['a_cell'],s['b_cell']] and row['selected']==sorted([s['a_cell'],s['b_cell']]) and row['mode']==1)
        check('no_coincident_or_refractory_conflicts',len(merged)==len(set(merged)) and all(np.all(np.diff([t for t,i in merged if i==source])>=22) for source in row['selected']))
        audit(d,row['selected'],1);attempt(row,d);blank,gr=gate_data[bg,sid]
        if kind=='blank':check('formal_blank_aliases_exact',row['gate_alias']==gr['key'] and same(row,gr) and all(np.array_equal(v,d[k]) for k,v in blank.items()))
        else:check('formal_stimulation_after_gates_and_pre_stimulus_blank',row['gate_alias'] is None and row['attempt']>=38 and all(np.array_equal(d[k][:1520+shift],blank[k][:1520+shift]) for k in ('v','ge','gi')))
        active=np.ones((5999,3),dtype=bool)
        for t,i in pairs(d):active[t:min(5999,t+24),d['cells'].tolist().index(i)]=False
        v=d['v'];expected=v[:-1]+.005*(-(v[:-1]+.052)-d['ge'][:-1]*v[:-1]-d['gi'][:-1]*(v[:-1]+.070));error=float(np.max(abs(v[1:][active]-expected[active])));max_error=max(max_error,error);steps+=int(active.sum());check('LIF_equation',error<1e-13)
    reconstructed=[]
    for bg in p['backgrounds']:
        for s in p['sites']:
            sid=s['id'];shift=shifts[bg,sid];get=lambda k,d:by[bg,sid,k,d];ab=get('AB',200)['audit'];ba=get('BA',200)['audit'];blank=get('blank',0)['audit']
            for edge in set(map(int,ab[:,2]))|set(map(int,ba[:,2])):
                a=ab[ab[:,2]==edge];b=ba[ba[:,2]==edge];z=blank[blank[:,2]==edge];check('balanced_edges_and_unchanged_background',len(a)==len(b)==len(z)+2 and np.array_equal(a[:,4:6],b[:,4:6]) and all({tuple(map(int,r[:6])) for r in z}<={tuple(map(int,r[:6])) for r in full} for full in (a,b)))
            fullj=(get('AB',200)['v']-get('A0',0)['v']-get('Bd',200)['v'])-(get('BA',200)['v']-get('B0',0)['v']-get('Ad',200)['v']);check('zero_interaction_before_second_arrival',np.max(abs(fullj[:1738+shift,0]))<1e-14)
            for wi,(start,end) in enumerate([(1520,2020)]+[(1538+shift,x+shift) for x in (2038,2338,2838,3838)]):
                v=lambda k,d:get(k,d)['v'][start:end,0]*1000;a=v('AB',200);b=v('BA',200);z=v('blank',0);j=(a-v('A0',0)-v('Bd',200)+z)-(b-v('B0',0)-v('Ad',200)+z);cnt=lambda k,d,c:sum(c==i and start<=t<end for t,i in pairs(get(k,d)));counts={k+str(d):cnt(k,d,s['target']) for k,d in SPECS}
                reconstructed.append({'background':bg,'site':sid,'subtype':s['subtype'],'shift_ticks':shift,'window_index':wi,'window_ticks':[start,end],'voltage_order_mean_mv':float(np.mean(a-b)),'voltage_order_peak_abs_mv':float(np.max(abs(a-b))),'voltage_interaction_mean_mv':float(j.mean()),'voltage_interaction_mean_abs_mv':float(abs(j).mean()),'voltage_interaction_peak_abs_mv':float(abs(j).max()),'voltage_response_peak_abs_mv':float(max(abs(a-z).max(),abs(b-z).max())),'target_spikes':counts,'input_spikes':{k:{part:cnt(k,200,s[part+'_cell']) for part in ('a','b')} for k in ('AB','BA')},'spike_interaction':counts['AB200']-counts['BA200']-counts['A00']-counts['Bd200']+counts['B00']+counts['Ad200']})
    check('all_160_metrics_independently_recomputed',len(reconstructed)==len(report['per_site'])==160 and all(set(a)==set(b) and all(np.isclose(a[k],b[k],rtol=1e-10,atol=1e-11) if isinstance(a[k],float) else a[k]==b[k] for k in a) for a,b in zip(reconstructed,report['per_site'])))
    reference=read(root/'reference-signs.json');sign=lambda x:0 if abs(x)<=1e-8 else (1 if x>0 else -1)
    check('fixed_reference_from_first_background_only',reference['protocol_sha256']==sha(root/'protocol.json') and reference['source_background']==785 and reference['sign_floor_mv']==1e-8 and reference['source_row_archives']=={r['key']:r['archive_sha256'] for r in rows if r['background']==785} and len(reference['reference'])==8)
    for sid,item in enumerate(reference['reference']):
        metric=next(r for r in reconstructed if r['background']==785 and r['site']==sid and r['window_index']==0)
        check('fixed_reference_from_first_background_only',item['site']==sid and np.isclose(item['J_mean_mv'],metric['voltage_interaction_mean_mv'],atol=1e-11,rtol=1e-10) and np.isclose(item['raw_order_mean_mv'],metric['voltage_order_mean_mv'],atol=1e-11,rtol=1e-10) and item['J_reference_sign']==sign(metric['voltage_interaction_mean_mv']) and item['raw_reference_sign']==sign(metric['voltage_order_mean_mv']))
    post=read(root/'postchecks.json');check('four_prespecified_restorations',len(post)==4 and p['restorations']==[[785,0,'AB'],[786,2,'BA'],[787,4,'AB'],[788,6,'BA']])
    for row,spec in zip(post,p['restorations']):
        d=load(root/'trials',row);ref=next(r for r in rows if r['key']==row['reference_key']);rd=loaded[ref['key']];check('restorations_exact',row['exact'] and [ref['background'],ref['site'],ref['kind']]==spec and ref['delay']==200 and same(row,ref) and all(np.array_equal(val,rd[k]) for k,val in d.items()));audit(d,row['selected'],1);attempt(row,d)
    check('all_attempts_accounted',used==set(range(1,266)))
    result={'all_passed':all(checks.values()),'checks':checks,'native_attempts':len(budget['attempts']),'actual_edge_writes_checked':writes,'voltage_steps_checked':steps,'max_voltage_equation_error_V':max_error,'report_sha256':sha(root/'report.json'),'verifier_sha256':sha(Path(__file__))};save(root/'verification.json',result);save(root/'independent-metrics.json',reconstructed);print(json.dumps(result,indent=2),flush=True)
    if not result['all_passed']:raise ValueError('P13 independent verification failed')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);a=parser.parse_args();verify(a.root)
