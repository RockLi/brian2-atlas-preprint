"""Independent P12b reconstruction: no imports from the study or its data/metric helpers."""
import argparse
import json
import struct
from pathlib import Path
import numpy as np
from brian2_rust.binary_topology import inspect_csr,csr_arrays
from .motion_refinement import read,sha,load_npz
from .multispeed_data import digest
from .run_experiment import save

SPECS=[('blank',0),('A0',0),('B0',0),('Ad',200),('Bd',200),('AB',0),('AB',200),('BA',200)]


def delivery_checks(actual,events,offsets,targets,weights):
    """Reconstruct writes from the full CSR, separately from the simulator/auditor."""
    expected=[];contact=[]
    for t,i in events:
        if t+18>=6000:continue
        for edge in range(int(offsets[i]),int(offsets[i+1])):
            expected.append((t+18,i,edge,int(targets[edge])));contact.append(float(weights[0,edge]))
    values=actual[:,4:].copy().view(np.float64);w=np.asarray(contact)
    ge=np.array([(.275/52)*(x*int(x>0)) for x in w]);gi=np.array([-((.275/52)*4)*(x*int(x<0)) for x in w])
    return {
        'actual_deliveries_full_CSR':np.array_equal(actual[:,:4],np.array(expected,dtype=np.uint64).reshape(-1,4)),
        'no_duplicate_injection':len(actual)==len({(int(r[0]),int(r[2])) for r in actual}),
        'actual_weights_and_state_writes':np.array_equal(values[:,0],ge) and np.array_equal(values[:,1],gi) and np.array_equal(values[:,2]+values[:,0],values[:,3]) and np.array_equal(values[:,4]+values[:,1],values[:,5]),
    }


def verify(root):
    p=read(root/'protocol.json');rows=read(root/'rows.json');report=read(root/'report.json');budget=read(root/'budget.json');parent=Path(p['parent']);oldrows={r['key']:r for r in read(parent/'formal-v1/rows.json')};oldp=read(parent/'formal-v1/protocol.json')
    p11=Path('brian2-rust/validation/flywire-vision-convergent-probe-v1');p11p=read(p11/'protocol.json');model=read(Path(p11p['artifact'])/'model.json');top=model['instance']['synapses'][1]['topology'];offsets,targets,weights=csr_arrays(inspect_csr(top['path']))
    checks={}
    def check(name,value):checks[name]=checks.get(name,True) and bool(value)
    def same(a,b):return all(a[k]==b[k] for k in ('events_sha256','states_sha256','trace_sha256'))
    def load(directory,row):
        data=load_npz(directory/(row['key']+'.npz'))
        if row['blank_key']:
            blank=load_npz(directory/(row['blank_key']+'.npz'))
            for k in ('v','ge','gi'):data[k]=np.bitwise_xor(data[k],blank[k].view(np.uint64)).view(np.float64)
        check('archives_hashes_shapes',sha(directory/(row['key']+'.npz'))==row['archive_sha256'] and {k:digest(v) for k,v in data.items()}==row['array_sha256'] and {k:list(v.shape) for k,v in data.items()}==row['array_shapes'])
        return data
    def pairs(data,prefix):return [(int(t),int(i)) for t,i in zip(data[prefix+'ticks'],data[prefix+'indices'])]
    def spike_bytes(events):
        return b'B2SPIK01'+struct.pack('<Q',len(events))+np.array([t for t,i in events],dtype='<u8').tobytes()+np.array([i for t,i in events],dtype='<u8').tobytes()
    check('complete_design_and_budget',report['status']=='complete' and len(rows)==256 and p['hard_limit']==budget['limit']==200 and len(budget['attempts'])==report['native_attempts']==p['planned_native_total']==98)
    check('ledger_unique_completed',all(a['status']=='completed' for a in budget['attempts']) and len({a['key'] for a in budget['attempts']})==len(budget['attempts']) and [a['number'] for a in budget['attempts']]==list(range(1,99)))
    check('protocol_frozen_before_all_runs',all(a['time_unix']>p['frozen_at_unix'] for a in budget['attempts']))
    check('prior_evidence_unchanged',all(sha(Path(f))==h for f,h in p['prior_evidence'].items()))
    check('parent_links',all(p['parent_'+n+'_sha256']==sha(parent/'formal-v1'/(n+'.json')) for n in ('protocol','report','verification')))
    check('frozen_sources',all(sha(Path(__file__).with_name(n))==h for n,h in p['sources'].items()) and sha(Path(__file__).with_name('test_background_clamp.py'))==p['unit_test_sha256'])
    for directory in (p11,Path('brian2-rust/validation/flywire-vision-timing-probe-v1')):
        check('prior_frozen_sources',all(sha(Path(__file__).with_name(n))==h for n,h in read(directory/'protocol.json')['source_sha256'].items()))
    for protocol in (oldp,read(parent/'validation-v1/protocol.json')):
        check('prior_frozen_sources',all(sha(Path(__file__).with_name(n))==h for n,h in protocol['sources'].items()))
    check('frozen_primary_window',p['primary_window_ticks']==[1520,2020])
    check('frozen_sites_backgrounds',p['sites']==oldp['sites']==p11p['sites'] and p['backgrounds']==[783,784])
    check('report_hash_links',report['protocol_sha256']==sha(root/'protocol.json') and report['rows_sha256']==sha(root/'rows.json') and report['postchecks_sha256']==sha(root/'postchecks.json') and p['design_sha256']==sha(root/'design.json'))
    check('native_identity',p['native_identity']==read(parent/'validation-v1/native/identity.json')==read(root/'executor/native/identity.json') and sha(root/'executor/native/b2-native')==p['native_identity']['binary_sha256'] and sha(root/'executor/native/main.rs')==p['native_identity']['source_sha256'])
    check('base_graph_background',sha(Path(p11p['parent'])/'intact/base.bin')==budget['attempts'][0]['base_sha256'] and sha(Path(top['path']))==top['sha256'] and sha(root/'outgoing-graph.npz')==p['graph_sha256'] and sha(root/'background-784.bin')==p['background_784_sha256']==sha(parent/'background-784.bin'))
    archived_graph=load_npz(root/'outgoing-graph.npz');expected_graph=[]
    for source in sorted({s[k] for s in p['sites'] for k in ('a_cell','b_cell')}):
        expected_graph.extend((source,e,int(targets[e]),float(weights[0,e])) for e in range(int(offsets[source]),int(offsets[source+1])))
    check('edge_archive_matches_full_CSR',all(np.array_equal(archived_graph[k],np.array([r[j] for r in expected_graph])) for j,k in enumerate(('source','edge','target','contacts'))))
    backgrounds={};shifts={};support=[]
    for b in p['background_records']:
        s=p['sites'][b['site']];old=oldrows[b['source_row_key']];data=load(parent/'formal-v1/trials',old);ev=[(t,i) for t,i in pairs(data,'') if i in (s['a_cell'],s['b_cell'])];stored=load_npz(root/'backgrounds'/b['key'])
        check('blank_background_provenance',old['condition']=='ordinary' and old['kind']=='blank' and old['background']==b['background'] and old['site']==b['site'] and b['source_archive_sha256']==old['archive_sha256'] and sha(root/'backgrounds'/b['key'])==b['sha256'] and ev==pairs(stored,'')==[tuple(x) for x in b['events']] and len(ev)==b['event_count'])
        possible=[]
        for offset in range(101):
            candidates=sorted(ev+[(t+offset,i) for i in (s['a_cell'],s['b_cell']) for t in (1520,1620,1720,1820)])
            by_source={i:[t for t,j in candidates if i==j] for i in (s['a_cell'],s['b_cell'])}
            if len(candidates)==len(set(candidates)) and all(all(v-u>=22 for u,v in zip(ts,ts[1:])) for ts in by_source.values()):possible.append(offset)
        check('minimal_shared_shift',bool(possible) and b['stimulus_shift_ticks']==possible[0])
        group=(b['background'],b['site']);backgrounds[group]=ev;shifts[group]=b['stimulus_shift_ticks'];support.append({'background':group[0],'site':group[1],'events':len(ev),'early_support':any(t+18<2020 for t,i in ev),'shift_ticks':shifts[group]})
    expected_design=[]
    for bg in (783,784):
        for condition in ('without_background','with_background'):
            for sid in range(8):
                for kind,d in SPECS:
                    shift=shifts[bg,sid];alias=None;gate=None
                    if condition=='without_background' or not backgrounds[bg,sid]:
                        if shift==0 or kind=='blank':alias=f'b{bg}-controlled-s{sid}-{kind}-{d}'
                    elif kind=='blank':gate=f'blank-replay-{bg}-s{sid}'
                    expected_design.append({'background':bg,'condition':condition,'site':sid,'kind':kind,'delay':d,'shift':shift,'key':f'b{bg}-{condition}-s{sid}-{kind}-{d}','p12_alias':alias,'gate_alias':gate})
    check('all_conditions_and_valid_alias_design',expected_design==read(root/'design.json')==[{k:r[k] for k in expected_design[0]} for r in rows])
    edge_writes=0;equation_steps=0;max_error=0.
    def audit(data,events):
        nonlocal edge_writes
        edge_writes+=len(data['audit'])
        for name,value in delivery_checks(data['audit'],events,offsets,targets,weights).items():check(name,value)
    used=set()
    def attempt_link(row,data,bg,sid):
        n=row['attempt'];used.add(n);a=budget['attempts'][n-1];s=p['sites'][sid];work=root/'executor/work'/a['key']
        check('attempt_result_identity',a['key']==(row.get('gate_alias') or row['key']) and same(a['result'],row) and a['result']['audit_sha256']==digest(data['audit']) and a['mode']==1 and a['selected']==sorted([s['a_cell'],s['b_cell']]) and a['background_sha256']==(p['background_784_sha256'] if bg==784 else None))
        check('attempt_model_identity',a['binary_sha256']==p['native_identity']['binary_sha256'] and a['base_sha256']==budget['attempts'][0]['base_sha256'])
        config=b'P12CFG01'+struct.pack('<QQ',1,2)+np.array(a['selected'],dtype='<u8').tobytes()
        check('attempt_sidecars_match_actual_events',sha(work/'visual.bin')==a['visual_sha256'] and sha(work/'outgoing.bin')==a['outgoing_sha256'] and (work/'visual.bin').read_bytes()==spike_bytes(pairs(data,'visual_')) and (work/'outgoing.bin').read_bytes()==spike_bytes(pairs(data,'outgoing_')) and (work/'config.bin').read_bytes()==config)
    gates=read(root/'gates/rows.json');gate_by={r['key']:r for r in gates}
    for row in gates:
        data=load(root/'gates/trials',row);old=oldrows[row['reference_key']];od=load(parent/'formal-v1/trials',old);ev=backgrounds[row['background'],row['site']]
        check('blank_replay_exact',row['exact'] and same(row,old) and all(np.array_equal(data[k],od[k]) for k in ('v','ge','gi','indices','ticks','audit')) and pairs(data,'outgoing_')==ev and pairs(data,'visual_')==[])
        audit(data,ev);attempt_link(row,data,row['background'],row['site'])
    formal_attempts=[r['attempt'] for r in rows if not r['p12_alias'] and not r['gate_alias']]
    check('all_gates_before_stimulated_runs',len(gates)==11 and len(formal_attempts)==84 and max(r['attempt'] for r in gates)<min(formal_attempts) and read(root/'gates/verification.json')['all_passed'] and read(root/'gates/verification.json')['rows_sha256']==sha(root/'gates/rows.json'))
    by={};loaded={};blanks={}
    for row in rows:
        data=load(root/'trials',row);key=(row['background'],row['condition'],row['site'],row['kind'],row['delay']);by[key]=data;loaded[row['key']]=data;s=p['sites'][row['site']];bg=row['background'];sid=row['site'];shift=shifts[bg,sid];kind=row['kind'];d=row['delay']
        parts={'blank':[],'A0':[('a',0)],'B0':[('b',0)],'Ad':[('a',d)],'Bd':[('b',d)],'AB':[('a',0),('b',d)],'BA':[('b',0),('a',d)]}[kind]
        vis=sorted((1520+shift+delta+t,s[part][0]) for part,delta in parts for t in (0,100));stim=sorted((1520+shift+delta+t,s[part+'_cell']) for part,delta in parts for t in (0,100));base=backgrounds[bg,sid] if row['condition']=='with_background' else [];merged=sorted(base+stim)
        check('literal_stimuli_fixed_background',vis==pairs(data,'visual_') and stim==pairs(data,'stimulus_') and base==pairs(data,'background_') and merged==pairs(data,'outgoing_') and data['cells'].tolist()==[s['target'],s['a_cell'],s['b_cell']])
        check('source_event_spacing',len(merged)==len(set(merged)) and all(np.all(np.diff([t for t,i in merged if i==source])>=22) for source in (s['a_cell'],s['b_cell'])))
        audit(data,merged)
        if row['p12_alias']:
            old=oldrows[row['p12_alias']];od=load(parent/'formal-v1/trials',old)
            check('old_alias_exact',row['attempt'] is None and row['p12_attempt']==old['attempt'] and same(row,old) and all(np.array_equal(v,data[k]) for k,v in od.items()))
        else:attempt_link(row,data,bg,sid)
        group=key[:3]
        if kind=='blank':
            blanks[group]=data
            if row['condition']=='with_background':
                ordinary=oldrows[f'b{bg}-ordinary-s{sid}-blank-0'];od=load(parent/'formal-v1/trials',ordinary)
                check('all_16_background_baselines',same(row,ordinary) and all(np.array_equal(data[k],od[k]) for k in ('v','ge','gi','indices','ticks','audit')))
        else:check('pre_stimulus_blank',all(np.array_equal(data[k][:1520+shift],blanks[group][k][:1520+shift]) for k in ('v','ge','gi')))
        active=np.ones((5999,3),dtype=bool)
        for t,i in pairs(data,''):active[t:min(5999,t+24),data['cells'].tolist().index(i)]=False
        v=data['v'];expected=v[:-1]+.005*(-(v[:-1]+.052)-data['ge'][:-1]*v[:-1]-data['gi'][:-1]*(v[:-1]+.070));error=float(np.max(abs(v[1:][active]-expected[active])));max_error=max(max_error,error);equation_steps+=int(active.sum());check('LIF_equation',error<1e-13)
    for bg in (783,784):
        for sid in range(8):
            first=min((t+18 for t,i in backgrounds[bg,sid]),default=6000)
            for kind,d in SPECS:
                left=by[bg,'without_background',sid,kind,d];right=by[bg,'with_background',sid,kind,d]
                check('no_background_effect_before_delivery',all(np.array_equal(left[k][:first],right[k][:first]) for k in ('v','ge','gi')))
    reconstructed=[]
    for bg in (783,784):
        for condition in ('without_background','with_background'):
            for s in p['sites']:
                sid=s['id'];shift=shifts[bg,sid];get=lambda k,d:by[bg,condition,sid,k,d];a=get('AB',200)['audit'];b=get('BA',200)['audit'];blank=get('blank',0)['audit']
                for edge in set(map(int,a[:,2]))|set(map(int,b[:,2])):
                    aa=a[a[:,2]==edge];bb=b[b[:,2]==edge];zz=blank[blank[:,2]==edge]
                    check('balanced_per_edge_increment_over_blank',len(aa)==len(bb)==len(zz)+2 and np.array_equal(aa[:,4:6],bb[:,4:6]))
                    for full in (aa,bb):
                        before={tuple(map(int,r[:6])) for r in zz};after={tuple(map(int,r[:6])) for r in full}
                        check('blank_edge_events_retained',before<=after and len(after-before)==2)
                all_j=(get('AB',200)['v']-get('A0',0)['v']-get('Bd',200)['v']+get('blank',0)['v'])-(get('BA',200)['v']-get('B0',0)['v']-get('Ad',200)['v']+get('blank',0)['v'])
                check('zero_interaction_before_second_arrival',np.max(abs(all_j[:1738+shift,0]))<1e-14)
                for ordinal,(start,end) in enumerate([(1520,2020)]+[(1538+shift,end+shift) for end in (2038,2338,2838,3838)]):
                    vv=lambda k,d:get(k,d)['v'][start:end,0]*1000;ab=vv('AB',200);ba=vv('BA',200);z=vv('blank',0);j=(ab-vv('A0',0)-vv('Bd',200)+z)-(ba-vv('B0',0)-vv('Ad',200)+z)
                    cnt=lambda k,d,i:sum(i==i0 and start<=t<end for t,i0 in pairs(get(k,d),''));counts={k+str(d):cnt(k,d,s['target']) for k,d in SPECS}
                    reconstructed.append({'background':bg,'condition':condition,'site':sid,'subtype':s['subtype'],'shift_ticks':shift,'window_index':ordinal,'window_ticks':[start,end],
                        'voltage_order_mean_mv':float(np.mean(ab-ba)),'voltage_order_peak_abs_mv':float(np.max(abs(ab-ba))),'voltage_interaction_mean_mv':float(np.mean(j)),
                        'voltage_interaction_mean_abs_mv':float(np.mean(abs(j))),'voltage_interaction_peak_abs_mv':float(np.max(abs(j))),
                        'voltage_response_peak_abs_mv':float(max(np.max(abs(ab-z)),np.max(abs(ba-z)))),'target_spikes':counts,
                        'input_spikes':{k:{part:cnt(k,200,s[part+'_cell']) for part in ('a','b')} for k in ('AB','BA')},
                        'spike_interaction':counts['AB200']-counts['BA200']-counts['A00']-counts['Bd200']+counts['B00']+counts['Ad200']})
    check('all_metrics_independently_recomputed',len(reconstructed)==len(report['per_site'])==160 and all(set(a)==set(b) and all(np.isclose(a[k],b[k],rtol=1e-10,atol=1e-11) if isinstance(a[k],float) else a[k]==b[k] for k in a) for a,b in zip(reconstructed,report['per_site'])))
    post=read(root/'postchecks.json')
    check('three_prespecified_postchecks',len(post)==3 and p['postchecks']==[[783,'with_background',0],[784,'with_background',7],[783,'without_background',0]])
    for row,expected in zip(post,p['postchecks']):
        data=load(root/'trials',row);ref=next(r for r in rows if r['key']==row['reference_key']);rd=loaded[ref['key']]
        check('exact_post_restoration',row['exact'] and [ref[k] for k in ('background','condition','site')]==expected and ref['kind']=='AB' and ref['delay']==200 and same(row,ref) and all(np.array_equal(v,rd[k]) for k,v in data.items()))
        audit(data,pairs(data,'outgoing_'));attempt_link(row,data,expected[0],expected[2])
    check('all_native_attempts_accounted',used==set(range(1,len(budget['attempts'])+1)))
    result={'all_passed':all(checks.values()),'checks':checks,'native_attempts':len(budget['attempts']),'actual_edge_writes_checked':edge_writes,'voltage_steps_checked':equation_steps,'max_voltage_equation_error_V':max_error,'background_support':support,'report_sha256':sha(root/'report.json'),'verifier_sha256':sha(Path(__file__))}
    save(root/'verification.json',result);save(root/'independent-metrics.json',reconstructed);print(json.dumps(result,indent=2),flush=True)
    if not result['all_passed']:raise ValueError('P12b independent verification failed')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();verify(args.root)
