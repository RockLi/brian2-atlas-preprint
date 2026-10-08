"""Independent P12 reconstruction from lossless traces, native writes and CSR."""
import argparse
import hashlib
import struct
from pathlib import Path
import numpy as np
from brian2_rust.binary_topology import inspect_csr,csr_arrays
from .motion_refinement import read,sha,load_npz
from .multispeed_data import digest
from .run_experiment import save
from .convergent_archive import decode


def verify(root):
    f=root/'formal-v1';g=root/'validation-v1';p=read(f/'protocol.json');report=read(f/'report.json');rows=read(f/'rows.json');budget=read(root/'budget.json');p11=Path('brian2-rust/validation/flywire-vision-convergent-probe-v1')
    oldp=read(p11/'protocol.json');model=read(Path(oldp['artifact'])/'model.json');top=model['instance']['synapses'][1]['topology'];offsets,targets,weights=csr_arrays(inspect_csr(top['path']))
    checks={'complete':report['status']=='complete' and len(rows)==256 and report['cumulative_native_runs']==len(budget['attempts'])==250,
        'budget':budget['limit']==300 and len(budget['attempts'])<=300 and all(a['status']=='completed' for a in budget['attempts']) and len({a['key'] for a in budget['attempts']})==len(budget['attempts']),
        'sources':all(sha(Path(__file__).with_name(n))==h for n,h in p['sources'].items()),
        'report_links':report['protocol_sha256']==sha(f/'protocol.json') and report['rows_sha256']==sha(f/'rows.json') and report['postchecks_sha256']==sha(f/'postchecks.json') and p['matrix_sha256']==sha(f/'design.json'),
        'targets':p['sites']==oldp['sites'],'graph':sha(Path(top['path']))==top['sha256'],'base':sha(Path(oldp['parent'])/'intact'/'base.bin')==read(g/'native'/'identity.json').get('base_sha256',budget['attempts'][0]['base_sha256']),
        'binary':sha(g/'native'/'b2-native')==p['native_identity']['binary_sha256'] and sha(g/'native'/'main.rs')==p['native_identity']['source_sha256'],
        'background':sha(root/'background-784.bin')==p['background_784_sha256'],
        'gates_before_freeze':read(g/'verification.json')['all_passed'] and sha(g/'verification.json')==p['gates_verification_sha256'] and (g/'verification.json').stat().st_mtime_ns<(f/'protocol.json').stat().st_mtime_ns}
    expected_base=read(Path(oldp['parent'])/'intact'/'identity.json')['base_sha256']
    checks['all_attempt_identities']=all(a['base_sha256']==expected_base and a['binary_sha256']==p['native_identity']['binary_sha256'] for a in budget['attempts'])
    checks['attempt_sidecars']=True
    for a in budget['attempts']:
        work=g/'work'/a['key'];expected_config=b'P12CFG01'+struct.pack('<QQ',a['mode'],len(a['selected']))+np.array(a['selected'],dtype='<u8').tobytes()
        checks['attempt_sidecars'] &= sha(work/'visual.bin')==a['visual_sha256'] and sha(work/'outgoing.bin')==a['outgoing_sha256'] and (work/'config.bin').read_bytes()==expected_config
    checks['parent_native_source']=p['native_identity']['parent_source_sha256']==sha(p11/'native'/'main.rs')
    gp=read(g/'protocol.json')
    checks['gate_sources']=all(sha(Path(__file__).with_name(n))==h for n,h in gp['sources'].items())
    checks['gate_protocol_before_runs']=all(a['time_unix']>(g/'protocol.json').stat().st_mtime for a in budget['attempts'][:9])
    bg_rows=[];rng=np.random.default_rng(int.from_bytes(hashlib.sha256(b'flywire-mnist-v1/784/background/0').digest()[:16],'little'))
    for channel in range(512):
        tick=-1
        while True:
            tick+=int(rng.geometric(.03))
            if tick>=6000:break
            bg_rows.append((tick,channel))
    bg_rows.sort();bg=np.array(bg_rows,dtype='<u8')
    expected_bg=b'B2SPIK01'+struct.pack('<Q',len(bg_rows))+bg[:,0].tobytes()+bg[:,1].tobytes()
    checks['independent_background_seed']=expected_bg==(root/'background-784.bin').read_bytes()
    checks['prior_evidence_unchanged']=all(sha(Path('brian2-rust/validation')/({'P10':'flywire-vision-timing-probe-v1','P11':'flywire-vision-convergent-probe-v1'}[key.split('/')[0]])/key.split('/')[1])==h for key,h in gp['prior_evidence'].items())
    for path in (Path('brian2-rust/validation/flywire-vision-timing-probe-v1'),p11):
        op=read(path/'protocol.json');checks['prior_evidence_unchanged'] &= all(sha(Path(__file__).with_name(n))==h for n,h in op['source_sha256'].items())
    graph=load_npz(root/'outgoing-graph.npz');expected_graph=[]
    for source in sorted({s[k] for s in p['sites'] for k in ('a_cell','b_cell')}):
        expected_graph += [(source,edge,int(targets[edge]),weights[0,edge]) for edge in range(int(offsets[source]),int(offsets[source+1]))]
    checks['selected_edge_archive']=all(np.array_equal(graph[k],np.array([a[j] for a in expected_graph])) for j,k in enumerate(('source','edge','target','contacts')))
    checks.update({k:True for k in ('trial_archives','literal_schedules','delivery_events','actual_weights_and_writes','no_duplicate_delivery','pre_stimulus','before_second_arrival','P11_ordinary_reproduction','metrics','equation','balanced_edges','attempt_links')})
    ar=read(p11/'traces'/'manifest.json');oldar={r['key']:r for r in ar['rows']};oldrows={(r['site'],r['kind'],r['delay']):r for r in read(p11/'rows.json') if r['condition']=='intact'}
    blanks={};loaded={};by={};max_equation_error=0.;equation_steps=0;edge_events=0
    def load(directory,row):
        data=load_npz(directory/(row['key']+'.npz'))
        if row['blank_key']:
            b=load_npz(directory/(row['blank_key']+'.npz'))
            for k in ('v','ge','gi'):data[k]=np.bitwise_xor(data[k],b[k].view(np.uint64)).view(np.float64)
        checks['trial_archives'] &= sha(directory/(row['key']+'.npz'))==row['archive_sha256'] and {k:digest(v) for k,v in data.items()}==row['array_sha256'] and {k:list(v.shape) for k,v in data.items()}==row['array_shapes']
        return data
    def audit_check(data,selected,mode):
        nonlocal edge_events
        events=zip(data['outgoing_ticks'],data['outgoing_indices']) if mode==1 else zip(data['ticks'],data['indices']);expected=[];w=[]
        for tick,source in events:
            source=int(source);tick=int(tick)
            if source not in selected or tick+18>=6000:continue
            for edge in range(int(offsets[source]),int(offsets[source+1])):
                expected.append((tick+18,source,edge,int(targets[edge])));w.append(weights[0,edge])
        actual=data['audit'];checks['delivery_events'] &= np.array_equal(actual[:,:4],np.array(expected,dtype=np.uint64).reshape(-1,4));edge_events+=len(actual)
        checks['no_duplicate_delivery'] &= len(actual)==len({(int(r[0]),int(r[2])) for r in actual})
        floats=actual[:,4:].copy().view(np.float64);w=np.asarray(w);trans=0. if mode==2 else 1.;ge=np.array([(.275/52)*(x*int(x>0)*trans) for x in w]);gi=np.array([-((.275/52)*4)*(x*int(x<0)*trans) for x in w])
        checks['actual_weights_and_writes'] &= len(actual)==len(ge) and np.array_equal(floats[:,0],ge) and np.array_equal(floats[:,1],gi) and np.array_equal(floats[:,3],floats[:,2]+floats[:,0]) and np.array_equal(floats[:,5],floats[:,4]+floats[:,1])
    # Independently inspect all gate records, including the transmission-zero reference.
    gate_data={};gate_rows={r['key']:r for r in read(g/'rows.json')}
    for key,row in gate_rows.items():
        data=load(g/'trials',row);gate_data[key]=data;audit_check(data,row['selected'],row['mode'])
    def same(a,b,skip=()):
        return a['events_sha256']==b['events_sha256'] and all(v==b['states_sha256'][k] for k,v in a['states_sha256'].items() if k not in skip)
    checks['gates_reconstructed']=True
    for a,b,skip in [('gate01-ab','gate02-native-replay',()),('gate01-ab','gate05-restored-ab',()),('gate03-cut-reference','gate04-empty-replacement',('transmission',)),('gate06-second-bg-ab','gate07-second-bg-replay',())]:
        checks['gates_reconstructed'] &= same(gate_rows[a],gate_rows[b],skip) and all(np.array_equal(gate_data[a][k],gate_data[b][k]) for k in ('v','ge','gi','indices','ticks'))
    for a,b in [('gate01-ab','gate02-native-replay'),('gate06-second-bg-ab','gate07-second-bg-replay')]:
        sel=gate_rows[b]['selected'];mask=np.isin(gate_data[a]['indices'],sel)
        checks['gates_reconstructed'] &= np.array_equal(gate_data[b]['outgoing_indices'],gate_data[a]['indices'][mask]) and np.array_equal(gate_data[b]['outgoing_ticks'],gate_data[a]['ticks'][mask]) and np.array_equal(gate_data[a]['audit'],gate_data[b]['audit'])
    expected_design=[]
    for bg in (783,784):
        for condition in ('ordinary','controlled'):
            for site in range(8):
                for kind,d in [('blank',0),('A0',0),('B0',0),('Ad',200),('Bd',200),('AB',0),('AB',200),('BA',200)]:
                    expected_design.append({'background':bg,'condition':condition,'site':site,'kind':kind,'delay':d,'key':f'b{bg}-{condition}-s{site}-{kind}-{d}'})
    checks['design']=expected_design==read(f/'design.json')==[{k:r[k] for k in expected_design[0]} for r in rows]
    for row in rows:
        data=load(f/'trials',row);loaded[row['key']]=data;key=(row['background'],row['condition'],row['site'],row['kind'],row['delay']);by[key]=(data,row);s=p['sites'][row['site']];pair=[s['a_cell'],s['b_cell']];kind=row['kind'];d=row['delay']
        parts={'blank':[],'A0':[('a',0)],'B0':[('b',0)],'Ad':[('a',d)],'Bd':[('b',d)],'AB':[('a',0),('b',d)],'BA':[('b',0),('a',d)]}[kind]
        vis=sorted((1500+offset+t,s[part][0]) for part,offset in parts for t in (20,120));ev=sorted((1500+offset+t,s[part+'_cell']) for part,offset in parts for t in (20,120)) if row['condition']=='controlled' else []
        checks['literal_schedules'] &= vis==list(zip(data['visual_ticks'],data['visual_indices'])) and ev==list(zip(data['outgoing_ticks'],data['outgoing_indices'])) and data['cells'].tolist()==[s['target'],s['a_cell'],s['b_cell']]
        audit_check(data,pair,int(row['condition']=='controlled'))
        attempt=budget['attempts'][row['attempt']-1];checks['attempt_links'] &= attempt['result']['events_sha256']==row['events_sha256'] and attempt['result']['states_sha256']==row['states_sha256'] and attempt['key']==(row['gate_alias'] or row['key'])
        if row['gate_alias'] is None:checks['attempt_links'] &= attempt['time_unix']>(f/'protocol.json').stat().st_mtime and attempt['mode']==int(row['condition']=='controlled') and attempt['selected']==sorted(pair) and attempt['background_sha256']==(p['background_784_sha256'] if row['background']==784 else None)
        group=key[:3]
        if kind=='blank':blanks[group]=data
        else:checks['pre_stimulus'] &= all(np.array_equal(data[k][:1520],blanks[group][k][:1520]) for k in ('v','ge','gi'))
        if row['background']==783 and row['condition']=='ordinary':
            old=oldrows[-1 if kind=='blank' else row['site'],kind,d];oldmeta=oldar[old['key']];olddata=decode(load_npz(p11/'traces'/(old['key']+'.npz')),load_npz(p11/'traces'/oldmeta['blank']));columns=[oldp['monitor_cells'].index(int(c)) for c in data['cells']]
            checks['P11_ordinary_reproduction'] &= same(row,old) and all(np.array_equal(data[k],olddata[k][:,columns]) for k in ('v','ge','gi'))
        active=np.ones((5999,3),dtype=bool)
        for tick,cell in zip(data['ticks'],data['indices']):active[int(tick):min(5999,int(tick)+24),list(data['cells']).index(cell)]=False
        vv=data['v'];expected=vv[:-1]+.005*(-(vv[:-1]+.052)-data['ge'][:-1]*vv[:-1]-data['gi'][:-1]*(vv[:-1]+.070));err=float(np.max(abs(vv[1:][active]-expected[active])));max_equation_error=max(max_equation_error,err);equation_steps+=int(active.sum());checks['equation'] &= err<1e-13
    reconstructed=[]
    for bg in (783,784):
        for condition in ('ordinary','controlled'):
            for s in p['sites']:
                sid=s['id'];get=lambda k,d:by[bg,condition,sid,k,d][0]
                if condition=='controlled':
                    ab=get('AB',200)['audit'];ba=get('BA',200)['audit']
                    # Exact count and signed injected increment budget at every real edge.
                    for edge in set(map(int,ab[:,2]))|set(map(int,ba[:,2])):
                        aa=ab[ab[:,2]==edge];bb=ba[ba[:,2]==edge];checks['balanced_edges'] &= len(aa)==len(bb)==2 and np.array_equal(aa[:,4:6],bb[:,4:6])
                for start,end in ((1520,2020),(1538,2038),(1538,2338),(1538,2838),(1538,3838)):
                    vv=lambda k,d:get(k,d)['v'][start:end,0]*1000;ab=vv('AB',200);ba=vv('BA',200);z=vv('blank',0)
                    ja=ab-(vv('A0',0)+vv('Bd',200)-z);jb=ba-(vv('B0',0)+vv('Ad',200)-z);j=ja-jb
                    cnt=lambda k,d,cell:sum(1 for cell0,t in zip(get(k,d)['indices'],get(k,d)['ticks']) if cell==cell0 and start<=t<end)
                    target={k+str(d):cnt(k,d,s['target']) for k,d in [('blank',0),('A0',0),('B0',0),('Ad',200),('Bd',200),('AB',0),('AB',200),('BA',200)]}
                    reconstructed.append({'background':bg,'condition':condition,'site':sid,'subtype':s['subtype'],'window_ticks':[start,end],
                        'voltage_order_mean_mv':float(np.mean(ab-ba)),'voltage_order_peak_abs_mv':float(np.max(abs(ab-ba))),
                        'voltage_interaction_mean_mv':float(np.mean(j)),'voltage_interaction_peak_abs_mv':float(np.max(abs(j))),'voltage_interaction_mean_abs_mv':float(np.mean(abs(j))),
                        'voltage_response_peak_abs_mv':float(max(np.max(abs(ab-z)),np.max(abs(ba-z)))),'target_spikes':target,
                        'input_spikes':{k:{part:cnt(k,200,s[part+'_cell']) for part in ('a','b')} for k in ('AB','BA')},
                        'spike_interaction':target['AB200']-target['BA200']-target['A00']-target['Bd200']+target['B00']+target['Ad200']})
                    checks['before_second_arrival'] &= np.max(abs(j[:max(0,1738-start)]))<1e-11
    checks['metrics']=len(reconstructed)==len(report['per_site']) and all(all(np.isclose(a[k],b[k],atol=1e-11,rtol=1e-10) if isinstance(a[k],float) else a[k]==b[k] for k in a) for a,b in zip(reconstructed,report['per_site']))
    checks['post_restoration']=True
    for row in read(f/'postchecks.json'):
        data=load(f/'trials',row);ref=loaded[row['reference_key']];rr=next(r for r in rows if r['key']==row['reference_key']);checks['post_restoration'] &= row['exact'] and same(row,rr) and all(np.array_equal(v,ref[k]) for k,v in data.items())
    postrows=read(f/'postchecks.json')
    used=set(r['attempt'] for r in gate_rows.values())|set(r['attempt'] for r in rows)|set(r['attempt'] for r in postrows)
    checks['all_attempts_accounted']=used==set(range(1,len(budget['attempts'])+1)) and len(postrows)==4 and sum(r['gate_alias'] is None for r in rows)==237
    checks['post_attempt_links']=all(budget['attempts'][r['attempt']-1]['key']==r['key'] and budget['attempts'][r['attempt']-1]['result']['events_sha256']==r['events_sha256'] for r in postrows)
    result={'all_passed':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'native_attempts':len(budget['attempts']),'actual_edge_writes_checked':edge_events,'voltage_steps_checked':equation_steps,'max_voltage_equation_error_V':max_equation_error,'report_sha256':sha(f/'report.json'),'verifier_sha256':sha(Path(__file__))}
    save(f/'verification.json',result);print(result,flush=True)
    if not result['all_passed']:raise ValueError('P12 verification failed')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();verify(a.root)
