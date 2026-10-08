"""Independent P13c schedule, raw data, forecasting chronology and metric checks."""
import argparse
import json
import hashlib
import struct
from pathlib import Path
import numpy as np
from brian2_rust.binary_topology import inspect_csr,csr_arrays
from .verify_background_clamp import delivery_checks


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def verify(root):
    p=read(root/'protocol.json');source=Path(p['source']);parent=read(source/'protocol.json');rows=read(root/'rows.json');oldrows=read(source/'rows.json');oldby={(r['background'],r['site'],r['kind'],r['delay']):r for r in oldrows};budget=read(root/'budget.json');checks={};used=set();writes=0;steps=0;max_error=0.
    top=read(Path(parent['artifact'])/'model.json')['instance']['synapses'][1]['topology'];offsets,targets,weights=csr_arrays(inspect_csr(top['path']))
    def check(n,x):checks[n]=checks.get(n,True) and bool(x)
    def pairs(d,k=''):return list(zip(map(int,d[k+'ticks']),map(int,d[k+'indices'])))
    def load(r,row):
        path=r/'trials'/(row['key']+'.npz')
        with np.load(path,allow_pickle=False) as f:d={k:f[k] for k in f.files}
        if row['blank_key']:
            with np.load(r/'trials'/(row['blank_key']+'.npz'),allow_pickle=False) as f:
                for k in ('v','ge','gi'):d[k]=np.bitwise_xor(d[k],f[k].view(np.uint64)).view(np.float64)
        check('lossless_archives',sha(path)==row['archive_sha256'] and {k:digest(v) for k,v in d.items()}==row['array_sha256'] and {k:list(v.shape) for k,v in d.items()}==row['array_shapes']);return d
    def identity(a,b):return all(a[k]==b[k] for k in ('events_sha256','states_sha256','trace_sha256'))
    def bytes_for(ev):return b'B2SPIK01'+struct.pack('<Q',len(ev))+np.array([t for t,i in ev],dtype='<u8').tobytes()+np.array([i for t,i in ev],dtype='<u8').tobytes()
    def native(row,d):
        nonlocal writes,steps,max_error
        n=row['attempt'];used.add(n);a=budget['attempts'][n-1];work=root/'executor/work'/a['key'];sel=row['selected'];cfg=b'P12CFG01'+struct.pack('<QQ',1,len(sel))+np.array(sel,dtype='<u8').tobytes()
        check('attempt_identity',a['key']==row['key'] and identity(a['result'],row) and a['selected']==sel and a['mode']==row['mode']==1 and a['result']['audit_sha256']==row['audit_sha256']==digest(d['audit']) and a['binary_sha256']==parent['native_identity']['binary_sha256'] and a['base_sha256']==read(Path(parent['parent'])/'intact/identity.json')['base_sha256'] and a['background_sha256']==parent['background_generator_sha256'][str(row['background'])])
        check('sidecars_exact',sha(work/'visual.bin')==a['visual_sha256'] and (work/'visual.bin').read_bytes()==bytes_for(pairs(d,'visual_')) and sha(work/'outgoing.bin')==a['outgoing_sha256'] and (work/'outgoing.bin').read_bytes()==bytes_for(pairs(d,'outgoing_')) and (work/'config.bin').read_bytes()==cfg)
        ev=pairs(d,'outgoing_')
        for k,x in delivery_checks(d['audit'],ev,offsets,targets,weights).items():check(k,x)
        writes+=len(d['audit'])
        for col,c in enumerate(d['cells']):
            ts=np.array([t for t,i in pairs(d) if i==c],dtype=int);available=np.ones(5999,dtype=bool);fired=np.zeros(5999,dtype=bool)
            for t in ts:
                if t<5999:fired[t]=True
                available[t+1:min(5999,t+22)]=False
            v=d['v'][:,col];pred=np.where(available,v[:-1]+.005*(-(v[:-1]+.052)-d['ge'][:-1,col]*v[:-1]-d['gi'][:-1,col]*(v[:-1]+.070)),v[:-1]);pred=np.where(fired,-.052,pred);e=float(np.max(abs(pred-v[1:])));max_error=max(max_error,e);steps+=5999;check('LIF_reset_refractory_equation',e<1e-13)
    manifest=read(source/'file-manifest.json')['sha256'];check('P13_all_evidence_unchanged',sha(source/'file-manifest.json')==p['P13_manifest_sha256'] and sha(source/'protocol.json')==p['P13_protocol_sha256'] and all(sha(source/n)==h for n,h in manifest.items()))
    check('frozen_sources',all(sha(Path(__file__).with_name(n))==h for n,h in p['sources'].items()))
    check('unchanged_native_and_model',all(sha(root/'executor/native'/n)==sha(source/'executor/native'/n) for n in ('main.rs','b2-native','identity.json')) and sha(Path(top['path']))==top['sha256'] and sha(Path(parent['parent'])/'intact/base.bin')==read(Path(parent['parent'])/'intact/identity.json')['base_sha256'])
    check('budget_complete',budget['limit']==p['hard_limit']==128 and len(budget['attempts'])==p['planned_native']['total']==116 and [a['number'] for a in budget['attempts']]==list(range(1,117)) and all(a['status']=='completed' and a['time_unix']>p['frozen_at_unix'] for a in budget['attempts']))
    check('fixed_scope',p['backgrounds']==[785,787] and p['delta_ticks']==500 and p['sites']==parent['sites'] and p['monitor_cells']==parent['monitor_cells'] and len(rows)==128)
    gates=read(root/'gates.json')
    check('two_initial_identity_gates',len(gates)==2 and [r['attempt'] for r in gates]==[1,2] and [r['background'] for r in gates]==[785,787])
    for row in gates:
        d=load(root,row);ref=oldby[row['background'],0,'AB',200];old=load(source,ref);check('identity_gates_exact',row['reference_key']==ref['key'] and row['exact'] and identity(row,ref) and all(np.array_equal(v,old[k]) for k,v in d.items()));native(row,d)
    specs=[('blank',0),('A0',0),('B0',0),('Ad',200),('Bd',200),('AB',0),('AB',200),('BA',200)];design=[];data={};shiftmap={};rowby={}
    for bg in [785,787]:
        for sid in range(8):
            shift=oldby[bg,sid,'blank',0]['shift']+500;shiftmap[bg,sid]=shift
            for kind,delay in specs:design.append({'background':bg,'site':sid,'kind':kind,'delay':delay,'shift':shift,'key':f'b{bg}-s{sid}-{kind}-{delay}'})
    check('complete_frozen_design',design==read(root/'design.json') and sha(root/'design.json')==p['design_sha256'] and design==[{k:r[k] for k in design[0]} for r in rows])
    for row in rows:
        bg=row['background'];sid=row['site'];s=p['sites'][sid];kind=row['kind'];delay=row['delay'];d=load(root,row);data[bg,sid,kind,delay]=d;rowby[bg,sid,kind,delay]=row;ref=oldby[bg,sid,kind,delay];old=load(source,ref);blank=load(source,oldby[bg,sid,'blank',0]);stim=[(t+500,i) for t,i in pairs(old,'stimulus_')];visual=[(t+500,i) for t,i in pairs(old,'visual_')];background=pairs(old,'background_');out=sorted(background+stim);pair=sorted([s['a_cell'],s['b_cell']])
        check('only_stimulus_onset_changed',pairs(d,'visual_')==visual and pairs(d,'stimulus_')==stim and pairs(d,'background_')==background and pairs(d,'outgoing_')==out and d['cells'].tolist()==[s['target'],s['a_cell'],s['b_cell']] and row['selected']==pair and row['mode']==1)
        check('no_event_conflicts',len(out)==len(set(out)) and all(np.all(np.diff([t for t,i in out if i==c])>=22) for c in pair))
        if kind=='blank':check('all_16_blank_aliases_exact',row['attempt'] is None and row['reference_blank']==oldby[bg,sid,'blank',0]['key'] and identity(row,oldby[bg,sid,'blank',0]) and all(np.array_equal(v,blank[k]) for k,v in d.items()))
        else:
            check('pre_stimulus_blank',all(np.array_equal(d[k][:1520+row['shift']],blank[k][:1520+row['shift']]) for k in ('v','ge','gi')));native(row,d)
    sign=lambda x:0 if abs(x)<=1e-8 else 1 if x>0 else -1;preds=read(root/'predictions.json');check('all_16_forecasts',len(preds)==16 and len({(x['background'],x['site']) for x in preds})==16);comparisons=[]
    for bg in [785,787]:
        for sid in range(8):
            shift=shiftmap[bg,sid];start=1538+shift;end=2038+shift;sl=slice(start,end);get=lambda k,d:data[bg,sid,k,d];v=lambda k,d:get(k,d)['v'][sl,0]*1000
            raw=v('AB',200)-v('BA',200);linear=(v('A0',0)+v('Bd',200))-(v('B0',0)+v('Ad',200));j=raw-linear;pred=next(x for x in preds if x['background']==bg and x['site']==sid);singles=[rowby[bg,sid,k,d] for k,d in [('A0',0),('B0',0),('Ad',200),('Bd',200)]];paired=[rowby[bg,sid,k,d] for k,d in [('AB',0),('AB',200),('BA',200)]]
            check('forecast_precedes_all_pairs',pred['before_paired_runs'] and max(budget['attempts'][r['attempt']-1]['time_unix']+budget['attempts'][r['attempt']-1]['seconds'] for r in singles)<pred['time_unix']<min(budget['attempts'][r['attempt']-1]['time_unix'] for r in paired) and pred['single_archive_hashes']=={r['key']:r['archive_sha256'] for r in singles})
            # Producer's serialized expression order is checked separately from independent grouped algebra.
            serialized=v('A0',0)+v('Bd',200)-v('B0',0)-v('Ad',200)
            check('forecasts_recomputed_from_singles',np.isclose(pred['linear_mean_mv'],linear.mean(),atol=1e-11,rtol=1e-10) and pred['linear_sign']==sign(linear.mean()) and pred['linear_waveform_sha256']==digest(serialized))
            oldstart=start-500;oldend=end-500;oldv=lambda k,d:load(source,oldby[bg,sid,k,d])['v'][oldstart:oldend,0]*1000;oldraw=oldv('AB',200)-oldv('BA',200);oldlin=oldv('A0',0)+oldv('Bd',200)-oldv('B0',0)-oldv('Ad',200);oldj=oldraw-oldlin;rms=lambda a:float(np.sqrt(np.mean(a*a)));norm=rms(raw)
            counts={k+str(d):sum(start<=t<end and i==p['sites'][sid]['target'] for t,i in pairs(get(k,d))) for k,d in specs}
            comparisons.append({'background':bg,'site':sid,'subtype':p['sites'][sid]['subtype'],'window_ticks':[start,end],'old_window_ticks':[oldstart,oldend],'raw_mean_mv':float(raw.mean()),'linear_mean_mv':float(linear.mean()),'J_mean_mv':float(j.mean()),'J_mean_abs_mv':float(abs(j).mean()),'raw_rms_mv':norm,'relative_RMS_error':rms(j)/norm if norm>1e-8 else None,'raw_sign':sign(raw.mean()),'linear_sign':sign(linear.mean()),'old_raw_mean_mv':float(oldraw.mean()),'old_raw_sign':sign(oldraw.mean()),'old_J_mean_abs_mv':float(abs(oldj).mean()),'old_J_sign':sign(oldj.mean()),'J_sign':sign(j.mean()),'target_spikes':counts,'target_count_interaction':counts['AB200']-counts['BA200']-counts['A00']-counts['Bd200']+counts['B00']+counts['Ad200']})
            audit_a=get('AB',200)['audit'];audit_b=get('BA',200)['audit'];check('balanced_pair_edge_budgets',np.array_equal(np.unique(audit_a[:,2],return_counts=True),np.unique(audit_b[:,2],return_counts=True)))
    post=read(root/'restorations.json');check('two_prespecified_restorations',len(post)==2 and p['restorations']==[[785,0,'AB'],[787,7,'BA']] and [r['attempt'] for r in post]==[115,116])
    for row,spec in zip(post,p['restorations']):
        d=load(root,row);ref=rowby[spec[0],spec[1],spec[2],200];old=data[spec[0],spec[1],spec[2],200];check('restorations_exact',row['exact'] and row['reference_key']==ref['key'] and identity(row,ref) and all(np.array_equal(v,old[k]) for k,v in d.items()));native(row,d)
    check('every_launch_accounted',used==set(range(1,117)))
    save(root/'comparison.json',comparisons);result={'all_passed':all(checks.values()),'checks':checks,'new_native_runs':len(budget['attempts']),'actual_edge_writes_checked':writes,'voltage_steps_checked':steps,'max_voltage_error_V':max_error,'verifier_sha256':sha(__file__),'comparison_sha256':sha(root/'comparison.json'),'protocol_sha256':sha(root/'protocol.json'),'rows_sha256':sha(root/'rows.json')};save(root/'verification.json',result);print(json.dumps(result,indent=2),flush=True)
    if not result['all_passed']:raise ValueError('onset diagnostic verification failed')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('root',type=Path);verify(a.parse_args().root)
