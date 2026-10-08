"""Post-hoc P13 response accounting, with no simulator launches or causal claims."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

P13=Path('brian2-rust/validation/flywire-vision-independent-repeat-v1')
SPECS=[('blank',0),('A0',0),('B0',0),('Ad',200),('Bd',200),('AB',0),('AB',200),('BA',200)]
J_COEF={('AB',200):1,('BA',200):-1,('A0',0):-1,('Bd',200):-1,('B0',0):1,('Ad',200):1}
L_COEF={('A0',0):1,('Bd',200):1,('B0',0):-1,('Ad',200):-1}

def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def decode(root,row):
    with np.load(root/'trials'/(row['key']+'.npz'),allow_pickle=False) as z:d={k:z[k] for k in z.files}
    assert sha(root/'trials'/(row['key']+'.npz'))==row['archive_sha256']
    if row['blank_key']:
        with np.load(root/'trials'/(row['blank_key']+'.npz'),allow_pickle=False) as z:
            for k in ('v','ge','gi'):d[k]=np.bitwise_xor(d[k],z[k].view(np.uint64)).view(np.float64)
    return d

def currents(d):
    v=d['v'][:,0];ge=d['ge'][:-1,0];gi=d['gi'][:-1,0];n=len(v)-1
    spikes=d['ticks'][d['indices']==d['cells'][0]].astype(int)
    active=np.ones(n,dtype=bool);fired=np.zeros(n,dtype=bool)
    for t in spikes:
        if t<n:fired[t]=True
        active[t+1:min(n,t+22)]=False
    inc={'leak':.005*(-v[:-1]-.052)*active,'exc':.005*(-ge*v[:-1])*active,'inh':.005*(-gi*(v[:-1]+.070))*active}
    predicted=v[:-1]+sum(inc.values());expected=np.where(fired,-.052,predicted)
    error=float(np.max(abs(v[1:]-expected)))
    if error>1e-13:raise ValueError('LIF/reset accounting mismatch')
    inc['reset']=np.where(fired,-.052-predicted,0.)
    cumulative={k:np.r_[0.,np.cumsum(a)]*1000 for k,a in inc.items()}
    return cumulative,error,spikes

def analyze(data,start,end):
    voltage={k:d['v'][:,0]*1000 for k,d in data.items()}
    op=lambda c,values:sum(n*values[k] for k,n in c.items())
    raw=voltage['AB',200]-voltage['BA',200];linear=op(L_COEF,voltage);j=op(J_COEF,voltage)
    assert np.max(abs(raw-linear-j))<1e-10
    accounting={};errors=[];spikes={}
    for k,d in data.items():accounting[k],err,spikes[k]=currents(d);errors.append(err)
    components={name:op(J_COEF,{k:a[name] for k,a in accounting.items()}) for name in ('leak','exc','inh','reset')}
    initial=op(J_COEF,{k:a[0] for k,a in voltage.items()});reconstruction=float(np.max(abs(j-initial-sum(components.values()))))
    assert reconstruction<1e-9
    w=slice(start,end);rms=lambda a:float(np.sqrt(np.mean(a*a)));sgn=lambda x:0 if abs(x)<=1e-8 else 1 if x>0 else -1
    nrm=rms(raw[w]);jrm=rms(j[w]);reset=components['reset'][w]
    localspikes={k[0]+str(k[1]):[int(t) for t in ts if start<=t<end] for k,ts in spikes.items()}
    first=min((int(t) for ts in spikes.values() for t in ts if start<=t<end),default=None)
    pre=slice(start,first+1 if first is not None else end)
    target=int(data['blank',0]['cells'][0]);baseline=data['blank',0]
    def divergence(name):
        a=op(J_COEF,{k:d[name][:,0] for k,d in data.items()});ix=np.flatnonzero(abs(a[start:end])>(1e-11 if name=='v' else 1e-12));return int(start+ix[0]) if len(ix) else None
    return {'window_ticks':[start,end],'target':target,'raw_mean_mv':float(raw[w].mean()),'linear_mean_mv':float(linear[w].mean()),'J_mean_mv':float(j[w].mean()),'raw_sign':sgn(raw[w].mean()),'linear_sign':sgn(linear[w].mean()),'J_sign':sgn(j[w].mean()),'raw_rms_mv':nrm,'J_rms_mv':jrm,'relative_superposition_error':jrm/nrm if nrm>1e-8 else None,'J_mean_abs_mv':float(abs(j[w]).mean()),'J_peak_abs_mv':float(abs(j[w]).max()),'J_before_first_target_spike_peak_mv':float(abs(j[pre]).max()),'component_J_mean_mv':{k:float(a[w].mean()) for k,a in components.items()},'component_J_rms_mv':{k:rms(a[w]) for k,a in components.items()},'any_target_spike':any(localspikes.values()),'target_spike_ticks':localspikes,'first_target_spike_tick':first,'baseline_v_mv':float(baseline['v'][start,0]*1000),'baseline_threshold_distance_mv':float(-45-baseline['v'][start,0]*1000),'baseline_ge':float(baseline['ge'][start,0]),'baseline_gi':float(baseline['gi'][start,0]),'first_interaction_tick':{k:divergence(k) for k in ('v','ge','gi')},'max_step_error_V':max(errors),'max_J_accounting_error_mv':reconstruction}

def build(output):
    output.mkdir(parents=True,exist_ok=False)
    manifest=read(P13/'file-manifest.json')['sha256']
    assert all(sha(P13/n)==h for n,h in manifest.items())
    protocol={'source':str(P13),'source_manifest_sha256':sha(P13/'file-manifest.json'),'analysis_source_sha256':sha(__file__),'type':'post-hoc descriptive accounting; source data previously inspected; not a held-out confirmation','new_native_runs':0,'scope':'all 8 targets and all 4 backgrounds; primary and first-arrival-aligned 50ms windows; D=linear+J; exact LIF leak/excitatory/inhibitory/reset accounting','limitations':'current components depend on observed voltage and interventions; accounting terms are not independent causal shares; numerical detection floors do not imply functional significance'}
    save(output/'protocol.json',protocol);p=read(P13/'protocol.json');rows=read(P13/'rows.json');index={(r['background'],r['site'],r['kind'],r['delay']):r for r in rows};out=[]
    for bg in p['backgrounds']:
        for s in p['sites']:
            sid=s['id'];data={k:decode(P13,index[bg,sid,*k]) for k in SPECS};shift=index[bg,sid,'blank',0]['shift']
            for wi,(a,b) in enumerate(((1520,2020),(1538+shift,2038+shift))):out.append({'background':bg,'site':sid,'subtype':s['subtype'],'window_index':wi,**analyze(data,a,b)})
    save(output/'metrics.json',out);primary=[r for r in out if r['window_index']==0];relative=[r['relative_superposition_error'] for r in primary if r['relative_superposition_error'] is not None]
    result={'all_accounting_checks_passed':True,'cases':len(primary),'windows':len(out),'raw_linear_same_sign':sum(r['raw_sign']==r['linear_sign']!=0 for r in primary),'relative_error_min_median_max':[float(f(relative)) for f in (np.min,np.median,np.max)],'spiking_cases':sum(r['any_target_spike'] for r in primary),'max_step_error_V':max(r['max_step_error_V'] for r in out),'max_J_accounting_error_mv':max(r['max_J_accounting_error_mv'] for r in out),'per_background':{str(bg):{'mean_abs_J_mv':float(np.mean([r['J_mean_abs_mv'] for r in primary if r['background']==bg])),'median_relative_error':float(np.median([r['relative_superposition_error'] for r in primary if r['background']==bg]))} for bg in p['backgrounds']}}
    assert all(sha(P13/n)==h for n,h in manifest.items());save(output/'summary.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('output',type=Path);build(a.parse_args().output)
