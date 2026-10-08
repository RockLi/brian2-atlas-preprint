"""Independent analytic-filter and 34-digit Decimal local-LIF reconstruction."""
import argparse
from decimal import Decimal, localcontext
import json
import hashlib
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def decode(source,row):
    with np.load(source/'trials'/(row['key']+'.npz'),allow_pickle=False) as z:d={k:z[k] for k in z.files}
    if row['blank_key']:
        with np.load(source/'trials'/(row['blank_key']+'.npz'),allow_pickle=False) as z:
            for k in ('v','ge','gi'):d[k]=np.bitwise_xor(d[k],z[k].view(np.uint64)).view(np.float64)
    return d

def analytic(events):
    x=np.zeros(6000);g=np.zeros(6000,dtype=np.longdouble);t=np.arange(6000)
    for arrival,w in events:
        n=t-arrival;mask=n>=2;x[mask]+=(.275/52)*w*.005*.052*1000*(.995**(n[mask]-1)-.98**(n[mask]-1))/.015
        mask=n>=1;g[mask]+=np.longdouble((.275/52)*w)*np.longdouble('.98')**(n[mask]-1)
    return x,g

def integrate(blank,g):
    # On this ARM host numpy.longdouble is only float64. Use real decimal precision.
    with localcontext() as ctx:
        ctx.prec=34;dt=Decimal('.005');rest=Decimal('-.052');inhib=Decimal('.070');threshold=Decimal('-.045')
        v=Decimal.from_float(float(blank['v'][0,0]));ge=[Decimal.from_float(float(x)) for x in blank['ge'][:,0]+g];gi=[Decimal.from_float(float(x)) for x in blank['gi'][:,0]];values=[];last=-1000;spikes=[]
        for t in range(6000):
            values.append(float(v*1000))
            if t-last>=22:
                v+=(rest-v-ge[t]*v-gi[t]*(v+inhib))*dt
                if v>threshold:last=t;spikes.append(t);v=rest
    return np.array(values),np.array(spikes,dtype=np.int64)

def same(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return b is not None and np.isclose(a,b,atol=1e-9,rtol=1e-10)
    return a==b

def verify(root):
    p=read(root/'protocol.json');rows=read(root/'prediction-rows.json');gates=read(root/'blank-gates.json');sealed=read(root/'prediction-seal.json');metrics=read(root/'metrics.json');checks={};linear_error=0.;local_error=0.;recomputed=[];expectedcases=[];sources=list(p['source_manifests']);sourceby={}
    def check(k,x):checks[k]=checks.get(k,True) and bool(x)
    check('fixed_models_scope',p['new_native_runs']==0 and p['variants']==['original','balanced','swapped'] and p['primary_window_index']==0 and p['criteria']['sign_fraction_min']==.875 and p['criteria']['median_relative_RMS_error_max']==.2)
    check('producer_sources_unchanged',all(sha(Path(__file__).with_name(n))==h for n,h in p['sources'].items()))
    for si,s in enumerate(sources):
        source=Path(s);sp=read(source/'protocol.json');sr=read(source/'rows.json');sourceby[s]={(r['background'],r['site'],r['kind'],r['delay']):r for r in sr}
        check('all_parent_evidence_unchanged',sha(source/'file-manifest.json')==p['source_manifests'][s] and all(sha(source/n)==h for n,h in read(source/'file-manifest.json')['sha256'].items()))
        for b in sr:
            if b['kind']=='blank':expectedcases.append((si,b['background'],b['site'],b['shift']))
    check('all_48_cases',len(rows)==48 and expectedcases==[(r['source_index'],r['background'],r['site'],r['shift']) for r in rows] and len(gates)==48)
    check('prediction_seal_identity',sealed['time_unix']>p['frozen_at_unix'] and sealed['protocol_sha256']==sha(root/'protocol.json') and sealed['prediction_rows_sha256']==sha(root/'prediction-rows.json') and sealed['all_predictions_before_loading_stimulated_responses'])
    for r in rows:
        source=Path(r['source']);sp=read(source/'protocol.json');s=sp['sites'][r['site']];by=sourceby[str(source)];br=by[r['background'],r['site'],'blank',0];blank=decode(source,br);path=root/'predictions'/(r['key']+'.npz')
        check('anatomical_strengths_and_blank_identity',r['contacts']==s['contacts'] and r['target']==s['target'] and r['blank_key']==br['key'] and r['blank_archive_sha256']==br['archive_sha256'] and sha(path)==r['archive_sha256'])
        base,bs=integrate(blank,np.zeros(6000,dtype=np.longdouble));gate=next(x for x in gates if x['key']==r['key']);check('all_blank_gates_reconstructed',gate['spikes_exact'] and gate['max_voltage_error_mv']<1e-9 and np.max(abs(base-blank['v'][:,0]*1000))<1e-8 and np.array_equal(bs,blank['ticks'][blank['indices']==s['target']]))
        with np.load(path,allow_pickle=False) as z:
            for variant in p['variants']:
                a,b=s['contacts'];a,b=(a,b) if variant=='original' else ((a+b)/2,(a+b)/2) if variant=='balanced' else (b,a)
                for order in ('AB','BA'):
                    weights=(a,b) if order=='AB' else (b,a);events=[(1538+r['shift']+offset+t,w) for offset,w in zip((0,200),weights) for t in (0,100)];prediction,g=analytic(events);err=float(np.max(abs(z[f'linear_{variant}_{order}']-prediction)));linear_error=max(linear_error,err);check('linear_analytic_solution',err<1e-8)
                    vv,ss=integrate(blank,g);err=float(np.max(abs(vv-z[f'local_{variant}_{order}'])));local_error=max(local_error,err);check('local_34_digit_decimal_solution',err<1e-8 and np.array_equal(ss,z[f'local_{variant}_{order}_spikes']))
            for model in ('linear','local'):
                check('balanced_zero_and_swapped_exchange',np.array_equal(z[f'{model}_balanced_AB'],z[f'{model}_balanced_BA']) and np.array_equal(z[f'{model}_swapped_AB'],z[f'{model}_original_BA']) and np.array_equal(z[f'{model}_swapped_BA'],z[f'{model}_original_AB']))
            get=lambda k,d:decode(source,by[r['background'],r['site'],k,d]);ad=get('AB',200);bd=get('BA',200);actual=(ad['v'][:,0]-bd['v'][:,0])*1000;single=((get('A0',0)['v'][:,0]-get('Ad',200)['v'][:,0])-(get('B0',0)['v'][:,0]-get('Bd',200)['v'][:,0]))*1000
            # Cross-check the actual two direct edge deliveries against anatomical strengths.
            for d in (ad,bd):
                audit=d['audit'];stim=set(zip(map(int,d['stimulus_ticks']+18),map(int,d['stimulus_indices'])))
                for sourcecell,contact in zip((s['a_cell'],s['b_cell']),s['contacts']):
                    relevant=[x for x in audit if int(x[3])==s['target'] and int(x[1])==sourcecell and (int(x[0]),int(x[1])) in stim]
                    check('direct_strengths_match_real_writes',len(relevant)==2 and all(np.array([x[4]],dtype=np.uint64).view(np.float64)[0]==(.275/52)*contact and np.array([x[5]],dtype=np.uint64).view(np.float64)[0]==0 for x in relevant))
            sign=lambda x:0 if abs(x)<=1e-8 else 1 if x>0 else -1
            for wi,end in enumerate((2038+r['shift'],2338+r['shift'],2838+r['shift'],3838+r['shift'],6000)):
                start=1538+r['shift'];w=slice(start,end);y=actual[w];den=float(np.sqrt(np.mean(y*y)));count=lambda d:int(np.sum((d['indices']==s['target'])&(d['ticks']>=start)&(d['ticks']<end)))
                m={'key':r['key'],'source_index':r['source_index'],'background':r['background'],'site':r['site'],'subtype':r['subtype'],'window_index':wi,'window_ticks':[start,end],'actual_mean_mv':float(y.mean()),'actual_integral_mv_ms':float(y.sum()*.1),'actual_rms_mv':den,'actual_sign':sign(y.mean()),'actual_spikes_AB_BA':[count(ad),count(bd)],'models':{}}
                for model in ('linear','local','single_superposition'):
                    pred=single if model=='single_superposition' else z[f'{model}_original_AB']-z[f'{model}_original_BA'];pp=pred[w];entry={'mean_mv':float(pp.mean()),'integral_mv_ms':float(pp.sum()*.1),'sign':sign(pp.mean()),'relative_RMS_error':float(np.sqrt(np.mean((pp-y)**2)))/den if den>1e-8 else None}
                    if model!='single_superposition':
                        for variant in ('balanced','swapped'):
                            u=(z[f'{model}_{variant}_AB']-z[f'{model}_{variant}_BA'])[w];entry[variant+'_mean_mv']=float(u.mean());entry[variant+'_peak_abs_mv']=float(abs(u).max())
                        if model=='local':entry['spikes_AB_BA']=[int(np.sum((z[f'local_original_{o}_spikes']>=start)&(z[f'local_original_{o}_spikes']<end))) for o in ('AB','BA')]
                    m['models'][model]=entry
                recomputed.append(m)
    check('all_240_metrics_recomputed',len(metrics)==len(recomputed)==240 and same(metrics,recomputed))
    expected_observed=[{'source':r['source'],'key':sourceby[r['source']][r['background'],r['site'],k,d]['key'],'sha256':sourceby[r['source']][r['background'],r['site'],k,d]['archive_sha256']} for r in rows for k,d in [('AB',200),('BA',200),('A0',0),('Bd',200),('B0',0),('Ad',200)]]
    check('all_observed_sources_declared',read(root/'observed-rows.json')==expected_observed)
    check('zero_new_native_runs',read(root/'run.json')['new_native_runs']==0 and not list(root.rglob('b2-native')))
    result={'all_passed':all(checks.values()),'checks':checks,'max_linear_reconstruction_error_mv':linear_error,'max_local_reconstruction_error_mv':local_error,'local_integrator_decimal_precision':34,'numpy_longdouble_mantissa_bits':int(np.finfo(np.longdouble).nmant),'verifier_sha256':sha(__file__),'metrics_sha256':sha(root/'metrics.json'),'prediction_rows_sha256':sha(root/'prediction-rows.json')};save(root/'verification.json',result);save(root/'independent-metrics.json',recomputed);print(json.dumps(result,indent=2),flush=True)
    if not result['all_passed']:raise ValueError('minimal-model verification failed')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('root',type=Path);verify(a.parse_args().root)
