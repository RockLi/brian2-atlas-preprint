"""P13d minimal-model competition: frozen descriptive reanalysis, zero native runs."""
import argparse
import time
from pathlib import Path
import numpy as np
from .response_diagnostic import read,save,sha,decode
from .minimal_response_models import linear,local,schedule,weights_for,N

SOURCES=[Path('brian2-rust/validation/flywire-vision-independent-repeat-v1'),Path('brian2-rust/validation/flywire-vision-onset-diagnostic-v1')]
VARIANTS=['original','balanced','swapped']

def run(root):
    root.mkdir(parents=True,exist_ok=False);(root/'predictions').mkdir()
    manifests={str(s):sha(s/'file-manifest.json') for s in SOURCES};source_names=['minimal_response_models.py','minimal_response_study.py','test_minimal_response_models.py','response_diagnostic.py']
    p={'stage':'P13d-minimal-response-v1','frozen_at_unix':time.time(),'source_manifests':manifests,'sources':{n:sha(Path(__file__).with_name(n)) for n in source_names},'new_native_runs':0,'sample':'all 32 P13 cases and 16 P13c cases; previously observed development data, not a held-out test; repeated target/background cases are not independent biological replicates','models':{'linear_rest':'two direct anatomical contact strengths, 5ms conductance decay and 20ms membrane decay, linearized at -52mV; no recorded stimulus responses or blank trajectories','local_blank':'one conductance-LIF target, exact frozen dt/threshold/reset/refractory, recorded target blank ge/gi and initial V, plus direct anatomical stimulus conductances; no stimulus-driven recurrent feedback; background traces retain full-network information','single_superposition':'same-time full-network single-input responses; contextual reference, not network-free'},'variants':VARIANTS,'variant_scope':'balanced and swapped strengths are interventions in the minimal models only, retaining total strength; never treated as actual full-network ablations','windows':'start first actual arrival; ends last arrival+20/50/100/200ms and recorded trial end600ms; all prespecified, no best-window selection','primary_window_index':0,'criteria':{'sign_fraction_min':.875,'median_relative_RMS_error_max':.2,'note':'inherited exploratory adequacy thresholds; evaluate each source and all cases separately; neither biological significance nor classification accuracy'},'numerical_floor_mv':1e-8,'limitations':['no independent new full-network trials','linear balanced/swap results are structural properties of the null model','blank conductance replay is open-loop and contains real-network background information','finite 600ms recording is not infinite response integration','weights represent contacts with the frozen model gain, not measured physiological synapse strength']}
    save(root/'protocol.json',p)
    allrows=[];gates=[]
    # Predictions use protocols and blanks only. Paired/single stimulus responses are read below, after this archive is sealed.
    for si,source in enumerate(SOURCES):
        sp=read(source/'protocol.json');rows=read(source/'rows.json');sites=sp['sites'];blanks=[r for r in rows if r['kind']=='blank'];old=read(SOURCES[0]/'protocol.json')
        assert all(sha(source/n)==h for n,h in read(source/'file-manifest.json')['sha256'].items())
        for b in blanks:
            bg=b['background'];sid=b['site'];s=sites[sid];shift=b['shift'];key=f'd{si}-b{bg}-s{sid}';blank=decode(source,b);ge=blank['ge'][:,0];gi=blank['gi'][:,0];initial=blank['v'][0,0]
            vb,sb=local(ge,gi,initial,np.zeros(N));reference_spikes=blank['ticks'][blank['indices']==s['target']];error=float(np.max(abs(vb-blank['v'][:,0]*1000)))
            assert error<1e-9 and np.array_equal(sb,reference_spikes),'blank replay failed'
            gates.append({'key':key,'max_voltage_error_mv':error,'spikes_exact':True,'blank_archive_sha256':b['archive_sha256']})
            arrays={}
            for variant in VARIANTS:
                weights=weights_for(s['contacts'],variant)
                for order in ('AB','BA'):
                    inp=schedule(shift,order,weights);arrays[f'linear_{variant}_{order}']=linear(inp);vv,ss=local(ge,gi,initial,inp);arrays[f'local_{variant}_{order}']=vv;arrays[f'local_{variant}_{order}_spikes']=ss
            np.savez_compressed(root/'predictions'/(key+'.npz'),**arrays)
            allrows.append({'key':key,'source':str(source),'source_index':si,'background':bg,'site':sid,'subtype':s['subtype'],'target':s['target'],'contacts':s['contacts'],'shift':shift,'archive_sha256':sha(root/'predictions'/(key+'.npz')),'blank_key':b['key'],'blank_archive_sha256':b['archive_sha256']})
    save(root/'prediction-rows.json',allrows);save(root/'blank-gates.json',gates);save(root/'prediction-seal.json',{'time_unix':time.time(),'prediction_rows_sha256':sha(root/'prediction-rows.json'),'protocol_sha256':sha(root/'protocol.json'),'all_predictions_before_loading_stimulated_responses':True})
    metrics=[];observed_rows=[]
    for r in allrows:
        source=Path(r['source']);by={(x['background'],x['site'],x['kind'],x['delay']):x for x in read(source/'rows.json')};get=lambda k,d:decode(source,by[r['background'],r['site'],k,d]);a=get('AB',200);b=get('BA',200);actual=(a['v'][:,0]-b['v'][:,0])*1000;singles=(get('A0',0)['v'][:,0]+get('Bd',200)['v'][:,0]-get('B0',0)['v'][:,0]-get('Ad',200)['v'][:,0])*1000
        observed_rows.extend({'source':str(source),'key':by[r['background'],r['site'],k,d]['key'],'sha256':by[r['background'],r['site'],k,d]['archive_sha256']} for k,d in [('AB',200),('BA',200),('A0',0),('Bd',200),('B0',0),('Ad',200)])
        with np.load(root/'predictions'/(r['key']+'.npz')) as z:
            sign=lambda x:0 if abs(x)<=1e-8 else 1 if x>0 else -1
            for wi,end in enumerate([2038+r['shift'],2338+r['shift'],2838+r['shift'],3838+r['shift'],6000]):
                start=1538+r['shift'];w=slice(start,end);y=actual[w];den=float(np.sqrt(np.mean(y*y)));target=r['target'];count=lambda d:sum(start<=t<end and i==target for t,i in zip(d['ticks'],d['indices']))
                m={'key':r['key'],'source_index':r['source_index'],'background':r['background'],'site':r['site'],'subtype':r['subtype'],'window_index':wi,'window_ticks':[start,end],'actual_mean_mv':float(y.mean()),'actual_integral_mv_ms':float(y.sum()*.1),'actual_rms_mv':den,'actual_sign':sign(y.mean()),'actual_spikes_AB_BA':[int(count(a)),int(count(b))],'models':{}}
                for model in ('linear','local','single_superposition'):
                    pred=singles if model=='single_superposition' else z[f'{model}_original_AB']-z[f'{model}_original_BA'];yy=pred[w];entry={'mean_mv':float(yy.mean()),'integral_mv_ms':float(yy.sum()*.1),'sign':sign(yy.mean()),'relative_RMS_error':float(np.sqrt(np.mean((yy-y)**2)))/den if den>1e-8 else None}
                    if model!='single_superposition':
                        for variant in ('balanced','swapped'):
                            c=(z[f'{model}_{variant}_AB']-z[f'{model}_{variant}_BA'])[w];entry[variant+'_mean_mv']=float(c.mean());entry[variant+'_peak_abs_mv']=float(abs(c).max())
                        if model=='local':entry['spikes_AB_BA']=[int(np.sum((z[f'local_original_{o}_spikes']>=start)&(z[f'local_original_{o}_spikes']<end))) for o in ('AB','BA')]
                    m['models'][model]=entry
                metrics.append(m)
    save(root/'metrics.json',metrics);save(root/'observed-rows.json',observed_rows)
    for source in SOURCES:assert all(sha(source/n)==h for n,h in read(source/'file-manifest.json')['sha256'].items())
    save(root/'run.json',{'status':'complete_pending_verification','cases':len(allrows),'metric_rows':len(metrics),'new_native_runs':0,'prediction_seal_sha256':sha(root/'prediction-seal.json'),'metrics_sha256':sha(root/'metrics.json')});print({'cases':len(allrows),'metrics':len(metrics),'new_native_runs':0},flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('root',type=Path);run(a.parse_args().root)
