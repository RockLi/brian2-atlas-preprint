"""P13c: frozen single-factor +50ms onset diagnostic using unchanged P13 native."""
import argparse
import os
import shutil
import time
from pathlib import Path
import numpy as np
from .response_diagnostic import P13, SPECS, L_COEF, read,save,sha,decode,analyze
from .event_clamp_engine import Runner,Budget,atomic_json
from .event_clamp_data import extract,store_trial,verify_delivery,load_trial
from .background_clamp import stimulus,safe_union,events_array
from .repeat_design import subset
from .multispeed_data import digest

BACKGROUNDS=[785,787]
DELTA=500


def prepare(root,offline):
    root.mkdir(parents=True,exist_ok=False)
    for name in ('trials','executor/native','offline'): (root/name).mkdir(parents=True)
    p=read(P13/'protocol.json');oldrows=read(P13/'rows.json');bgrows=read(P13/'background-records.json')
    for n in ('main.rs','b2-native','identity.json'):
        os.link(P13/'executor/native'/n,root/'executor/native'/n)
    for f in offline.iterdir():
        if f.is_file():shutil.copy2(f,root/'offline'/f.name)
    design=[]
    for bg in BACKGROUNDS:
        for s in p['sites']:
            record=next(b for b in bgrows if b['background']==bg and b['site']==s['id']);base=decode(P13,next(r for r in oldrows if r['background']==bg and r['site']==s['id'] and r['kind']=='blank'));background=(base['background_indices'],base['background_ticks'])
            possible=events_array(sorted((1520+record['shift']+DELTA+t,c) for t in (0,100,200,300) for c in (s['a_cell'],s['b_cell'])))
            safe_union(background,possible)
            for kind,delay in SPECS:design.append({'background':bg,'site':s['id'],'kind':kind,'delay':delay,'shift':record['shift']+DELTA,'key':f'b{bg}-s{s["id"]}-{kind}-{delay}'})
    names=['onset_diagnostic.py','response_diagnostic.py','test_response_diagnostic.py','event_clamp_engine.py','event_clamp_data.py','background_clamp.py','repeat_design.py']
    protocol={'stage':'P13c-onset-diagnostic-v1','frozen_at_unix':time.time(),'source':str(P13),'P13_protocol_sha256':sha(P13/'protocol.json'),'P13_manifest_sha256':sha(P13/'file-manifest.json'),'offline_summary':read(offline/'summary.json'),'backgrounds':BACKGROUNDS,'background_selection':'785 and 787 deliberately contrast high/low prior average J; post-hoc development selection, not unseen-background validation','sites':p['sites'],'monitor_cells':p['monitor_cells'],'delta_ticks':DELTA,'duration_ms':600,'only_intervention':'shift all visual and controlled source stimulus events +50ms from corresponding P13 condition; preserve all fixed background output events and full-network background generators, constants, weights, sources and target indices','conflict_policy':'all 16 pairs prechecked at exactly +500ticks; stop on conflict, no additional shifting/deletion','primary_window':'first actual stimulus arrival through 20ms after last arrival: [1538+shift,2038+shift); old reference same relative window','prediction':'L=(Aearly+Blate)-(Bearly+Alate), formed from new single-stimulus trials before any paired-stimulus run for that case; no fit, no new label/window selection','criteria':{'sign_prediction_min_cases':14,'total_cases':16,'median_waveform_relative_RMS_error_max':0.2,'old_to_new_raw_sign_min_cases':14,'note':'predeclared exploratory engineering adequacy thresholds, not biological significance or a statistically powered confirmation; assess each criterion separately; no topology stage automatically'},'planned_native':{'identity_gates':2,'single_and_paired_stimuli':112,'restorations':2,'total':116},'hard_limit':128,'restorations':[[785,0,'AB'],[787,7,'BA']],'sources':{n:sha(Path(__file__).with_name(n)) for n in names},'design_sha256':None,'limitations':['same 8 development targets, 2 previously observed backgrounds, 1 shifted onset','background output replay is open-loop','onset changes background phase and elapsed network state together; does not isolate one conductance or feedback mechanism','inherited P13 T4d/785 +3ms remains in both onsets','linear prediction uses contemporaneous single-input calibrations; not a blind classifier from pixels','nonlinear interaction is neither necessary nor sufficient for useful direction response']}
    save(root/'design.json',design);protocol['design_sha256']=sha(root/'design.json');save(root/'protocol.json',protocol);atomic_json(root/'budget.json',{'limit':128,'attempts':[]});atomic_json(root/'checkpoint.json',{'stage':'frozen','new_native_runs':0});print({'frozen':True,'planned_native':116,'hard_limit':128},flush=True)


class OnsetRunner(Runner):
    def __init__(self,root,p):
        parent=read(P13/'protocol.json');super().__init__(root/'executor',P13,Path(parent['artifact']),Path(parent['parent']),Budget(root/'budget.json'))


def run(root):
    p=read(root/'protocol.json');assert not read(root/'budget.json')['attempts'],'existing ledger: do not restart'
    assert all(sha(Path(__file__).with_name(n))==h for n,h in p['sources'].items())
    runner=OnsetRunner(root,p);rows=[];gates=[];predictions=[];oldrows=read(P13/'rows.json');graph=np.load(P13/'outgoing-graph.npz');reference={(r['background'],r['site'],r['kind'],r['delay']):r for r in oldrows}
    def execute(key,s,bg,visual,outgoing,cells):
        pair=sorted([s['a_cell'],s['b_cell']]);result,audit,summary,work=runner.run(key,pair,visual,outgoing,1,P13/'backgrounds'/f'generator-{bg}.bin')
        checked=verify_delivery(audit,outgoing,pair,1,graph);data=extract(result,audit,cells,runner.cells,visual,outgoing);del result
        return data,{**summary,'delivery_check':checked,'background':bg,'site':s['id'],'selected':pair,'mode':1},work
    # Re-execute an old actual stimulated case in each selected background, before new trials.
    for bg in BACKGROUNDS:
        s=p['sites'][0];ref=reference[bg,0,'AB',200];d=decode(P13,ref);key=f'identity-{bg}';got,summary,work=execute(key,s,bg,(d['visual_indices'],d['visual_ticks']),(d['outgoing_indices'],d['outgoing_ticks']),d['cells'].tolist());exact=all(summary[k]==ref[k] for k in ('events_sha256','states_sha256','trace_sha256')) and all(np.array_equal(v,d[k]) for k,v in got.items());gates.append({**summary,**store_trial(root/'trials',key,got),'reference_key':ref['key'],'exact':exact});save(root/'gates.json',gates);shutil.rmtree(work/'result');assert exact,'identity gate failed'
    case_data={}
    for row in read(root/'design.json'):
        bg=row['background'];sid=row['site'];s=p['sites'][sid];cells=[s['target'],s['a_cell'],s['b_cell']];refblank=reference[bg,sid,'blank',0];blank=decode(P13,refblank);background=(blank['background_indices'],blank['background_ticks']);visual,stim=stimulus(s,row['kind'],row['delay'],row['shift']);outgoing=safe_union(background,stim);blankkey=f'b{bg}-s{sid}-blank-0';work=None
        if row['kind']=='blank':
            data={k:v.copy() for k,v in blank.items()};summary={k:refblank[k] for k in ('events_sha256','states_sha256','trace_sha256','audit_sha256','selected','mode')};summary.update(attempt=None,reference_blank=refblank['key']);stored=store_trial(root/'trials',row['key'],data)
        else:
            data,summary,work=execute(row['key'],s,bg,visual,outgoing,cells);data.update(background_indices=background[0],background_ticks=background[1],stimulus_indices=stim[0],stimulus_ticks=stim[1]);stored=store_trial(root/'trials',row['key'],data,blankkey,blank)
        rows.append({**row,**summary,**stored});save(root/'rows.json',rows);case_data[row['kind'],row['delay']]=data
        if work:shutil.rmtree(work/'result')
        if row['kind']=='Bd':
            a=1538+row['shift'];b=2038+row['shift'];linear=sum(n*case_data[k]['v'][a:b,0]*1000 for k,n in L_COEF.items());mean=float(linear.mean());predictions.append({'background':bg,'site':sid,'time_unix':time.time(),'before_paired_runs':True,'linear_mean_mv':mean,'linear_sign':0 if abs(mean)<=1e-8 else 1 if mean>0 else -1,'linear_waveform_sha256':digest(linear),'single_archive_hashes':{r['key']:r['archive_sha256'] for r in rows if r['background']==bg and r['site']==sid and r['kind'] in ('A0','B0','Ad','Bd')}});save(root/'predictions.json',predictions)
        if row['kind']=='BA':case_data={}
        atomic_json(root/'checkpoint.json',{'stage':'running','records':len(rows),'attempts':len(read(root/'budget.json')['attempts'])})
        if len(rows)%16==0:print({'records':len(rows),'attempts':len(read(root/'budget.json')['attempts'])},flush=True)
    post=[]
    for bg,sid,kind in p['restorations']:
        row=next(r for r in rows if (r['background'],r['site'],r['kind'],r['delay'])==(bg,sid,kind,200));d=decode(root,row);s=p['sites'][sid];key=f'restore-{bg}-s{sid}-{kind}';got,summary,work=execute(key,s,bg,(d['visual_indices'],d['visual_ticks']),(d['outgoing_indices'],d['outgoing_ticks']),d['cells'].tolist());exact=all(summary[k]==row[k] for k in ('events_sha256','states_sha256','trace_sha256')) and all(np.array_equal(v,d[k]) for k,v in got.items());post.append({**summary,**store_trial(root/'trials',key,got),'reference_key':row['key'],'exact':exact});save(root/'restorations.json',post);shutil.rmtree(work/'result');assert exact,'restoration failed'
    atomic_json(root/'checkpoint.json',{'stage':'complete_pending_verification','records':len(rows),'attempts':len(read(root/'budget.json')['attempts'])});print({'complete':True,'attempts':len(read(root/'budget.json')['attempts'])},flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('root',type=Path);a.add_argument('--prepare',type=Path);args=a.parse_args();prepare(args.root,args.prepare) if args.prepare else run(args.root)
