"""P13 bounded new-target/new-background repetition; no classifier or parameter search."""
import argparse
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
from .repeat_design import selection,change_monitor_only,subset,SPECS,BACKGROUNDS,CENTERS
from .event_clamp_engine import Budget,Runner,write_spikes,atomic_json
from .event_clamp_data import outgoing_graph,verify_delivery,extract,store_trial,load_trial
from .background_clamp import events_array,safe_union,shift_for_background,stimulus
from .convergent_probe import monitored_model
from .motion_refinement import read,sha,load_npz
from .run_experiment import save
from flywire_mnist.encoding import keyed_rng,schedule as background_schedule

P11=Path('brian2-rust/validation/flywire-vision-convergent-probe-v1')
P12=Path('brian2-rust/validation/flywire-vision-event-clamp-v1')
P12B=Path('brian2-rust/validation/flywire-vision-background-replay-v1')


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    for d in ('trials','gates/trials','baselines/trials','backgrounds','executor/native'): (root/d).mkdir(parents=True)
    Budget(root/'budget.json');old=read(P11/'protocol.json');artifact=Path(old['artifact']);parent=Path(old['parent']);sites,log=selection(artifact,parent,old['sites']);cells=sorted({s[k] for s in sites for k in ('target','a_cell','b_cell')});model,ni=monitored_model(read(artifact/'model.json'),cells);save(root/'monitor-definition.json',model['definition']);save(root/'selection.json',log)
    graph=outgoing_graph(artifact,sorted({s[k] for s in sites for k in ('a_cell','b_cell')}));np.savez_compressed(root/'outgoing-graph.npz',**graph)
    source=change_monitor_only((P12/'validation-v1/native/main.rs').read_text(),old['monitor_cells'],cells);native=root/'executor/native';(native/'main.rs').write_text(source)
    flags=['--edition=2021','-C','opt-level=3','-C','codegen-units=1','-C','panic=abort'];subprocess.run(['rustc',*flags,str(native/'main.rs'),'-o',str(native/'b2-native')],capture_output=True,check=True)
    subprocess.run(['rustc','--edition=2021','--test',str(native/'main.rs'),'-o',str(native/'unit-tests')],capture_output=True,check=True);test=subprocess.run([str(native/'unit-tests')],capture_output=True,text=True,check=True);(native/'unit-tests.log').write_text(test.stdout+test.stderr)
    identity={'source_sha256':sha(native/'main.rs'),'binary_sha256':sha(native/'b2-native'),'parent_source_sha256':sha(P12/'validation-v1/native/main.rs'),'parent_binary_sha256':sha(P12/'validation-v1/native/b2-native'),'change':'only 72 passive monitor cell-index literals; all dynamics and clamp logic byte-identical outside those statements','flags':flags};save(native/'identity.json',identity)
    background_hashes={}
    for seed in BACKGROUNDS:
        ii,tt=background_schedule(np.full(512,300.),0,6000,.1,keyed_rng(seed,'background'));path=root/'backgrounds'/f'generator-{seed}.bin';write_spikes(path,ii,tt);background_hashes[str(seed)]=sha(path)
    source_names=('repeat_design.py','repeat_study.py','test_repeat_study.py','background_clamp.py','event_clamp_engine.py','event_clamp_data.py','convergent_probe.py')
    p={'schema':'flywire-p13-independent-repeat-v1','frozen_at_unix':time.time(),'artifact':str(artifact),'parent':str(parent),'p12b':str(P12B),'sites':sites,'monitor_cells':cells,'backgrounds':list(BACKGROUNDS),
        'selection':'nearest eligible target to fixed cardinal centers per subtype; target >=0.15 from every old P11 target; target and sources exclude all old 24 monitor cells and are distinct across new probes; pair distance 0.02..0.08; maximize minimum positive same-family contacts, then balance/distance/index; no response-based selection; fail rather than relax',
        'centers':CENTERS,'selection_sha256':sha(root/'selection.json'),'monitor_definition_sha256':sha(root/'monitor-definition.json'),'native_identity':identity,'background_generator_sha256':background_hashes,'graph_sha256':sha(root/'outgoing-graph.npz'),
        'intervention':'fixed ordinary blank-derived selected-source outgoing events plus two balanced stimulus events per source; P12 replacement mode 1, native selected output suppressed; full-network background remains enabled',
        'conflicts':'unchanged P12b rule: keep all background events; smallest common offset 0..100 ticks allowing >=22 ticks same-source spacing across all possible stimulus schedules; one offset for both sources/all orders/singles/simultaneous; stop if impossible',
        'primary_window_ticks':[1520,2020],'supplementary_windows':'first stimulus arrival 1538+shift through 2038/2338/2838/3838+shift; same 20 ms inter-packet delay, 10 ms intra-packet gap',
        'primary':'signed mean target voltage order interaction J=(AB-Aearly-Blate+blank)-(BA-Bearly-Alate+blank); report mean absolute and peak J, raw AB-BA, target and live-source spikes, reset-related limitations',
        'repetition_analysis':'background 785 fixes each new target signed J and raw AB-BA reference signs separately; backgrounds 786/787/788 only test those signs; disclose all magnitudes, minimum/maximum amplitude ratio, early background event support and all weak/silent targets; numerical sign floor 1e-8 mV is only a discrimination tolerance, not functional significance',
        'decision':'exploratory new-target repetition, not a biological confirmation set; recommend a bounded topology control only if coherent raw order and J across multiple new subtypes/backgrounds with substantial absolute effects, not merely nonzero J or a higher sign fraction; if effects collapse or remain background-specific, defer expansion and revisit input/dynamics in a separately frozen study; no topology/parameter/visualization work in this run',
        'limitations':['one new target per subtype from the same connectome','four event backgrounds, fixed initial state and background mapping','no additional no-replay matrix: across-seed differences include the entire network background, not solely selected-source background output','no trainable readout or biological effect-size threshold calibrated in this pilot','new cardinal locations and possibly different spatial axes; no direct old/new subtype signed pooling'],
        'planned_native':{'monitor_gate':1,'ordinary_new_background_blanks':4,'pair_replay_blanks':32,'stimulated_formal':224,'restorations':4,'total':265},'planned_formal_records':256,'hard_limit':300,
        'restorations':[[785,0,'AB'],[786,2,'BA'],[787,4,'AB'],[788,6,'BA']],
        'sources':{n:sha(Path(__file__).with_name(n)) for n in source_names},'prior_evidence':{str(f):sha(f) for directory in (Path('brian2-rust/validation/flywire-vision-timing-probe-v1'),P11,P12,P12B) for f in directory.rglob('*') if f.is_file()}}
    save(root/'protocol.json',p);atomic_json(root/'checkpoint.json',{'stage':'protocol_frozen','attempts':0});print({'frozen':True,'sites':[(s['subtype'],s['target'],s['a_cell'],s['b_cell']) for s in sites],'planned_native':265},flush=True)


class RepeatRunner(Runner):
    def __init__(self,root,p):
        super().__init__(root/'executor',P11,Path(p['artifact']),Path(p['parent']),Budget(root/'budget.json'))
        if read(root/'budget.json')['limit']!=300:raise ValueError('unexpected budget')
        self.cells=p['monitor_cells'];self.model=monitored_model(read(Path(p['artifact'])/'model.json'),self.cells)[0]


def metrics(root):
    p=read(root/'protocol.json');rows=read(root/'rows.json');by={(r['background'],r['site'],r['kind'],r['delay']):r for r in rows};result=[]
    for bg in BACKGROUNDS:
        for s in p['sites']:
            sid=s['id'];data={(k,d):load_trial(root/'trials',by[bg,sid,k,d]) for k,d in SPECS};shift=by[bg,sid,'blank',0]['shift']
            for wi,(start,end) in enumerate([(1520,2020)]+[(1538+shift,x+shift) for x in (2038,2338,2838,3838)]):
                v=lambda k,d:data[k,d]['v'][start:end,0]*1000;ab=v('AB',200);ba=v('BA',200);z=v('blank',0);j=ab-ba-v('A0',0)-v('Bd',200)+v('B0',0)+v('Ad',200)
                cnt=lambda k,d,c:int(np.sum((data[k,d]['indices']==c)&(data[k,d]['ticks']>=start)&(data[k,d]['ticks']<end)));counts={k+str(d):cnt(k,d,s['target']) for k,d in SPECS}
                result.append({'background':bg,'site':sid,'subtype':s['subtype'],'shift_ticks':shift,'window_index':wi,'window_ticks':[start,end],'voltage_order_mean_mv':float(np.mean(ab-ba)),'voltage_order_peak_abs_mv':float(np.max(abs(ab-ba))),'voltage_interaction_mean_mv':float(j.mean()),'voltage_interaction_mean_abs_mv':float(abs(j).mean()),'voltage_interaction_peak_abs_mv':float(abs(j).max()),'voltage_response_peak_abs_mv':float(max(abs(ab-z).max(),abs(ba-z).max())),'target_spikes':counts,'input_spikes':{k:{part:cnt(k,200,s[part+'_cell']) for part in ('a','b')} for k in ('AB','BA')},'spike_interaction':counts['AB200']-counts['BA200']-counts['A00']-counts['Bd200']+counts['B00']+counts['Ad200']})
    return result


def run(root):
    p=read(root/'protocol.json')
    if read(root/'budget.json')['attempts']:raise ValueError('already started; inspect existing process and ledger, never restart blindly')
    for n,h in p['sources'].items():
        if sha(Path(__file__).with_name(n))!=h:raise ValueError('frozen source changed')
    runner=RepeatRunner(root,p);graph=load_npz(root/'outgoing-graph.npz');selected=sorted({s[k] for s in p['sites'] for k in ('a_cell','b_cell')});empty=events_array([]);base_rows=[];base_data={};gates=[];gate_data={};rows=[]
    def execute(key,sel,visual,outgoing,mode,bg,cells):
        result,audit,summary,work=runner.run(key,sel,visual,outgoing,mode,root/'backgrounds'/f'generator-{bg}.bin' if bg in BACKGROUNDS else None);pop=result['populations'][1]
        emitted=outgoing if mode==1 else (pop['indices'],pop['spike_ticks']);check=verify_delivery(audit,emitted,sel,mode,graph);data=extract(result,audit,cells,runner.cells,visual,outgoing);del pop,result
        return data,{**summary,'delivery_check':check,'background':bg,'mode':mode,'selected':sel},work
    data,summary,work=execute('monitor-control',selected,empty,empty,0,783,p['monitor_cells']);ref=next(r for r in read(P12/'formal-v1/rows.json') if r['background']==783 and r['condition']=='ordinary' and r['site']==0 and r['kind']=='blank');exact=all(summary[k]==ref[k] for k in ('events_sha256','states_sha256'));save(root/'monitor-control.json',{**summary,**store_trial(root/'baselines/trials','monitor-control',data),'exact':exact,'reference_key':ref['key']});shutil.rmtree(work/'result');assert exact,'passive monitor changed dynamics'
    for bg in BACKGROUNDS:
        key=f'ordinary-blank-{bg}';data,summary,work=execute(key,selected,empty,empty,0,bg,p['monitor_cells']);base_rows.append({**summary,**store_trial(root/'baselines/trials',key,data)});save(root/'baselines/rows.json',base_rows);base_data[bg]=data;shutil.rmtree(work/'result')
    backgrounds=[];matrix=[]
    for bg in BACKGROUNDS:
        for s in p['sites']:
            pair=[s['a_cell'],s['b_cell']];mask=np.isin(base_data[bg]['indices'],pair);ev=(base_data[bg]['indices'][mask],base_data[bg]['ticks'][mask]);shift=shift_for_background(ev,pair);key=f'pair-{bg}-s{s["id"]}';np.savez_compressed(root/'backgrounds'/(key+'.npz'),indices=ev[0],ticks=ev[1]);backgrounds.append({'background':bg,'site':s['id'],'key':key,'sha256':sha(root/'backgrounds'/(key+'.npz')),'shift':shift,'event_count':len(ev[0]),'early_support':bool(np.any(ev[1]+18<2020))})
            for k,d in SPECS:matrix.append({'background':bg,'site':s['id'],'kind':k,'delay':d,'shift':shift,'key':f'b{bg}-s{s["id"]}-{k}-{d}'})
    save(root/'background-records.json',backgrounds);save(root/'design.json',matrix);save(root/'formal-freeze.json',{'time_unix':time.time(),'protocol_sha256':sha(root/'protocol.json'),'background_records_sha256':sha(root/'background-records.json'),'design_sha256':sha(root/'design.json'),'baseline_rows_sha256':sha(root/'baselines/rows.json'),'before_stimulated_runs':True});print({'backgrounds_frozen':True,'shifted_pairs':[(b['background'],b['site'],b['shift']) for b in backgrounds if b['shift']]},flush=True)
    for b in backgrounds:
        bg=b['background'];s=p['sites'][b['site']];pair=sorted([s['a_cell'],s['b_cell']]);ev=load_npz(root/'backgrounds'/(b['key']+'.npz'));outgoing=(ev['indices'],ev['ticks']);key=f'replay-blank-{bg}-s{s["id"]}';cells=[s['target'],s['a_cell'],s['b_cell']];data,summary,work=execute(key,pair,empty,outgoing,1,bg,cells);old=subset(base_data[bg],cells,pair);oldrow=next(r for r in base_rows if r['background']==bg);exact=all(summary[k]==oldrow[k] for k in ('events_sha256','states_sha256','trace_sha256')) and all(np.array_equal(data[k],old[k]) for k in ('v','ge','gi','indices','ticks','audit'))
        gates.append({**summary,**store_trial(root/'gates/trials',key,data),'site':s['id'],'exact':exact,'reference_key':oldrow['key']});gate_data[bg,s['id']]=data;save(root/'gates/rows.json',gates);shutil.rmtree(work/'result');assert exact,'background replay changed blank'
    save(root/'gates/verification.json',{'all_passed':all(r['exact'] for r in gates),'count':len(gates),'rows_sha256':sha(root/'gates/rows.json')});print({'replay_gates_passed':len(gates),'attempts':len(read(root/'budget.json')['attempts'])},flush=True)
    for row in matrix:
        s=p['sites'][row['site']];bg=row['background'];cells=[s['target'],s['a_cell'],s['b_cell']];base=gate_data[bg,s['id']];background=(base['outgoing_indices'],base['outgoing_ticks']);visual,stim=stimulus(s,row['kind'],row['delay'],row['shift']);outgoing=safe_union(background,stim);work=None
        if row['kind']=='blank':
            data={k:v.copy() for k,v in base.items()};g=next(g for g in gates if g['background']==bg and g['site']==s['id']);summary={k:g[k] for k in ('attempt','seconds','events_sha256','states_sha256','trace_sha256','audit_sha256','delivery_check','mode','selected')};alias=g['key'];blank_key=None;blank=None
        else:data,summary,work=execute(row['key'],sorted([s['a_cell'],s['b_cell']]),visual,outgoing,1,bg,cells);alias=None;blank_key=f'b{bg}-s{s["id"]}-blank-0';blank=base
        data.update(background_indices=background[0],background_ticks=background[1],stimulus_indices=stim[0],stimulus_ticks=stim[1]);stored=store_trial(root/'trials',row['key'],data,blank_key,blank);rows.append({**row,**summary,**stored,'gate_alias':alias});save(root/'rows.json',rows)
        if work:shutil.rmtree(work/'result')
        atomic_json(root/'checkpoint.json',{'stage':'formal_running','records':len(rows),'attempts':len(read(root/'budget.json')['attempts'])})
        if len(rows)%16==0:print({'records':len(rows),'attempts':len(read(root/'budget.json')['attempts'])},flush=True)
    post=[]
    for bg,sid,kind in p['restorations']:
        row=next(r for r in rows if r['background']==bg and r['site']==sid and r['kind']==kind and r['delay']==200);ref=load_trial(root/'trials',row);s=p['sites'][sid];key=f'restore-{bg}-s{sid}-{kind}';data,summary,work=execute(key,row['selected'],(ref['visual_indices'],ref['visual_ticks']),(ref['outgoing_indices'],ref['outgoing_ticks']),1,bg,ref['cells'].tolist());exact=all(summary[k]==row[k] for k in ('events_sha256','states_sha256','trace_sha256')) and all(np.array_equal(v,ref[k]) for k,v in data.items());post.append({**summary,**store_trial(root/'trials',key,data),'site':sid,'reference_key':row['key'],'exact':exact});save(root/'postchecks.json',post);shutil.rmtree(work/'result');assert exact,'restoration failed'
    save(root/'report.json',{'status':'complete','native_attempts':len(read(root/'budget.json')['attempts']),'protocol_sha256':sha(root/'protocol.json'),'rows_sha256':sha(root/'rows.json'),'postchecks_sha256':sha(root/'postchecks.json'),'per_site':metrics(root)});atomic_json(root/'checkpoint.json',{'stage':'complete_pending_independent_verification','records':256,'attempts':len(read(root/'budget.json')['attempts'])});print({'complete':True,'attempts':len(read(root/'budget.json')['attempts'])},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--prepare',action='store_true');a=parser.parse_args();prepare(a.root) if a.prepare else run(a.root)
