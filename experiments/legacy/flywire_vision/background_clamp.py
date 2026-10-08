"""P12b: fixed blank-derived background plus count-balanced stimulus output."""
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
from .event_clamp_engine import Budget,Runner,atomic_json
from .event_clamp_data import load_trial,store_trial,extract,verify_delivery
from .event_clamp_study import SPECS
from .motion_refinement import read,sha,load_npz
from .multispeed_data import digest
from .run_experiment import save

P12=Path('brian2-rust/validation/flywire-vision-event-clamp-v1')


class Budget200(Budget):
    def __init__(self,path):
        if not path.exists():atomic_json(path,{'limit':200,'attempts':[]})
        super().__init__(path)
        if read(path)['limit']!=200:raise ValueError('P12b budget must be 200')
    def reserve(self,key,metadata):
        if len(read(self.path)['attempts'])>=200:raise RuntimeError('P12b 200-run limit reached')
        return super().reserve(key,metadata)


def events_array(pairs):
    pairs=sorted((int(t),int(i)) for t,i in pairs)
    if len(pairs)!=len(set(pairs)):raise ValueError('coincident source events')
    return np.array([i for t,i in pairs],dtype=np.int64),np.array([t for t,i in pairs],dtype=np.int64)


def safe_union(background,stimulus,refractory=22):
    result=events_array([*zip(background[1],background[0]),*zip(stimulus[1],stimulus[0])])
    for source in np.unique(result[0]):
        if np.any(np.diff(result[1][result[0]==source])<refractory):raise ValueError('source event gap below refractory constraint')
    return result


def shift_for_background(background,pair):
    # One offset per probe/background, shared by both sources and every order/control.
    # All four possible per-source stimulus ticks are considered before responses.
    for shift in range(101):
        try:
            safe_union(background,events_array((t+shift,source) for source in pair for t in (1520,1620,1720,1820)))
            return shift
        except ValueError:pass
    raise ValueError('cannot retain background and balanced stimulation within 10 ms common offset')


def stimulus(site,kind,delay,shift):
    parts={'blank':[],'A0':[('a',0)],'B0':[('b',0)],'Ad':[('a',delay)],'Bd':[('b',delay)],'AB':[('a',0),('b',delay)],'BA':[('b',0),('a',delay)]}[kind]
    visual=events_array((1500+shift+offset+t,site[part][0]) for part,offset in parts for t in (20,120))
    outgoing=events_array((1500+shift+offset+t,site[part+'_cell']) for part,offset in parts for t in (20,120))
    return visual,outgoing


def prepare(root):
    root.mkdir(parents=True,exist_ok=False);Budget200(root/'budget.json');(root/'trials').mkdir();(root/'backgrounds').mkdir();(root/'executor').mkdir()
    p=read(P12/'formal-v1/protocol.json');oldrows=read(P12/'formal-v1/rows.json');backgrounds=[]
    for bg in (783,784):
        for s in p['sites']:
            old=next(r for r in oldrows if r['background']==bg and r['condition']=='ordinary' and r['site']==s['id'] and r['kind']=='blank');data=load_trial(P12/'formal-v1/trials',old);pair=[s['a_cell'],s['b_cell']];mask=np.isin(data['indices'],pair);ev=(data['indices'][mask],data['ticks'][mask]);shift=shift_for_background(ev,pair)
            name=f'b{bg}-s{s["id"]}.npz';np.savez_compressed(root/'backgrounds'/name,indices=ev[0],ticks=ev[1])
            backgrounds.append({'background':bg,'site':s['id'],'key':name,'sha256':sha(root/'backgrounds'/name),'source_row_key':old['key'],'source_archive_sha256':old['archive_sha256'],
                'events':list(map(list,zip(ev[1].tolist(),ev[0].tolist()))),'event_count':len(ev[0]),'stimulus_shift_ticks':shift})
    matrix=[]
    for bg in (783,784):
        for condition in ('without_background','with_background'):
            for s in p['sites']:
                background=next(b for b in backgrounds if b['background']==bg and b['site']==s['id']);shift=background['stimulus_shift_ticks']
                for k,d in SPECS:
                    oldalias=None;gate=None
                    if condition=='without_background' or background['event_count']==0:
                        if shift==0 or k=='blank':oldalias=f'b{bg}-controlled-s{s["id"]}-{k}-{d}'
                    elif k=='blank':gate=f'blank-replay-{bg}-s{s["id"]}'
                    matrix.append({'background':bg,'condition':condition,'site':s['id'],'kind':k,'delay':d,'shift':shift,'key':f'b{bg}-{condition}-s{s["id"]}-{k}-{d}','p12_alias':oldalias,'gate_alias':gate})
    shutil.copytree(P12/'validation-v1/native',root/'executor/native')
    shutil.copy2(P12/'background-784.bin',root/'background-784.bin');shutil.copy2(P12/'outgoing-graph.npz',root/'outgoing-graph.npz')
    save(root/'design.json',matrix)
    # Current inputs permit 11 replay gates, 84 new matrix runs and 3 restores.
    formal_new=sum(r['p12_alias'] is None and r['gate_alias'] is None for r in matrix);gate_new=sum(b['event_count']>0 for b in backgrounds)
    protocol={'frozen_at_unix':time.time(),'schema':'flywire-p12b-background-replay-v1','parent':str(P12),'parent_protocol_sha256':sha(P12/'formal-v1/protocol.json'),'parent_report_sha256':sha(P12/'formal-v1/report.json'),'parent_verification_sha256':sha(P12/'formal-v1/verification.json'),
        'prior_evidence':{str(f):sha(f) for name in ('flywire-vision-timing-probe-v1','flywire-vision-convergent-probe-v1','flywire-vision-event-clamp-v1') for f in (Path('brian2-rust/validation')/name).rglob('*') if f.is_file()},
        'sites':p['sites'],'backgrounds':[783,784],'background_records':backgrounds,'design_sha256':sha(root/'design.json'),
        'conflict_rule':'keep all blank-derived background events exactly; select smallest shared nonnegative integer offset 0..100 ticks such that every source has >=22 ticks between all background and possible early/late stimulus events; apply same offset to both sources, every order/single/simultaneous and both comparison conditions; no background deletion, no per-order shifts; stop if none exists',
        'external_drive':'same shifted external pulses in both conditions; live source neurons remain active, their native output is suppressed by unchanged P12 mode 1',
        'with_background':'exact blank-derived selected-source output for full 600 ms plus balanced stimulus; background is identical across blank/singles/simultaneous/AB/BA, feedback may affect other cells but cannot rewrite selected-source output',
        'without_background':'P12 stimulus-only output replacement; reuse original records only when schedules, selected cells, background generator and immutable engine/base all match; otherwise acquire matching shifted comparator',
        'primary_window_ticks':[1520,2020],'supplementary_windows':'[1538+shift,2038/2338/2838/3838+shift), matched within probe; primary stays identical to P12 absolute window',
        'primary_metric':'signed mean target J=(AB-Aearly-Blate+blank)-(BA-Bearly-Alate+blank); also mean absolute J, peak absolute J, raw order, actual live input/target spikes, ge/gi traces',
        'inference':'all 16 pairs and all windows; background sensitivity and sign flips; ratios descriptive, undefined for ordinary signed mean magnitude below 1e-6 mV; no mechanism contribution percentages or population significance claim',
        'support_groups':'all 16 pairs; separately pairs with at least one blank-derived selected-source delivery before tick 2020, defined only by events before any new response; empty or late-only background equality is not independent early support',
        'p13_decision':'recommend independent repetition only as a new stage; assess early residual absolute effect, signs across both backgrounds, sparse background support, dependence on shifted probe and late effects; numerical nonzero alone is insufficient; no P13 run in this goal',
        'planned_formal_records':len(matrix),'planned_new_formal':formal_new,'planned_new_gates':gate_new,'planned_postchecks':3,'planned_native_total':formal_new+gate_new+3,'hard_limit':200,
        'postchecks':[[783,'with_background',0],[784,'with_background',7],[783,'without_background',0]],
        'sources':{n:sha(Path(__file__).with_name(n)) for n in ('background_clamp.py','event_clamp_engine.py','event_clamp_data.py','event_clamp_study.py')},
        'native_identity':read(root/'executor/native/identity.json'),'background_784_sha256':sha(root/'background-784.bin'),'unit_test_sha256':sha(Path(__file__).with_name('test_background_clamp.py')),'graph_sha256':sha(root/'outgoing-graph.npz')}
    save(root/'protocol.json',protocol);atomic_json(root/'checkpoint.json',{'stage':'protocol_frozen_before_any_new_response','attempts':0,'planned_native_total':protocol['planned_native_total']})
    print({'frozen':True,'new_gates':gate_new,'new_formal':formal_new,'postchecks':3,'planned_total':protocol['planned_native_total'],'shifts':[(b['background'],b['site'],b['stimulus_shift_ticks']) for b in backgrounds if b['stimulus_shift_ticks']]},flush=True)


def metric_records(root):
    p=read(root/'protocol.json');rows=read(root/'rows.json');by={(r['background'],r['condition'],r['site'],r['kind'],r['delay']):r for r in rows};result=[]
    for bg in (783,784):
        for condition in ('without_background','with_background'):
            for s in p['sites']:
                sid=s['id'];shift=next(b['stimulus_shift_ticks'] for b in p['background_records'] if b['background']==bg and b['site']==sid);data={(k,d):load_trial(root/'trials',by[bg,condition,sid,k,d]) for k,d in SPECS}
                windows=[(1520,2020)]+[(1538+shift,end+shift) for end in (2038,2338,2838,3838)]
                for ordinal,(start,end) in enumerate(windows):
                    v=lambda k,d:data[k,d]['v'][start:end,0]*1000;ab=v('AB',200);ba=v('BA',200);z=v('blank',0);j=ab-ba-v('A0',0)-v('Bd',200)+v('B0',0)+v('Ad',200)
                    cnt=lambda k,d,cell:int(np.sum((data[k,d]['indices']==cell)&(data[k,d]['ticks']>=start)&(data[k,d]['ticks']<end)))
                    counts={k+str(d):cnt(k,d,s['target']) for k,d in SPECS}
                    result.append({'background':bg,'condition':condition,'site':sid,'subtype':s['subtype'],'shift_ticks':shift,'window_index':ordinal,'window_ticks':[start,end],
                        'voltage_order_mean_mv':float(np.mean(ab-ba)),'voltage_order_peak_abs_mv':float(np.max(abs(ab-ba))),'voltage_interaction_mean_mv':float(np.mean(j)),
                        'voltage_interaction_mean_abs_mv':float(np.mean(abs(j))),'voltage_interaction_peak_abs_mv':float(np.max(abs(j))),
                        'voltage_response_peak_abs_mv':float(max(np.max(abs(ab-z)),np.max(abs(ba-z)))),'target_spikes':counts,
                        'input_spikes':{k:{part:cnt(k,200,s[part+'_cell']) for part in ('a','b')} for k in ('AB','BA')},
                        'spike_interaction':counts['AB200']-counts['BA200']-counts['A00']-counts['Bd200']+counts['B00']+counts['Ad200']})
    return result


def run(root):
    p=read(root/'protocol.json');matrix=read(root/'design.json')
    if (root/'rows.json').exists() or read(root/'budget.json')['attempts']:raise ValueError('already started; inspect live process/checkpoint rather than restart')
    for n,h in p['sources'].items():
        if sha(Path(__file__).with_name(n))!=h:raise ValueError('frozen source changed')
    oldp=read(P12/'formal-v1/protocol.json');p11=Path('brian2-rust/validation/flywire-vision-convergent-probe-v1');p11p=read(p11/'protocol.json')
    runner=Runner(root/'executor',p11,Path(p11p['artifact']),Path(p11p['parent']),Budget200(root/'budget.json'));graph=load_npz(root/'outgoing-graph.npz');oldrows={r['key']:r for r in read(P12/'formal-v1/rows.json')};backgrounds={}
    for b in p['background_records']:
        data=load_npz(root/'backgrounds'/b['key']);backgrounds[b['background'],b['site']]=(data['indices'],data['ticks'])
    empty=(np.array([],dtype=np.int64),np.array([],dtype=np.int64));gates=[];gate_data={};(root/'gates').mkdir();(root/'gates/trials').mkdir()
    def execute(key,bg,s,visual,events):
        pair=sorted([s['a_cell'],s['b_cell']]);result,audit,summary,work=runner.run(key,pair,visual,events,1,root/'background-784.bin' if bg==784 else None)
        check=verify_delivery(audit,events,pair,1,graph);data=extract(result,audit,[s['target'],s['a_cell'],s['b_cell']],runner.cells,visual,events);del result
        return data,{**summary,'delivery_check':check},work
    # Every nonempty pair gets a new blank replay gate before any stimulated response.
    for b in p['background_records']:
        if b['event_count']==0:continue
        s=p['sites'][b['site']];bg=b['background'];key=f'blank-replay-{bg}-s{s["id"]}';data,summary,work=execute(key,bg,s,empty,backgrounds[bg,s['id']]);old=oldrows[b['source_row_key']];od=load_trial(P12/'formal-v1/trials',old)
        exact=all(summary[k]==old[k] for k in ('events_sha256','states_sha256','trace_sha256')) and all(np.array_equal(data[k],od[k]) for k in ('v','ge','gi','indices','ticks','audit'))
        stored=store_trial(root/'gates/trials',key,data);gates.append({**summary,**stored,'background':bg,'site':s['id'],'reference_key':old['key'],'exact':exact});save(root/'gates/rows.json',gates);gate_data[key]=data
        shutil.rmtree(work/'result');assert exact,'fixed blank background replay changed baseline'
    save(root/'gates/verification.json',{'all_passed':all(r['exact'] for r in gates),'new_gate_count':len(gates),'rows_sha256':sha(root/'gates/rows.json')})
    rows=[];blanks={}
    for row in matrix:
        s=p['sites'][row['site']];bg=row['background'];visual,stim=stimulus(s,row['kind'],row['delay'],row['shift']);base=backgrounds[bg,s['id']] if row['condition']=='with_background' else empty;events=safe_union(base,stim);work=None
        if row['p12_alias']:
            old=oldrows[row['p12_alias']];data=load_trial(P12/'formal-v1/trials',old)
            assert np.array_equal(data['outgoing_indices'],events[0]) and np.array_equal(data['outgoing_ticks'],events[1]) and np.array_equal(data['visual_indices'],visual[0]) and np.array_equal(data['visual_ticks'],visual[1])
            summary={k:old[k] for k in ('seconds','events_sha256','states_sha256','trace_sha256')};summary['p12_attempt']=old['attempt'];summary['attempt']=None
        elif row['gate_alias']:
            data=gate_data[row['gate_alias']];old=next(g for g in gates if g['key']==row['gate_alias']);summary={k:old[k] for k in ('attempt','seconds','events_sha256','states_sha256','trace_sha256')}
        else:data,summary,work=execute(row['key'],bg,s,visual,events)
        # Store background/stimulus provenance separately from merged real outgoing events.
        data={**data,'background_indices':base[0].copy(),'background_ticks':base[1].copy(),'stimulus_indices':stim[0].copy(),'stimulus_ticks':stim[1].copy()}
        check=verify_delivery(data['audit'],events,sorted([s['a_cell'],s['b_cell']]),1,graph)
        group=(bg,row['condition'],s['id']);blank_key=f'b{bg}-{row["condition"]}-s{s["id"]}-blank-0'
        if row['kind']=='blank':blanks[group]=data;blank_key=None;blank=None
        else:blank=blanks[group]
        stored=store_trial(root/'trials',row['key'],data,blank_key,blank);rows.append({**row,**summary,**stored,'delivery_check':check});save(root/'rows.json',rows)
        if work:shutil.rmtree(work/'result')
        atomic_json(root/'checkpoint.json',{'stage':'formal_running','records':len(rows),'attempts':len(read(root/'budget.json')['attempts'])})
        if len(rows)%16==0:print({'records':len(rows),'native_attempts':len(read(root/'budget.json')['attempts'])},flush=True)
    post=[]
    for bg,condition,sid in p['postchecks']:
        ref=next(r for r in rows if r['background']==bg and r['condition']==condition and r['site']==sid and r['kind']=='AB' and r['delay']==200);old=load_trial(root/'trials',ref);s=p['sites'][sid];key=f'restore-{bg}-{condition}-s{sid}'
        data,summary,work=execute(key,bg,s,(old['visual_indices'],old['visual_ticks']),(old['outgoing_indices'],old['outgoing_ticks']))
        exact=all(summary[k]==ref[k] for k in ('events_sha256','states_sha256','trace_sha256')) and all(np.array_equal(v,old[k]) for k,v in data.items());stored=store_trial(root/'trials',key,data)
        post.append({**summary,**stored,'reference_key':ref['key'],'exact':exact});save(root/'postchecks.json',post);shutil.rmtree(work/'result');assert exact,'restore changed'
    report={'status':'complete','native_attempts':len(read(root/'budget.json')['attempts']),'records':len(rows),'protocol_sha256':sha(root/'protocol.json'),'rows_sha256':sha(root/'rows.json'),'postchecks_sha256':sha(root/'postchecks.json'),'per_site':metric_records(root)}
    save(root/'report.json',report);atomic_json(root/'checkpoint.json',{'stage':'complete_pending_independent_verification','records':len(rows),'attempts':report['native_attempts']});print({k:v for k,v in report.items() if k!='per_site'},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--prepare',action='store_true');a=ap.parse_args();prepare(a.root) if a.prepare else run(a.root)
