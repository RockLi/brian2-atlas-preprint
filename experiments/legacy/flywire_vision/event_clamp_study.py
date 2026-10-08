"""Frozen P12 matrix: native drive versus matched outgoing events, two backgrounds."""
import argparse
from pathlib import Path
import shutil
import numpy as np
from .event_clamp_engine import Budget,Runner,atomic_json
from .event_clamp_data import verify_delivery,extract,store_trial,load_trial
from .motion_refinement import read,sha,load_npz
from .multispeed_data import digest
from .timing_probe import schedule
from .run_experiment import save

SPECS=(('blank',0),('A0',0),('B0',0),('Ad',200),('Bd',200),('AB',0),('AB',200),('BA',200))
WINDOWS=((1520,2020),(1538,2038),(1538,2338),(1538,2838),(1538,3838))


def design():
    return [{'background':bg,'condition':c,'site':s,'kind':k,'delay':d,'key':f'b{bg}-{c}-s{s}-{k}-{d}'} for bg in (783,784) for c in ('ordinary','controlled') for s in range(8) for k,d in SPECS]


def gate_alias(row):
    if row['condition']=='ordinary':
        if row['kind']=='blank':return 'gate00-blank' if row['background']==783 else 'gate08-second-bg-blank'
        if row['site']==0 and row['kind']=='AB' and row['delay']==200:return 'gate01-ab' if row['background']==783 else 'gate06-second-bg-ab'
    if row['condition']=='controlled' and row['site']==0 and row['background']==783 and row['kind']=='blank':return 'gate04-empty-replacement'
    return None


def metrics(root):
    p=read(root/'protocol.json');rows=read(root/'rows.json');by={(r['background'],r['condition'],r['site'],r['kind'],r['delay']):r for r in rows};result=[]
    for bg in (783,784):
        for c in ('ordinary','controlled'):
            for site in p['sites']:
                s=site['id'];data={(k,d):load_trial(root/'trials',by[bg,c,s,k,d]) for k,d in SPECS}
                for start,end in WINDOWS:
                    sl=slice(start,end);v=lambda k,d:data[k,d]['v'][sl,0]*1000
                    ab=v('AB',200);ba=v('BA',200);j=ab-ba-v('A0',0)-v('Bd',200)+v('B0',0)+v('Ad',200);blank=v('blank',0)
                    count=lambda k,d,cell:int(np.sum((data[k,d]['indices']==cell)&(data[k,d]['ticks']>=start)&(data[k,d]['ticks']<end)))
                    result.append({'background':bg,'condition':c,'site':s,'subtype':site['subtype'],'window_ticks':[start,end],
                        'voltage_order_mean_mv':float(np.mean(ab-ba)),'voltage_order_peak_abs_mv':float(np.max(abs(ab-ba))),
                        'voltage_interaction_mean_mv':float(np.mean(j)),'voltage_interaction_peak_abs_mv':float(np.max(abs(j))),
                        'voltage_interaction_mean_abs_mv':float(np.mean(abs(j))),
                        'voltage_response_peak_abs_mv':float(max(np.max(abs(ab-blank)),np.max(abs(ba-blank)))),
                        'target_spikes':{k+str(d):count(k,d,site['target']) for k,d in SPECS},
                        'input_spikes':{k:{part:count(k,200,site[part+'_cell']) for part in ('a','b')} for k in ('AB','BA')},
                        'spike_interaction':count('AB',200,site['target'])-count('BA',200,site['target'])-count('A0',0,site['target'])-count('Bd',200,site['target'])+count('B0',0,site['target'])+count('Ad',200,site['target'])})
    return result


def execute(root):
    v=root/'validation-v1';g=read(v/'verification.json')
    if not g['all_passed'] or g['native_attempts']!=9:raise ValueError('pre-formal gates missing')
    p11=Path('brian2-rust/validation/flywire-vision-convergent-probe-v1');old=read(p11/'protocol.json');out=root/'formal-v1';out.mkdir(exist_ok=False);(out/'trials').mkdir()
    matrix=design();gate_rows={r['key']:r for r in read(v/'rows.json')};p={'schema':'flywire-p12-event-clamp-v1','sites':old['sites'],
        'backgrounds':[783,784],'background_783':'original P11 background events','background_784':'only background event train uses keyed_rng(784, background); all mappings, initial states and neural constants unchanged',
        'initial_state_and_mapping_seed':783,'background_784_sha256':sha(root/'background-784.bin'),'graph_archive_sha256':sha(root/'outgoing-graph.npz'),
        'conditions':{'ordinary':'unchanged external input pulses and endogenous outgoing spikes',
            'controlled':'same external drive into living selected cells; replace all outgoing events from only the two selected cells for the entire run with the four prescribed source events; no endogenous outgoing events added; all real outgoing edges, contacts, delay retained'},
        'blank':'separate per background, condition and probe; controlled blank suppresses those two cells outgoing background events; no background-event replay added to primary replacement',
        'source_event_ticks':'packet onset 1500 and +200; each packet offsets 20,120 ticks; native recurrent delay 18 ticks is applied exactly once',
        'primary_window_ticks':[1520,2020],'supplementary_windows_ticks':[list(w) for w in WINDOWS[1:]],
        'window_scope':'primary identical absolute window to P11; first supplement is 20 ms after prescribed last arrival (1838+200); ordinary mode uses same absolute windows, not its last endogenous background spike',
        'primary':'P11 signed target voltage order interaction mean; also raw order, interaction peak and mean absolute magnitude, actual input/target spikes; all sites and backgrounds reported',
        'retention':'per-site paired controlled minus ordinary; descriptive magnitude ratios only when ordinary absolute mean >= 1e-6 mV, otherwise NA; report sign and near-zero denominators; no pooled signed cancellation or success threshold',
        'limits':['background outgoing activity of selected cells is removed in controlled condition','two backgrounds, eight previously seen targets are descriptive mechanism screening, not independent biological replication','two controlled source events need not equal endogenous evoked counts','residual feedback remains; full per-edge injected balance does not balance all recurrent target input','no direction accuracy, topology advantage, weight or time-constant optimization'],
        'formal_records':256,'new_formal_native_runs':237,'gate_alias_records':19,'planned_postchecks':4,'planned_total_native_runs':250,'hard_total_limit':300,
        'sources':{n:sha(Path(__file__).with_name(n)) for n in ('event_clamp_engine.py','event_clamp_data.py','event_clamp_gates.py','event_clamp_study.py','timing_probe.py')},
        'gates_verification_sha256':sha(v/'verification.json'),'native_identity':read(v/'native'/'identity.json'),'parent_p11_protocol_sha256':sha(p11/'protocol.json'),
        'matrix_sha256':None,'output':'lossless full 6000-tick v/ge/gi for each probe target and its two stimulated neurons; their actual spikes; prescribed schedules; actual per-edge state-write log; all-network event/state/24-cell monitor hashes'}
    save(out/'design.json',matrix);p['matrix_sha256']=sha(out/'design.json');save(out/'protocol.json',p)
    atomic_json(root/'checkpoint.json',{'stage':'formal_protocol_frozen','formal_started':False,'full_graph_attempts':len(read(root/'budget.json')['attempts']),'protocol_sha256':sha(out/'protocol.json')})
    runner=Runner(v,p11,Path(old['artifact']),Path(old['parent']),Budget(root/'budget.json'));graph=load_npz(root/'outgoing-graph.npz');rows=[];empty=(np.array([],dtype=np.int64),np.array([],dtype=np.int64));blanks={}
    for row in matrix:
        s=p['sites'][row['site']];pair=sorted([s['a_cell'],s['b_cell']]);cells=[s['target'],s['a_cell'],s['b_cell']];visual=schedule(s,row['kind'],row['delay']);mode=int(row['condition']=='controlled')
        events=(np.array([{s['a'][0]:s['a_cell'],s['b'][0]:s['b_cell']}[int(i)] for i in visual[0]],dtype=np.int64),visual[1].copy()) if mode else empty
        # Source ids have a different order to channel ids. Native schedule is tick/source sorted.
        if mode:
            order=np.lexsort((events[0],events[1]));events=(events[0][order],events[1][order])
        alias=gate_alias(row)
        if alias:
            ar=gate_rows[alias];gd=load_trial(v/'trials',ar);columns=[list(gd['cells']).index(c) for c in cells];keep=np.isin(gd['indices'],cells);audit=gd['audit'][np.isin(gd['audit'][:,1],pair)]
            data={**{k:gd[k][:,columns].copy() for k in ('v','ge','gi')},'cells':np.array(cells,dtype=np.int64),'indices':gd['indices'][keep],'ticks':gd['ticks'][keep],
                'visual_indices':visual[0],'visual_ticks':visual[1],'outgoing_indices':events[0],'outgoing_ticks':events[1],'audit':audit.copy()}
            summary={k:ar[k] for k in ('attempt','seconds','events_sha256','states_sha256','trace_sha256')};work=None
        else:
            result,audit,summary,work=runner.run(row['key'],pair,visual,events,mode,root/'background-784.bin' if row['background']==784 else None)
            data=extract(result,audit,cells,runner.cells,visual,events);del result
        emitted=events if mode else (data['indices'],data['ticks']);check=verify_delivery(audit,emitted,pair,mode,graph)
        group=(row['background'],row['condition'],row['site']);blank_key=f"b{row['background']}-{row['condition']}-s{row['site']}-blank-0"
        if row['kind']=='blank':blank=None;blanks[group]=data;blank_key=None
        else:blank=blanks[group]
        stored=store_trial(out/'trials',row['key'],data,blank_key,blank);rows.append({**row,**summary,**stored,'gate_alias':alias,'delivery_check':check});save(out/'rows.json',rows)
        if work is not None:shutil.rmtree(work/'result')
        atomic_json(root/'checkpoint.json',{'stage':'formal_running','formal_started':True,'formal_records':len(rows),'full_graph_attempts':len(read(root/'budget.json')['attempts']),'last_key':row['key']})
        if len(rows)%16==0:print({'formal_records':len(rows),'native_total':len(read(root/'budget.json')['attempts'])},flush=True)
    post=[]
    for bg in (783,784):
        for c in ('ordinary','controlled'):
            ref=next(r for r in rows if r['background']==bg and r['condition']==c and r['site']==0 and r['kind']=='AB' and r['delay']==200);data=load_trial(out/'trials',ref);s=p['sites'][0];pair=sorted([s['a_cell'],s['b_cell']])
            vi=(data['visual_indices'],data['visual_ticks']);ev=(data['outgoing_indices'],data['outgoing_ticks']);mode=int(c=='controlled');key=f'post-{bg}-{c}'
            result,audit,summary,work=runner.run(key,pair,vi,ev,mode,root/'background-784.bin' if bg==784 else None);new=extract(result,audit,data['cells'].tolist(),runner.cells,vi,ev)
            exact=all(summary[k]==ref[k] for k in ('events_sha256','states_sha256','trace_sha256')) and all(digest(new[k])==digest(data[k]) for k in data)
            stored=store_trial(out/'trials',key,new);post.append({**summary,**stored,'reference_key':ref['key'],'exact':exact});save(out/'postchecks.json',post)
            del result;shutil.rmtree(work/'result');assert exact,'post-restoration changed'
    report={'status':'complete','formal_records':len(rows),'new_formal_native_runs':sum(r['gate_alias'] is None for r in rows),'cumulative_native_runs':len(read(root/'budget.json')['attempts']),
        'protocol_sha256':sha(out/'protocol.json'),'rows_sha256':sha(out/'rows.json'),'postchecks_sha256':sha(out/'postchecks.json'),'per_site':metrics(out)}
    save(out/'report.json',report);atomic_json(root/'checkpoint.json',{'stage':'formal_complete_pending_independent_verification','formal_records':len(rows),'full_graph_attempts':len(read(root/'budget.json')['attempts'])});print({k:v for k,v in report.items() if k!='per_site'},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();execute(a.root)
