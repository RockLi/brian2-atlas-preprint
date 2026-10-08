"""Pre-formal P12 full-graph gates. Every native launch uses the shared ledger."""
import argparse
from pathlib import Path
import shutil
import numpy as np
from .event_clamp_engine import Budget,Runner,write_spikes,atomic_json
from .event_clamp_data import outgoing_graph,verify_delivery,extract,store_trial
from .motion_refinement import read,sha,load_npz
from .run_experiment import save
from .timing_probe import schedule
from .convergent_archive import decode
from flywire_mnist.encoding import keyed_rng,schedule as bg_schedule


def gates(root):
    p11=Path('brian2-rust/validation/flywire-vision-convergent-probe-v1');p10=Path('brian2-rust/validation/flywire-vision-timing-probe-v1')
    p=read(p11/'protocol.json');artifact=Path(p['artifact']);parent=Path(p['parent']);v=root/'validation-v1'
    if (v/'protocol.json').exists():raise ValueError('gates already started; inspect checkpoint, do not restart')
    sites=p['sites'];selected=sorted({s[k] for s in sites for k in ('a_cell','b_cell')});allcells=p['monitor_cells'];site=sites[0];pair=sorted([site['a_cell'],site['b_cell']]);empty=(np.array([],dtype=np.int64),np.array([],dtype=np.int64))
    graph=outgoing_graph(artifact,selected);np.savez_compressed(root/'outgoing-graph.npz',**graph)
    ii,tt=bg_schedule(np.full(512,300.),0,6000,.1,keyed_rng(784,'background'));write_spikes(root/'background-784.bin',ii,tt)
    save(v/'protocol.json',{'stage':'pre-formal implementation gates','sources':{n:sha(Path(__file__).with_name(n)) for n in ('event_clamp_engine.py','event_clamp_data.py','event_clamp_gates.py')},
        'prior_evidence':{name+'/'+f:sha(path/f) for name,path in [('P10',p10),('P11',p11)] for f in ('protocol.json','report.json','verification.json')},
        'sites':sites,'background':'original seed 783; second background event generator seed 784 only, unchanged initial state and background graph','background_sha256':sha(root/'background-784.bin'),
        'planned_gates':['ordinary blank monitor equality','ordinary AB20 monitor equality','native outgoing replay equality','transmission-zero reference blank','empty replacement equals reference blank','ordinary restoration','second-background ordinary AB20','second-background outgoing replay','second-background ordinary blank'],
        'max_full_graph':300,'graph_sha256':str(graph['graph_sha256'])})
    runner=Runner(v,p11,artifact,parent,Budget(root/'budget.json'));out=v/'trials';out.mkdir();rows=[];cached={}
    oldrows=read(p11/'rows.json');oldarchive=read(p11/'traces'/'manifest.json');oldby={r['key']:r for r in oldarchive['rows']}
    def old_data(row):
        ar=oldby[row['key']];return decode(load_npz(p11/'traces'/(row['key']+'.npz')),load_npz(p11/'traces'/ar['blank']))
    def execute(key,sel,visual,events,mode,bg):
        result,audit,summary,work=runner.run(key,sel,visual,events,mode,bg);pop=result['populations'][1]
        emitted=events if mode==1 else (pop['indices'],pop['spike_ticks'])
        verified=verify_delivery(audit,emitted,sel,mode,graph)
        data=extract(result,audit,allcells,allcells,visual,events);record={**summary,**store_trial(out,key,data),'mode':mode,'selected':sel,'delivery_check':verified}
        rows.append(record);save(v/'rows.json',rows);cached[key]=(data,summary)
        del result,pop;shutil.rmtree(work/'result')
        return data,summary
    def compare(new,old,skip=()):
        return new['events_sha256']==old['events_sha256'] and all(h==old['states_sha256'][k] for k,h in new['states_sha256'].items() if k not in skip)
    checks={}
    data,rec=execute('gate00-blank',selected,empty,empty,0,None);old=next(r for r in oldrows if r['condition']=='intact' and r['kind']=='blank');oldraw=old_data(old)
    checks['ordinary_blank_matches_p11']=compare(rec,old) and all(np.array_equal(data[k],oldraw[k]) for k in ('v','ge','gi'));assert checks['ordinary_blank_matches_p11']
    visual=schedule(site,'AB',200);data,rec=execute('gate01-ab',pair,visual,empty,0,None);old=next(r for r in oldrows if r['condition']=='intact' and r['site']==0 and r['kind']=='AB' and r['delay']==200);oldraw=old_data(old)
    checks['ordinary_ab_matches_p11']=compare(rec,old) and all(np.array_equal(data[k],oldraw[k]) for k in ('v','ge','gi'));assert checks['ordinary_ab_matches_p11']
    mask=np.isin(data['indices'],pair);events=(data['indices'][mask],data['ticks'][mask]);replay,rr=execute('gate02-native-replay',pair,visual,events,1,None)
    checks['native_replay_exact']=compare(rr,rec) and all(np.array_equal(replay[k],data[k]) for k in ('v','ge','gi','indices','ticks','audit'));assert checks['native_replay_exact']
    reference,ref=execute('gate03-cut-reference',pair,empty,empty,2,None);blank,br=execute('gate04-empty-replacement',pair,empty,empty,1,None)
    checks['empty_replacement_matches_cut']=compare(br,ref,('transmission',)) and all(np.array_equal(blank[k],reference[k]) for k in ('v','ge','gi','indices','ticks'));assert checks['empty_replacement_matches_cut']
    restored,rs=execute('gate05-restored-ab',pair,visual,empty,0,None)
    checks['restoration_exact']=compare(rs,rec) and all(np.array_equal(restored[k],data[k]) for k in ('v','ge','gi','indices','ticks','audit'));assert checks['restoration_exact']
    bg=root/'background-784.bin';data2,rec2=execute('gate06-second-bg-ab',pair,visual,empty,0,bg);mask=np.isin(data2['indices'],pair)
    replay2,rr2=execute('gate07-second-bg-replay',pair,visual,(data2['indices'][mask],data2['ticks'][mask]),1,bg)
    checks['second_background_replay_exact']=compare(rr2,rec2) and all(np.array_equal(replay2[k],data2[k]) for k in ('v','ge','gi','indices','ticks','audit'));assert checks['second_background_replay_exact']
    execute('gate08-second-bg-blank',selected,empty,empty,0,bg)
    save(v/'verification.json',{'all_passed':all(checks.values()),'checks':checks,'native_attempts':len(read(root/'budget.json')['attempts']),'protocol_sha256':sha(v/'protocol.json'),'rows_sha256':sha(v/'rows.json')})
    atomic_json(root/'checkpoint.json',{'stage':'gates_passed','full_graph_attempts':len(read(root/'budget.json')['attempts']),'formal_started':False})
    print(read(v/'verification.json'),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();gates(a.root)
