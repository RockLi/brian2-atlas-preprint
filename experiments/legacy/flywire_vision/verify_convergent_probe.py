"""Independent numerical and provenance checks for the convergent probe."""
import argparse
import io
import hashlib
from pathlib import Path
import struct
import numpy as np
from .motion_refinement import read, sha, load_npz
from .run_experiment import save
from .convergent_probe import select_sites


def verify(root,archive=None):
    hashes={}
    if archive is not None:
        from .convergent_archive import decode
        archived={r['key']:r for r in read(archive/'manifest.json')['rows']}
        blanks={name:load_npz(archive/name) for name in {r['blank'] for r in archived.values()}}
    def fetch(key):
        if archive is None:
            path=root/'trials'/(key+'.npz');hashes[key]=sha(path);return load_npz(path)
        row=archived[key];path=archive/(key+'.npz');assert sha(path)==row['encoded_sha256']
        data=decode(load_npz(path),blanks[row['blank']]);buffer=io.BytesIO();np.savez_compressed(buffer,**data)
        hashes[key]=hashlib.sha256(buffer.getvalue()).hexdigest();assert hashes[key]==row['original_npz_sha256'];return data
    p=read(root/'protocol.json');r=read(root/'report.json');rows=read(root/'rows.json');artifact=Path(p['artifact']);parent=Path(p['parent']);p10=Path(p['p10'])
    checks={'complete':r['status']=='complete' and r['native_runs']==116 and len(rows)==114,
        'hashes':r['protocol_sha256']==sha(root/'protocol.json') and r['rows_sha256']==sha(root/'rows.json'),
        'frozen_sources':all(sha(Path(__file__).with_name(n))==h for n,h in p['source_sha256'].items()),
        'selection':p['sites']==select_sites(artifact,parent),'design':read(root/'design.json')==[{k:v[k] for k in ('condition','site','kind','delay')} for v in rows],
        'model_source':sha(artifact/'model.json')==p['artifact_model_sha256'] and sha(parent/'protocol.json')==p['parent_protocol_sha256']}
    olddef=read(artifact/'model.json')['definition'];newdef=read(root/'monitor-definition.json')
    for definition in (olddef,newdef):
        definition.pop('schedule')
        for pop in definition['populations']:
            if pop['name']=='flywire_neurons':pop.pop('state_monitors');pop.pop('monitor')
    checks['only_passive_definition_change']=olddef==newdef
    for c in ('intact','cut_input'):
        ident=read(root/c/'identity.json');checks[c+'_identity']=ident['base_sha256']==read(parent/c/'identity.json')['base_sha256']==sha(root/c/'base.bin') and ident['binary_sha256']==sha(root/'native'/'b2-native')
        checks[c+'_predeclared']=(root/'protocol.json').stat().st_mtime_ns<(root/c/'identity.json').stat().st_mtime_ns
    by={};cells=np.array(p['monitor_cells']);checks.update({k:True for k in ('raw_files','schedules','shapes','before_input','target_cut_blank','metrics','lif_equation')});max_error=0.;checked_steps=0
    for row in rows:
        data=fetch(row['key']);by[row['condition'],row['site'],row['kind'],row['delay']]=data
        checks['raw_files'] &= hashes[row['key']]==row['data_sha256']
        s=p['sites'][max(0,row['site'])];d=row['delay'];kind=row['kind']
        parts={'blank':[], 'A0':[('a',0)],'B0':[('b',0)],'Ad':[('a',d)],'Bd':[('b',d)],'AB':[('a',0),('b',d)],'BA':[('b',0),('a',d)]}[kind]
        events=sorted((1500+offset+t,s[patch][0]) for patch,offset in parts for t in (20,120))
        checks['schedules'] &= events==list(zip(data['input_ticks'],data['input_indices']))
        checks['shapes'] &= all(data[k].shape==(6000,len(cells)) and np.isfinite(data[k]).all() for k in ('v','ge','gi')) and np.all(np.isin(data['indices'],cells))
        # At start-of-tick, Euler v update uses that tick's v/ge/gi. Exclude
        # reset ticks and all possible refractory steps following a saved spike.
        active=np.ones((5999,len(cells)),dtype=bool);lookup={int(c):i for i,c in enumerate(cells)}
        for tick,cell in zip(data['ticks'],data['indices']):active[int(tick):min(5999,int(tick)+24),lookup[int(cell)]]=False
        v=data['v'];ge=data['ge'];gi=data['gi'];expected=v[:-1]+.005*(-(v[:-1]+.052)-ge[:-1]*v[:-1]-gi[:-1]*(v[:-1]+.070))
        error=float(np.max(np.abs(v[1:][active]-expected[active])));max_error=max(max_error,error);checked_steps+=int(active.sum());checks['lif_equation'] &= error<1e-13
    target_cols=[p['monitor_cells'].index(s['target']) for s in p['sites']]
    for row in rows:
        data=by[row['condition'],row['site'],row['kind'],row['delay']];blank=by[row['condition'],-1,'blank',0];cutoff=int(data['input_ticks'][0]) if len(data['input_ticks']) else 6000
        checks['before_input'] &= all(np.array_equal(data[k][:cutoff],blank[k][:cutoff]) for k in ('v','ge','gi'))
        if row['condition']=='cut_input':checks['target_cut_blank'] &= all(np.array_equal(data[k][:,target_cols],blank[k][:,target_cols]) for k in ('v','ge','gi'))
    expected_records=[];diagnostics=[];checks['no_interaction_before_second_packet']=True
    for c in ('intact','cut_input'):
        for site in p['sites']:
            sid=site['id'];col=p['monitor_cells'].index(site['target']);get=lambda k,d:by[c,sid,k,d]
            for after in (20,50,100,200):
                end=1820+10*after;sl=slice(1520,end);v=lambda k,d:get(k,d)['v'][sl,col]*1000
                blank=by[c,-1,'blank',0]['v'][sl,col]*1000;ab=v('AB',200);ba=v('BA',200)
                ja=ab-(v('A0',0)+v('Bd',200)-blank);jb=ba-(v('B0',0)+v('Ad',200)-blank);j=ja-jb
                count=lambda k,d:sum(1 for i,t in zip(get(k,d)['indices'],get(k,d)['ticks']) if i==site['target'] and 1520<=t<end)
                na,nb=count('AB',200),count('BA',200)
                checks['no_interaction_before_second_packet'] &= float(np.max(abs(j[:200])))<1e-11
                if c=='intact' and after==20:
                    target_counts={k+str(d):count(k,d) for k,d in [('A0',0),('B0',0),('Ad',200),('Bd',200),('AB',0),('AB',200),('BA',200)]}
                    drive_counts={k:{patch:sum(1 for i,t in zip(get(k,200)['indices'],get(k,200)['ticks']) if i==site[patch+'_cell'] and 1520<=t<end) for patch in ('a','b')} for k in ('AB','BA')}
                    diagnostics.append({'site':sid,'target_spikes_all_controls':target_counts,'input_neuron_spikes':drive_counts,'target_max_voltage_mv':float(max(np.max(ab),np.max(ba)))})
                expected_records.append({'condition':c,'site':sid,'subtype':site['subtype'],'after_last_pulse_ms':after,
                    'voltage_order_mean_mv':float(np.mean(ab-ba)),'voltage_order_peak_abs_mv':float(np.max(abs(ab-ba))),
                    'voltage_interaction_mean_mv':float(np.mean(j)),'voltage_interaction_peak_abs_mv':float(np.max(abs(j))),
                    'voltage_response_peak_abs_mv':float(max(np.max(abs(ab-blank)),np.max(abs(ba-blank)))),
                    'ab_spikes':na,'ba_spikes':nb,'spike_order':na-nb,'spike_interaction':na-nb-count('A0',0)-count('Bd',200)+count('B0',0)+count('Ad',200)})
    checks['metrics'] &= len(expected_records)==len(r['per_site']) and all(all(np.isclose(a[k],b[k],atol=1e-11,rtol=1e-10) if isinstance(a[k],float) else a[k]==b[k] for k in a) for a,b in zip(expected_records,r['per_site']))
    old=next(v for v in read(p10/'rows.json') if v['condition']=='intact' and v['kind']=='AB' and v['delay']==200);control=read(root/'monitor-control.json')
    fetch('monitor_control')
    checks['passive_monitor']=all(old[k]==control[k] for k in ('events_sha256','states_sha256')) and hashes['monitor_control']==control['data_sha256']
    first=next(v for v in rows if v['condition']=='intact' and v['kind']=='AB' and v['delay']==200);restored=read(root/'restoration.json');data=fetch('restoration');original=by['intact',first['site'],'AB',200]
    checks['restoration']=all(first[k]==restored[k] for k in ('events_sha256','states_sha256')) and all(np.array_equal(data[k],original[k]) for k in data) and hashes['restoration']==restored['data_sha256']
    result={'archive_used':str(archive) if archive is not None else None,'all_passed':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'primary_diagnostics':diagnostics,'lif_checked_steps':checked_steps,'lif_max_error_volts':max_error,'report_sha256':sha(root/'report.json'),'verifier_sha256':sha(Path(__file__))}
    save(root/'verification.json',result);print(result,flush=True)
    if not result['all_passed']:raise ValueError('verification failed')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--archive',type=Path);a=ap.parse_args();verify(a.root,a.archive)
