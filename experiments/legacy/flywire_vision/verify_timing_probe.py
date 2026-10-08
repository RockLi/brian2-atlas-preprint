"""Rebuild schedules, event histograms and contrasts from saved P10 records."""
import argparse
from pathlib import Path
import numpy as np
from .motion_refinement import read,sha,load_npz
from .multispeed_data import digest
from .timing_probe import choose_sites,design,DELAYS
from .run_experiment import save


def verify(root):
    p=read(root/'protocol.json');r=read(root/'report.json');rows=read(root/'rows.json');parent=Path(p['parent']);artifact=Path(p['artifact'])
    mapping=load_npz(root/'mapping.npz');channels=read(artifact/'channels.json');bykey={};byrow={}
    checks={'complete':r['status']=='complete' and len(rows)==155 and r['native_runs']==156,
        'provenance':sha(parent/'protocol.json')==p['parent_protocol_sha256'] and sha(artifact/'channels.json')==p['channels_sha256'] and sha(artifact/'groups.json')==p['groups_sha256'] and sha(root/'mapping.npz')==p['mapping_sha256'],
        'frozen_sources':all(sha(Path(__file__).with_name(n))==h for n,h in p['source_sha256'].items()),
        'report_links':all(sha(root/(k+'.json'))==r[k+'_sha256'] for k in ('protocol','rows','design')),
        'declared_design':read(root/'design.json')==design(p['sites'])==[{k:row[k] for k in ('condition','site','kind','delay')} for row in rows],
        'spatial_selection':choose_sites(channels)==p['sites']}
    original=load_npz(parent/'transform.npz')
    checks['inherited_mapping']=all(np.array_equal(mapping[k],original[k]) for k in ('cells','subtype','family','xy','valid'))
    for condition in ('intact','cut_input','matched_cut'):
        identity=read(root/condition/'identity.json');old=read(parent/condition/'identity.json')
        checks[condition+'_identity']=sha(root/condition/'base.bin')==identity['base_sha256']==old['base_sha256'] and sha(artifact/'compile/native/b2-native')==identity['binary_sha256']==old['binary_sha256']
        checks[condition+'_locked']= (root/'protocol.json').stat().st_mtime_ns < (root/condition/'identity.json').stat().st_mtime_ns
    checks.update({k:True for k in ('input_schedules','event_histograms','raw_hashes','pair_counts','single_packets','simultaneous','cut_equals_blank','metric_records')})
    for row in rows:
        data=load_npz(root/'trials'/(row['key']+'.npz'));bykey[row['key']]=data;byrow[row['condition'],row['site'],row['kind'],row['delay']]=data
        site=p['sites'][max(row['site'],0)];kind=row['kind'];d=row['delay'];events=[]
        if kind!='blank':
            if kind in ('AB','BA'):
                patches=[('a',0),('b',d)] if kind=='AB' else [('b',0),('a',d)]
            else:patches=[('a' if kind.startswith('A') else 'b',0 if kind.endswith('0') else d)]
            for patch,offset in patches:
                events.extend((1500+offset+time,ch) for ch in site[patch] for time in (20,120))
        expected=np.array(sorted(events),dtype=np.int64).reshape(-1,2);actual=np.stack([data['input_ticks'],data['input_indices']],1)
        checks['input_schedules'] &= np.array_equal(actual,expected) and len(expected)==row['external_spikes'] and len(set(map(tuple,actual)))==len(actual)
        checks['raw_hashes'] &= sha(root/'trials'/(row['key']+'.npz'))==row['data_sha256'] and digest(actual.astype('<i8'))==row['input_sha256']
        slots=data['slots'];ticks=data['ticks'];valid=np.all((slots>=0)&(slots<len(mapping['cells']))) and np.all((ticks>=0)&(ticks<6000))
        hist=np.bincount(slots,minlength=len(mapping['cells']));flat=(ticks//10)*9+mapping['subtype'][slots]
        trace=np.bincount(flat,minlength=5400).reshape(600,9)
        checks['event_histograms'] &= valid and np.array_equal(hist,data['totals']) and np.array_equal(trace,data['trace']) and hist.sum()==len(ticks)
    checks['no_response_before_input']=True
    for row in rows:
        data=bykey[row['key']];blank=byrow[row['condition'],-1,'blank',0]
        cutoff=int(data['input_ticks'][0]) if len(data['input_ticks']) else 6000
        a=data['ticks']<cutoff;b=blank['ticks']<cutoff
        checks['no_response_before_input'] &= np.array_equal(data['ticks'][a],blank['ticks'][b]) and np.array_equal(data['slots'][a],blank['slots'][b])
    totals=lambda c,s,k,d:byrow[c,s,k,d]['totals'].astype(np.int64)
    inputs=lambda c,s,k,d:byrow[c,s,k,d]['input_indices']
    saved={(a['site'],a['onset_interval_ms']):a for a in r['per_site']}
    reconstructed=[];cuts=[]
    for site in p['sites']:
        s=site['id'];family=0 if site['family']=='Mi1' else 1;mask=(mapping['family']==family)&mapping['valid']&(np.linalg.norm(mapping['xy']-site['center'],axis=1)<=.25)
        blank=totals('intact',-1,'blank',0);a0=totals('intact',s,'A0',0);b0=totals('intact',s,'B0',0)
        simultaneous=byrow['intact',s,'AB',0]
        for d in DELAYS:
            ab=totals('intact',s,'AB',d);ba=totals('intact',s,'BA',d);ad=totals('intact',s,'Ad',d);bd=totals('intact',s,'Bd',d)
            jab=ab-(a0+bd-blank);jba=ba-(b0+ad-blank);j=jab-jba;v=ab-ba
            result={'site':s,'family':site['family'],'axis':site['axis'],'onset_interval_ms':d/10,'local_cells':int(mask.sum()),
                'local_response_l1':float((np.abs(ab-blank)[mask].sum()+np.abs(ba-blank)[mask].sum())/2),
                'local_order_l1':int(np.abs(v[mask]).sum()),'local_interaction_l1':int(np.abs(j[mask]).sum()),
                'local_order_changed_cells':int(np.count_nonzero(v[mask])),'local_interaction_cells':int(np.count_nonzero(j[mask])),
                'all_readout_order_l1':int(np.abs(v).sum()),'all_readout_interaction_l1':int(np.abs(j).sum()),
                'subtype_signed_order':[int(v[mapping['subtype']==k].sum()) for k in range(9)],
                'subtype_signed_interaction':[int(j[mapping['subtype']==k].sum()) for k in range(9)]}
            checks['metric_records'] &= result==saved[s,d/10];reconstructed.append(result)
            counts=[np.bincount(inputs('intact',s,k,d),minlength=len(channels)) for k in ('AB','BA')]
            checks['pair_counts'] &= np.array_equal(*counts) and np.array_equal(counts[0],np.bincount(simultaneous['input_indices'],minlength=len(channels)))
            for pair,early,late in (('AB','A0','Bd'),('BA','B0','Ad')):
                data=byrow['intact',s,pair,d];one=byrow['intact',s,early,0];two=byrow['intact',s,late,d]
                expected=sorted([*zip(one['input_ticks'],one['input_indices']),*zip(two['input_ticks'],two['input_indices'])])
                checks['single_packets'] &= list(zip(data['input_ticks'],data['input_indices']))==expected
        a=byrow['intact',s,'A0',0];b=byrow['intact',s,'B0',0]
        checks['simultaneous'] &= list(zip(simultaneous['input_ticks'],simultaneous['input_indices']))==sorted([*zip(a['input_ticks'],a['input_indices']),*zip(b['input_ticks'],b['input_indices'])])
        for c in ('intact','cut_input','matched_cut'):
            v=totals(c,s,'AB',200)-totals(c,s,'BA',200);cuts.append({'site':s,'condition':c,'local_order_l1':int(abs(v[mask]).sum()),'all_readout_order_l1':int(abs(v).sum())})
            for k in ('AB','BA'):
                data=byrow[c,s,k,200];intact=byrow['intact',s,k,200]
                checks['pair_counts'] &= np.array_equal(data['input_indices'],intact['input_indices']) and np.array_equal(data['input_ticks'],intact['input_ticks'])
                if c=='cut_input':
                    blank_data=byrow[c,-1,'blank',0]
                    checks['cut_equals_blank'] &= all(np.array_equal(data[key],blank_data[key]) for key in ('totals','trace','slots','ticks'))
    means={str(d/10):{k:float(np.mean([v[k] for v in reconstructed if v['onset_interval_ms']==d/10])) for k in ('local_response_l1','local_order_l1','local_interaction_l1','local_order_changed_cells','local_interaction_cells')} for d in DELAYS}
    checks['summary']=means==r['interval_means'] and cuts==r['cut_per_site'] and {c:float(np.mean([v['local_order_l1'] for v in cuts if v['condition']==c])) for c in ('intact','cut_input','matched_cut')}==r['cut_means']
    first=next(row for row in rows if row['kind']=='AB' and row['delay']==200);restored=read(root/'restoration.json');data=load_npz(root/'restoration.npz')
    checks['restoration']=r['reset_exact'] and all(first[k]==restored[k] for k in ('input_sha256','events_sha256','states_sha256')) and sha(root/'restoration.npz')==restored['data_sha256'] and all(np.array_equal(v,bykey[first['key']][k]) for k,v in data.items())
    result={'all_passed':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'report_sha256':sha(root/'report.json'),'verifier_sha256':sha(Path(__file__))}
    save(root/'verification.json',result);print(result,flush=True)
    if not result['all_passed']:raise ValueError('P10 verification failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);a=parser.parse_args();verify(a.root)
