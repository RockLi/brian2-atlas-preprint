"""Supplementary response-window audit; does not change stimulus duration."""
import argparse
from pathlib import Path
import numpy as np
from .motion_refinement import read,sha,load_npz
from .run_experiment import save


def count_until(data,stop,cells):
    return np.bincount(data['slots'][data['ticks']<stop],minlength=cells).astype(np.int64)


def analyze(root):
    p=read(root/'protocol.json');report=read(root/'report.json');rows=read(root/'rows.json');mapping=load_npz(root/'mapping.npz')
    if report['status']!='complete' or not read(root/'verification.json')['all_passed']:raise ValueError('verified complete study required')
    data={(r['site'],r['kind'],r['delay']):load_npz(root/'trials'/(r['key']+'.npz')) for r in rows if r['condition']=='intact'}
    records=[]
    for site in p['sites']:
        s=site['id'];family=0 if site['family']=='Mi1' else 1
        local=(mapping['family']==family)&mapping['valid']&(np.linalg.norm(mapping['xy']-site['center'],axis=1)<=.25)
        for delay in (200,500,1000):
            for after in (20,50,100,200):
                stop=1500+delay+120+after*10
                cnt=lambda kind,d:count_until(data[s,kind,d],stop,len(local))
                ab=cnt('AB',delay);ba=cnt('BA',delay);a=cnt('A0',0);b=cnt('B0',0);ad=cnt('Ad',delay);bd=cnt('Bd',delay)
                interaction=(ab-a-bd)-(ba-b-ad)
                blank=count_until(data[-1,'blank',0],stop,len(local))
                records.append({'site':s,'family':site['family'],'onset_interval_ms':delay/10,'after_last_pulse_ms':after,
                    'observation_end_ms':stop/10,'local_response_l1':float((np.abs(ab-blank)[local].sum()+np.abs(ba-blank)[local].sum())/2),'local_order_l1':int(np.abs(ab-ba)[local].sum()),
                    'local_interaction_l1':int(np.abs(interaction)[local].sum())})
    summary=[{'onset_interval_ms':delay,'after_last_pulse_ms':after,
        **{key:float(np.mean([r[key] for r in records if r['onset_interval_ms']==delay and r['after_last_pulse_ms']==after])) for key in ('local_response_l1','local_order_l1','local_interaction_l1')}} for delay in (20,50,100) for after in (20,50,100,200)]
    simultaneous=[]
    for site in p['sites']:
        sid=site['id'];family=0 if site['family']=='Mi1' else 1
        local=(mapping['family']==family)&mapping['valid']&(np.linalg.norm(mapping['xy']-site['center'],axis=1)<=.25)
        for stop in (1820,2120,2620,3620,6000):
            cnt=lambda kind:count_until(data[sid,kind,0],stop,len(local))
            blank=count_until(data[-1,'blank',0],stop,len(local))
            joint=cnt('AB');error=joint-cnt('A0')-cnt('B0')+blank
            simultaneous.append({'site':sid,'observation_end_ms':stop/10,'after_last_pulse_ms':(stop-1620)/10,
                'local_response_l1':int(np.abs(joint-blank)[local].sum()),
                'local_additivity_error_l1':int(np.abs(error)[local].sum())})
    simultaneous_means=[{'observation_end_ms':stop/10,'after_last_pulse_ms':(stop-1620)/10,
        **{key:float(np.mean([row[key] for row in simultaneous if row['observation_end_ms']==stop/10])) for key in ('local_response_l1','local_additivity_error_l1')}} for stop in (1820,2120,2620,3620,6000)]
    return {'scope':'post-hoc supporting analysis proposed after inspecting the first complete site; all fixed windows reported; observation duration changes only, not stimulus or model; cumulative counts from tick zero to an exclusive endpoint after the final input pulse',
        'report_sha256':sha(root/'report.json'),'rows_sha256':sha(root/'rows.json'),'source_sha256':sha(Path(__file__)),
        'per_site':records,'means':summary,'simultaneous':simultaneous,'simultaneous_means':simultaneous_means}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--verify',action='store_true');a=parser.parse_args()
    result=analyze(a.root);path=a.root/'window-audit.json'
    if a.verify:
        if read(path)!=result:raise ValueError('window audit changed')
        save(a.root/'window-audit-verification.json',{'all_passed':True,'audit_sha256':sha(path)})
        print({'all_passed':True},flush=True)
    else:
        if path.exists():raise ValueError('window audit already exists')
        save(path,result);print(result['means'],flush=True)
