"""Complete native primary raw audit, with delayed publication after guard success.

Run the audit under a finite Linux resource/analysis guard on node23's data
volume. It writes only a pending report; publish requires the terminal audit
guard too. No simulation, source deletion or automatic retry is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from mam_collect_native_primary_raw import LABEL, BASE, BUILD, DESTINATION, NODES, IPS, TAG, MAX_FILE, MAX_HOST, MAX_ALL, make_catalog, reference_identity, PROJECT
from mam_collect_native_primary_resources import read, CAP
from mam_native_primary_resources import audit as audit_resources, NEURONS, EDGES, PARAMETERS, LAYOUT_SHA, CPUS
from mam_collection_transfer import validate, source_path, release_cache
from mam_native_rank_audit import audit_rank
from mam_primary_resources import require, counters, number

PRIOR='nest-mam-long-v1-metastable-seed1729-10500ms'
PRIOR_CATALOG_SHA='5dfa2a430cced978caa65dc55b6afb5f95e9b41504e78d13c74eea00991d7a35'
PRIOR_SUMMARY_SHA='66dd679db5520a13c9309d2e0557af881f8732019cea60ac33eaef3b6bd35c86'
PRIOR_SPIKES=527018677
PENDING='summary.pending.json'
SUMMARY='summary.json'


def audit_spec(reference_seed):
    if reference_seed is None:
        return dict(label=LABEL,seed=1729,tag=TAG,destination=DESTINATION,
                    prior=PRIOR,prior_duration_ms=10500,prior_spikes=PRIOR_SPIKES,
                    prior_catalog_sha256=PRIOR_CATALOG_SHA,prior_summary_sha256=PRIOR_SUMMARY_SHA)
    label,tag,destination,_=reference_identity(reference_seed)
    pins={
        1730:('97d39707a9e7bfa35377c37029d2996a6c1ec85873da8e34c1882086ab45e6ce',
              '91bb8def78749ddcd68d322edbe31130917ef0f241725d1b4fb160a86205b48a',134632862),
        1731:('0b2b12345426ba3f691ab19f85d55a32bcdbb71bc9f7be12e91cdb76e577423a',
              'd0f7777066a516341cdccadf31e37e00b51eee778133c15b3eff878b5ef451ef',112262847),
    }
    catalog,summary,spikes=pins[reference_seed]
    return dict(label=label,seed=reference_seed,tag=tag,destination=destination,
                prior=f'nest-mam-ensemble-v1-seed{reference_seed}-2500ms',prior_duration_ms=2500,
                prior_spikes=spikes,prior_catalog_sha256=catalog,prior_summary_sha256=summary)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_file(path, expected=None, cap=CAP):
    raw=read(path,cap)
    if expected is not None:
        require(len(raw)==expected['bytes'] and sha(raw)==expected['sha256'],'control bytes/hash differ: '+str(path))
    return json.loads(raw)


def small_file_hash(path):
    require(path.stat().st_size<=8*2**20, 'unexpected large metadata file')
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(2**20):h.update(block)
        release_cache(f,0,False)
    return h.hexdigest()


def transfer_guard(g,host,name,receiver,*,reference_seed=None):
    tag=audit_spec(reference_seed)['tag']
    require(g['schema']=='b2-mpi-resource-guard-v1' and g['host']==host and g['uid']==1000
            and g['cgroup']=='/system.slice/b2mpi-'+tag+'-'+name+('-receive' if receiver else '-send')+'.service'
            and g['admitted'] is True and type(g.get('returncode')) is int and g['returncode']==0
            and not g.get('error'),'transfer guard identity/terminal failure')
    number(g['wall_seconds'],'transfer wall',positive=True)
    require(g['wall_seconds']<=900,'transfer wall exceeded')
    for phase in ['before','after']:
        row=g[phase];events=counters(row['memory.events'])
        require(row['memory.max']==str(4*2**30) and row['memory.swap.max']=='0'
                and row['pids.max']=='64' and row['cpuset.cpus.effective']=='8-9'
                and all(events[k]==0 for k in ['max','oom','oom_kill','oom_group_kill']),
                'transfer pressure or resource limits differ')
        quota,period=map(int,row['cpu.max'].split());require(period>0 and quota==2*period,'transfer CPU quota differs')
    require(0<int(g['after']['memory.peak'])<=4*2**30,'transfer peak exceeds limit')
    reserve=(1280 if receiver else 128)*2**30
    require(g['data_volume']==('/data/brick2' if receiver else '/')
            and (g['data_device']!=g['root_device'] if receiver else g['data_device']==g['root_device'])
            and g['minimum_free_bytes']==reserve and g['minimum_observed_free_bytes']>=reserve
            and g['file_limit_bytes']==3*2**30,'transfer storage limit differs')


def controls(directory,*,reference_seed=None):
    spec=audit_spec(reference_seed)
    hashes=json_file(directory/'input-sha256.json',cap=2**20)
    expected={'admission.json','launch.json','parameters.json','layout.json','stage.json','report.json'}|{'host-'+str(i)+'.json' for i in range(6)}
    require(set(hashes)==expected,'terminal control hash coverage differs')
    raw={name:read(directory/name,CAP+1) for name in expected}
    require(all(sha(raw[n])==h for n,h in hashes.items()),'terminal controls changed')
    require(sha(raw['parameters.json'])==PARAMETERS and sha(raw['layout.json'])==LAYOUT_SHA,'native parameter/layout pin differs')
    values={name:json.loads(data) for name,data in raw.items()}
    identity={}
    if reference_seed is not None:
        from mam_launch_native_full_reference import PROTOCOL_SHA, CAMPAIGN
        a=values['admission.json']
        require(a['seed']==reference_seed and a['label']==spec['label']
                and a['protocol_sha256']==PROTOCOL_SHA and a['campaign']==CAMPAIGN
                and a['source_project']==PROJECT and a['source_project_is_retained_primary'] is True,
                'reference resource admission identity differs')
        identity=dict(label=spec['label'],seed=reference_seed)
    hosts=[values['host-'+str(i)+'.json'] for i in range(6)]
    resource=audit_resources(values['admission.json'],values['launch.json'],hosts,
                             values['parameters.json'],values['layout.json'],**identity)
    require(resource==values['report.json'],'native resource decision differs')
    return hosts,resource,hashes['report.json']


def collection_gate(directory,host_controls,resource_sha,outer_control_guard,*,reference_seed=None):
    spec=audit_spec(reference_seed);destination=spec['destination'];tag=spec['tag']
    r=json_file(directory/'report.json',cap=2**20)
    require(r['schema']=='b2-mam-native-primary-collection-v1' and r['ready'] is True
            and r['complete'] is True and r['resource_report_sha256']==resource_sha
            and r['destination']==destination and r['source_archive_created'] is False
            and r['automatic_retry'] is False and r['shared_collection_budget_seconds']==7200,
            'native collection decision differs')
    if reference_seed is not None:
        from mam_launch_native_full_reference import PROTOCOL_SHA
        require(r['label']==spec['label'] and r['seed']==reference_seed
                and r['protocol_sha256']==PROTOCOL_SHA and r['transfer_tag']==tag
                and r['source_project']==PROJECT, 'reference collection identity differs')
    require(r['control_outer_guard_sha256']==sha(read(outer_control_guard,2**20)), 'original outer guard pin differs')
    g=json_file(outer_control_guard,cap=2**20)
    if reference_seed is not None:
        command=g['command']
        require(command and Path(command[0]).name=='mam_collect_native_primary_resources.py'
                and command.count('--reference-seed')==1
                and command[command.index('--reference-seed')+1]==str(reference_seed),
                'reference control guard seed differs')
    require(type(g['returncode']) is int and g['returncode']==0 and g['stop_reason'] is None
            and 0<g['wall_seconds']<=g['wall_limit_seconds']<=180
            and 0<g['sampled_peak_rss_bytes']<=g['sampled_rss_limit_bytes']==1536*2**20,
            'original control collection guard failed')
    prior=number(r['previous_collection_seconds'],'previous collection time')
    elapsed=number(r['bulk_collection_seconds'],'bulk collection time',positive=True)
    require(prior>=g['wall_seconds'] and prior+elapsed<7200,'shared collection budget exceeded')
    require([row['host'] for row in r['nodes']]==NODES,'native collection host coverage differs')
    catalogs=[];total=0
    for i,host in enumerate(host_controls):
        name='node'+NODES[i].rsplit('-',1)[-1]
        catalog=json_file(directory/(name+'-catalog.json'),cap=2**20)
        progress={v['path']:{k:v[k] for k in ['bytes','sha256']} for v in catalog['files'] if v['path'].endswith('.progress.jsonl')}
        require(catalog==make_catalog(i,host,progress,run_label=spec['label']),'native catalog differs from terminal report set')
        size,digest=validate(catalog,MAX_FILE,MAX_HOST);row=r['nodes'][i];total+=size
        require(row['catalog_sha256']==digest and row['bytes']==size and row['files']==len(catalog['files']), 'collection node totals differ')
        receipt=json_file(directory/(name+'-receipt.json'),cap=2**20)
        require(receipt['schema']=='mam-direct-collection-v1' and receipt['complete'] is True
                and receipt['catalog_sha256']==digest and receipt['bytes']==size
                and receipt['files']==len(catalog['files']) and receipt['peer']==IPS[i]
                and receipt['output']==destination+'/'+name
                and receipt['max_file_bytes']==MAX_FILE and receipt['max_total_bytes']==MAX_HOST
                and receipt['reserve_bytes']==1280*2**30 and receipt['file_cache_release_supported'] is True
                and receipt['source_archive_created'] is False,'native receipt differs')
        for receiver,phase in [(True,'receive'),(False,'send')]:
            guard=json_file(directory/(tag+'-'+name+'-'+phase+'-guard.json'),cap=2**20)
            transfer_guard(guard,'hk-prod-model-ae02-23' if receiver else NODES[i],name,receiver,reference_seed=reference_seed)
            require(guard['wall_seconds']==row[phase+'_wall_seconds'],'transfer guard wall evidence differs')
        catalogs.append(catalog)
    require(total==r['bytes'] and 0<total<=MAX_ALL,'native collection total differs')
    return catalogs,r


def verify_inventory(root,catalog):
    actual=[];directories=[];entries=0
    require(root.is_dir() and not root.is_symlink(),'native collection directory required')
    for p in root.rglob('*'):
        entries+=1;require(entries<=512,'unexpected collection tree expansion')
        require(not p.is_symlink() and (p.is_file() or p.is_dir()),'nonregular collection entry')
        if p.is_file():actual.append(str(p.relative_to(root)))
        else:directories.append(str(p.relative_to(root)))
    require(sorted(actual)==[r['path'] for r in catalog['files']], 'collection file set differs')
    expected_dirs={str(p) for row in catalog['files'] for p in Path(row['path']).parents if str(p)!='.'}
    require(set(directories)==expected_dirs, 'collection directory set differs')
    require(not (root/'.collection-parts').exists(), 'partial collection cannot pass')
    for row in catalog['files']:
        p=source_path(root,row['path']);require(p.stat().st_size==row['bytes'],'collected file size differs')
        if not row['path'].endswith('.events.bin'):
            require(small_file_hash(p)==row['sha256'],'collected metadata checksum differs')


def summarize(rank_results,resource,chunk_counts,chunk_simulation,chunk_rss,*,reference_seed=None):
    spec=audit_spec(reference_seed)
    require([r['rank'] for r in rank_results]==list(range(48)),'raw rank coverage differs')
    require(all(r['rank_event_stream_verified'] is True and r['raw_byte_prefix_exact'] is True
                and r['construction_ledger_exact'] is True and r['duration_ms']==100500
                and r['chunks']==2010 and r['prefix_duration_ms']==spec['prior_duration_ms'] and r['linux_cache_release'] is True
                for r in rank_results),'incomplete native raw rank audit')
    if reference_seed is not None:
        require(all(r['seed']==reference_seed for r in rank_results)
                and resource['label']==spec['label'],'reference raw/resource seed identity differs')
    bins=np.zeros(2010,dtype=np.int64);terminal=spikes=prefix=0
    for r in rank_results:
        require(all(type(r[k]) is int and r[k]>=0 for k in ['terminal_tick_events','spikes','event_bytes','prefix_bytes']),
                'invalid native raw count')
        require(r['prefix_bytes']<=r['event_bytes']<=2*2**30, 'native recording/prefix quota differs')
        values=r['physical_50ms_bin_counts']
        require(len(values)==2010 and all(type(v) is int and v>=0 for v in values),'invalid physical bins')
        require(sum(values)+r['terminal_tick_events']==r['spikes'] and r['event_bytes']==8*r['spikes'], 'per-rank physical count differs')
        bins+=np.asarray(values,dtype=np.int64);terminal+=r['terminal_tick_events'];spikes+=r['spikes'];prefix+=r['prefix_bytes']
    require(prefix==spec['prior_spikes']*8 and spikes==resource['reported_spikes']
            and spikes*8==resource['reported_event_bytes'] and int(chunk_counts.sum())==spikes,
            'raw/resource/prefix global totals differ')
    require(resource['terminal_resource_audit_passed'] is True and resource['construction_ledger_passed'] is True
            and (resource['neurons'],resource['recurrent_edges'],resource['projections'])==(NEURONS,EDGES,8344),
            'complete native resource/construction audit missing')
    return dict(schema='b2-mam-native-primary-output-audit-v1',passed=False,
        raw_output_audit_passed=True,terminal_resource_audit_passed=True,audit_guard_passed=False,
        label=spec['label'],parameters_sha256=PARAMETERS,neurons=NEURONS,recurrent_edges=EDGES,projections=8344,seed=spec['seed'],
        duration_ms=100500,spikes=spikes,event_bytes=spikes*8,terminal_tick_events=terminal,
        physical_50ms_bin_counts=bins.tolist(),physical_50ms_bin_rates_hz=(bins/(NEURONS*.05)).tolist(),
        physical_window='[0,100500) ms; terminal tick excluded from bins',
        all_projection_counts_exact=True,**{f"all_first_{spec['prior_duration_ms']}ms_event_prefixes_exact":True},all_event_rank_ownership_exact=True,
        retained_prefix_spikes=spec['prior_spikes'],
        prefix_checks=[{k:r[k] for k in ['rank','prefix_bytes','prefix_sha256','raw_byte_prefix_exact','construction_ledger_exact']} for r in rank_results],
        chunks=[dict(end_ms=(i+1)*50,spikes=int(chunk_counts[i]),
            recorded_count_rate_hz=float(chunk_counts[i]/(NEURONS*.05)),max_simulation_seconds=float(chunk_simulation[i]),
            max_rank_rss_kib=int(chunk_rss[i])) for i in range(2010)],
        scientific_equivalence=False,performance_cost_acceptance=False,all_actual_openmp_workers_observed=False,
        scope='Complete native recording consistency and exact retained-byte prefix. Recorder drain chunks differ from physical-time bins; continuation may record additional earlier physical spikes. NEST dynamics are not independently reproduced. Scientific and tuned speed/cost acceptance remain open.')


def run(root,control_directory,collection_directory,outer_control_guard,prior_catalog_path,prior_summary_path,*,reference_seed=None):
    started=time.monotonic()
    spec=audit_spec(reference_seed)
    require(root.resolve()==Path(spec['destination']) and Path('/data/brick2').is_mount(),'native audit must use node23 data volume')
    require(hasattr(os,'posix_fadvise'),'Linux cache release required')
    require(not (root/SUMMARY).exists() and not (root/PENDING).exists(),'native audit already attempted/published')
    checkpoint=root/'primary-raw-audit-attempt.json'
    require(not checkpoint.exists(),'native raw audit was already attempted')
    hosts,resource,resource_sha=controls(control_directory,reference_seed=reference_seed)
    catalogs,collection=collection_gate(collection_directory,hosts,resource_sha,outer_control_guard,reference_seed=reference_seed)
    old_raw=read(prior_catalog_path,2**20);old_summary_raw=read(prior_summary_path,2**20)
    require(sha(old_raw)==spec['prior_catalog_sha256'] and sha(old_summary_raw)==spec['prior_summary_sha256'],'retained native evidence pins differ')
    old_catalog=json.loads(old_raw);old_summary=json.loads(old_summary_raw)
    require(old_summary['passed'] is True and old_summary['spikes']==spec['prior_spikes']
            and old_summary['parameters_sha256']==PARAMETERS,'retained native summary differs')
    if reference_seed is not None:
        require(old_summary['seed']==reference_seed and old_summary['label']==spec['prior'], 'retained reference seed differs')
    with checkpoint.open('x') as f:f.write(json.dumps(dict(started=True,automatic_retry=False,resource_report_sha256=resource_sha))+'\n')
    ranks=[];chunk_counts=np.zeros(2010,dtype=np.int64);chunk_simulation=np.zeros(2010);chunk_rss=np.zeros(2010,dtype=np.int64)
    for i,host in enumerate(hosts):
        name='node'+NODES[i].rsplit('-',1)[-1];node=root/name
        require(json_file(root/(name+'-catalog.json'),cap=2**20)==catalogs[i], 'destination catalog differs')
        require(json_file(root/(name+'-receipt.json'),cap=2**20)==json_file(collection_directory/(name+'-receipt.json'),cap=2**20),'destination receipt differs')
        verify_inventory(node,catalogs[i])
        for rank in range(i*8,(i+1)*8):
            current=node/'runs'/spec['label'];stem='rank'+str(rank)
            report=json_file(current/(stem+'.json'),cap=4*2**20)
            require(report==host['rank_reports'][str(rank)],'collected rank report differs from accepted control')
            progress_raw=read(current/(stem+'.progress.jsonl'),4*2**20)
            progress=[json.loads(line) for line in progress_raw.splitlines()]
            relative=spec['prior']+'-audit/'+name+'/runs/'+spec['prior']+'/'+stem
            previous=json_file(Path(BUILD)/(relative+'.json'),old_catalog[relative+'.json'],4*2**20)
            require(previous['host']==NODES[i] and previous['event_bytes']==old_catalog[relative+'.events.bin']['bytes']
                    and previous['event_sha256']==old_catalog[relative+'.events.bin']['sha256'],'retained rank recording pin differs')
            result=audit_rank(report,progress,current/(stem+'.events.bin'),rank=rank,duration_ms=100500,neurons=NEURONS,
                parameters_sha256=PARAMETERS,allowed_cpus=[CPUS[(rank%8)*4]],prefix_report=previous,
                prefix_events=Path(BUILD)/(relative+'.events.bin'),prefix_allowed_cpus=list(range(32)),expected_seed=spec['seed'])
            ranks.append(result)
            chunk_counts+=np.array([c['spikes'] for c in report['chunks']],dtype=np.int64)
            chunk_simulation=np.maximum(chunk_simulation,[c['simulation_seconds'] for c in report['chunks']])
            chunk_rss=np.maximum(chunk_rss,[report['phase_memory'][f'after_{(j+1)*50}ms']['rss_kib'] for j in range(2010)])
            print(json.dumps(dict(event='native_rank_raw_verified',rank=rank,spikes=result['spikes'],prefix_bytes=result['prefix_bytes'])),flush=True)
    report=summarize(ranks,resource,chunk_counts,chunk_simulation,chunk_rss,reference_seed=reference_seed)
    report.update(resource_report_sha256=resource_sha,collection_report_sha256=sha(read(collection_directory/'report.json')),
                  prior_catalog_sha256=spec['prior_catalog_sha256'],prior_summary_sha256=spec['prior_summary_sha256'],
                  audit_seconds=time.monotonic()-started)
    with (root/PENDING).open('x') as f:
        f.write(json.dumps(report,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    return dict(raw_output_audit_passed=True,passed=False,pending_guard_acceptance=True,pending_report_sha256=sha(read(root/PENDING)))


def publish(root,guard_path,expected_pending_sha,*,reference_seed=None):
    spec=audit_spec(reference_seed)
    require(root.resolve()==Path(spec['destination']), 'native publication must target admitted data path')
    raw=read(root/PENDING,8*2**20);require(sha(raw)==expected_pending_sha,'pending native report changed')
    r=json.loads(raw);g=json_file(guard_path,cap=2**20)
    require(r['schema']=='b2-mam-native-primary-output-audit-v1' and r['label']==spec['label'] and r['seed']==spec['seed']
            and r['raw_output_audit_passed'] is True and r['terminal_resource_audit_passed'] is True
            and r['passed'] is False and r['audit_guard_passed'] is False,'not a pending successful native audit')
    require(g['schema']=='b2-mpi-resource-guard-v1' and g['host']=='hk-prod-model-ae02-23'
            and g['uid']==1000 and g['admitted'] is True and type(g.get('returncode')) is int
            and g['returncode']==0 and not g.get('error'),'native audit guard not terminal success')
    require(g['data_volume']=='/data/brick2' and g['data_device']!=g['root_device']
            and g['minimum_free_bytes']>=1280*2**30 and g['minimum_observed_free_bytes']>=g['minimum_free_bytes']
            and g['disk_check_interval_seconds']==5 and 0<g['file_limit_bytes']<=512*2**20,
            'native audit disk reserve failed')
    require(number(g['wall_seconds'],'audit guard wall',positive=True)<=10800,'native analysis budget exceeded')
    command=g['command']
    require(any(Path(arg).name=='mam_native_primary_output_audit.py' for arg in command)
            and command.count('--root')==1 and command[command.index('--root')+1]==str(root)
            and 'audit' in command,'guard not bound to this native raw audit')
    if reference_seed is not None:
        require(command.count('--reference-seed')==1
                and command[command.index('--reference-seed')+1]==str(reference_seed), 'raw audit guard seed differs')
    else:
        require('--reference-seed' not in command,'reference raw guard cannot publish as primary')
    for phase in ['before','after']:
        facts=g[phase];events=counters(facts['memory.events'])
        require(0<int(facts['memory.max'])<=16*2**30 and facts['memory.swap.max']=='0'
                and facts['pids.max']=='64' and facts['cpuset.cpus.effective']=='8-9'
                and all(events[k]==0 for k in ['max','oom','oom_kill','oom_group_kill']),'native audit pressure or limits differ')
        quota,period=map(int,facts['cpu.max'].split());require(period>0 and 0<quota<=2*period,'native audit CPU limit differs')
    require(0<int(g['after']['memory.peak'])<=int(g['after']['memory.max']), 'native audit peak exceeds memory cap')
    require(0<=r['audit_seconds']<=g['wall_seconds'], 'native audit timing exceeds guard')
    r.update(passed=True,audit_guard_passed=True,audit_guard_sha256=sha(read(guard_path,2**20)),
             analysis_wall_seconds_consumed=g['wall_seconds'],shared_analysis_budget_seconds=10800)
    temporary=root/'summary.publishing.json'
    with temporary.open('x') as f:f.write(json.dumps(r,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    os.link(temporary,root/SUMMARY);temporary.unlink()
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(passed=True,summary_sha256=sha(read(root/SUMMARY)),scientific_equivalence=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('audit')
    for name in ['root','controls','collection','control-guard','prior-catalog','prior-summary']:a.add_argument('--'+name,type=Path,required=True)
    a.add_argument('--reference-seed',type=int,choices=[1730,1731])
    a=sub.add_parser('publish');a.add_argument('--root',type=Path,required=True);a.add_argument('--guard',type=Path,required=True);a.add_argument('--pending-sha256',required=True)
    a.add_argument('--reference-seed',type=int,choices=[1730,1731])
    a=p.parse_args()
    if a.mode=='audit':result=run(a.root,a.controls,a.collection,a.control_guard,a.prior_catalog,a.prior_summary,reference_seed=a.reference_seed)
    else:result=publish(a.root,a.guard,a.pending_sha256,reference_seed=a.reference_seed)
    print(json.dumps(result))
