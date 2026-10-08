"""One bounded native primary collection, reusing the verified catalog protocol.

Control admission must already pass. No simulation is started, no source tar
is created, and partial failures are preserved without automatic retry. Raw
format/physical-time/prefix and scientific audits remain subsequent gates.
"""
import argparse
import ast
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time

from mam_launch_native_primary import BASE, LABEL, PROJECT, NODES, IPS, PARAMETERS, LAYOUT_SHA
from mam_native_primary_resources import audit
from mam_collect_native_primary_resources import read, CAP
from mam_primary_resources import require, counters
from mam_collection_transfer import SCHEMA, validate

BUILD='/data/brick2/brian2-mpi-region-20260907'
DESTINATION=BUILD+'/'+LABEL+'-audit'
LEADER='hk-prod-model-ae02-23'
LEADER_IP='192.168.20.23'
TAG='native-primary-collection-v1'
MAX_FILE=2*2**30
MAX_HOST=17*2**30
MAX_ALL=102*2**30
SOURCE_NAMES=['mam_collection_transfer.py','mam_direct_transfer_v3.py','mpi_resource_guard.py']


def remaining(start, previous):
    value=7200-previous-(time.monotonic()-start)-90
    require(value>60, 'shared native collection budget exhausted')
    return min(900,math.floor(value))


def reference_identity(reference_seed):
    if reference_seed is None:
        return LABEL, TAG, DESTINATION, {}
    from mam_launch_native_full_reference import label_for
    label=label_for(reference_seed)
    return label, f'native-reference-collection-v1-seed{reference_seed}', BUILD+'/'+label+'-audit', dict(label=label,seed=reference_seed)


def control_gate(directory, outer_guard, *, reference_seed=None):
    label,_,_,identity_options=reference_identity(reference_seed)
    required=['admission.json','launch.json','parameters.json','layout.json','stage.json',
              'report.json','input-sha256.json','collection-controller.json']+['host-'+str(i)+'.json' for i in range(6)]
    if not all((directory/n).exists() for n in required) or not outer_guard.exists():
        return None
    raw={n:read(directory/n,CAP+1) for n in required}
    hashes=json.loads(raw['input-sha256.json'])
    expected=set(required)-{'input-sha256.json','collection-controller.json'}
    require(set(hashes)==expected and all(hashlib.sha256(raw[n]).hexdigest()==h for n,h in hashes.items()),
            'native control hashes incomplete or changed')
    require(hashlib.sha256(raw['parameters.json']).hexdigest()==PARAMETERS
            and hashlib.sha256(raw['layout.json']).hexdigest()==LAYOUT_SHA,'native parameter/layout pin differs')
    values={n:json.loads(v) for n,v in raw.items()}
    if reference_seed is not None:
        from mam_launch_native_full_reference import CAMPAIGN, PROTOCOL_SHA
        a=values['admission.json']
        require(a['campaign']==CAMPAIGN and a['protocol_sha256']==PROTOCOL_SHA
                and a['seed']==reference_seed and a['label']==label
                and a['source_project']==PROJECT and a['source_project_is_retained_primary'] is True,
                'native reference control identity differs')
    hosts=[values['host-'+str(i)+'.json'] for i in range(6)]
    result=audit(values['admission.json'],values['launch.json'],hosts,
                 values['parameters.json'],values['layout.json'],**identity_options)
    require(result==values['report.json'],'native resource decision differs')
    guard=json.loads(read(outer_guard,2**20))
    require(type(guard['returncode']) is int and guard['returncode']==0 and guard['stop_reason'] is None
            and guard['sampled_rss_limit_bytes']==1536*2**20
            and 0<guard['sampled_peak_rss_bytes']<=guard['sampled_rss_limit_bytes']
            and guard['kernel_cpu_limit_seconds']==120 and guard['kernel_file_limit_bytes']==64*2**20
            and 0<guard['wall_seconds']<=guard['wall_limit_seconds']<=180,
            'native control collection outer guard failed or changed')
    command=guard['command']
    require(command and Path(command[0]).name=='mam_collect_native_primary_resources.py'
            and command.count('--output')==1 and command[command.index('--output')+1]==str(directory),
            'outer guard is not bound to this native control collection')
    timing=values['collection-controller.json']
    if reference_seed is not None:
        require(command.count('--reference-seed')==1
                and command[command.index('--reference-seed')+1]==str(reference_seed)
                and timing['label']==label and timing['seed']==reference_seed
                and timing['protocol_sha256']==PROTOCOL_SHA and timing['source_project']==PROJECT
                and timing.get('recovery') is None, 'reference outer guard or collection identity differs')
    else:
        require('--reference-seed' not in command, 'reference collection cannot pass as primary')
    require(timing['ready'] is True and timing['terminal_resource_audit_passed'] is True
            and timing['shared_collection_budget_seconds']==7200
            and 0<=timing.get('stage_elapsed_seconds',timing['elapsed_seconds'])<600
            and 0<=timing['elapsed_seconds']<7200-90 and timing['automatic_retry'] is False,
            'native collection timing missing or invalid')
    prior_seconds=0.
    recovery=timing.get('recovery')
    if recovery is not None:
        require(recovery==json.loads(read(directory/'recovery.json'))
                and recovery['recovery_attempts']==1 and recovery['automatic_retry'] is False,
                'native recovery provenance differs')
        prior=Path(recovery['prior_attempt']);prior_guard=Path(recovery['prior_guard'])
        require(not (prior/'recovery.json').exists(), 'recursive native recovery')
        require(hashlib.sha256(read(prior/'failure.json')).hexdigest()==recovery['failure_sha256']
                and hashlib.sha256(read(prior_guard)).hexdigest()==recovery['guard_sha256'],
                'native failed attempt evidence changed')
        failed=json.loads(read(prior/'failure.json'));old_guard=json.loads(read(prior_guard))
        prior_seconds=recovery['charged_prior_seconds']
        require(old_guard['returncode']!=0 and old_guard['stop_reason'] is None
                and recovery['failed_attempt_seconds']==max(failed['elapsed_seconds'],old_guard['wall_seconds'])
                and recovery['diagnosis_seconds']>=0
                and prior_seconds==recovery['failed_attempt_seconds']+recovery['diagnosis_seconds']
                and timing['elapsed_seconds']==timing['stage_elapsed_seconds']+prior_seconds,
                'native recovery budget excludes failed attempt or diagnosis')
        require(command.count('--prior-attempt')==1 and command[command.index('--prior-attempt')+1]==str(prior)
                and command.count('--prior-guard')==1 and command[command.index('--prior-guard')+1]==str(prior_guard),
                'native recovery guard is not bound to prior attempt')
        if 'offline_from' in recovery:
            old=Path(recovery['offline_from']);old_gp=Path(recovery['offline_guard'])
            require(hashlib.sha256(read(old/'failure.json')).hexdigest()==recovery['offline_failure_sha256']
                    and hashlib.sha256(read(old_gp)).hexdigest()==recovery['offline_guard_sha256']
                    and command.count('--offline-from')==1 and command[command.index('--offline-from')+1]==str(old)
                    and command.count('--offline-guard')==1 and command[command.index('--offline-guard')+1]==str(old_gp),
                    'offline native audit provenance differs')
            require(all(read(old/('host-'+str(i)+'.json'),CAP)==raw['host-'+str(i)+'.json'] for i in range(6)),
                    'offline native controls changed')
    else:
        require(timing['elapsed_seconds']==timing.get('stage_elapsed_seconds',timing['elapsed_seconds'])
                and timing['elapsed_seconds']<600,'unexpected native control budget')
    return dict(hosts=hosts,previous_seconds=max(timing['elapsed_seconds'],prior_seconds+guard['wall_seconds']),
                resource_report_sha256=hashes['report.json'],control_outer_guard_sha256=hashlib.sha256(read(outer_guard)).hexdigest())


def expected_sources(index, host, *, run_label=LABEL):
    require(host['host']==NODES[index], 'wrong native source host')
    sources={n:dict(v) for n,v in host['source_files'].items()}
    expected={LABEL+'/'+n for n in ['mam_nest_reference.py','mam_nest_rank_affinity.py',
        'mpi_resource_guard.py','layout.json','planned-budget.json','parameters.json']}
    prefix_roles=(['controller'] if index==0 else [])+['proxy-'+str(index)]
    for role in prefix_roles:
        name=host['guards'][role]['cgroup'].split('/')[-1].removesuffix('.service')
        expected.add('guards/'+name+'.json')
    for rank in range(index*8,(index+1)*8):
        expected.update(['runs/'+run_label+'/rank'+str(rank)+'.json','affinity/'+run_label+'/rank'+str(rank)+'.json'])
    require(set(sources)==expected, 'native source control file coverage differs')
    for rank in range(index*8,(index+1)*8):
        r=host['rank_reports'][str(rank)]
        sources['runs/'+run_label+'/rank'+str(rank)+'.events.bin']=dict(bytes=r['event_bytes'],sha256=r['event_sha256'])
    return sources


def make_catalog(index, host, progress, *, run_label=LABEL):
    files=expected_sources(index,host,run_label=run_label)
    expected={'runs/'+run_label+'/rank'+str(rank)+'.progress.jsonl' for rank in range(index*8,(index+1)*8)}
    require(set(progress)==expected, 'native progress coverage differs')
    require(all(0<v['bytes']<=4*2**20 for v in progress.values()), 'oversized/empty native progress file')
    files.update(progress)
    catalog=dict(schema=SCHEMA,files=[dict(path=n,**r) for n,r in sorted(files.items())])
    validate(catalog,MAX_FILE,MAX_HOST)
    return catalog


def remote(node, code, data=None):
    ast.parse(code)
    result=subprocess.run(['tsh','ssh','rock@'+node,shlex.join(['taskset','-c','8,9','python3','-c',code])],
                          input=data,capture_output=True,check=True,timeout=45)
    require(len(result.stdout)<=2**20 and len(result.stderr)<=2**20,'oversized control response')
    return json.loads(result.stdout) if result.stdout.strip() else None


def probe_code(index, expected, *, run_label=LABEL):
    node=LEADER if index is None else NODES[index]
    base=BUILD if index is None else BASE
    reserve=1280 if index is None else 128
    return f'''from pathlib import Path
import os,sys,json,hashlib,subprocess,time,resource,datetime
resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(30,30))
b=Path({base!r});assert os.uname().nodename=={node!r}
assert b.resolve().is_relative_to(Path({'/data/brick2' if index is None else '/home/rock'!r}))
assert Path({'/data/brick2' if index is None else '/'!r}).is_mount() and hasattr(os,'posix_fadvise')
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5);assert not units.strip(),units
free=os.statvfs(b).f_bavail*os.statvfs(b).f_frsize;assert free>({reserve}*2**30+{MAX_ALL if index is None else 2**20})
mem={{k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}};assert mem['MemAvailable']>=80*2**30
def ticks():return {{int(x[0][3:]):list(map(int,x[1:9])) for line in Path('/proc/stat').read_text().splitlines() if (x:=line.split())[0].startswith('cpu') and x[0][3:].isdigit()}}
a=ticks();time.sleep(1);z=ticks();busy={{}}
for c in [8,9]:
 d=[y-x for x,y in zip(a[c],z[c])];assert sum(d)>0;busy[c]=100*(1-(d[3]+d[4])/sum(d))
assert max(busy.values())<50 and sum(busy.values())/2<25,busy
def path(n):
 p=b/n;assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(b.resolve());return p
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while raw:=f.read(2**20):h.update(raw)
  os.posix_fadvise(f.fileno(),0,f.tell(),os.POSIX_FADV_DONTNEED)
 return h.hexdigest()
for n,row in {expected!r}.items():
 p=path(n);assert p.stat().st_size==row['bytes']
 if not n.endswith('.events.bin'):assert row['bytes']<=8*2**20 and sha(p)==row['sha256'],n
progress={{}}
if {index is not None!r}:
 names={{'rank'+str(r)+suffix for r in range({(index or 0)*8},{((index or 0)+1)*8}) for suffix in ['.json','.progress.jsonl','.events.bin']}}
 assert {{p.name for p in (b/'runs'/{run_label!r}).iterdir()}}==names
 for rank in range({(index or 0)*8},{((index or 0)+1)*8}):
  n='runs/'+{run_label!r}+'/rank'+str(rank)+'.progress.jsonl';p=path(n);assert 0<p.stat().st_size<=4*2**20
  progress[n]=dict(bytes=p.stat().st_size,sha256=sha(p))
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=free,available_memory_bytes=mem['MemAvailable'],cpu_busy_percent=busy,progress=progress,raw_recording_bytes_read=0,active_own_units=[])))'''


def stage(node, base, bundle, catalog=None, *, tag=TAG):
    code=f'''from pathlib import Path
import sys,json,hashlib
b=Path({base!r});p=b/{tag+'-source'!r};raw=json.loads(sys.stdin.read());modules=raw['modules']
assert set(modules)=={set(SOURCE_NAMES)!r}
if p.exists():
 assert {{q.name for q in p.iterdir()}}==set(modules)
 for n,s in modules.items():assert (p/n).read_text()==s
else:
 p.mkdir()
 for n,s in modules.items():(p/n).open('x').write(s)
if raw['catalog'] is not None:
 c=b/({tag!r}+'-controls');c.mkdir(exist_ok=True)
 (c/'expected.json').open('x').write(json.dumps(raw['catalog']))
print(json.dumps(dict(source_sha256={{n:hashlib.sha256(s.encode()).hexdigest() for n,s in modules.items()}})))'''
    return remote(node,code,json.dumps(dict(modules=bundle,catalog=catalog)).encode())


def guard_command(base, label, timeout, app, receiver, *, tag=TAG):
    volume='/data/brick2' if receiver else '/'
    command=['systemd-run','--expand-environment=no','--quiet','--wait','--pipe','--collect',
        '--unit=b2mpi-'+label,'--uid=rock','--service-type=exec','--property=MemoryMax=4096M',
        '--property=MemorySwapMax=0','--property=CPUQuota=200%','--property=AllowedCPUs=8-9',
        '--property=TasksMax=64','--property=RuntimeMaxSec='+str(timeout+5),'--property=TimeoutStopSec=5',
        '--property=KillMode=control-group','--property=OOMPolicy=continue',
        '/usr/bin/python3',base+'/'+tag+'-source/mpi_resource_guard.py',
        '--output',base+'/guards/'+label+'.json','--volume',volume,
        '--memory-mib','4096','--cpu-percent','200','--file-mib','3072',
        '--min-free-gib','1280' if receiver else '128','--timeout',str(timeout)]
    if not receiver:command.append('--allow-root-volume')
    return command+['--',*app]


def guarded(node, base, label, timeout, app, receiver, directory, *, tag=TAG):
    with (directory/(label+'.log')).open('x') as log:
        result=subprocess.run(['tsh','ssh','root@'+node,shlex.join(guard_command(base,label,timeout,app,receiver,tag=tag))],
                              stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=timeout+20)
    # Preserve guard evidence even when the child or SSH fails.
    g=remote(node,f"import json;from pathlib import Path;print(Path({base+'/guards/'+label+'.json'!r}).read_text())")
    (directory/(label+'-guard.json')).write_text(json.dumps(g,indent=2)+'\n')
    require(result.returncode==0 and g['admitted'] is True and type(g.get('returncode')) is int
            and g['returncode']==0 and not g.get('error'),'native transfer guard failed')
    require(g['host']==node and g['uid']==1000 and g['cgroup']=='/system.slice/b2mpi-'+label+'.service',
            'transfer guard identity differs')
    require(g['data_volume']==('/data/brick2' if receiver else '/')
            and (g['data_device']!=g['root_device'] if receiver else g['data_device']==g['root_device']),
            'transfer storage device differs')
    for phase in ['before','after']:
        facts=g[phase];events=counters(facts['memory.events'])
        require(all(events[k]==0 for k in ['max','oom','oom_kill','oom_group_kill'])
                and facts['memory.max']==str(4*2**30) and facts['memory.swap.max']=='0'
                and facts['pids.max']=='64' and facts['cpuset.cpus.effective']=='8-9', 'transfer resource pressure/limits')
        quota,period=map(int,facts['cpu.max'].split());require(period>0 and quota==2*period,'transfer CPU limit differs')
    reserve=(1280 if receiver else 128)*2**30
    require(g['minimum_free_bytes']==reserve and g['minimum_observed_free_bytes']>=reserve
            and g['file_limit_bytes']==3*2**30 and 0<g['wall_seconds']<=timeout,'transfer disk/time bounds')
    return g


def run(controls, outer_guard, output, *, reference_seed=None):
    label,tag,destination,_=reference_identity(reference_seed)
    start=time.monotonic();gate=control_gate(controls,outer_guard,reference_seed=reference_seed)
    if gate is None:return dict(ready=False,collection_started=False,reason='Native terminal controls and clean outer collection guard are required')
    require(Path('/Volumes/T7').is_mount() and output.resolve().is_relative_to(Path('/Volumes/T7').resolve()),'T7 output required')
    require(os.statvfs(output.parent).f_bavail*os.statvfs(output.parent).f_frsize>=128*2**30,'T7 reserve')
    previous=gate['previous_seconds'];remaining(start,previous)
    output.mkdir(parents=True,exist_ok=False)
    modules={n:(Path(__file__).parent/n).read_text() for n in SOURCE_NAMES}
    rows=[];total=0
    try:
        leader=remote(LEADER,probe_code(None,{}))
        (output/'leader-admission.json').write_text(json.dumps(leader,indent=2)+'\n')
        stage(LEADER,BUILD,modules,tag=tag)
        remote(LEADER,f"from pathlib import Path;Path({destination!r}).mkdir(exist_ok=False)")
        for index,host in enumerate(gate['hosts']):
            node=NODES[index];name='node'+node.rsplit('-',1)[-1]
            admitted=time.monotonic()
            source=remote(node,probe_code(index,expected_sources(index,host,run_label=label),run_label=label))
            leader=remote(LEADER,probe_code(None,{}))
            catalog=make_catalog(index,host,source['progress'],run_label=label);size,digest=validate(catalog,MAX_FILE,MAX_HOST)
            total+=size;require(total<=MAX_ALL,'native global collection exceeds cap')
            stage(node,BASE,modules,catalog,tag=tag)
            (output/(name+'-catalog.json')).write_text(json.dumps(catalog,indent=2)+'\n')
            (output/(name+'-admission.json')).write_text(json.dumps(dict(source=source,destination=leader),indent=2)+'\n')
            xp=destination+'/'+name+'-catalog.json';ready=destination+'/'+name+'-ready.json';receipt=destination+'/'+name+'-receipt.json'
            remote(LEADER,f"from pathlib import Path;import sys;Path({xp!r}).open('xb').write(sys.stdin.buffer.read())",json.dumps(catalog).encode())
            require(time.monotonic()-admitted<60,'native transfer admission stale')
            timeout=remaining(start,previous);node_start=time.monotonic()
            app=['python3',BUILD+'/'+tag+'-source/mam_collection_transfer.py','receive','--bind',LEADER_IP,
                '--peer',IPS[index],'--catalog',xp,'--output',destination+'/'+name,'--ready',ready,'--receipt',receipt,
                '--max-file-bytes',str(MAX_FILE),'--max-total-bytes',str(MAX_HOST),'--reserve-bytes',str(1280*2**30)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                receiver=pool.submit(guarded,LEADER,BUILD,tag+'-'+name+'-receive',timeout,app,True,output,tag=tag)
                deadline=time.monotonic()+45
                while True:
                    if receiver.done():
                        receiver.result()
                        raise RuntimeError('native receiver ended before readiness')
                    control=remote(LEADER,f"from pathlib import Path;import json;p=Path({ready!r});print(p.read_text() if p.exists() else '{{}}')")
                    if control:break
                    require(time.monotonic()<deadline,'native receiver readiness timeout')
                    time.sleep(.5)
                control_path=BASE+'/'+tag+'-controls/ready.json'
                remote(node,f"import os,sys;fd=os.open({control_path!r},os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)\nwith os.fdopen(fd,'wb') as f:f.write(sys.stdin.buffer.read())",json.dumps(control).encode())
                send_timeout=min(remaining(start,previous),math.floor(timeout-(time.monotonic()-node_start)-10))
                require(send_timeout>30,'native sender has insufficient remaining deadline')
                send=['python3',BASE+'/'+tag+'-source/mam_collection_transfer.py','send','--root',BASE,
                      '--catalog',BASE+'/'+tag+'-controls/expected.json','--ready',control_path]
                errors=[]
                try:send_guard=guarded(node,BASE,tag+'-'+name+'-send',send_timeout,send,False,output,tag=tag)
                except Exception as exc:errors.append('sender: '+str(exc))
                try:receive_guard=receiver.result(timeout=max(1,timeout-(time.monotonic()-node_start)+20))
                except Exception as exc:errors.append('receiver: '+str(exc))
                require(not errors, '; '.join(errors))
            result=remote(LEADER,f"from pathlib import Path;print(Path({receipt!r}).read_text())")
            require(result['complete'] is True and result['catalog_sha256']==digest and result['bytes']==size
                    and result['files']==len(catalog['files']) and result['file_cache_release_supported'] is True
                    and result['source_archive_created'] is False,'native transfer receipt differs')
            (output/(name+'-receipt.json')).write_text(json.dumps(result,indent=2)+'\n')
            rows.append(dict(host=node,catalog_sha256=digest,bytes=size,files=len(catalog['files']),
                             send_wall_seconds=send_guard['wall_seconds'],receive_wall_seconds=receive_guard['wall_seconds']))
            print(json.dumps(dict(event='native_node_collected',**rows[-1])),flush=True)
        remaining(start,previous)
        report=dict(schema='b2-mam-native-primary-collection-v1',ready=True,collection_started=True,complete=True,
            nodes=rows,bytes=total,destination=destination,resource_report_sha256=gate['resource_report_sha256'],
            control_outer_guard_sha256=gate['control_outer_guard_sha256'],previous_collection_seconds=previous,
            bulk_collection_seconds=time.monotonic()-start,shared_collection_budget_seconds=7200,
            source_archive_created=False,automatic_retry=False,raw_format_prefix_audit_passed=False,
            scientific_acceptance=False,independent_device_backup=False,
            source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in modules.items()})
        if reference_seed is not None:
            from mam_launch_native_full_reference import PROTOCOL_SHA
            report.update(label=label,seed=reference_seed,protocol_sha256=PROTOCOL_SHA,
                          transfer_tag=tag,source_project=PROJECT)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps(dict(error=str(exc),previous_collection_seconds=previous,
            bulk_collection_seconds=time.monotonic()-start,completed_nodes=rows,automatic_retry=False),indent=2)+'\n')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controls',type=Path,required=True);p.add_argument('--control-guard',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-seed',type=int,choices=[1730,1731]);args=p.parse_args()
    result=run(args.controls,args.control_guard,args.output,reference_seed=args.reference_seed);print(json.dumps(result))
    raise SystemExit(0 if result['ready'] else 2)
