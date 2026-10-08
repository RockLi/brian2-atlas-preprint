"""Prepare and launch the one frozen Rust target after accepted NEST output.

No deployment, admission, output directory or network call occurs while the
NEST terminal/raw prerequisite is absent. Simulations are never retried.
"""
import argparse
import ast
import base64
import concurrent.futures
import json
import math
import os
from pathlib import Path
import shlex
import time

from mam_launch_native_primary import read,sha,check,remote
from mam_launch_performance_tuning import write,PROTOCOL_SHA
from mam_primary_resources import CPUS,MODEL,PLAN,NODES
from mam_rust_terminal_sync import EXE_SHA
from mam_rust_cpu_placement import ALTERNATE_CPUS,revision,candidate_topologies,previous_window_gate

BASE='/atlas-home/0003/workspace/brian2-mpi-primary-20260909'
BRICK='/data/brick2/brian2-mpi-region-20260907/primary-host-v1'
OLD='full32-n1-k1-spool-primary-v1-100500ms'
LABEL='rust-mam-perf-v1-seed1729-100500ms'
SOURCE='mam-rust-performance-v1-source-20260911'
CASE='rust32-target'
IPS=['192.168.20.23','192.168.30.81','192.168.30.83','192.168.30.71']
PACKAGE_SHA='c5a84eb948949df4820d94529c9eceb71b788c3e1266d6affe76e535f563f552'
SYNC_COMPLETION_SHA='0414679d47bcaee667df47d7d19dcbf55be0918473bb54562f18ede73f7b143c'
DEFERRED_SHA='33e27deea91875c53afcd6478ad8e387f8b566a3f333c0f117fa921fa8fa32c9'
RECOVERY_DEFERRED_SHA='253c7d782355075127ac2c530630a90fac6f5264af06d26b205e5fe8f0f37273'


def deferred_gate(out,options,catalog,prerequisite,*,resource_window=False,alternate_cpus=False):
    """Only resume the pinned CPU preflight deferral, never a launched run."""
    check(out.is_dir() and not out.is_symlink(),'deferred case required')
    check(not (out/'admission.json').exists() and not (out/'launch.json').exists()
          and not options['output'].exists(),'admitted or launched target cannot resume')
    if resource_window or alternate_cpus:
        if alternate_cpus:previous_window_gate(out)
        else:check(not (out/'resource-window-v2').exists(),'resource window already attempted')
        check(sha(out/'recovery-deferred.json')==RECOVERY_DEFERRED_SHA,'recovery evidence changed')
        recovery=read(out/'recovery-deferred.json')
        check(recovery['recovery_attempt_terminal'] is True and recovery['admission_created'] is False
              and recovery['neural_runs_started']==0 and recovery['launch_started'] is False,
              'only terminal pre-admission recovery can enter window')
        expected={'resume-preflight-v1/'+name for name in
                  ['failure.json','intent.json']+[f'preflight-{i}.json' for i in range(7)]}
        check(set(recovery['input_sha256'])==expected,'recovery input coverage')
        for name,digest in recovery['input_sha256'].items():check(sha(out/name)==digest,'recovery input changed')
    else:
        check(not (out/'resume-preflight-v1').exists(),'one explicit preflight recovery only')
    deferred=out/'preflight-deferred.json'
    check(sha(deferred)==DEFERRED_SHA,'deferred evidence changed')
    proof=read(deferred)
    check(proof['neural_runs_started']==0 and proof['launch_started'] is False
          and proof['source_staging_complete'] is True
          and proof['nest_completion_sha256']==prerequisite['completion_sha256'],
          'only unlaunched CPU deferral can resume')
    check(sha(out/'cpu-availability.json')==proof['cpu_availability_sha256'],'CPU evidence changed')
    for name,digest in proof['diagnostics'].items():
        check(Path(name).parts[0]=='preflight-diagnosis' and len(Path(name).parts)==2,
              'bounded diagnostic path required')
        check(sha(out/name)==digest,'preflight diagnostic changed')
    check(read(out/'failure.json')['error_type']=='CalledProcessError','wrong failure stage')
    intent=read(out/'intent.json')
    expected={k:str(v) if isinstance(v,Path) else v for k,v in options.items()}
    check(intent['case_id']==CASE and intent['protocol_sha256']==PROTOCOL_SHA
          and intent['nest_prerequisite']==prerequisite and intent['source_catalog']==catalog
          and intent['launch_options']==expected and intent['maximum_neural_runs']==1
          and intent['automatic_retry'] is False,'deferred launch contract changed')
    staged=[read(out/f'stage-{i}.json') for i in range(4)]
    for i,row in enumerate(staged):
        check(row['host']==NODES[i] and row['source_catalog']==catalog
              and row['neural_simulations']==0
              and row['base_resolved']==(BRICK if i==0 else BASE),'staged source receipt changed')
    return staged


def nest_gate(evidence):
    case=evidence/'performance-runs-v1/nest-selected-target'
    if not (case/'completion.json').is_file():return None
    completion=read(case/'completion.json')
    check(completion['protocol_sha256']==PROTOCOL_SHA and completion['case_id']=='nest-selected-target'
          and completion['terminal_resource_audit_passed'] is True
          and completion['raw_output_audit_passed'] is True,'NEST target not accepted')
    check(set(completion['input_sha256'])=={'admission.json','launch.json','terminal/report.json','raw/report.json','raw/intent.json'},
          'NEST completion input coverage')
    for name,digest in completion['input_sha256'].items():check(sha(case/name)==digest,'NEST target input changed')
    check(sha(case/'admission.json')=='3b9af4c512babb202ccd0dc55096e11d3f2c237fe99a34390f153f9a4dfc5a0e',
          'wrong NEST target admission')
    terminal=read(case/'terminal/report.json')
    raw=read(case/'raw/report.json')
    check(terminal['duration_ms']==raw['duration_ms']==100500
          and len(terminal['ranks'])==48
          and [r['rank'] for host in raw['hosts'] for r in host['ranks']]==list(range(48))
          and all(len(r['physical_50ms_bin_counts'])==2010 for host in raw['hosts'] for r in host['ranks']),
          'full NEST time/rank coverage')
    for item in terminal['collection_files']:check(sha(case/'terminal'/item['file'])==item['sha256'],'NEST source changed')
    return dict(completion_sha256=sha(case/'completion.json'),terminal_sha256=sha(case/'terminal/report.json'),
                raw_sha256=sha(case/'raw/report.json'))


def launch_options(output,*,cpu_ids=CPUS,source=SOURCE,label=LABEL):
    project=BASE+'/'+OLD+'/mpi'
    wrapper=[ '/usr/bin/python3',BASE+'/'+source+'/mam_rust_terminal_sync.py',
             '--executable',project+'/b2-mpi','--instance',project+'/instance.bin',
             '--output',BRICK+'/runs/'+label,'--receipt-directory',BASE+'/performance-receipts/'+label]
    command='exec env PYTHONDONTWRITEBYTECODE=1 B2_THREAD_AFFINITY=required B2_MPI_MAX_LOCAL_EDGES=1024000000 B2_MPI_MAX_PREBUILT_TOPOLOGY_BYTES=134217728 /usr/bin/time -v -o '
    command+=shlex.quote(BASE+'/metrics/'+label+'-rank')+'"${PMI_RANK}.time" '+shlex.join(wrapper)
    return dict(nodes=NODES,ips=IPS,ranks_per_node=8,remote_base=BASE,application=['sh','-c',command],
        output=output,mpi_prefix=BASE+'/mpi',timeout=64800,login='root',
        guard_script=BASE+'/'+source+'/mpi_resource_guard.py',guard_volume='/',guard_allow_root_volume=True,
        guard_memory_mib=262144,guard_cpu_percent=800,guard_cpu_count=8,guard_cpu_ids=cpu_ids,
        guard_file_mib=65504,guard_min_free_gib=128,
        guard_node_overrides={NODES[0]:dict(volume='/data/brick2',remote_base=BRICK,
                                         script=BRICK+'/'+source+'/mam_rust_guard23.py')})


def stage_code(index,payload,catalog,*,source=SOURCE):
    return f'''from pathlib import Path
import os,json,hashlib,base64,subprocess,resource,signal
signal.alarm(30);resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20));resource.setrlimit(resource.RLIMIT_CPU,(15,15))
assert os.uname().nodename=={NODES[index]!r}
b=Path({BASE!r});leader={index==0!r}
assert b.resolve()==Path({BRICK!r}) if leader else b.resolve().is_relative_to(Path('/home/rock'))
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5);assert not units.strip(),units
s=os.statvfs(b);assert s.f_bavail*s.f_frsize>=(2048 if leader else 192)*2**30
p=b/{source!r};p.mkdir();actual={{}}
for name,encoded in {payload!r}.items():
 raw=base64.b64decode(encoded);assert len(raw)<=65536
 with (p/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 actual[name]=dict(bytes=len(raw),sha256=hashlib.sha256((p/name).read_bytes()).hexdigest())
assert actual=={catalog!r}
(b/'performance-receipts').mkdir(exist_ok=True)
assert (b/'performance-receipts').resolve().is_relative_to(b.resolve()) and (b/'metrics').is_dir() and (not leader or (b/'runs').is_dir())
for q in [p,b/'performance-receipts',b]:
 fd=os.open(q,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(host=os.uname().nodename,source_catalog=actual,base_resolved=str(b.resolve()),neural_simulations=0)))'''


def preflight_code(index,package,catalog,topology,*,cpu_ids=CPUS,source=SOURCE,label=LABEL):
    return f'''from pathlib import Path
import os,json,hashlib,subprocess,time,datetime,resource,signal
signal.alarm(40);resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(30,30))
assert os.uname().nodename=={NODES[index]!r}
b=Path({BASE!r});p=b/{OLD!r};leader={index==0!r};v=Path('/data/brick2') if leader else Path('/')
assert b.resolve()==Path({BRICK!r}) if leader else b.resolve().is_relative_to(Path('/home/rock'))
assert v.is_mount();s=os.statvfs(v);free=s.f_bavail*s.f_frsize;assert free>=(2048 if leader else 192)*2**30
m={{k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in ['MemTotal','MemAvailable']}};assert m['MemAvailable']>=(576 if leader else 320)*2**30
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5);assert not units.strip(),units
assert not (b/'performance-receipts'/{label!r}).exists()
assert not list((b/'metrics').glob({label+'-rank*.time'!r}))
assert not (Path({BRICK!r})/'runs'/{label!r}).exists()
selected=[line.split(',') for line in subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET,NODE,ONLINE'],text=True).splitlines() if not line.startswith('#') and int(line.split(',')[0]) in {cpu_ids!r}]
assert selected=={topology!r}
def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for name,item in {package['catalog']!r}.items():
 path=p/name;assert path.resolve().is_relative_to(p.resolve()) and not path.is_symlink()
 assert path.stat().st_size==item['bytes'] and digest(path)==item['sha256'],name
for name,expected in {package['mpi_runtime']!r}.items():
 path=b/'mpi'/name;assert path.resolve().is_relative_to((b/'mpi').resolve()) and digest(path)==expected
for name,item in {catalog!r}.items():
 path=b/{source!r}/name;assert not path.is_symlink() and path.stat().st_size==item['bytes'] and digest(path)==item['sha256']
if leader:assert digest(Path('/data/brick2/brian2-mpi-region-20260907')/{OLD!r}/'model.json')=={MODEL!r}
def ticks():return {{int(x[0][3:]):list(map(int,x[1:9])) for line in Path('/proc/stat').read_text().splitlines() if (x:=line.split())[0].startswith('cpu') and x[0][3:].isdigit()}}
a=ticks();time.sleep(1);z=ticks();busy={{}}
for c in {cpu_ids!r}:
 d=[y-x for x,y in zip(a[c],z[c])];assert sum(d)>0;busy[c]=100*(1-(d[3]+d[4])/sum(d))
assert max(busy.values())<50 and sum(busy.values())/8<25,busy
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),volume=str(v),free_bytes=free,memory=m,selected_cpu_topology=selected,selected_cpu_busy_percent=busy,active_units=[],all_artifact_hashes_verified=True,source_catalog_verified=True)))'''


def cpu_busy_error(error):
    """Only a numeric CPU occupancy assertion can be re-observed in a window."""
    if error.get('error_type')!='CalledProcessError':return False
    for line in error.get('stderr','').splitlines():
        if not line.startswith('AssertionError: {'):continue
        try:busy=ast.literal_eval(line.removeprefix('AssertionError: '))
        except (ValueError,SyntaxError):return False
        return (isinstance(busy,dict) and set(busy)==set(CPUS)
            and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=100 for v in busy.values())
            and (max(busy.values())>=50 or sum(busy.values())/8>=25))
    return False


def perform_preflight(attempt,package,catalog,old_admission,protocol,*,cpu_ids=CPUS,source=SOURCE,label=LABEL):
    start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        futures=[pool.submit(remote,node,preflight_code(i,package,catalog,old_admission['preflight'][i]['selected_cpu_topology'],cpu_ids=cpu_ids,source=source,label=label)) for i,node in enumerate(NODES)]
        extra_nodes=[n for n in protocol['native']['nodes'] if n not in NODES]
        absent_code="import subprocess,json;u=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5);assert not u.strip(),u;print(json.dumps({'active_units':[]}))"
        extras=[pool.submit(remote,node,absent_code) for node in extra_nodes]
        rows=[];extra=[];errors=[]
        for index,(node,future) in enumerate(zip(NODES+extra_nodes,futures+extras,strict=True)):
            try:
                row=future.result()
                (rows if index<len(NODES) else extra).append(row if index<len(NODES) else dict(host=node,**row))
                write(attempt/f'preflight-{index}.json',dict(host=node,passed=True,result=row))
            except Exception as error:
                detail=getattr(error,'stderr',b'') or b''
                if isinstance(detail,bytes):detail=detail.decode(errors='replace')
                errors.append(dict(host=node,error_type=type(error).__name__,stderr=detail[-16384:]))
                write(attempt/f'preflight-{index}.json',dict(passed=False,**errors[-1]))
    return rows,extra,errors,start


def bounded_window(attempt,perform,*,clock=time.monotonic,sleeper=time.sleep):
    began=clock()
    for index in range(3):
        check(clock()-began<540,'resource window lacks time for a fresh check')
        directory=attempt/f'check-{index+1}';directory.mkdir(exist_ok=False)
        rows,extra,errors,start=perform(directory)
        check(clock()-began<600,'resource window elapsed budget')
        if not errors:
            check(clock()-start<60,'preflight stale within resource window')
            return rows,extra,start
        check(all(cpu_busy_error(error) for error in errors),'non-CPU preflight failure; no repeated check')
        if index<2:
            print(json.dumps(dict(event='resource_window_cpu_busy',check=index+1,next_check_after_seconds=30)),flush=True)
            sleeper(30)
    raise ValueError('resource window exhausted its three checks; no model launched')


def run(args):
    prerequisite=nest_gate(args.evidence)
    if prerequisite is None:
        return dict(ready=False,launch_started=False,reason='NEST target terminal/resource/raw acceptance is pending')
    protocol_path=args.evidence/'performance-protocol-v1/protocol.json'
    check(sha(protocol_path)==PROTOCOL_SHA,'protocol changed')
    protocol=read(protocol_path);case=next(c for c in protocol['cases'] if c['id']==CASE)
    check(case['duration_ms']==100500 and case['ranks']==32 and case['threads']==1
          and case['wall_seconds']==64800 and case['total_final_output_bytes']==128*2**30,'Rust target changed')
    sync=args.evidence/'rust-terminal-sync-v1/completion.json'
    check(sha(sync)==SYNC_COMPLETION_SHA,'sync validation changed')
    root=Path(__file__).resolve().parents[1]
    for name,item in read(sync)['files'].items():check(sha(root/name)==item['sha256'],'sync proof input changed')
    package_path=args.evidence/'primary-run/package.json';check(sha(package_path)==PACKAGE_SHA,'artifact package changed')
    package=read(package_path);check(package['build']['executable_sha256']==EXE_SHA,'executable changed')
    old_admission=read(args.evidence/'primary-run/admission.json')
    files=[Path(__file__).with_name(n) for n in ['mam_rust_terminal_sync.py','mpi_resource_guard.py','mam_rust_guard23.py']]+[protocol_path]
    raw={p.name:p.read_bytes() for p in files}
    import hashlib
    catalog={n:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for n,data in raw.items()}
    payload={n:base64.b64encode(data).decode() for n,data in raw.items()}
    check(Path('/Volumes/T7').is_mount() and args.t7.resolve().is_relative_to(Path('/Volumes/T7')),'T7 required')
    s=os.statvfs(args.t7);check(s.f_bavail*s.f_frsize>=512*2**30,'T7 start reserve')
    out=args.evidence/'performance-runs-v1'/CASE
    alternate=getattr(args,'alternate_cpus_v3',False)
    cpu_ids=ALTERNATE_CPUS if alternate else CPUS
    options=launch_options(args.t7/'logs'/CASE,cpu_ids=cpu_ids);check(not options['output'].exists(),'Rust logs already exist')
    window=getattr(args,'resource_window_v2',False)
    check(sum(map(bool,[window,alternate,getattr(args,'resume_preflight',False)]))<=1,'choose one recovery mode')
    resume=getattr(args,'resume_preflight',False) or window or alternate
    if resume:
        staged=deferred_gate(out,launch_options(options['output']),catalog,prerequisite,
                             resource_window=window,alternate_cpus=alternate)
        if alternate:
            old_admission=dict(preflight=[dict(selected_cpu_topology=t) for t in candidate_topologies(args.evidence)])
        attempt=out/('alternate-cpus-v3' if alternate else 'resource-window-v2' if window else 'resume-preflight-v1');attempt.mkdir(exist_ok=False)
        write(attempt/'intent.json',dict(deferred_sha256=DEFERRED_SHA,
            recovery_deferred_sha256=RECOVERY_DEFERRED_SHA if window else None,neural_runs_before=0,
            maximum_neural_runs=1,automatic_retry=False,restage_source=False,
            maximum_preflight_checks=3 if window else 1,maximum_window_seconds=600 if window else 60,
            repeated_checks_only_for_cpu_occupancy=window,cpu_thresholds_unchanged=True,
            launcher_sha256=sha(Path(__file__)),nest_prerequisite=prerequisite,
            cpu_placement_revision=revision() if alternate else None))
    else:
        out.mkdir(exist_ok=False);attempt=out;staged=[]
        write(out/'intent.json',dict(case_id=CASE,protocol_sha256=PROTOCOL_SHA,nest_prerequisite=prerequisite,
            maximum_neural_runs=1,automatic_retry=False,source_catalog=catalog,
            launch_options={k:str(v) if isinstance(v,Path) else v for k,v in options.items()}))
    try:
        if not resume:
            for i,node in enumerate(NODES):
                result=remote(node,stage_code(i,payload,catalog));staged.append(result);write(out/f'stage-{i}.json',result)
        perform=lambda directory:perform_preflight(directory,package,catalog,old_admission,protocol,cpu_ids=cpu_ids)
        if window:
            rows,extra,start=bounded_window(attempt,perform)
        else:
            rows,extra,errors,start=perform(attempt)
            check(not errors,'preflight failed; bounded remote error details retained')
        check(time.monotonic()-start<60,'preflight stale')
        admission=dict(schema='b2-mam-rust-performance-admission-v1',case_id=CASE,label=LABEL,admitted=True,
            protocol_sha256=PROTOCOL_SHA,nest_prerequisite=prerequisite,model_sha256=MODEL,plan_sha256=PLAN,
            executable_sha256=EXE_SHA,package_sha256=PACKAGE_SHA,source_catalog=catalog,
            identity=dict(ranks=32,threads=1,seed=1729,duration_ms=100500,dt_ms=.1),
            preflight=rows,other_native_hosts=extra,source_staging=staged,
            resources=dict(nodes=NODES,ranks_per_node=8,cpu_ids=cpu_ids,memory_bytes_per_service=256*2**30,
                cpu_quota_cores_per_service=8,zero_swap=True,pids_max=64,file_limit_bytes=65504*2**20,
                minimum_free_bytes_by_host={n:(1280 if i==0 else 128)*2**30 for i,n in enumerate(NODES)},
                total_final_output_bytes=128*2**30,spool_bytes=96*2**30,wall_seconds=64800),
            output=BRICK+'/runs/'+LABEL,receipt_directory=BASE+'/performance-receipts/'+LABEL,
            launch_options={k:str(v) if isinstance(v,Path) else v for k,v in options.items()},
            launcher_sha256=sha(Path(__file__)),scientific_acceptance=False,performance_cost_acceptance=False)
        if alternate:admission['cpu_placement_revision']=revision()
        write(out/'admission.json',admission)
        write(args.t7/'artifacts/performance-protocol-v1'/(CASE+'-admission.json'),admission)
        check(time.monotonic()-start<60,'preflight expired before launch')
        print(json.dumps(dict(event='rust_target_admitted',case_id=CASE,wall_seconds=64800,admission_sha256=sha(out/'admission.json'))),flush=True)
        from mpi_teleport_launch import launch
        result=launch(**options);write(out/'launch.json',result)
        return dict(ready=True,launch_started=True,launcher_terminal_success=True,all_output_and_resource_audits_required=True)
    except BaseException as error:
        if (options['output']/'launch.json').exists() and not (out/'launch.json').exists():write(out/'launch.json',read(options['output']/'launch.json'))
        write(attempt/'failure.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--t7',type=Path,required=True)
    recovery=p.add_mutually_exclusive_group()
    recovery.add_argument('--resume-preflight',action='store_true',help='Once only, resume the pinned CPU deferral before any admission or model launch')
    recovery.add_argument('--resource-window-v2',action='store_true',help='One pinned 600-second availability window, at most three CPU-only rechecks; never retry a simulation')
    recovery.add_argument('--alternate-cpus-v3',action='store_true',help='One fresh preflight and launch on the diagnosed, same-NUMA alternative CPUs; no retries')
    result=run(p.parse_args());print(json.dumps(result));raise SystemExit(0 if result['ready'] else 2)
