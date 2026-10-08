"""One resource-bounded 1750 replication; no automatic retry or batch launch.

This acquires a new descriptive scientific sample. Statistical equivalence,
publication reproduction and performance/cost acceptance require later evidence.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import time
from mam_confirmation_identity import identity,deployment_catalog,NODES,IPS,HOME,BRICK,BUILD,LABEL
from mam_confirmation_terminal_sync import IDENTITY_SHA,parse,require
from mam_confirmation_profile import (REPLICATE,selected,IDENTITY_SHA_1751,READINESS_SHA_1751,DEPLOYMENT_SHA_1751,TOPOLOGY_SHA_1751)
IDENTITY_SHA=selected(IDENTITY_SHA,IDENTITY_SHA_1751)
from mam_collect_native_primary_raw import remote
from mam_rust_cpu_placement import ALTERNATE_CPUS,candidate_topologies
from mpi_teleport_launch import launch

CASE=f'confirmation-run-v1-seed{REPLICATE}'
LABEL=f'rust-mam-confirmation-v1-replicate{REPLICATE}-100500ms'
SOURCE=f'mam-confirmation-run-v1-seed{REPLICATE}-source'
READINESS='f1534dfb6c3f0c47fe1e001631da99ca85b38e740f8cc3d6dd08c6b4d0d059c5'
READINESS=selected(READINESS,READINESS_SHA_1751)
DEPLOYMENT='fca401e32a2011f4f0376a761c307013416c0c042ef7cc655df43e53a5afb843'
DEPLOYMENT=selected(DEPLOYMENT,DEPLOYMENT_SHA_1751)
TOPOLOGY='a5c25b2fc8ff6a46f5bee3c14e8067136cf1e0a2fe9edec1faada0174c451874'
TOPOLOGY=selected(TOPOLOGY,TOPOLOGY_SHA_1751)
ESTIMATORS='8f45dce3c1863371e32b6e00b863ed6dfbf3ffc80e3032ceada6ceb0a775b18b'
PACKAGE='c5a84eb948949df4820d94529c9eceb71b788c3e1266d6affe76e535f563f552'
BACKUP_NODE=selected('hk-prod-model-ae02-24','hk-prod-model-ae08-82')
BACKUP='/atlas-home/0003/backups/brian2-mpi/'+CASE
LIMIT=64800


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p,digest=None):
    require(p.is_file() and not p.is_symlink() and p.stat().st_size<2**20,'bounded control required')
    if digest:require(sha(p)==digest,'control changed: '+str(p))
    return parse(p.read_bytes())


def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def prerequisite(evidence):
    require(READINESS is not None,'selected readiness has not been pinned')
    v=identity(evidence,REPLICATE)
    source=evidence/f'confirmation-topology-v1-seed{REPLICATE}/identity.json'
    require(read(source,IDENTITY_SHA)==v,'new identity differs')
    r=read(evidence/f'confirmation-readiness-v1-seed{REPLICATE}/report.json',READINESS)
    require(r['identity_sha256']==IDENTITY_SHA and r['neural_runs']==0 and not r['launch_admitted'],'readiness identity')
    for folder,digest in [(selected('confirmation-deployment-recovery-v1-seed1750','confirmation-deployment-v1-seed1751'),DEPLOYMENT),
                          (f'confirmation-topology-v1-seed{REPLICATE}',TOPOLOGY)]:
        p=evidence/folder;c=read(p/'complete.json',digest)
        require(c['complete'] and c['neural_runs']==0,'prerequisite incomplete')
        for name,h in c['evidence_sha256'].items():
            q=p/name;require(q.resolve().is_relative_to(p.resolve()) and q.is_file()
                and q.stat().st_size<2**20 and sha(q)==h,'prerequisite evidence changed')
    d=read(evidence/(selected('confirmation-deployment-recovery-v1-seed1750','confirmation-deployment-v1-seed1751')+'/complete.json'))
    require(d['all_32_owned_shards_deployed'] and len(d['nodes'])==4,'deployment incomplete')
    for index,row in enumerate(d['nodes']):
        require(row['host']==NODES[index] and row['full_readback'] and row['owned_ranks']==list(range(index*8,index*8+8))
                and row['project']==v['project'],'deployment rank coverage')
    t=read(evidence/f'confirmation-topology-v1-seed{REPLICATE}/pending.json')
    for key in ['model_sha256','instance_sha256','plan_sha256']:
        require(t[key]==v[key],'topology identity differs')
    require(t['exact_target_histograms'] and t['full_shared_projection_coverage']
            and t['counts_sha256']==r['counts_sha256'],'topology audit incomplete')
    read(evidence/'native-full-reference-v1/protocol.json',ESTIMATORS)
    package=read(evidence/'primary-run/package.json',PACKAGE)
    return v,source.read_bytes(),r,package['mpi_runtime']


def protocol(v,r):
    return dict(schema='b2-mam-single-replication-protocol-v1',replicate=REPLICATE,label=LABEL,
        purpose='One new independent random-input realization, fixed descriptive estimators; equivalence decision unresolved.',
        maximum_neural_runs=1,automatic_retry=False,replacement_seed_allowed=False,
        identity_sha256=IDENTITY_SHA,readiness_sha256=READINESS,model_sha256=v['model_sha256'],
        plan_sha256=v['plan_sha256'],instance_sha256=v['instance_sha256'],
        replicate_is_not_runtime_rng_key=True,
        duration_ms=100500,transient_ms=500,observation_ms=100000,dt_ms=.1,
        neurons=4129924,synapses=24126516728,populations=254,areas=32,
        estimators_protocol_sha256=ESTIMATORS,
        endpoints=['population firing rates','LvR','within-population pair correlation','area PSD','functional connectivity','interarea propagation lags'],
        estimator_changes_allowed=False,formal_equivalence_margin=None,formal_sample_size=None,
        prior_confirmation_design_five_by_five_batch_admitted=False,
        unresolved_science=['scientifically justified equivalence margins and primary inferential endpoints',
            'LvR historical denominator and V1 subsampling comparability','state occupancy/switching primary estimator'],
        identity_paths={k:v[k] for k in ['output','receipts','metrics','analysis','raw_audit']},
        full_raw_histogram=dict(path=BUILD+f'/confirmation-topology-v1-seed{REPLICATE}/exact-counts.jsonl',
            sha256=r['counts_sha256'],topology_completion_sha256=TOPOLOGY),
        budgets=dict(simulation_seconds=LIMIT,full_raw_audit_seconds=7200,
            scientific_analysis_seconds=10800,backup_seconds=7200,controls_seconds=1200,total_seconds=91200,
            maximum_full_raw_audits=1,maximum_six_metric_analyses=1,maximum_raw_backups=1,
            stage_budget_exhaustion='Stop that path and report; no automatic extension or rerun'),
        resources=dict(nodes=NODES,ranks_per_node=8,ranks=32,threads_per_rank=1,cpu_ids=ALTERNATE_CPUS,
            memory_bytes_per_service=256*2**30,cpu_quota_cores_per_service=8,zero_swap=True,pids_max=64,
            file_limit_bytes=65504*2**20,total_final_output_bytes=128*2**30,spool_bytes=96*2**30,
            start_available_memory_bytes_by_host={n:(576 if i==0 else 320)*2**30 for i,n in enumerate(NODES)},
            start_free_bytes_by_host={n:(2048 if i==0 else 192)*2**30 for i,n in enumerate(NODES)},
            minimum_free_bytes_by_host={n:(1280 if i==0 else 128)*2**30 for i,n in enumerate(NODES)}),
        raw_backup=dict(node=BACKUP_NODE,path=BACKUP,maximum_bytes=128*2**30,
            start_free_bytes=272*2**30,minimum_free_bytes=128*2**30,control_allowance_bytes=16*2**30,
            different_host=True,physical_failure_domain_independence_verified=False,
            fresh_readmission_required_at_copy=True),
        performance_comparison='Report descriptive wall time, measured and allocated core-hours, shared node-hours and peaks; no equal-core or money claim.',
        scientific_acceptance=False,performance_cost_acceptance=False,overall_goal_complete=False)


def launch_options(v,t7):
    wrapper=['/usr/bin/python3',HOME+'/'+SOURCE+'/mam_confirmation_terminal_sync.py',
             '--identity',HOME+'/'+SOURCE+'/identity.json']
    command='exec env PYTHONDONTWRITEBYTECODE=1 B2_THREAD_AFFINITY=required B2_MPI_MAX_LOCAL_EDGES=1024000000 B2_MPI_MAX_PREBUILT_TOPOLOGY_BYTES=134217728 /usr/bin/time -v -o '
    command+=shlex.quote(v['metrics']+'-rank')+'"${PMI_RANK}.time" '+shlex.join(wrapper)
    return dict(nodes=NODES,ips=IPS,ranks_per_node=8,remote_base=HOME,application=['sh','-c',command],
        output=t7/'logs'/CASE,mpi_prefix=HOME+'/mpi',timeout=LIMIT,login='root',
        guard_script=HOME+'/'+SOURCE+'/mpi_resource_guard.py',guard_volume='/',guard_allow_root_volume=True,
        guard_memory_mib=262144,guard_cpu_percent=800,guard_cpu_count=8,guard_cpu_ids=ALTERNATE_CPUS,
        guard_file_mib=65504,guard_min_free_gib=128,
        guard_node_overrides={NODES[0]:dict(volume='/data/brick2',remote_base=BRICK,
            script=BRICK+'/'+SOURCE+'/mam_rust_guard23.py')})


def stage(index,bundle,catalog):
    code=f'''from pathlib import Path
import json,sys,os,hashlib,signal,resource
signal.alarm(25);resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20));resource.setrlimit(resource.RLIMIT_CPU,(15,15))
assert os.uname().nodename=={NODES[index]!r}
b=Path({HOME!r});assert b.resolve()==Path({BRICK!r}) if {index==0!r} else b.resolve().is_relative_to(Path('/home/rock'))
s=os.statvfs(b);assert s.f_bavail*s.f_frsize>=(2048 if {index==0!r} else 192)*2**30
r=json.loads(sys.stdin.read(2**20));assert len(r)==5
p=b/{SOURCE!r};p.mkdir();actual={{}}
for name,text in r.items():
 assert Path(name).name==name and len(text.encode())<2**20
 with (p/name).open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
 actual[name]=dict(bytes=(p/name).stat().st_size,sha256=hashlib.sha256((p/name).read_bytes()).hexdigest())
assert actual=={catalog!r}
for name in ['confirmation-receipts','confirmation-metrics']:
 q=b/name;q.mkdir(exist_ok=True);assert q.resolve().is_relative_to(b.resolve())
for q in [p,b/'confirmation-receipts',b/'confirmation-metrics',b]:
 fd=os.open(q,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(host=os.uname().nodename,source_catalog=actual,base_resolved=str(b.resolve()),neural_runs=0)))'''
    return remote(NODES[index],code,json.dumps(bundle).encode())


def preflight_code(index,v,catalog,mpi_runtime,topology,counts_sha):
    owned={x['path']:{k:x[k] for k in ['bytes','sha256']} for x in deployment_catalog(v,index)['files']}
    return f'''from pathlib import Path
import os,json,hashlib,subprocess,time,datetime,resource,signal
signal.alarm(40);resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(30,30))
assert os.uname().nodename=={NODES[index]!r}
b=Path({HOME!r});p=Path({v['project']!r});leader={index==0!r};volume=Path('/data/brick2') if leader else Path('/')
assert b.resolve()==Path({BRICK!r}) if leader else b.resolve().is_relative_to(Path('/home/rock'))
assert p.resolve().is_relative_to(b.resolve()) and volume.is_mount()
if leader:assert volume.stat().st_dev!=Path('/').stat().st_dev
s=os.statvfs(volume);free=s.f_bavail*s.f_frsize;assert free>=(2048 if leader else 192)*2**30
memory={{k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in ['MemTotal','MemAvailable']}}
assert memory['MemAvailable']>=(576 if leader else 320)*2**30
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5);assert not units.strip(),units
assert not Path({v['receipts']!r}).exists()
assert not list(Path({str(Path(v['metrics']).parent)!r}).glob({Path(v['metrics']).name+'-rank*.time'!r}))
if leader:
 for target in {[v[k] for k in ['output','analysis','raw_audit']]!r}:assert not Path(target).exists()
 assert Path({v['output']!r}).parent.resolve()==Path({v['output']!r}).parent and Path({v['output']!r}).parent.is_dir()
selected=[line.split(',') for line in subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET,NODE,ONLINE'],text=True,timeout=5).splitlines() if not line.startswith('#') and int(line.split(',')[0]) in {ALTERNATE_CPUS!r}]
assert selected=={topology!r}
def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
actual={{str(q.relative_to(p)) for q in (p/'mpi').iterdir()}}
assert actual==set({owned!r}),actual
for name,item in {owned!r}.items():
 q=p/name;assert not q.is_symlink() and q.resolve().is_relative_to(p.resolve())
 assert q.stat().st_size==item['bytes'] and digest(q)==item['sha256'],name
assert os.access(p/'mpi/b2-mpi',os.X_OK)
for name,h in {mpi_runtime!r}.items():
 q=b/'mpi'/name;assert q.resolve().is_relative_to((b/'mpi').resolve()) and digest(q)==h
for name,item in {catalog!r}.items():
 q=b/{SOURCE!r}/name;assert not q.is_symlink() and q.stat().st_size==item['bytes'] and digest(q)==item['sha256']
if leader:
 assert digest(Path({v['source_artifact']!r})/'model.json')=={v['model_sha256']!r}
 assert digest(Path({BUILD+f'/confirmation-topology-v1-seed{REPLICATE}/exact-counts.jsonl'!r}))=={counts_sha!r}
def ticks():return {{int(x[0][3:]):list(map(int,x[1:9])) for line in Path('/proc/stat').read_text().splitlines() if (x:=line.split())[0].startswith('cpu') and x[0][3:].isdigit()}}
a=ticks();time.sleep(1);z=ticks();busy={{}}
for cpu in {ALTERNATE_CPUS!r}:
 d=[y-x for x,y in zip(a[cpu],z[cpu])];assert sum(d)>0;busy[cpu]=100*(1-(d[3]+d[4])/sum(d))
assert max(busy.values())<50 and sum(busy.values())/8<25,busy
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),volume=str(volume),free_bytes=free,memory=memory,
 selected_cpu_topology=selected,selected_cpu_busy_percent=busy,active_units=[],owned_input_files=len(actual),all_artifact_hashes_verified=True,source_catalog_verified=True)))'''


def backup_preflight_code():
    return f'''import os,json,subprocess,datetime,signal
from pathlib import Path
signal.alarm(20);assert os.uname().nodename=={BACKUP_NODE!r}
b=Path('/atlas-home/0003/backups/brian2-mpi');assert b.is_dir() and b.resolve().is_relative_to(Path('/home/rock'))
assert not Path({BACKUP!r}).exists()
s=os.statvfs(b);free=s.f_bavail*s.f_frsize;assert free>=272*2**30,free
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5);assert not units.strip(),units
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=free,active_units=[],planned_destination={BACKUP!r},reservation_is_not_exclusive=True)))'''


def archive(case,t7):
    dest=t7/'artifacts'/CASE
    for p in case.rglob('*'):
        if not p.is_file():continue
        require(not p.is_symlink() and p.stat().st_size<2**20,'bounded run control')
        q=dest/p.relative_to(case);q.parent.mkdir(parents=True,exist_ok=True)
        if not q.exists():
            with p.open('rb') as source,q.open('xb') as target:
                shutil.copyfileobj(source,target);target.flush();os.fsync(target.fileno())
        require(sha(q)==sha(p),'archive differs')


def run(evidence,t7):
    v,identity_raw,r,mpi_runtime=prerequisite(evidence)
    require(Path('/Volumes/T7').is_mount() and t7.resolve().is_relative_to(Path('/Volumes/T7')),'T7 required')
    s=os.statvfs(t7);require(s.f_bavail*s.f_frsize>=512*2**30,'T7 reserve')
    opts=launch_options(v,t7);require(not opts['output'].exists(),'existing launch log; no rerun')
    case=evidence/CASE;case.mkdir();begin=time.monotonic()
    write(case/'protocol.json',protocol(v,r));(case/'identity.json').open('xb').write(identity_raw)
    tools=Path(__file__).parent
    names=['mam_confirmation_terminal_sync.py','mpi_resource_guard.py','mam_rust_guard23.py']
    bundle={n:(tools/n).read_text() for n in names}
    bundle.update({'protocol.json':(case/'protocol.json').read_text(),'identity.json':identity_raw.decode()})
    catalog={n:dict(bytes=len(raw.encode()),sha256=hashlib.sha256(raw.encode()).hexdigest()) for n,raw in bundle.items()}
    source=case/'source';source.mkdir()
    for n,text in bundle.items():(source/n).open('x').write(text)
    write(case/'intent.json',dict(protocol_sha256=sha(case/'protocol.json'),identity_sha256=IDENTITY_SHA,
        launcher_sha256=sha(Path(__file__)),transport_launcher_sha256=sha(tools/'mpi_teleport_launch.py'),
        source_catalog=catalog,maximum_launches=1,automatic_retry=False,preparation_budget_seconds=300))
    try:
        staged=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(stage,i,bundle,catalog) for i in range(4)]
            failures=[]
            for i,future in enumerate(futures):
                try:
                    row=future.result();staged.append(row);write(case/f'stage-{i}.json',row)
                except Exception as error:
                    failures.append(str(error));write(case/f'stage-{i}-failure.json',dict(error=str(error),stderr=str(getattr(error,'stderr',''))[-16384:]))
            require(not failures,'source stage failed; no automatic retry')
        require(time.monotonic()-begin<240,'source preparation exhausted budget')
        topologies=candidate_topologies(evidence);started=time.monotonic();rows=[];errors=[]
        requests=[(node,preflight_code(i,v,catalog,mpi_runtime,topologies[i],r['counts_sha256'])) for i,node in enumerate(NODES)]
        requests.append((BACKUP_NODE,backup_preflight_code()))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures=[pool.submit(remote,node,code) for node,code in requests]
            for i,((node,_),future) in enumerate(zip(requests,futures,strict=True)):
                try:
                    row=future.result();rows.append(row);write(case/f'preflight-{i}.json',dict(passed=True,result=row))
                except Exception as error:
                    errors.append(str(error));write(case/f'preflight-{i}.json',dict(passed=False,host=node,error_type=type(error).__name__,error=str(error),stderr=str(getattr(error,'stderr',''))[-16384:]))
        require(not errors,'resource preflight failed; no simulation or automatic retry')
        admission=dict(schema='b2-mam-confirmation-admission-v1',admitted=True,replicate=REPLICATE,label=LABEL,
            protocol_sha256=sha(case/'protocol.json'),identity_sha256=IDENTITY_SHA,
            readiness_sha256=READINESS,source_catalog=catalog,source_staging=staged,
            model_sha256=v['model_sha256'],plan_sha256=v['plan_sha256'],instance_sha256=v['instance_sha256'],
            executable_sha256=v['executable_sha256'],mpi_runtime=mpi_runtime,
            preflight=rows[:4],backup_preflight=rows[4],resources=protocol(v,r)['resources'],
            launch_options={k:str(x) if isinstance(x,Path) else x for k,x in opts.items()},
            scientific_acceptance=False,performance_cost_acceptance=False)
        write(case/'admission.json',admission);archive(case,t7)
        require(time.monotonic()-started<60 and time.monotonic()-begin<300,'preflight stale before launch')
        print(json.dumps(dict(event='confirmation_admitted',replicate=REPLICATE,admission_sha256=sha(case/'admission.json'))),flush=True)
        result=launch(**opts);write(case/'launch.json',result);archive(case,t7)
        print(json.dumps(dict(event='launcher_terminal',all_terminal_raw_science_audits_required=True,scientific_acceptance=False)),flush=True)
    except BaseException as error:
        p=opts['output']/'launch.json'
        if p.exists() and not (case/'launch.json').exists():write(case/'launch.json',read(p))
        write(case/'failure.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False,
            launch_log_exists=opts['output'].exists(),elapsed_seconds=time.monotonic()-begin))
        archive(case,t7);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--t7',type=Path,required=True)
    a=parser.parse_args();run(a.evidence,a.t7)
