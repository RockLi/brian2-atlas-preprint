"""One guarded six-metric analysis after independently accepted confirmation raw data.

The three-hour deadline includes preparation, computation and compact evidence
collection. Large derived arrays remain on brick2; their complete catalogs and
reports are collected and verified. No scientific equivalence is inferred.
"""
from mam_confirmation_profile import REPLICATE,selected,worker_environment,IDENTITY_SHA_1751
import argparse,base64,hashlib,io,json,math,os,shlex,subprocess,tarfile,time
from pathlib import Path
from mam_confirmation_analysis import accepted_raw,stages,REQUIRED,NORMALIZATION
from mam_confirmation_raw import terminal_names
from mam_confirmation_terminal_sync import IDENTITY_SHA,require
from mam_launch_confirmation_run import CASE,LABEL,read,sha,write
from mam_launch_native_analysis import interarea_files
from mam_benchmark_terminal import guard
IDENTITY_SHA=selected(IDENTITY_SHA,IDENTITY_SHA_1751)

BASE=Path('/data/brick2/brian2-mpi-region-20260907')
SOURCE=BASE/f'confirmation-analysis-source-v1-seed{REPLICATE}'
OUTPUT=BASE/'confirmation-analysis'/LABEL
GUARD=BASE/f'guards/confirmation-analysis-v1-seed{REPLICATE}.json'
PREFIX=f'b2mpi-confirm{REPLICATE}-analysis-v1'
UNIT=PREFIX+'-science'
NODE='hk-prod-model-ae02-23'
PYTHON='/atlas-home/0003/workspace/brian2-lk-20260906/.venv/bin/python'
TOTAL=10800
NORM_ARCHIVE_SHA='de2035e9977196acf659017870490a5c0bf02cfef054a367cb2d4ba0b99da04b'
NORMS={NORMALIZATION:dict(bytes=34499,sha256='b9dda7098372ed14a94c8a3c4483657cb92d66272c532926e8e78787377f21bc'),
    'normalization/mam-normalization-generated-data-v1.json':dict(bytes=6746809,sha256='8c66bb68d55cf2bff222c67716952a2b6260576d6ba0a3650cc150af49f7c2cb')}


def bundle(root,case,t7,completion_sha):
    identity,raw,terminal=accepted_raw(case,completion_sha)
    require(identity['analysis']==str(OUTPUT),'analysis output path differs')
    names=terminal_names(read(case/'terminal/report.json'))+['completion.json']
    names+=['raw/'+n for n in ['binding.json','intent.json','admission.json','controller.json','pending.json','guard.json','attempt.json','report.json']]
    files={}
    for name in names:
        p=case/name;require(p.is_file() and not p.is_symlink() and p.stat().st_size<64*2**20,'bounded ordinary case control')
        files['case/'+name]=p.read_bytes()
    for parent in ['tools','python']:
        for p in (root/parent).rglob('*.py'):
            if p.name.startswith('._'):continue
            require(not p.is_symlink() and p.stat().st_size<16*2**20,'bounded ordinary source')
            files[str(p.relative_to(root))]=p.read_bytes()
    norm=t7/'artifacts/primary-postrun-v1/source.tar.gz'
    require(norm.stat().st_size<2*2**20 and sha(norm)==NORM_ARCHIVE_SHA,'retained normalization source archive differs')
    with tarfile.open(norm) as archive:
        for name,item in NORMS.items():
            member=archive.getmember(name);require(member.isfile() and member.size==item['bytes'],'normalization member size')
            data=archive.extractfile(member).read(item['bytes']+1)
            require(len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],'normalization digest differs')
            files[name]=data
    for name,p in interarea_files(t7/'artifacts').items():files[name]=p.read_bytes()
    require(sum(map(len,files.values()))<64*2**20,'expanded analysis bundle exceeds budget')
    catalog={n:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for n,data in files.items()}
    files['catalog.json']=(json.dumps(catalog,indent=2)+'\n').encode();buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name,data in sorted(files.items()):
            member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
    payload=buffer.getvalue();require(len(payload)<32*2**20,'compressed analysis bundle exceeds budget')
    return payload,catalog,identity,raw,terminal


def remote(code,payload=None):
    result=subprocess.run(['tsh','ssh','rock@'+NODE,shlex.join(['taskset','-c','8,9','python3','-c',code])],
        input=payload,capture_output=True,timeout=45)
    require(result.returncode==0,'analysis control failed: '+result.stderr[:2000].decode(errors='replace'))
    require(len(result.stdout)<12*2**20,'bounded analysis control reply')
    return json.loads(result.stdout)


def preflight_code():
    return f'''from pathlib import Path
import os,json,time,signal,resource,subprocess,datetime
signal.alarm(30);resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20));resource.setrlimit(resource.RLIMIT_CPU,(15,15))
assert os.uname().nodename=={NODE!r};b=Path({str(BASE)!r})
assert Path('/data/brick2').is_mount() and b.resolve().is_relative_to(Path('/data/brick2')) and b.stat().st_dev!=Path('/').stat().st_dev
for p in {list(map(str,[SOURCE,OUTPUT,GUARD]))!r}:assert not Path(p).exists(),p
assert not subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5).strip()
s=os.statvfs(b);free=s.f_bavail*s.f_frsize;assert free>=1536*2**30
memory={{k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k=='MemAvailable'}}
assert memory['MemAvailable']>=80*2**30
def ticks():return {{int(x[0][3:]):list(map(int,x[1:9])) for line in Path('/proc/stat').read_text().splitlines() if (x:=line.split())[0].startswith('cpu') and x[0][3:].isdigit()}}
a=ticks();time.sleep(1);z=ticks();busy={{}}
for cpu in [8,9]:
 d=[y-x for x,y in zip(a[cpu],z[cpu])];assert sum(d)>0;busy[cpu]=100*(1-(d[3]+d[4])/sum(d))
assert max(busy.values())<50 and sum(busy.values())/2<25,busy
assert Path({PYTHON!r}).is_file()
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=free,memory=memory,cpu_busy_percent=busy,active_units=[])))'''


def stage_code(payload_sha,catalog):
    return f'''from pathlib import Path
import os,sys,json,tarfile,hashlib,io,signal,resource
signal.alarm(35);resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(25,25))
raw=sys.stdin.buffer.read(32*2**20+1);assert len(raw)<32*2**20 and hashlib.sha256(raw).hexdigest()=={payload_sha!r}
b=Path({str(SOURCE)!r});b.mkdir();expected_catalog={catalog!r}
with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
 members=archive.getmembers();assert len(members)==len(expected_catalog)+1
 assert set(x.name for x in members)==set(expected_catalog)|{{'catalog.json'}} and all(x.isfile() for x in members)
 assert sum(x.size for x in members)<65*2**20;archive.extractall(b,filter='data')
for name,item in expected_catalog.items():
 p=b/name;assert not p.is_symlink() and p.resolve().is_relative_to(b.resolve()) and p.stat().st_size==item['bytes']
 with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
assert json.loads((b/'catalog.json').read_text())==expected_catalog
for p in [Path({str(OUTPUT.parent)!r}),b/'tmp',b/'mpl']:
 p.mkdir(exist_ok=True);assert p.resolve()==p
print(json.dumps(dict(source_sha256={payload_sha!r},verified_files={len(catalog)})))'''


def application(completion_sha,previous):
    return ['env','PYTHONDONTWRITEBYTECODE=1','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',
        'TMPDIR='+str(SOURCE/'tmp'),'MPLCONFIGDIR='+str(SOURCE/'mpl'),
        *worker_environment(),'PYTHONPATH='+str(SOURCE/'tools')+':'+str(SOURCE/'python'),PYTHON,
        str(SOURCE/'tools/mam_confirmation_analysis.py'),'--source',str(SOURCE),
        '--completion-sha256',completion_sha,'--previous-seconds',str(previous)]


def command(completion_sha,previous,limit):
    require(0<=previous<600 and 0<limit<=math.floor(TOTAL-previous-180),'finite shared science budget')
    return ['systemd-run','--expand-environment=no','--quiet','--wait','--pipe','--collect','--unit='+UNIT,
        '--uid=rock','--service-type=exec','--property=MemoryMax=16384M','--property=MemorySwapMax=0',
        '--property=CPUQuota=200%','--property=AllowedCPUs=8-9','--property=TasksMax=64',
        '--property=RuntimeMaxSec='+str(limit+5),'--property=TimeoutStopSec=5',
        '--property=KillMode=control-group','--property=OOMPolicy=continue','/usr/bin/python3',
        str(SOURCE/'tools/mpi_resource_guard.py'),'--output',str(GUARD),'--volume','/data/brick2',
        '--memory-mib','16384','--cpu-percent','200','--file-mib','512','--min-free-gib','1280',
        '--timeout',str(limit),'--',*application(completion_sha,previous)]


def collect_code(catalog):
    reports=dict(activity='activity.json',cell='paper-cell-metrics.json',correlation='correlation.json',series='time-series.json',fc='fc.json',lags='lags.json')
    files={'pending.json':str(OUTPUT/'pending.json'),'guard.json':str(GUARD),'stages.jsonl':str(OUTPUT/'stages.jsonl')}
    for name,report in reports.items():
        for f in [report,'catalog.json']:files[name+'/'+f]=str(OUTPUT/name/f)
    return f'''from pathlib import Path
import json,os,sys,signal,resource,hashlib,base64,subprocess
signal.alarm(40);resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(30,30))
assert not subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend',{UNIT+'.service'!r}],text=True,timeout=5).strip()
b=Path({str(SOURCE)!r})
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for name,item in {catalog!r}.items():
 p=b/name;assert not p.is_symlink() and p.resolve().is_relative_to(b) and p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
assert not (Path({str(OUTPUT)!r})/'failure.json').exists()
# Reopen and hash every derived artifact, including the large arrays, without loading arrays into memory.
for name in {list(REQUIRED)!r}:
 root=Path({str(OUTPUT)!r})/name;cat=json.loads((root/'catalog.json').read_text())
 for n,row in cat.items():
  assert Path(n).name==n;p=root/n
  assert p.is_file() and not p.is_symlink() and 0<p.stat().st_size==row['bytes']<=512*2**20 and sha(p)==row['sha256']
result={{}};total=0
for name,path in {files!r}.items():
 p=Path(path);assert p.is_file() and not p.is_symlink() and p.stat().st_size<2*2**20
 raw=p.read_bytes();total+=len(raw);assert total<8*2**20
 result[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
print(json.dumps(result))'''


def publish(pending,g,controller,*,completion_sha,identity,raw_report_sha,previous,limit,catalog,elapsed):
    require(type(controller['returncode']) is int and controller['returncode']==0 and controller['error'] is None
        and controller['command']==command(completion_sha,previous,limit),'science controller did not complete')
    require(g['command']==application(completion_sha,previous),'science guard command differs')
    spec=dict(host=NODE,role='science',memory_bytes=16*2**30,pids_max=64,cpu_ids=[8,9],cpu_quota_cores=2,
        volume='/data/brick2',allow_root_volume=False,file_limit_bytes=512*2**20,
        minimum_free_bytes=1280*2**30,reserved_host_memory_bytes=64*2**30)
    accounting=guard(g,spec,PREFIX,limit)
    require(pending['schema']=='b2-mam-confirmation-six-metrics-pending-v1' and pending['replicate']==REPLICATE
        and pending['identity_sha256']==IDENTITY_SHA and pending['completion_sha256']==completion_sha
        and pending['source_catalog_sha256']==hashlib.sha256((json.dumps(catalog,indent=2)+'\n').encode()).hexdigest(),
        'science pending identity/source differs')
    require(pending['model_sha256']==identity['model_sha256']
        and type(pending['runtime_random_key']) is int and pending['runtime_random_key']==identity['random_keys']['runtime_input']
        and pending['raw_report_sha256']==raw_report_sha,'science model/random key/raw report differs')
    require(pending['full_descriptive_analysis_complete'] is True and pending['analysis_guard_passed'] is False
        and pending['pending_guard_acceptance'] is True and all(pending[k] is False for k in
        ['scientific_acceptance','performance_cost_acceptance','formal_equivalence_acceptance']),'unsupported scientific acceptance')
    names=list(REQUIRED);rows=pending['stages']
    require(len(rows)==12 and [r['stage'] for r in rows]==[n for n in names for _ in range(2)]
        and [r['state'] for r in rows]==['started','complete']*6 and set(pending['catalogs'])==set(names),
        'six complete stages required')
    for index,(name,cap,expected) in enumerate(stages(SOURCE,identity)):
        row=rows[index*2]
        require(row['command']==[PYTHON,*expected[1:]] and 0<row['timeout_seconds']<=cap,'estimator command or time cap differs')
    require(pending['previous_seconds']==previous and pending['shared_budget_seconds']==TOTAL
        and 0<pending['science_seconds']<=g['wall_seconds']+.02 and pending['total_accounted_seconds']<=elapsed<=TOTAL,
        'whole science budget exceeded')
    return dict(pending,analysis_guard_passed=True,pending_guard_acceptance=False,
        full_derived_artifact_readback_verified=True,elapsed_through_collection_seconds=elapsed,
        resource_accounting=accounting,automatic_retry=False,large_arrays_retained_on_brick2=True)


def run(evidence,t7):
    case=evidence/CASE
    if not (case/'completion.json').exists():return dict(ready=False,analysis_started=False,reason='Full engineering acceptance is pending.')
    began=time.monotonic();completion_sha=sha(case/'completion.json')
    payload,catalog,identity,raw,terminal=bundle(Path(__file__).resolve().parents[1],case,t7,completion_sha)
    require(Path('/Volumes/T7').is_mount() and t7.resolve().is_relative_to(Path('/Volumes/T7')),'T7 required')
    s=os.statvfs(t7);require(s.f_bavail*s.f_frsize>=512*2**30,'T7 reserve')
    local=case/'science';backup=t7/'artifacts'/CASE/'science'
    require(not local.exists() and not backup.exists(),'science already attempted; no retry')
    local.mkdir();backup.mkdir(parents=True)
    write(local/'intent.json',dict(completion_sha256=completion_sha,total_seconds=TOTAL,attempts=1,automatic_retry=False,
        memory_gib=16,cpu_ids=[8,9],file_limit_mib=512,minimum_free_gib=1280,full_estimator_changes=False))
    try:
        fresh=time.monotonic();admission=remote(preflight_code());payload_sha=hashlib.sha256(payload).hexdigest()
        staged=remote(stage_code(payload_sha,catalog),payload)
        with (backup/'source.tar.gz').open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        require(sha(backup/'source.tar.gz')==payload_sha,'analysis source backup differs')
        previous=time.monotonic()-began;limit=math.floor(TOTAL-previous-180)
        cmd=command(completion_sha,previous,limit)
        admission.update(previous_seconds=previous,limit_seconds=limit,command=cmd,source=staged,source_catalog=catalog)
        write(local/'admission.json',admission)
        require(time.monotonic()-fresh<60,'science admission stale')
        rc=None;error=None
        try:
            with (backup/'controller.log').open('x') as log:
                rc=subprocess.run(['tsh','ssh','root@'+NODE,shlex.join(cmd)],stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,timeout=limit+45).returncode
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        controller=dict(returncode=rc,error=error,command=cmd,observation_failure_is_not_remote_completion=True)
        write(local/'controller.json',controller)
        require(rc==0 and error is None,'science controller unsuccessful; inspect same job, never restart')
        collected=remote(collect_code(catalog));values={}
        for name,item in collected.items():
            data=base64.b64decode(item['base64'],validate=True)
            require(len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],'science control transfer differs')
            p=local/name;p.parent.mkdir(parents=True,exist_ok=True);p.open('xb').write(data)
            if name in ['pending.json','guard.json']:values[name]=json.loads(data)
        pending=values['pending.json']
        for name,digest in pending['catalogs'].items():require(sha(local/name/'catalog.json')==digest,'derived catalog changed')
        result=publish(pending,values['guard.json'],controller,completion_sha=completion_sha,identity=identity,raw_report_sha=sha(case/'raw/report.json'),previous=previous,
            limit=limit,catalog=catalog,elapsed=time.monotonic()-began)
        result['evidence_sha256']={str(p.relative_to(local)):sha(p) for p in local.rglob('*') if p.is_file()}
        write(local/'report.json',result)
        for p in local.rglob('*'):
            if not p.is_file():continue
            q=backup/p.relative_to(local);q.parent.mkdir(parents=True,exist_ok=True)
            with q.open('xb') as f:f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
            require(sha(q)==sha(p),'science evidence backup differs')
        require(time.monotonic()-began<=TOTAL,'science total deadline including archive exceeded')
        completion=dict(full_descriptive_analysis_complete=True,replicate=REPLICATE,report_sha256=sha(local/'report.json'),
            total_seconds=time.monotonic()-began,scientific_acceptance=False,performance_cost_acceptance=False,
            arrays_retained_on_node23=True,compact_controls_archived_on_t7=True)
        write(local/'complete.json',completion);write(backup/'complete.json',completion)
        return dict(ready=True,analysis_started=True,full_descriptive_analysis_complete=True,scientific_acceptance=False)
    except BaseException as error:
        write(local/'failure.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False,
            elapsed_seconds=time.monotonic()-began,observation_failure_is_not_remote_completion=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--t7',type=Path,required=True)
    a=p.parse_args();result=run(a.evidence,a.t7);print(json.dumps(result));raise SystemExit(0 if result['ready'] else 2)
