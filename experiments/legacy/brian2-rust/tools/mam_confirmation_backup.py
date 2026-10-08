"""One identity-bound 1750 output backup from23 to24, after full raw acceptance.

Reuses the private collection protocol. No source deletion or automatic retry.
Controller handles only small controls; payload stays on the cluster.
"""
from mam_confirmation_profile import REPLICATE,selected,worker_environment,IDENTITY_SHA_1751
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import time

from mam_collection_transfer import SCHEMA, validate
from mam_cluster_raw_backup import readback

TAG = f'confirmation-run-v1-seed{REPLICATE}'
CASE = TAG
PREFIX = 'b2mpi-' + TAG
SOURCE = 'hk-prod-model-ae02-23'
DEST = selected('hk-prod-model-ae02-24','hk-prod-model-ae08-82')
DEST_IP=selected('192.168.20.24','192.168.30.82')
BACKUP_DIRECTORY=selected('backup24','backup82')
ROOT = f'/data/brick2/brian2-mpi-region-20260907/primary-host-v1/runs/rust-mam-confirmation-v1-replicate{REPLICATE}-100500ms'
BASES = {SOURCE: '/data/brick2/brian2-mpi-region-20260907/' + TAG,
         DEST: '/atlas-home/0003/backups/brian2-mpi/' + TAG}
IDENTITY_SHA = '9a5a3258ca7b33b3d5296bc74398ba19e3455f3e3e61ffcb872f7a6b374165de'
IDENTITY_SHA=selected(IDENTITY_SHA,IDENTITY_SHA_1751)
TRANSFER = 3600
READBACK = 1800
TOTAL = 7200
MODULES = ['mam_collection_transfer.py', 'mam_direct_transfer_v3.py', 'mpi_resource_guard.py',
           'mam_cluster_raw_backup.py','mam_confirmation_backup.py','mam_confirmation_profile.py']
HELPERS = {
 'mam_cluster_raw_backup.py':'7e7e5ba8a173d9e4234a3ae95dd02c927ee0bf1a1d16a0a715292ab397122b7b',
 'mam_collection_transfer.py':'ecfe7c39af688f2a31d263c604a5de02949fdd086c3acd40cdfec67adcebcc66',
 'mam_direct_transfer_v3.py':'dc5c598c71b326c83209b4ca0e290d3172ead8ec9b09ffc261f502cddc7399cd',
 'mpi_resource_guard.py':'630fc35f0b38729fc202925b09b4eea55cc965cfeee396ace6085a3a98004014'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def catalog_for(identity,raw,terminal,leader):
    need(identity['replicate']==REPLICATE and identity['output']==ROOT,'pinned source output required')
    need(raw['replicate']==REPLICATE and raw['identity_sha256']==IDENTITY_SHA
         and raw['raw_output_audit_passed'] is True and raw['audit_guard_passed'] is True
         and raw['pending_guard_acceptance'] is False,'complete new raw audit required')
    for key in ['model_sha256','plan_sha256','executable_sha256']:
        need(raw[key]==identity[key],'raw model identity differs')
    need(set(raw['dump_sha256'])==set(raw['dump_bytes'])=={'results.bin','events.bin'},'binary coverage')
    files=[]
    for name in ['events.bin','mpi-runtime.json','results.bin','summary.json']:
        if name.endswith('.bin'):
            item=dict(path=name,bytes=raw['dump_bytes'][name],sha256=raw['dump_sha256'][name])
        else:
            data=leader['runtime_json' if name=='mpi-runtime.json' else 'summary_json'].encode()
            need(0<len(data)<=32*2**20,'bounded output metadata')
            item=dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        need(0<item['bytes']==terminal['output_bytes'][name],'terminal output size differs')
        files.append(item)
    catalog=dict(schema=SCHEMA,files=files);validate(catalog,64*2**30,128*2**30)
    return catalog


def source_gate(evidence):
    case=evidence/CASE
    if not (case/'completion.json').is_file():return None
    from mam_confirmation_analysis import accepted_raw
    from mam_launch_confirmation_run import read,sha
    import gzip,io
    completion_sha=sha(case/'completion.json');identity,raw,terminal=accepted_raw(case,completion_sha)
    report=read(case/'terminal/report.json')
    row=report['collection_files'][0];need(row['file']=='host-0.json.gz' and row['host']==SOURCE,'leader control identity')
    p=case/'terminal'/row['file'];need(p.stat().st_size==row['bytes']<=64*2**20,'bounded leader controls')
    data=p.read_bytes();need(hashlib.sha256(data).hexdigest()==row['sha256'],'leader controls changed')
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:expanded=stream.read(64*2**20+1)
    need(len(expanded)<=64*2**20,'expanded leader control ceiling')
    leader=json.loads(expanded)['leader_outputs']
    return dict(completion_sha256=completion_sha,identity_sha256=IDENTITY_SHA,
        raw_report_sha256=sha(case/'raw/report.json'),catalog=catalog_for(identity,raw,terminal,leader))


def command(node, role, timeout, app):
    base = BASES[node]; source = node == SOURCE
    result = ['systemd-run', '--expand-environment=no', '--quiet', '--wait', '--pipe', '--collect',
        '--unit='+PREFIX+'-'+role, '--uid=rock', '--service-type=exec',
        '--property=MemoryMax=4096M', '--property=MemorySwapMax=0', '--property=CPUQuota=200%',
        '--property=AllowedCPUs=8-9', '--property=TasksMax=64',
        '--property=RuntimeMaxSec='+str(timeout+5), '--property=TimeoutStopSec=5',
        '--property=KillMode=control-group', '--property=OOMPolicy=continue',
        '/usr/bin/python3', base+'/source/mpi_resource_guard.py', '--output', base+'/'+role+'-guard.json',
        '--volume', '/data/brick2' if source else '/', '--memory-mib', '4096', '--cpu-percent', '200',
        '--file-mib', '65536', '--min-free-gib', '1280' if source else '128', '--timeout', str(timeout)]
    if not source: result.append('--allow-root-volume')
    return result + ['--', *app]


def remote(node, code, data=None):
    from mam_collect_native_primary_raw import remote as call
    return call(node, code, data)


def preflight(node, catalog):
    source = node == SOURCE; volume = '/data/brick2' if source else '/home/rock'
    return remote(node, f'''import os,json,time,resource,signal,subprocess,datetime
from pathlib import Path
signal.alarm(20);resource.setrlimit(resource.RLIMIT_AS,(128*2**20,128*2**20));resource.setrlimit(resource.RLIMIT_CPU,(5,5))
assert os.uname().nodename=={node!r}
p=Path({volume!r});assert p.resolve()==p
assert not Path({BASES[node]!r}).exists()
assert not subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5).strip()
s=os.statvfs(p);free=s.f_bavail*s.f_frsize;assert free>={1281 if source else 272}*2**30
assert (p.stat().st_dev!=Path('/').stat().st_dev)=={source!r}
m={{k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())}}
assert m['MemAvailable']>=80*2**30
if {source!r}:
 for row in {catalog['files']!r}:
  q=Path({ROOT!r})/row['path'];assert q.is_file() and not q.is_symlink() and q.stat().st_size==row['bytes']
def ticks():return {{int(x[0][3:]):list(map(int,x[1:9])) for l in Path('/proc/stat').read_text().splitlines() if (x:=l.split())[0].startswith('cpu') and x[0][3:].isdigit()}}
a=ticks();time.sleep(1);z=ticks();busy={{}}
for c in [8,9]:
 d=[y-x for x,y in zip(a[c],z[c])];assert sum(d)>0;busy[c]=100*(1-(d[3]+d[4])/sum(d))
assert max(busy.values())<50 and sum(busy.values())/2<25,busy
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=free,available_memory_bytes=m['MemAvailable'],cpu_busy_percent=busy,active_own_units=[])))''')


def stage(node, catalog):
    bundle={n:(Path(__file__).parent/n).read_text() for n in MODULES}
    result=remote(node, f'''import json,sys,hashlib,os
from pathlib import Path
raw=json.loads(sys.stdin.read());b=Path({BASES[node]!r})
assert not b.exists();b.parent.mkdir(parents=True,exist_ok=True)
assert b.parent.resolve().is_relative_to(Path({'/data/brick2' if node==SOURCE else '/home/rock'!r}))
b.mkdir(mode=0o700);(b/'source').mkdir()
for name,value in raw['modules'].items():(b/'source'/name).open('x').write(value)
(b/'catalog.json').open('x').write(json.dumps(raw['catalog']))
print(json.dumps(dict(source_sha256={{n:hashlib.sha256((b/'source'/n).read_bytes()).hexdigest() for n in raw['modules']}})))''', json.dumps(dict(modules=bundle,catalog=catalog)).encode())
    need(result['source_sha256']=={n:hashlib.sha256(s.encode()).hexdigest() for n,s in bundle.items()},'staged code differs')
    return result


def guarded(node, role, timeout, app, output):
    from mam_benchmark_terminal import guard
    cmd=command(node,role,timeout,app)
    write(output/(role+'-command.json'),dict(command=cmd,timeout_seconds=timeout))
    with (output/(role+'.log')).open('x') as log:
        rc=subprocess.run(['tsh','ssh','root@'+node,shlex.join(cmd)],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,timeout=timeout+25).returncode
    g=remote(node,f"from pathlib import Path;print(Path({BASES[node]+'/'+role+'-guard.json'!r}).read_text())")
    write(output/(role+'-guard.json'),g)
    need(rc==0 and g['command']==app,'SSH/guard command failed')
    source=node==SOURCE
    spec=dict(host=node,role=role,memory_bytes=4*2**30,pids_max=64,cpu_ids=[8,9],cpu_quota_cores=2,
        volume='/data/brick2' if source else '/',allow_root_volume=not source,file_limit_bytes=64*2**30,
        minimum_free_bytes=(1280 if source else 128)*2**30,reserved_host_memory_bytes=64*2**30)
    return guard(g,spec,PREFIX,timeout)


def archive_backup(output,t7):
    """Archive only this backup's controls, not previously collected large files."""
    need(output.name==BACKUP_DIRECTORY,'dedicated backup controls required')
    destination=t7/'artifacts'/CASE/BACKUP_DIRECTORY;total=0
    for p in output.iterdir():
        need(p.is_file() and not p.is_symlink() and p.stat().st_size<=8*2**20,'bounded backup control')
        total+=p.stat().st_size;need(total<=16*2**20,'backup control archive ceiling')
        q=destination/p.name;q.parent.mkdir(parents=True,exist_ok=True)
        data=p.read_bytes()
        if not q.exists():
            with q.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        need(q.read_bytes()==data,'backup control archive differs')


def terminal_code(node,expected,catalog):
    return f"""import json,hashlib,subprocess,signal
from pathlib import Path
signal.alarm(20);b=Path({BASES[node]!r})
u=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend',{PREFIX+'-*'!r}],text=True,timeout=5).strip()
assert not u,u
actual={{}}
for name,digest in {expected!r}.items():
 p=b/'source'/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size<2**20
 actual[name]=hashlib.sha256(p.read_bytes()).hexdigest();assert actual[name]==digest
assert json.loads((b/'catalog.json').read_text())=={catalog!r}
print(json.dumps(dict(active=u,source_sha256=actual,catalog_verified=True)))"""


def run(evidence, t7):
    started=time.monotonic();case=evidence/CASE;output=case/BACKUP_DIRECTORY
    proof=source_gate(evidence)
    if proof is None:return dict(ready=False,backup_started=False,reason='Selected complete raw output is not yet accepted')
    for name,digest in HELPERS.items():
        need(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==digest,'verified transfer/readback helper changed')
    need(Path('/Volumes/T7').is_mount() and t7.resolve().is_relative_to(Path('/Volumes/T7')),'T7 controls archive required')
    capacity=os.statvfs(t7);need(capacity.f_bavail*capacity.f_frsize>=512*2**30,'T7 controls reserve')
    need(not output.exists(),'attempt exists; automatic retry forbidden')
    catalog=proof['catalog']
    total,digest=validate(catalog,64*2**30,128*2**30)
    output.mkdir()
    write(output/'intent.json',dict(catalog=catalog,source=SOURCE,destination=DEST,source_root=ROOT,
        destination_root=BASES[DEST]+'/raw',completion_sha256=proof['completion_sha256'],identity_sha256=IDENTITY_SHA,raw_report_sha256=proof['raw_report_sha256'],transfer_seconds=TRANSFER,
        readback_seconds=READBACK,whole_seconds=TOTAL,automatic_retry=False,max_attempts=1,
        output_files=4,replicate=REPLICATE,source_deletion=False,physical_failure_domain_independence_audited=False,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try:
        admission_start=time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs={n:pool.submit(preflight,n,catalog) for n in [SOURCE,DEST]}
            admission={n:f.result() for n,f in jobs.items()}
        write(output/'admission.json',admission)
        staged={n:stage(n,catalog) for n in [SOURCE,DEST]};write(output/'stage.json',staged)
        archive_backup(output,t7)
        need(time.monotonic()-admission_start<60,'admission stale')
        dst=BASES[DEST];src=BASES[SOURCE]
        receive=['env','PYTHONDONTWRITEBYTECODE=1','python3',dst+'/source/mam_collection_transfer.py','receive','--bind',DEST_IP,
            '--peer','192.168.20.23','--catalog',dst+'/catalog.json','--output',dst+'/raw',
            '--ready',dst+'/ready.json','--receipt',dst+'/receipt.json','--max-file-bytes',str(64*2**30),
            '--max-total-bytes',str(128*2**30),'--reserve-bytes',str(128*2**30)]
        transfer_start=time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(guarded,DEST,'receive',TRANSFER,receive,output)
            deadline=time.monotonic()+45
            while True:
                need(not future.done(),'receiver ended before ready')
                control=remote(DEST,f"from pathlib import Path;p=Path({dst+'/ready.json'!r});print(p.read_text() if p.exists() else '{{}}')")
                if control:break
                need(time.monotonic()<deadline,'receiver readiness timeout');time.sleep(.5)
            # Capability never enters logs, local evidence, or process arguments.
            remote(SOURCE,f"import os,sys;fd=os.open({src+'/ready.json'!r},os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)\nwith os.fdopen(fd,'wb') as f:f.write(sys.stdin.buffer.read())",json.dumps(control).encode())
            timeout=int(TRANSFER-(time.monotonic()-transfer_start)-15)
            need(timeout>30,'transfer budget exhausted')
            send=['env','PYTHONDONTWRITEBYTECODE=1','python3',src+'/source/mam_collection_transfer.py','send','--root',ROOT,
                  '--catalog',src+'/catalog.json','--ready',src+'/ready.json']
            sender=guarded(SOURCE,'send',timeout,send,output)
            receiver=future.result()
        receipt=remote(DEST,f"from pathlib import Path;print(Path({dst+'/receipt.json'!r}).read_text())")
        write(output/'receipt.json',receipt)
        need(receipt['complete'] is True and receipt['bytes']==total and receipt['catalog_sha256']==digest
             and receipt['files']==len(catalog['files']) and receipt['file_cache_release_supported'] is True
             and receipt['output']==dst+'/raw' and receipt['peer']=='192.168.20.23','transfer receipt mismatch')
        print(json.dumps(dict(event='transfer_complete',bytes=total,seconds=time.monotonic()-transfer_start)),flush=True)
        need(time.monotonic()-started+READBACK+90<TOTAL,'readback exceeds whole budget')
        app=['env','PYTHONDONTWRITEBYTECODE=1',*worker_environment(),'python3',dst+'/source/mam_confirmation_backup.py','readback','--root',dst+'/raw',
             '--catalog',dst+'/catalog.json','--receipt',dst+'/readback.json']
        audit=guarded(DEST,'readback',READBACK,app,output)
        result=remote(DEST,f"from pathlib import Path;print(Path({dst+'/readback.json'!r}).read_text())")
        write(output/'readback.json',result)
        need(result['complete'] is True and result['full_destination_readback'] is True
             and result['catalog_sha256']==digest and result['bytes']==total and result['files']==catalog['files'],'readback proof differs')
        stopped={n:remote(n,terminal_code(n,staged[n]['source_sha256'],catalog)) for n in [SOURCE,DEST]}
        write(output/'terminal-state.json',stopped);need(all(not v['active'] for v in stopped.values()),'backup services still running')
        need(time.monotonic()-started<TOTAL,'whole budget exceeded')
        archive_backup(output,t7)
        need(time.monotonic()-started<TOTAL,'whole backup including compact archive exceeded')
        report=dict(complete=True,different_host_verified_backup=True,physical_failure_domain_independence_audited=False,
            catalog=catalog,source=SOURCE,destination=DEST,destination_root=dst+'/raw',
            source_completion_sha256=proof['completion_sha256'],identity_sha256=IDENTITY_SHA,raw_report_sha256=proof['raw_report_sha256'],guards=[sender,receiver,audit],wall_seconds=time.monotonic()-started,
            replicate=REPLICATE,compact_controls_archived=True,accounting_scope='Preparation through transfer, independent readback and compact evidence archive; final completion receipt writes excluded.',
            new_neural_runs=0,scientific_acceptance=False,automatic_retry=False,
            evidence_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
        write(output/'complete.json',report)
        write(t7/'artifacts'/CASE/(BACKUP_DIRECTORY+'/complete.json'),report)
        need((output/'complete.json').read_bytes()==(t7/'artifacts'/CASE/(BACKUP_DIRECTORY+'/complete.json')).read_bytes(),'completion archive differs')
        print(json.dumps(report),flush=True)
        return dict(ready=True,backup_started=True,different_host_verified_backup=True,scientific_acceptance=False)
    except BaseException as exc:
        write(output/'failure.json',dict(error=str(exc),wall_seconds=time.monotonic()-started,automatic_retry=False,complete=False))
        archive_backup(output,t7)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','readback'])
    p.add_argument('--evidence',type=Path);p.add_argument('--t7',type=Path)
    p.add_argument('--root',type=Path);p.add_argument('--catalog',type=Path);p.add_argument('--receipt',type=Path)
    a=p.parse_args()
    if a.mode=='readback':
        need(not a.receipt.exists(),'readback receipt exists')
        write(a.receipt,readback(a.root,json.loads(a.catalog.read_text())))
    else:
        result=run(a.evidence,a.t7);print(json.dumps(result))
        raise SystemExit(0 if result['ready'] else 2)
