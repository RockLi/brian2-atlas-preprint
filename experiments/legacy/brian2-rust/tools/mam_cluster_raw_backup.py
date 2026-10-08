"""One 23-to-24 raw backup; only complete.json accepts guarded full readback.

Reuses the private collection protocol. No source deletion or automatic retry.
Controller handles only small controls; payload stays on the cluster.
"""
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

from mam_collection_transfer import SCHEMA, validate, source_path, release_cache

TAG = 'rust32-raw-backup-24-v1'
PREFIX = 'b2mpi-' + TAG
SOURCE = 'hk-prod-model-ae02-23'
DEST = 'hk-prod-model-ae02-24'
ROOT = '/data/brick2/brian2-mpi-region-20260907/primary-host-v1/runs/rust-mam-perf-v2-seed1729-100500ms'
BASES = {SOURCE: '/data/brick2/brian2-mpi-region-20260907/' + TAG,
         DEST: '/atlas-home/0003/backups/brian2-mpi/' + TAG}
PIN = '01438e2e820aa6a7de6e882e33f94eefb9937cf608496ce957abff61d8ed4350'
TRANSFER = 3600
READBACK = 1800
TOTAL = 6000
MODULES = ['mam_collection_transfer.py', 'mam_direct_transfer_v3.py', 'mpi_resource_guard.py',
           'mam_cluster_raw_backup.py']


def need(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def readback(root, catalog):
    """Reopen every completed file, hash all bytes, and reject changed files."""
    total, digest = validate(catalog, 64*2**30, 128*2**30)
    start = time.monotonic(); rows = []
    for item in catalog['files']:
        p = source_path(root, item['path'])
        h = hashlib.sha256(); count = 0; released = 0
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as f:
            before = os.fstat(f.fileno())
            need(stat.S_ISREG(before.st_mode) and before.st_size == item['bytes'], 'readback type/size')
            while block := f.read(2**20):
                count += len(block); h.update(block)
                need(count <= item['bytes'], 'file grew during readback')
                if count - released >= 64*2**20:
                    released = release_cache(f, released, False)
            release_cache(f, released, False)
            after = os.fstat(f.fileno())
        need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'file changed during readback')
        need(count == item['bytes'] and h.hexdigest() == item['sha256'], 'readback digest mismatch')
        rows.append(dict(path=item['path'], bytes=count, sha256=h.hexdigest()))
        print(json.dumps(dict(event='file_readback_passed', **rows[-1])), flush=True)
    # Persist directory entries created by the receiver before accepting the copy.
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(complete=True, bytes=total, catalog_sha256=digest, files=rows,
                full_destination_readback=True, wall_seconds=time.monotonic()-start)


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


def run(evidence, output):
    from mam_rust_science_reuse import accepted_new_raw
    started=time.monotonic()
    need(not output.exists(),'attempt exists; automatic retry forbidden')
    raw=accepted_new_raw(evidence,PIN);need(raw is not None,'accepted full raw proof required')
    catalog=dict(schema=SCHEMA,files=[dict(path=n,bytes=raw['dump_bytes'][n],sha256=raw['dump_sha256'][n])
                                    for n in ['events.bin','results.bin']])
    total,digest=validate(catalog,64*2**30,128*2**30)
    output.mkdir()
    write(output/'intent.json',dict(catalog=catalog,source=SOURCE,destination=DEST,source_root=ROOT,
        destination_root=BASES[DEST]+'/raw',completion_sha256=PIN,transfer_seconds=TRANSFER,
        readback_seconds=READBACK,whole_seconds=TOTAL,automatic_retry=False,max_attempts=1,
        source_deletion=False,physical_failure_domain_independence_audited=False,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try:
        admission_start=time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs={n:pool.submit(preflight,n,catalog) for n in [SOURCE,DEST]}
            admission={n:f.result() for n,f in jobs.items()}
        write(output/'admission.json',admission)
        staged={n:stage(n,catalog) for n in [SOURCE,DEST]};write(output/'stage.json',staged)
        need(time.monotonic()-admission_start<60,'admission stale')
        dst=BASES[DEST];src=BASES[SOURCE]
        receive=['python3',dst+'/source/mam_collection_transfer.py','receive','--bind','192.168.20.24',
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
            send=['python3',src+'/source/mam_collection_transfer.py','send','--root',ROOT,
                  '--catalog',src+'/catalog.json','--ready',src+'/ready.json']
            sender=guarded(SOURCE,'send',timeout,send,output)
            receiver=future.result()
        receipt=remote(DEST,f"from pathlib import Path;print(Path({dst+'/receipt.json'!r}).read_text())")
        write(output/'receipt.json',receipt)
        need(receipt['complete'] is True and receipt['bytes']==total and receipt['catalog_sha256']==digest
             and receipt['files']==2 and receipt['file_cache_release_supported'] is True
             and receipt['output']==dst+'/raw' and receipt['peer']=='192.168.20.23','transfer receipt mismatch')
        print(json.dumps(dict(event='transfer_complete',bytes=total,seconds=time.monotonic()-transfer_start)),flush=True)
        need(time.monotonic()-started+READBACK+90<TOTAL,'readback exceeds whole budget')
        app=['python3',dst+'/source/mam_cluster_raw_backup.py','readback','--root',dst+'/raw',
             '--catalog',dst+'/catalog.json','--receipt',dst+'/readback.json']
        audit=guarded(DEST,'readback',READBACK,app,output)
        result=remote(DEST,f"from pathlib import Path;print(Path({dst+'/readback.json'!r}).read_text())")
        write(output/'readback.json',result)
        need(result['complete'] is True and result['full_destination_readback'] is True
             and result['catalog_sha256']==digest and result['bytes']==total and result['files']==catalog['files'],'readback proof differs')
        stopped={n:remote(n,"import json,subprocess;print(json.dumps(dict(active=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend',"+repr(PREFIX+'-*')+"],text=True,timeout=5).strip())))") for n in [SOURCE,DEST]}
        write(output/'terminal-state.json',stopped);need(all(not v['active'] for v in stopped.values()),'backup services still running')
        need(time.monotonic()-started<TOTAL,'whole budget exceeded')
        report=dict(complete=True,different_host_verified_backup=True,physical_failure_domain_independence_audited=False,
            catalog=catalog,source=SOURCE,destination=DEST,destination_root=dst+'/raw',
            source_completion_sha256=PIN,guards=[sender,receiver,audit],wall_seconds=time.monotonic()-started,
            new_neural_runs=0,scientific_acceptance=False,automatic_retry=False,
            evidence_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
        write(output/'complete.json',report)
        print(json.dumps(report),flush=True)
    except BaseException as exc:
        write(output/'failure.json',dict(error=str(exc),wall_seconds=time.monotonic()-started,automatic_retry=False,complete=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','readback'])
    p.add_argument('--evidence',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--root',type=Path);p.add_argument('--catalog',type=Path);p.add_argument('--receipt',type=Path)
    a=p.parse_args()
    if a.mode=='readback':
        need(not a.receipt.exists(),'readback receipt exists')
        write(a.receipt,readback(a.root,json.loads(a.catalog.read_text())))
    else:run(a.evidence,a.output)
