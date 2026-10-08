"""One bounded Teleport stream and independent T7 readback of accepted raw data.

Only complete.json publishes acceptance. Failed/incomplete attempts retain all
bytes. No overwrite, simulation retry, source deletion, compression or listener.
"""
import argparse,fcntl,hashlib,json,os,re,resource,select,shlex,signal,stat,subprocess,sys,time
from pathlib import Path
from mam_launch_native_primary import read,sha
from mam_rust_science_reuse import accepted_new_raw
from mam_launch_rust_recovery import CASE,LABEL
from mam_launch_rust_performance import BRICK,NODES
from mam_launch_rust_benchmark_raw import remote
from mam_benchmark_terminal import guard

BLOCK=2**20
MEMORY=512*2**20
FILE_LIMIT=64*2**30
TOTAL_SECONDS=10800
TRANSFER_SECONDS=7200
RESERVE=1536*2**30
PREFIX='b2mpi-t7-raw-rust32-v1'
BASE=Path('/data/brick2/brian2-mpi-region-20260907')
GUARD_SOURCE=BASE/'rust-target-raw-v1-source/tools/mpi_resource_guard.py'
GUARD_SHA='630fc35f0b38729fc202925b09b4eea55cc965cfeee396ace6085a3a98004014'

def need(ok,message):
    if not ok:raise ValueError(message)

def write_new(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def disk(path,reserve,extra=0):
    s=os.statvfs(path);free=s.f_bavail*s.f_frsize
    need(free>=reserve+extra,'destination reserve would be crossed')
    return free

def no_cache(fd):
    if sys.platform=='darwin':fcntl.fcntl(fd,48,1) # F_NOCACHE, macOS sys/fcntl.h

def memory_admission(text):
    total=re.findall(r'The system has ([0-9]+) ',text)
    free=re.findall(r'System-wide memory free percentage: ([0-9]+)%',text)
    need(len(total)==len(free)==1,'unrecognized local memory-pressure output')
    total,free=int(total[0]),int(free[0]);need(0<=free<=100,'invalid local free-memory percentage')
    estimate=total*free//100
    need(estimate>=MEMORY+2*2**30,'insufficient local memory headroom')
    return dict(total_bytes=total,free_percentage=free,estimated_free_bytes=estimate,
        required_estimated_free_bytes=MEMORY+2*2**30,source='memory_pressure -Q')

def copy_payload(reader,path,expected,deadline,check_disk,progress=lambda count:None):
    """reader(size, timeout) must return bounded bytes, b'' only at EOF."""
    need(type(expected['bytes']) is int and 0<expected['bytes']<=FILE_LIMIT,'file budget')
    count=0;digest=hashlib.sha256();next_sync=64*BLOCK
    with path.open('xb') as f:
        no_cache(f.fileno())
        while count<expected['bytes']:
            remaining=deadline-time.monotonic();need(remaining>0,'transfer deadline')
            size=min(BLOCK,expected['bytes']-count)
            block=reader(size,min(30,remaining));need(0<len(block)<=size,'truncated or oversized payload')
            check_disk(len(block));need(f.write(block)==len(block),'short destination write')
            count+=len(block);digest.update(block)
            if count>=next_sync:
                f.flush();os.fsync(f.fileno());next_sync=count+64*BLOCK
            progress(count)
        need(deadline>time.monotonic(),'transfer deadline')
        need(reader(1,min(30,deadline-time.monotonic()))==b'','trailing payload')
        f.flush();os.fsync(f.fileno())
    need(digest.hexdigest()==expected['sha256'],'transferred payload digest mismatch')
    return dict(bytes=count,sha256=digest.hexdigest())

def readback(path,expected,deadline,progress=lambda count:None):
    digest=hashlib.sha256();count=0
    with path.open('rb') as f:
        no_cache(f.fileno())
        before=os.fstat(f.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size==expected['bytes'],'readback file size/type')
        while block:=f.read(BLOCK):
            need(time.monotonic()<deadline,'readback deadline')
            count+=len(block);digest.update(block);progress(count)
        after=os.fstat(f.fileno())
        need((before.st_ino,before.st_size,before.st_mtime_ns)==
             (after.st_ino,after.st_size,after.st_mtime_ns),'destination changed during readback')
    need(count==expected['bytes'] and digest.hexdigest()==expected['sha256'],'independent readback mismatch')
    return dict(bytes=count,sha256=digest.hexdigest(),independent_readback=True)

def sender_code(source,expected,receipt):
    return f'''import hashlib,json,os,stat,sys,time
from pathlib import Path
p=Path({str(source)!r});root=Path({BRICK!r})
assert p.resolve().is_relative_to(root) and not p.is_symlink()
fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);start=time.monotonic();digest=hashlib.sha256();count=0;released=0
with os.fdopen(fd,'rb') as f:
 before=os.fstat(f.fileno());assert stat.S_ISREG(before.st_mode) and before.st_size=={expected['bytes']}
 while block:=f.read({BLOCK}):
  count+=len(block);assert count<={expected['bytes']};digest.update(block)
  assert sys.stdout.buffer.write(block)==len(block)
  if count-released>=64*{BLOCK}:
   os.posix_fadvise(f.fileno(),released,count-released,os.POSIX_FADV_DONTNEED);released=count
 sys.stdout.buffer.flush()
 os.posix_fadvise(f.fileno(),released,count-released,os.POSIX_FADV_DONTNEED)
 after=os.fstat(f.fileno());assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
assert count=={expected['bytes']} and digest.hexdigest()=={expected['sha256']!r}
q=Path({str(receipt)!r});q.parent.mkdir(parents=True,exist_ok=True)
with q.open('x') as f:
 json.dump(dict(source=str(p),bytes=count,sha256=digest.hexdigest(),wall_seconds=time.monotonic()-start,source_unchanged=True),f);f.flush();os.fsync(f.fileno())
'''

def source_command(index,expected,seconds):
    name=PREFIX+'-file'+str(index);g=BASE/'guards'/(name+'.json');receipt=BASE/'bulk-backup-receipts'/(name+'.json')
    source=Path(BRICK)/'runs'/LABEL/expected['name']
    app=['/usr/bin/python3','-c',sender_code(source,expected,receipt)]
    cmd=['systemd-run','--expand-environment=no','--quiet','--wait','--pipe','--collect',
        '--unit='+name,'--uid=rock','--service-type=exec','--property=MemoryMax=512M',
        '--property=MemorySwapMax=0','--property=CPUQuota=100%','--property=AllowedCPUs=8-9',
        '--property=TasksMax=64','--property=RuntimeMaxSec='+str(seconds+5),'--property=TimeoutStopSec=5',
        '--property=KillMode=control-group','--property=OOMPolicy=continue','/usr/bin/python3',str(GUARD_SOURCE),
        '--output',str(g),'--volume','/data/brick2','--memory-mib','512','--cpu-percent','100',
        '--file-mib','1','--min-free-gib','1280','--timeout',str(seconds),'--',*app]
    return cmd,g,receipt,app

def preflight(files):
    paths=[str(BASE/'guards'/(PREFIX+'-file'+str(i)+'.json')) for i in range(2)]
    paths += [str(BASE/'bulk-backup-receipts'/(PREFIX+'-file'+str(i)+'.json')) for i in range(2)]
    return remote(f'''import os,json,signal,resource,hashlib,subprocess,datetime,time
from pathlib import Path
signal.alarm(20);resource.setrlimit(resource.RLIMIT_AS,(128*2**20,128*2**20));resource.setrlimit(resource.RLIMIT_CPU,(5,5))
assert os.uname().nodename=={NODES[0]!r}
b=Path('/data/brick2');assert b.is_mount() and b.stat().st_dev!=Path('/').stat().st_dev
s=os.statvfs(b);assert s.f_bavail*s.f_frsize>=1280*2**30
m={{k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines()) if k=='MemAvailable'}};assert m['MemAvailable']>=65*2**30
assert not subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5).strip()
assert hashlib.sha256(Path({str(GUARD_SOURCE)!r}).read_bytes()).hexdigest()=={GUARD_SHA!r}
for name in {paths!r}:assert not Path(name).exists()
for item in {files!r}:
 p=Path({BRICK!r})/'runs'/{LABEL!r}/item['name'];assert not p.is_symlink() and p.is_file() and p.stat().st_size==item['bytes']
def ticks():return {{int(x[0][3:]):list(map(int,x[1:9])) for line in Path('/proc/stat').read_text().splitlines() if (x:=line.split())[0].startswith('cpu') and x[0][3:].isdigit()}}
a=ticks();time.sleep(1);z=ticks();busy={{}}
for c in [8,9]:
 d=[y-x for x,y in zip(a[c],z[c])];assert sum(d)>0;busy[c]=100*(1-(d[3]+d[4])/sum(d))
assert max(busy.values())<50 and sum(busy.values())/2<25
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),available_memory_bytes=m['MemAvailable'],free_bytes=s.f_bavail*s.f_frsize,cpu_busy_percent=busy,active_own_units=[],guard_sha256={GUARD_SHA!r})))''')

def collect_guard(index,expected,seconds,command,gpath,rpath,app):
    result=remote(f'''import json,signal,resource,subprocess
from pathlib import Path
signal.alarm(15);resource.setrlimit(resource.RLIMIT_AS,(128*2**20,128*2**20));resource.setrlimit(resource.RLIMIT_CPU,(5,5))
assert not subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend',{PREFIX+'-file'+str(index)+'.service'!r}],text=True,timeout=5).strip()
out={{}}
for n,p in {dict(guard=str(gpath),sender=str(rpath))!r}.items():
 q=Path(p);assert q.is_file() and not q.is_symlink() and q.stat().st_size<2**20;out[n]=json.loads(q.read_text())
print(json.dumps(out))''')
    need(result['guard']['command']==app,'sender guard command differs')
    spec=dict(host=NODES[0],role='file'+str(index),memory_bytes=MEMORY,pids_max=64,cpu_ids=[8,9],cpu_quota_cores=1,
        volume='/data/brick2',allow_root_volume=False,file_limit_bytes=2**20,minimum_free_bytes=1280*2**30,reserved_host_memory_bytes=64*2**30)
    result['accounting']=guard(result['guard'],spec,PREFIX,seconds)
    r=result['sender'];need(r['source_unchanged'] and r['bytes']==expected['bytes'] and r['sha256']==expected['sha256'],'sender receipt differs')
    return result

def worker(evidence,t7,completion_sha):
    resource.setrlimit(resource.RLIMIT_CPU,(1200,1200));resource.setrlimit(resource.RLIMIT_FSIZE,(FILE_LIMIT,FILE_LIMIT))
    def expire(*_):raise TimeoutError('whole backup deadline')
    signal.signal(signal.SIGALRM,expire);signal.alarm(TOTAL_SECONDS)
    start=time.monotonic();deadline=start+TOTAL_SECONDS
    case=evidence/'performance-runs-v1'/CASE
    raw=accepted_new_raw(evidence,completion_sha);need(raw is not None,'accepted complete raw audit required')
    volume=Path('/Volumes/T7');need(volume.is_mount() and t7.resolve().is_relative_to(volume.resolve()),'mounted T7 destination required')
    memory=memory_admission(subprocess.check_output(['/usr/bin/memory_pressure','-Q'],text=True,timeout=5))
    files=[dict(name=n,bytes=raw['dump_bytes'][n],sha256=raw['dump_sha256'][n]) for n in ['results.bin','events.bin']]
    need(sum(f['bytes'] for f in files)<=128*2**30,'whole raw backup ceiling')
    output=t7/'backups'/(CASE+'-v1');output.parent.mkdir(exist_ok=True)
    need(not output.parent.is_symlink() and output.parent.resolve().is_relative_to(volume.resolve()),'backup parent outside T7')
    need(not output.exists(),'backup attempt already exists; no retry')
    free=disk(output.parent,RESERVE,sum(f['bytes'] for f in files));output.mkdir(mode=0o700)
    write_new(output/'intent.json',dict(case_id=CASE,completion_sha256=completion_sha,files=files,
        whole_seconds=TOTAL_SECONDS,transfer_seconds=TRANSFER_SECONDS,reserve_bytes=RESERVE,
        free_bytes_before=free,local_memory_admission=memory,automatic_retry=False,implementation_sha256=sha(Path(__file__))))
    process=None;transfer_used=0.;receipts=[];last_progress=0.
    try:
        controls=output/'controls';controls.mkdir()
        names=['completion.json','admission.json','launch.json','terminal/report.json','raw/report.json']
        names += ['terminal/host-'+str(i)+'.json.gz' for i in range(4)]
        copied_controls={}
        for name in names:
            source=case/name;need(source.stat().st_size<=16*2**20,'bounded backup control')
            target=controls/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(source.read_bytes());f.flush();os.fsync(f.fileno())
            need(sha(source)==sha(target),'backup control readback mismatch')
            copied_controls[name]=dict(bytes=target.stat().st_size,sha256=sha(target))
        admission=preflight(files);write_new(output/'admission.json',admission)
        for index,item in enumerate(files):
            seconds=int(min(TRANSFER_SECONDS-transfer_used,deadline-time.monotonic()))
            need(seconds>0,'remaining transfer budget exhausted')
            cmd,gpath,rpath,app=source_command(index,item,seconds)
            write_new(output/f'file-{index}.command.json',dict(command=cmd,seconds=seconds))
            def progress(count,phase='transfer'):
                nonlocal last_progress
                if time.monotonic()-last_progress>=30:
                    print(json.dumps(dict(event='backup_progress',file=item['name'],phase=phase,bytes=count,total_bytes=item['bytes'])),flush=True)
                    last_progress=time.monotonic()
            tstart=time.monotonic()
            with (output/f'file-{index}.stderr').open('xb') as err:
                process=subprocess.Popen(['tsh','ssh','root@'+NODES[0],shlex.join(cmd)],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=err,env={**os.environ,'GOMAXPROCS':'1'})
                def reader(size,timeout):
                    ready,_,_=select.select([process.stdout],[],[],timeout)
                    if not ready:raise TimeoutError('payload idle timeout')
                    return os.read(process.stdout.fileno(),size)
                copied=copy_payload(reader,output/item['name'],item,min(deadline,tstart+seconds),
                    lambda extra:disk(output,RESERVE,extra),progress)
                rc=process.wait(timeout=max(1,min(30,deadline-time.monotonic())))
                need(rc==0,'Teleport sender unsuccessful')
            transfer_used+=time.monotonic()-tstart
            resource_receipt=collect_guard(index,item,seconds,cmd,gpath,rpath,app)
            write_new(output/f'file-{index}.remote.json',resource_receipt)
            verified=readback(output/item['name'],item,deadline,lambda count:progress(count,'readback'))
            row=dict(**item,transfer=copied,readback=verified,remote_receipt_sha256=sha(output/f'file-{index}.remote.json'))
            write_new(output/f'file-{index}.verified.json',row);receipts.append(row)
        need(time.monotonic()<deadline,'whole backup budget exhausted')
        result=dict(schema='b2-mam-independent-raw-backup-v1',case_id=CASE,files=receipts,
            complete=True,independent_full_raw_backup=True,source_files_deleted=False,
            controls=copied_controls,scope='Two complete raw output files plus audit controls; other source/analysis artifacts remain in their separately verified T7 archives.',
            completion_sha256=completion_sha,transfer_seconds=transfer_used,wall_seconds=time.monotonic()-start,
            scientific_acceptance=False,performance_cost_acceptance=False)
        write_new(output/'pending.json',result)
        return result
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        write_new(output/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),wall_seconds=time.monotonic()-start,
            automatic_retry=False,partial_files_retained=True,observation_failure_is_not_remote_completion=True))
        raise

def run(args):
    local=args.evidence/'performance-runs-v1'/CASE/'full-raw-backup-v1';local.mkdir(exist_ok=False)
    output=args.t7/'backups'/(CASE+'-v1')
    start=time.monotonic();peak=0;reason=None
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--evidence',str(args.evidence),'--t7',str(args.t7),'--completion-sha256',args.completion_sha256]
    child=subprocess.Popen(cmd,start_new_session=True)
    try:
        while child.poll() is None:
            r=subprocess.run(['/bin/ps','-axo','pid=,ppid=,rss='],capture_output=True,text=True,timeout=5,check=True)
            rows=[list(map(int,line.split())) for line in r.stdout.splitlines() if line.strip()]
            owned={child.pid}
            for _ in range(8):owned.update(pid for pid,parent,rss in rows if parent in owned)
            peak=max(peak,sum(rss*1024 for pid,parent,rss in rows if pid in owned))
            if peak>MEMORY:reason='sampled own-process-tree RSS limit'
            if time.monotonic()-start>TOTAL_SECONDS+30:reason='whole supervisor deadline'
            if reason:os.killpg(child.pid,signal.SIGKILL);break
            time.sleep(.5)
        rc=child.wait()
    except BaseException:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
    control=dict(returncode=rc,stop_reason=reason,sampled_peak_tree_rss_bytes=peak,sampled_tree_rss_limit_bytes=MEMORY,
        wall_seconds=time.monotonic()-start,wall_limit_seconds=TOTAL_SECONDS+30,command=cmd,
        automatic_retry=False,memory_limit_kind='sampled local process tree; remote cgroup enforced')
    write_new(local/'supervisor.json',control)
    need(rc==0 and reason is None,'backup worker unsuccessful; retain attempt, no retry')
    pending=read(output/'pending.json');need(pending['complete'] and pending['independent_full_raw_backup'],'pending backup missing')
    pending['supervisor']=control;pending['supervisor_sha256']=sha(local/'supervisor.json')
    write_new(output/'complete.json',pending);write_new(local/'complete.json',pending)
    print(json.dumps(dict(complete=True,independent_full_raw_backup=True,path=str(output),wall_seconds=pending['wall_seconds'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--t7',type=Path,required=True);p.add_argument('--completion-sha256',required=True)
    p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:worker(a.evidence,a.t7,a.completion_sha256)
    else:run(a)
