"""One bounded deployment of a pinned replica, only each host's owned rank shards."""
import argparse
import concurrent.futures
import hashlib
import inspect
import json
from pathlib import Path
import shlex
import subprocess
import time

from mam_confirmation_identity import identity,deployment_catalog,NODES,IPS,HOME,BRICK,SOURCE,PROJECT,need
from mam_collection_transfer import validate
from mam_collect_native_primary_raw import remote
from mam_benchmark_terminal import guard

REPLICATE=1750
TAG='confirm1750-deploy-v1'
PREFIX='b2mpi-'+TAG
CONTROLS=[BRICK+'/confirmation-deployment-v1']+[HOME+'/confirmation-deployment-v1']*3
OUTPUTS=[BRICK+'/confirmation-v1/seed1750']+[PROJECT]*3
MODULES=['mam_collection_transfer.py','mam_direct_transfer_v3.py','mpi_resource_guard.py','mam_cluster_raw_backup.py']
TOTAL=1800


def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def preflight(index,catalog,source=False):
    node=NODES[index];base=BRICK if index==0 else HOME;reserve=1296 if index==0 else 129
    return remote(node,f'''import os,json,time,datetime,signal,resource,subprocess,hashlib
from pathlib import Path
signal.alarm(25);resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20));resource.setrlimit(resource.RLIMIT_CPU,(10,10))
b=Path({base!r});assert b.resolve()==b and os.uname().nodename=={node!r}
assert (b.stat().st_dev!=Path('/').stat().st_dev)=={index==0!r}
assert not subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5).strip()
s=os.statvfs(b);free=s.f_bavail*s.f_frsize;assert free>={reserve}*2**30
m={{k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())}};assert m['MemAvailable']>=80*2**30
if not {source!r}:assert not Path({OUTPUTS[index]!r}).exists()
if {source!r}:
 for row in {catalog['files']!r}:
  p=Path({SOURCE!r})/row['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes']
  with p.open('rb') as f:
   assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256'];os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
def ticks():return {{int(x[0][3:]):list(map(int,x[1:9])) for l in Path('/proc/stat').read_text().splitlines() if (x:=l.split())[0].startswith('cpu') and x[0][3:].isdigit()}}
a=ticks();time.sleep(1);z=ticks();busy={{}}
for c in [8,9]:
 d=[y-x for x,y in zip(a[c],z[c])];assert sum(d)>0;busy[c]=100*(1-(d[3]+d[4])/sum(d))
assert max(busy.values())<50 and sum(busy.values())/2<25,busy
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=free,available_memory_bytes=m['MemAvailable'],cpu_busy_percent=busy,active_units=[],source_hashes_checked={source!r})))''')


CACHE_HASHES={
 'mam_direct_transfer_v3.cpython-312.pyc':'0f91cd41fcf0e5dfdfe736b4857c9befc74495bfee91c8178bfc77ea91f1ab4a',
 'mam_collection_transfer.cpython-312.pyc':'39b02ce902d8b28993117eb7ecdb7fe0d10c923f25a6899392b3b019a767c866'}


def check_staged_source(source,modules,cache_hashes):
    names={p.name for p in source.iterdir()}
    if names-set(modules) not in [set(),{'__pycache__'}] or not set(modules)<=names:
        raise ValueError('unexpected staged source members')
    for name,text in modules.items():
        p=source/name
        if p.is_symlink() or not p.is_file() or p.read_text()!=text:
            raise ValueError('staged source changed')
    if '__pycache__' in names:
        cache=source/'__pycache__'
        if cache.is_symlink() or not cache.is_dir():raise ValueError('invalid bytecode cache')
        for p in cache.iterdir():
            if p.name not in cache_hashes or p.is_symlink() or not p.is_file() or p.stat().st_size>65536:
                raise ValueError('unexpected bytecode cache entry')
            if hashlib.sha256(p.read_bytes()).hexdigest()!=cache_hashes[p.name]:
                raise ValueError('bytecode cache changed')


def stage(index,which,catalog,bundle):
    base=CONTROLS[index];node=NODES[index]
    return remote(node,f'''import json,sys,hashlib
from pathlib import Path
{inspect.getsource(check_staged_source)}
raw=json.loads(sys.stdin.read());b=Path({base!r});b.mkdir(exist_ok=True);source=b/'source'
assert b.resolve().is_relative_to(Path({'/data/brick2' if index==0 else '/home/rock'!r}))
if source.exists():
 check_staged_source(source,raw['modules'],{CACHE_HASHES!r})
else:
 source.mkdir()
 for n,s in raw['modules'].items():(source/n).open('x').write(s)
(b/{('catalog-node'+str(which)+'.json')!r}).open('x').write(json.dumps(raw['catalog']))
parent=Path({OUTPUTS[index]!r}).parent;parent.mkdir(exist_ok=True)
assert parent.resolve().is_relative_to(Path({'/data/brick2' if index==0 else '/home/rock'!r}))
print(json.dumps(dict(source_sha256={{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in raw['modules']}})))''',json.dumps(dict(modules=bundle,catalog=catalog)).encode())


def guarded(index,role,seconds,app,directory):
    app=['env','PYTHONDONTWRITEBYTECODE=1',*app]
    base=CONTROLS[index];node=NODES[index]
    cmd=['systemd-run','--expand-environment=no','--quiet','--wait','--pipe','--collect',
        '--unit='+PREFIX+'-'+role,'--uid=rock','--service-type=exec',
        '--property=MemoryMax=4096M','--property=MemorySwapMax=0','--property=CPUQuota=200%',
        '--property=AllowedCPUs=8-9','--property=TasksMax=64','--property=RuntimeMaxSec='+str(seconds+5),
        '--property=TimeoutStopSec=5','--property=KillMode=control-group','--property=OOMPolicy=continue',
        '/usr/bin/python3',base+'/source/mpi_resource_guard.py','--output',base+'/'+role+'-guard.json',
        '--volume','/data/brick2' if index==0 else '/','--memory-mib','4096','--cpu-percent','200',
        '--file-mib','128','--min-free-gib','1280' if index==0 else '128','--timeout',str(seconds)]
    if index:cmd.append('--allow-root-volume')
    cmd+=['--',*app];write(directory/(role+'-command.json'),dict(command=cmd,timeout_seconds=seconds))
    with (directory/(role+'.log')).open('x') as log:
        rc=subprocess.run(['tsh','ssh','root@'+node,shlex.join(cmd)],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,timeout=seconds+25).returncode
    g=remote(node,f"from pathlib import Path;print(Path({base+'/'+role+'-guard.json'!r}).read_text())")
    write(directory/(role+'-guard.json'),g);need(rc==0 and g['command']==app,'transfer command failed')
    spec=dict(host=node,role=role,memory_bytes=4*2**30,pids_max=64,cpu_ids=[8,9],cpu_quota_cores=2,
        volume='/data/brick2' if index==0 else '/',allow_root_volume=index!=0,file_limit_bytes=128*2**20,
        minimum_free_bytes=(1280 if index==0 else 128)*2**30,reserved_host_memory_bytes=64*2**30)
    return guard(g,spec,PREFIX,seconds)


def recovery_gate(prior,value):
    need(prior.name=='confirmation-deployment-v1-seed1750' and not (prior/'complete.json').exists(),'only fixed staging failure may resume')
    raw=(prior/'failure.json').read_bytes()
    need(hashlib.sha256(raw).hexdigest()=='883ee7e811f3a44e4bdd8ba811e7c436999a3f93bbc8bf4fb7be2f4643382116','prior failure changed')
    failed=json.loads(raw);need(len(failed['completed'])==1,'unexpected prior completed host count')
    need(json.loads((prior/'identity.json').read_text())==value,'prior artifact identity changed')
    row=failed['completed'][0];catalog=deployment_catalog(value,0);total,digest=validate(catalog,64*2**20,128*2**20)
    need(row['host']==NODES[0] and row['owned_ranks']==list(range(8)) and row['bytes']==total and row['catalog_sha256']==digest,'prior host0 coverage')
    directory=prior/'node0';actual=json.loads((directory/'readback.json').read_text())
    need(actual['complete'] and actual['files']==catalog['files'] and actual['catalog_sha256']==digest and actual['executable_mode']=='0o755','prior host0 readback')
    need(json.loads((directory/'catalog.json').read_text())==catalog,'prior catalog differs')
    accounting=[]
    for operation in ['send','receive','readback']:
        role='0-'+operation;g=json.loads((directory/(role+'-guard.json')).read_text());cmd=json.loads((directory/(role+'-command.json')).read_text())
        need(g['command']==cmd['command'][cmd['command'].index('--')+1:],'prior guard command')
        spec=dict(host=NODES[0],role=role,memory_bytes=4*2**30,pids_max=64,cpu_ids=[8,9],cpu_quota_cores=2,
            volume='/data/brick2',allow_root_volume=False,file_limit_bytes=128*2**20,
            minimum_free_bytes=1280*2**30,reserved_host_memory_bytes=64*2**30)
        accounting.append(guard(g,spec,PREFIX,cmd['timeout_seconds']))
    need(accounting==row['guards'],'prior host0 resource accounting')
    diagnosis=json.loads((prior/'source-stage-diagnosis.json').read_text())
    need(diagnosis['active_units']=='' and diagnosis['node1_catalog_exists'] is False,'failure was after transfer or job still live')
    # Charge repair and observation time too, conservatively from original intent creation.
    previous=time.time()-(prior/'intent.json').stat().st_mtime
    need(failed['elapsed_seconds']<=previous<TOTAL-1080,'insufficient original budget for the three unstarted hosts')
    proof=dict(prior=str(prior),charged_prior_wall_seconds=previous,
        original_intent_mtime_unix=(prior/'intent.json').stat().st_mtime,
        completed_hosts=1,remaining_private_collections=3,remaining_readbacks=3,
        prior_files={str(p.relative_to(prior)):hashlib.sha256(p.read_bytes()).hexdigest() for p in prior.rglob('*') if p.is_file()})
    return [row],previous,proof


def configure(value):
    global REPLICATE,TAG,PREFIX,CONTROLS,OUTPUTS,SOURCE,PROJECT
    REPLICATE=value['replicate']
    need(type(REPLICATE) is int and REPLICATE in (1750,1751),'unsupported deployment replicate')
    TAG=f'confirm{REPLICATE}-deploy-v1';PREFIX='b2mpi-'+TAG
    suffix='/confirmation-deployment-v1'+(f'-seed{REPLICATE}' if REPLICATE!=1750 else '')
    CONTROLS=[BRICK+suffix]+[HOME+suffix]*3
    SOURCE=value['source_artifact'];PROJECT=value['project']
    OUTPUTS=[BRICK+f'/confirmation-v1/seed{REPLICATE}']+[PROJECT]*3


def run(evidence,out,prior=None,replicate=1750):
    need(prior is None or replicate==1750,'historical recovery is restricted to 1750')
    value=identity(evidence,replicate);configure(value)
    need(not out.exists(),'deployment attempt exists; no retry')
    catalogs=[deployment_catalog(value,i) for i in range(4)]
    need(sum(validate(c,64*2**20,128*2**20)[0] for c in catalogs)<=512*2**20,'deployment batch size')
    completed=[];previous=0.;proof=None
    if prior is not None:completed,previous,proof=recovery_gate(prior,value)
    out.mkdir();write(out/'identity.json',value)
    if proof is not None:write(out/'recovery-binding.json',proof)
    bundle={n:(Path(__file__).parent/n).read_text() for n in MODULES}
    expected={n:hashlib.sha256(s.encode()).hexdigest() for n,s in bundle.items()}
    write(out/'intent.json',dict(maximum_private_collections=4,maximum_readbacks=4,maximum_neural_runs=0,
        transfer_seconds_each=180,readback_seconds_each=90,whole_seconds=TOTAL,total_bytes_limit=512*2**20,
        memory_gib_per_service=4,cpu_cores_per_service=2,source_sha256=expected,automatic_retry=False,
        remaining_private_collections=4-len(completed),charged_prior_wall_seconds=previous))
    started=time.monotonic()
    try:
        for i in range(len(completed),4):
            catalog=catalogs[i]
            need(previous+time.monotonic()-started<TOTAL-360,'whole deployment budget insufficient for next host')
            directory=out/('node'+str(i));directory.mkdir();write(directory/'catalog.json',catalog)
            admitted=time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                source=pool.submit(preflight,0,catalog,True);dest=pool.submit(preflight,i,catalog,False)
                checks=dict(source=source.result(),destination=dest.result())
            write(directory/'admission.json',checks)
            staged={}
            for j in sorted({0,i}):
                staged[str(j)]=stage(j,i,catalog,bundle)
                need(staged[str(j)]['source_sha256']==expected,'deployed transfer code differs')
            write(directory/'stage.json',staged);need(time.monotonic()-admitted<60,'deployment admission stale')
            src=CONTROLS[0];dst=CONTROLS[i];name='catalog-node'+str(i)+'.json'
            ready=dst+'/ready-node'+str(i)+'.json';receipt=dst+'/receipt-node'+str(i)+'.json'
            app=['python3',dst+'/source/mam_collection_transfer.py','receive','--catalog',dst+'/'+name,
                '--ready',ready,'--bind',IPS[i],'--peer',IPS[0],'--output',OUTPUTS[i],'--receipt',receipt,
                '--max-file-bytes',str(64*2**20),'--max-total-bytes',str(128*2**20),
                '--reserve-bytes',str((1280 if i==0 else 128)*2**30)]
            first=time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                receiver=pool.submit(guarded,i,str(i)+'-receive',180,app,directory)
                deadline=time.monotonic()+45
                while True:
                    need(not receiver.done(),'receiver exited before ready')
                    control=remote(NODES[i],f"from pathlib import Path;p=Path({ready!r});print(p.read_text() if p.exists() else '{{}}')")
                    if control:break
                    need(time.monotonic()<deadline,'readiness deadline');time.sleep(.5)
                target=src+'/ready-to-node'+str(i)+'.json'
                remote(NODES[0],f"import os,sys;fd=os.open({target!r},os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)\nwith os.fdopen(fd,'wb') as f:f.write(sys.stdin.buffer.read())",json.dumps(control).encode())
                seconds=int(180-(time.monotonic()-first)-10);need(seconds>30,'sender budget exhausted')
                send=['python3',src+'/source/mam_collection_transfer.py','send','--catalog',src+'/'+name,'--ready',target,'--root',SOURCE]
                sg=guarded(0,str(i)+'-send',seconds,send,directory);rg=receiver.result()
            received=remote(NODES[i],f"from pathlib import Path;print(Path({receipt!r}).read_text())")
            total,digest=validate(catalog,64*2**20,128*2**20)
            need(received['complete'] is True and received['bytes']==total and received['catalog_sha256']==digest
                 and received['files']==16 and received['output']==OUTPUTS[i],'deployment receipt differs')
            write(directory/'receipt.json',received)
            audit_path=dst+'/readback-node'+str(i)+'.json'
            app=['python3',dst+'/source/mam_cluster_raw_backup.py','readback','--root',OUTPUTS[i],
                 '--catalog',dst+'/'+name,'--receipt',audit_path]
            ag=guarded(i,str(i)+'-readback',90,app,directory)
            audit=remote(NODES[i],f'''import json,subprocess,os,stat
from pathlib import Path
p=Path({OUTPUTS[i]!r});wanted={{r['path'] for r in {catalog['files']!r}}}
assert {{str(q.relative_to(p)) for q in p.rglob('*') if q.is_file()}}==wanted
assert not subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend',{(PREFIX+'-*')!r}],text=True,timeout=5).strip()
r=json.loads(Path({audit_path!r}).read_text())
q=p/'mpi/b2-mpi';assert not q.is_symlink();q.chmod(0o755)
with q.open('rb') as f:os.fsync(f.fileno())
r['executable_mode']=oct(stat.S_IMODE(q.stat().st_mode));assert os.access(q,os.X_OK)
print(json.dumps(r))''')
            need(audit['complete'] is True and audit['full_destination_readback'] is True
                 and audit['executable_mode']=='0o755' and audit['files']==catalog['files'] and audit['catalog_sha256']==digest and audit['bytes']==total,'deployment readback differs')
            write(directory/'readback.json',audit)
            row=dict(host=NODES[i],project=PROJECT,resolved_output=OUTPUTS[i],owned_ranks=list(range(i*8,(i+1)*8)),
                files=16,bytes=total,catalog_sha256=digest,full_readback=True,executable_mode=audit['executable_mode'],guards=[sg,rg,ag])
            completed.append(row);print(json.dumps(dict(event='host_deployed',host=NODES[i],ranks=row['owned_ranks'],bytes=total)),flush=True)
        need(previous+time.monotonic()-started<TOTAL,'whole deployment budget')
        report=dict(complete=True,identity_sha256=hashlib.sha256((out/'identity.json').read_bytes()).hexdigest(),
            replicate=REPLICATE,nodes=completed,all_32_owned_shards_deployed=True,neural_runs=0,launch_admitted=False,
            scientific_acceptance=False,elapsed_seconds=previous+time.monotonic()-started,automatic_retry=False,
            recovery_binding_sha256=hashlib.sha256((out/'recovery-binding.json').read_bytes()).hexdigest() if proof else None,
            evidence_sha256={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()})
        write(out/'complete.json',report);print(json.dumps(dict(complete=True,elapsed_seconds=report['elapsed_seconds'],bytes=sum(r['bytes'] for r in completed))),flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(error=str(exc),completed=completed,elapsed_seconds=previous+time.monotonic()-started,automatic_retry=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-prior',type=Path)
    p.add_argument('--replicate',type=int,choices=[1750,1751],default=1750)
    a=p.parse_args();run(a.evidence,a.output,a.resume_prior,a.replicate)
