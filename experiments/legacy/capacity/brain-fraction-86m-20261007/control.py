"""Bounded 30-node experiment; reuse tested launcher and guard, never old runs."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, importlib.util, io, json, os, shlex, subprocess, sys, tarfile, time
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'brain-fraction-20261007'
spec=importlib.util.spec_from_file_location('old_trial_control',OLD/'control.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
selection=json.loads((HERE/'selection.json').read_text())
NODES=selection['nodes'];BASE=selection['base'];REAL=selection['leader_real']
c.HERE=HERE;c.BASE=BASE;c.REAL=REAL;c.NODES=NODES;c.IPS=selection['ips']
c.ENV.update(TMPDIR=BASE+'/tmp',MPLCONFIGDIR=BASE+'/mpl-cache',LD_LIBRARY_PATH=BASE+'/mpi/lib',RUSTUP_AUTO_INSTALL='0',B2_MAX_NEURONS='86000000',B2_MAX_INITIAL_VALUES='400000000',B2_MAX_IR_BYTES=str(8*2**30))

def remote(*a,**kw):return c.remote(*a,**kw)
def record(*a):return c.record(*a)

def fetch_runtime():
    code="""import io,sys,tarfile
from pathlib import Path
b=Path('/atlas-home/0003/workspace/brian2-mpi-primary-20260909/mpi')
s=io.BytesIO()
with tarfile.open(fileobj=s,mode='w:gz') as t:
 for p in b.rglob('*'):
  if p.is_file() or p.is_symlink():t.add(p,arcname=str(Path('mpi')/p.relative_to(b)),recursive=False)
sys.stdout.buffer.write(s.getvalue())"""
    r=subprocess.run(['tsh','ssh','rock@'+NODES[0],shlex.join(['python3','-c',code])],capture_output=True,timeout=60)
    assert r.returncode==0,r.stderr;assert len(r.stdout)<32*2**20
    (HERE/'mpi-runtime.tar.gz').write_bytes(r.stdout)
    record('mpi-runtime-identity.json',{'bytes':len(r.stdout),'sha256':hashlib.sha256(r.stdout).hexdigest()})

def stage():
    source=(HERE/'source.tar.gz').read_bytes();runtime=(HERE/'mpi-runtime.tar.gz').read_bytes()
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w:gz') as t:
        for name,data in [('source.tar.gz',source),('mpi-runtime.tar.gz',runtime)]:
            m=tarfile.TarInfo(name);m.size=len(data);t.addfile(m,io.BytesIO(data))
        data=(HERE/'source/tests/test_resource_limits.py').read_bytes();m=tarfile.TarInfo('test_resource_limits.py');m.size=len(data);t.addfile(m,io.BytesIO(data))
    raw=payload.getvalue();digest=hashlib.sha256(raw).hexdigest();rows=[]
    for node in NODES:
        real=REAL if node==NODES[0] else '/data/brick2/atlas-capacity-86m-20261007' if node in selection['data_nodes'] else BASE
        volume='/data/brick2' if node in selection['data_nodes'] else '/'
        init=f'''from pathlib import Path
import json,os,pwd
p=Path({real!r});b=Path({BASE!r});assert not p.exists() and not b.exists()
v=Path({volume!r});assert v.is_mount();assert os.statvfs(v).f_bavail*os.statvfs(v).f_frsize>128*2**30
u=pwd.getpwnam('rock');p.mkdir();os.chown(p,u.pw_uid,u.pw_gid)
if b!=p:b.symlink_to(p);os.lchown(b,u.pw_uid,u.pw_gid)
print(json.dumps({{'host':os.uname().nodename,'base':str(b),'real':str(b.resolve()),'volume':str(v)}}))'''
        location=remote(node,init,root=True)
        code=f'''from pathlib import Path
import io,json,sys,tarfile,hashlib,os
b=Path({BASE!r});raw=sys.stdin.buffer.read(64*2**20);assert hashlib.sha256(raw).hexdigest()=={digest!r}
with tarfile.open(fileobj=io.BytesIO(raw)) as t:
 assert all(m.isfile() and Path(m.name).name==m.name for m in t.getmembers());t.extractall(b,filter='data')
s=b/'source';s.mkdir()
with tarfile.open(b/'source.tar.gz') as t:
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(s,filter='data')
(s/'tests').mkdir();(s/'tests/test_resource_limits.py').write_bytes((b/'test_resource_limits.py').read_bytes())
with tarfile.open(b/'mpi-runtime.tar.gz') as t:
 for m in t.getmembers():
  assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  assert m.isfile() or (m.issym() and not Path(m.linkname).is_absolute() and '..' not in Path(m.linkname).parts)
 t.extractall(b,filter='data')
for name in ['tmp','mpl-cache']:(b/name).mkdir()
p=b/'mpi/lib/libmpi.so.12.4.3';h=hashlib.sha256(p.read_bytes()).hexdigest();assert h=='c87d6e80f83078df26e08626e1b515b07116ac93225df43f68c943f205b6c4b4'
print(json.dumps({{'host':os.uname().nodename,'payload_sha256':{digest!r},'mpi_sha256':h}}))'''
        result=remote(node,code,raw);rows.append({'location':location,'result':result})
        (HERE/'stage-progress.json').write_text(json.dumps(rows,indent=2)+'\n');print(node,'staged',flush=True)
    record('stage.json',{'complete':len(rows)==30,'rows':rows})

def build():
    command=['sh','-c','cd '+BASE+'/source && /atlas-home/0003/workspace/brian2-lk-20260906/.cargo/bin/cargo build --release --offline --bin b2-runner -j 4 && /atlas-home/0003/workspace/brian2-lk-20260906/.cargo/bin/cargo test --offline resource_limits::tests -j 4']
    c.guarded('build-86m',command,16384,4,1200)

def prepare(case):
    if case.startswith('major'):
        assert json.loads((HERE/'pilot-p12-validation.json').read_text())['passed']
        assert json.loads((HERE/'pilot-p12-resources.json').read_text())['passed']
    n=24000 if case.startswith('pilot') else 86000000
    script='prepare-p12.py' if case.endswith('p12') else 'prepare.py'
    command=[c.PYTHON,BASE+'/source/experiment/'+script,'--source',BASE+'/source','--output',BASE+'/'+case,'--neurons',str(n),'--ranks','240','--compile']
    # Large model JSON is ~5 GiB; the stage needs an explicit larger file cap.
    original=c.guarded
    guarded('prepare-'+case,command,65536,4,1800)

def guarded(stage,command,memory,cpu,timeout):
    args=['systemd-run','--unit=b2mpi-86m-'+stage,'--uid=rock','--collect','--property=MemoryMax='+str(memory)+'M','--property=MemorySwapMax=0','--property=CPUQuota='+str(cpu*100)+'%','--property=AllowedCPUs=8-11','--property=TasksMax=64','--property=RuntimeMaxSec='+str(timeout+10),'--property=KillMode=control-group','--property=StandardOutput=file:'+BASE+'/'+stage+'.log','--property=StandardError=file:'+BASE+'/'+stage+'.log','env',*[k+'='+v for k,v in c.ENV.items()],'/usr/bin/python3',BASE+'/source/tools/mpi_resource_guard.py','--output',REAL+'/'+stage+'-guard.json','--volume','/data/brick2','--memory-mib',str(memory),'--cpu-percent',str(cpu*100),'--file-mib','8192','--min-free-gib','128','--timeout',str(timeout),'--',*command]
    r=remote(NODES[0],f"import subprocess,json\np=subprocess.run({args!r},capture_output=True,text=True);assert p.returncode==0,p.stderr\nprint(json.dumps({{'message':p.stderr,'command':{command!r}}}))",root=True)
    record(stage+'-start.json',r)

def status(stage):
    code=f'''from pathlib import Path
import json
b=Path({BASE!r});p=b/({stage!r}+'-guard.json')
g=json.loads(p.read_text()) if p.exists() else None
cg=Path('/sys/fs/cgroup')/g['cgroup'].lstrip('/') if g else None
live={{n:(cg/n).read_text().strip() for n in ['memory.current','memory.peak','memory.events','cpu.stat','pids.current'] if cg and (cg/n).exists()}}
print(json.dumps({{'guard':g,'live':live,'log':(b/({stage!r}+'.log')).read_text()[-3000:] if (b/({stage!r}+'.log')).exists() else ''}}))'''
    record(stage+'-status.json',remote(NODES[0],code))

def deploy(case,ranks_per_node=8):
    code=f'''from pathlib import Path
import io,json,sys,tarfile
p=Path({BASE!r})/{case!r};assert json.loads((p/'prepared.json').read_text())['compiled']
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 for f in (p/'mpi').iterdir():
  if f.is_file() and (f.name.startswith('instance.rank-') or f.name in ['b2-mpi','manifest.json','build.json','instance.bin','execution-plan.json']):t.add(f,arcname=f.name,recursive=False)
sys.stdout.buffer.write(buf.getvalue())'''
    r=subprocess.run(['tsh','ssh','rock@'+NODES[0],shlex.join(['python3','-c',code])],capture_output=True,timeout=120)
    assert r.returncode==0,r.stderr;assert len(r.stdout)<256*2**20
    (HERE/(case+'-mpi.tar.gz')).write_bytes(r.stdout)
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:
        files={m.name:t.extractfile(m).read() for m in t.getmembers()}
    def worker(i):
        buf=io.BytesIO()
        with tarfile.open(fileobj=buf,mode='w:gz') as t:
            for name,data in files.items():
                if name.startswith('instance.rank-') and not i*ranks_per_node<=int(name.split('-')[1].split('.')[0])<(i+1)*ranks_per_node:continue
                m=tarfile.TarInfo(name);m.size=len(data);t.addfile(m,io.BytesIO(data))
        raw=buf.getvalue();digest=hashlib.sha256(raw).hexdigest()
        code=f'''from pathlib import Path
import hashlib,io,json,os,sys,tarfile
b=Path({BASE!r});p=b/{case!r}/'mpi';assert not p.exists()
assert os.statvfs(b).f_bavail*os.statvfs(b).f_frsize>128*2**30
raw=sys.stdin.buffer.read(256*2**20);assert hashlib.sha256(raw).hexdigest()=={digest!r}
p.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(raw)) as t:
 assert all(m.isfile() and Path(m.name).name==m.name for m in t.getmembers());t.extractall(p,filter='data')
(p/'b2-mpi').chmod(0o755)
print(json.dumps({{'host':os.uname().nodename,'files':{{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file()}},'owned_ranks':list(range({i*ranks_per_node},{(i+1)*ranks_per_node}))}}))'''
        row=remote(NODES[i],code,raw);print(NODES[i],'deployed',flush=True);return i,row
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(worker,range(1,len(NODES))))
    record(case+'-deployment.json',[row for _,row in sorted(rows)])

def reference(case='pilot'):
    guarded('reference-'+case,[BASE+'/source/target/release/b2-runner',BASE+'/'+case+'/model.json',BASE+'/'+case+'/reference'],4096,1,300)

def budget_tests():
    guarded('budget-tests',['env','B2_RUNNER='+BASE+'/source/target/release/b2-runner',c.PYTHON,'-m','pytest','-q',BASE+'/source/tests/test_resource_limits.py','--junitxml='+BASE+'/budget-tests.xml','--basetemp='+BASE+'/tmp/budget-tests'],4096,2,180)

def validate_pilot(case='pilot',expected_ranks=240):
    if case=='pilot':return c.validate_pilot()
    code=f'''from pathlib import Path
import sys,json,numpy as np
b=Path({BASE!r});p=b/{case!r};sys.path.insert(0,str(b/'source/python'))
from brian2_rust.results import load_results
m=json.loads((p/'model.json').read_text());a=load_results(m,p/'reference');z=load_results(m,p/'result');checks=[];maximum=0
for x,y in zip(a['populations'],z['populations'],strict=True):
 for k in ['spike_ticks','indices','counts','last_spikes']:checks.append(bool(np.array_equal(x[k],y[k])))
 for kind in ['states','trace','refractory']:
  for k,v in x[kind].items():
   w=y[kind][k];assert np.shape(v)==np.shape(w);error=float(np.max(np.abs(np.asarray(v,dtype=float)-np.asarray(w,dtype=float)),initial=0));maximum=max(maximum,error);checks.append(error<=1e-12)
checks.extend([a['synaptic_events']==z['synaptic_events'],z['synaptic_events']>0,z['metadata']['spike_count']>0])
r=json.loads((p/'result/mpi-runtime.json').read_text());assert r['ranks']=={expected_ranks} and len(set(r['processor_names']))==30
print(json.dumps({{'passed':all(checks),'case':{case!r},'checks':len(checks),'max_state_absolute_difference':maximum,'spikes':z['metadata']['spike_count'],'synaptic_events':z['synaptic_events'],'layers':m['protocol']['layers'],'runtime':r}}))'''
    record(case+'-validation.json',remote(NODES[0],code,python=c.PYTHON))

def collect(case,ranks_per_node=8):
    launch=json.loads((HERE/(case+'-launch-result.json')).read_text());prefix=launch['resource_guard']['unit_prefix']
    def worker(i):
        roles=['proxy-'+str(i)]+(['controller'] if i==0 else [])
        code=f'''from pathlib import Path
import json,os
b=Path({BASE!r});roles={roles!r};prefix={prefix!r}
guards={{role:json.loads((b/'guards'/(prefix+'-'+role+'.json')).read_text()) for role in roles}}
assert all(g.get('returncode')==0 and g['admitted'] and 'oom_kill 0' in g['after']['memory.events'] for g in guards.values())
times={{p.name:p.read_text() for p in (b/{case!r}).glob('rank-*.time')}}
assert len(times)=={ranks_per_node} and all('Exit status: 0' in t for t in times.values())
r={{'host':os.uname().nodename,'guards':guards,'rank_times':times}}
if {i==0!r}:r.update(prepared=json.loads((b/{case!r}/'prepared.json').read_text()),summary=json.loads((b/{case!r}/'result/summary.json').read_text()),runtime=json.loads((b/{case!r}/'result/mpi-runtime.json').read_text()))
print(json.dumps(r))'''
        return remote(NODES[i],code)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(worker,range(len(NODES))))
    record(case+'-resources.json',{'passed':True,'rows':rows})

def monitor():
    c.monitor()

def start_audit(case='major-p12'):
    raw=(HERE/'audit.py').read_bytes();h=hashlib.sha256(raw).hexdigest()
    code=f'''from pathlib import Path
import sys,json,hashlib
p=Path({BASE!r})/'audit.py';assert not p.exists()
a=sys.stdin.buffer.read(65536);assert hashlib.sha256(a).hexdigest()=={h!r};p.write_bytes(a)
print(json.dumps({{'sha256':{h!r}}}))'''
    remote(NODES[0],code,raw)
    guarded('audit-'+case,[c.PYTHON,BASE+'/audit.py','--base',BASE,'--case',case],131072,4,1200)

def collect_audit(case='major-p12'):
    c.collect_audit(case)

def launch(case,memory=None,timeout=None,ranks_per_node=8,cpu_ids=None,edge_limit=450000000):
    assert json.loads((HERE/'budget-tests-corrected-status.json').read_text())['guard']['returncode']==0
    assert json.loads((HERE/'rust-budget-tests-status.json').read_text())['guard']['returncode']==0
    assert json.loads((HERE/'stage.json').read_text())['complete']
    assert json.loads((HERE/(case+'-deployment-audit.json')).read_text())['passed']
    if case.startswith('major'):assert json.loads((HERE/'pilot-p12-validation.json').read_text())['passed']
    sys.path.insert(0,str(OLD/'frozen-tools'))
    from mpi_teleport_launch import launch as run
    memory=memory if memory is not None else 4096 if case.startswith('pilot') else 262144
    timeout=timeout if timeout is not None else 300 if case.startswith('pilot') else 1800
    code=f'''from pathlib import Path
import json,os,subprocess
b=Path({BASE!r});m={{k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines()) if k=='MemAvailable'}}
assert m['MemAvailable']>{memory}*2**20+64*2**30
assert os.statvfs(b).f_bavail*os.statvfs(b).f_frsize>128*2**30
assert not subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True).strip()
print(json.dumps({{'host':os.uname().nodename,'memory':m}}))'''
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:admission=list(pool.map(lambda n:remote(n,code),NODES))
    record(case+'-admission.json',admission)
    cpu_ids=cpu_ids or [1,13,25,37,49,61,73,85]
    assert len(cpu_ids)==ranks_per_node and edge_limit<=2**32-1
    command='exec env B2_MPI_MAX_LOCAL_EDGES='+str(edge_limit)+' B2_THREAD_AFFINITY=required /usr/bin/time -v -o '+BASE+'/'+case+'/rank-${PMI_RANK}.time '+BASE+'/'+case+'/mpi/b2-mpi '+BASE+'/'+case+'/mpi/instance.bin '+BASE+'/'+case+'/result'
    overrides={node:{'volume':'/data/brick2','remote_base':REAL if node==NODES[0] else '/data/brick2/atlas-capacity-86m-20261007','script':(REAL if node==NODES[0] else '/data/brick2/atlas-capacity-86m-20261007')+'/source/tools/mpi_resource_guard.py'} for node in selection['data_nodes']}
    try:
        run(nodes=NODES,ips=selection['ips'],ranks_per_node=ranks_per_node,remote_base=BASE,application=['sh','-c',command],output=HERE/(case+'-launch'),mpi_prefix=BASE+'/mpi',timeout=timeout,login='root',guard_script=BASE+'/source/tools/mpi_resource_guard.py',guard_volume='/',guard_allow_root_volume=True,guard_memory_mib=memory,guard_cpu_percent=ranks_per_node*100,guard_cpu_count=ranks_per_node,guard_cpu_ids=cpu_ids,guard_file_mib=8192,guard_min_free_gib=128,guard_node_overrides=overrides)
    finally:
        path=HERE/(case+'-launch/launch.json')
        if path.exists():record(case+'-launch-result.json',json.loads(path.read_text()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fetch-runtime','stage','build','prepare','status','deploy','reference','validate-pilot','launch','budget-tests','collect','monitor','start-audit','collect-audit']);p.add_argument('case',nargs='?');a=p.parse_args()
    {'fetch-runtime':fetch_runtime,'stage':stage,'build':build,'prepare':lambda:prepare(a.case),'status':lambda:status(a.case),'deploy':lambda:deploy(a.case),'reference':lambda:reference(a.case or 'pilot'),'validate-pilot':lambda:validate_pilot(a.case or 'pilot'),'launch':lambda:launch(a.case),'budget-tests':budget_tests,'collect':lambda:collect(a.case),'monitor':monitor,'start-audit':lambda:start_audit(a.case or 'major-p12'),'collect-audit':lambda:collect_audit(a.case or 'major-p12')}[a.action]()
