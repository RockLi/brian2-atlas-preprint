"""Isolated, bounded scaling controller; previous experiments stay immutable."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
P = json.loads((HERE/'protocol.json').read_text())
SELECTION = json.loads((HERE/'selection.json').read_text())
BASE, REAL = SELECTION['base'], SELECTION['leader_real']
SOURCE = BASE+'/source-scale1pct-v1'
PYTHON = '/atlas-home/0003/workspace/brian2-lk-20260906/.venv/bin/python'
spec = importlib.util.spec_from_file_location('scale_base', HERE.parent/'brain-fraction-256m-20261007/base_control.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.HERE = HERE
base.c.HERE = HERE
ENV = dict(base.c.ENV, B2_MAX_NEURONS='860000000', B2_MAX_INITIAL_VALUES='4000000000',
           B2_MAX_IR_BYTES=str(64*2**30))
NODES, IPS = [], []


def record(name, value):
    (HERE/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'record':name,'passed':value.get('passed') if isinstance(value,dict) else None}),flush=True)


base.c.record = record


def configure(hosts):
    global NODES, IPS
    assert hosts in [3,6,12,24,30]
    NODES, IPS = SELECTION['nodes'][:hosts], SELECTION['ips'][:hosts]
    base.NODES = NODES
    base.c.NODES = NODES
    base.c.IPS = IPS


def remote(node, code, data=None, root=False, timeout=90, python='python3'):
    return base.remote(node,code,data,root=root,timeout=timeout,python=python)


def pilot(hosts):
    return f'pilot-weak1pct-h{hosts:02d}-v1'


def major(hosts):
    return f'weak1pct-h{hosts:02d}-'+('v2' if hosts in [24,30] else 'v1')



def resource_caps(case, hosts):
    if case.startswith('pilot-'):
        return [32]+[16]*(hosts-1)
    if case.endswith('-v2'):
        revision=json.loads((HERE/'protocol-revision-v2.json').read_text())
        assert hosts in revision['applies_to_hosts'] and revision['reserve_gib']==64
        return [768]+[660 if i%3==2 else 655 for i in range(1,hosts)]
    return [768]+[665]*(hosts-1)


def stage():
    raw = (HERE/'source.tar.gz').read_bytes()
    identity = json.loads((HERE/'source-identity.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == identity['archive_sha256']
    code = f'''from pathlib import Path
import hashlib,io,json,sys,tarfile
p=Path({SOURCE!r});assert not p.exists();raw=sys.stdin.buffer.read(16*2**20)
assert hashlib.sha256(raw).hexdigest()=={identity['archive_sha256']!r};p.mkdir()
with tarfile.open(fileobj=io.BytesIO(raw)) as t:
 members=t.getmembers();assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members);t.extractall(p,filter='data')
print(json.dumps({{'passed':True,'source_archive_sha256':{identity['archive_sha256']!r}}}))'''
    record('source-stage.json',remote(NODES[0],code,raw))


def guarded(stage, command, memory_gib, cpu, timeout):
    assert 1 <= memory_gib <= 768 and 1 <= cpu <= 4 and 1 <= timeout <= 7200
    args = ['systemd-run','--unit=b2mpi-'+stage,'--uid=rock','--collect',
        '--property=MemoryMax='+str(memory_gib*1024)+'M','--property=MemorySwapMax=0',
        '--property=CPUQuota='+str(cpu*100)+'%','--property=AllowedCPUs=8-11',
        '--property=TasksMax=64','--property=RuntimeMaxSec='+str(timeout+10),
        '--property=KillMode=control-group','--property=StandardOutput=file:'+BASE+'/'+stage+'.log',
        '--property=StandardError=file:'+BASE+'/'+stage+'.log','env',
        *[k+'='+v for k,v in ENV.items()],'/usr/bin/python3',SOURCE+'/tools/mpi_resource_guard.py',
        '--output',REAL+'/'+stage+'-guard.json','--volume','/data/brick2',
        '--memory-mib',str(memory_gib*1024),'--cpu-percent',str(cpu*100),
        '--file-mib',str(P['file_cap_gib']*1024),'--min-free-gib',str(P['disk_reserve_gib']),
        '--timeout',str(timeout),'--',*command]
    code = f'''from pathlib import Path
import subprocess,json
assert not (Path({BASE!r})/({stage!r}+'-guard.json')).exists()
p=subprocess.run({args!r},capture_output=True,text=True);assert p.returncode==0,p.stderr
print(json.dumps({{'command':{command!r},'message':p.stderr}}))'''
    record(stage+'-start.json',remote(NODES[0],code,root=True))


def status(stage):
    base.status(stage)


def build():
    assert json.loads((HERE/'source-stage.json').read_text())['passed']
    cargo = '/atlas-home/0003/workspace/brian2-lk-20260906/.cargo/bin/cargo'
    cmd = 'cd '+shlex.quote(SOURCE)+' && '+shlex.join([cargo,'build','--offline','--release','--bin','b2-runner','-j','4'])
    cmd += ' && '+shlex.join([cargo,'test','--offline','resource_limits::tests','-j','4'])
    guarded('build-scale1pct-v1',['sh','-c',cmd],16,4,1200)


def gate_tests():
    for stage_name in ['build-scale1pct-v1','budget-scale1pct-v1']:
        g = json.loads((HERE/(stage_name+'-status.json')).read_text())['guard']
        assert g['returncode'] == 0 and SOURCE in ' '.join(g['command'])
    assert json.loads((HERE/'launcher-tests.json').read_text())['passed']


def tests():
    assert json.loads((HERE/'build-scale1pct-v1-status.json').read_text())['guard']['returncode'] == 0
    guarded('budget-scale1pct-v1',['env','B2_RUNNER='+SOURCE+'/target/release/b2-runner',
        PYTHON,'-m','pytest','-q',SOURCE+'/tests/test_resource_limits_corrected_n860.py',
        '--junitxml='+BASE+'/budget-scale1pct-v1.xml','--basetemp='+BASE+'/tmp/budget-scale1pct-v1'],4,2,300)


def readback():
    expected = json.loads((HERE/'source-identity.json').read_text())['files']
    code = f'''from pathlib import Path
import hashlib,json
b=Path({SOURCE!r});names={list(expected)!r}
print(json.dumps({{'files':{{n:hashlib.sha256((b/n).read_bytes()).hexdigest() for n in names}},'reference_binary_sha256':hashlib.sha256((b/'target/release/b2-runner').read_bytes()).hexdigest()}}))'''
    r = remote(NODES[0],code)
    assert r['files'] == {n:v['sha256'] for n,v in expected.items()}
    record('engine-source-readback.json',dict(passed=True,files_checked=len(expected),**r))


def prepare(case, hosts):
    gate_tests()
    small = case == pilot(hosts)
    assert small or case == major(hosts)
    if not small:
        assert json.loads((HERE/(pilot(hosts)+'-validation.json')).read_text())['passed']
        assert json.loads((HERE/(pilot(hosts)+'-resources.json')).read_text())['passed']
        if hosts != 3:
            previous = {6:3,12:6,24:12,30:24}[hosts]
            assert json.loads((HERE/(major(previous)+'-accepted.json')).read_text())['passed']
    command = [PYTHON,SOURCE+'/experiment/prepare-scaling.py','--source',SOURCE,
               '--output',BASE+'/'+case,'--hosts',str(hosts),'--compile']
    if small:
        command.append('--pilot')
    guarded('prepare-'+case,command,P['pilot_preparation_memory_gib'] if small else P['prep_memory_gib'],4,P['prepare_timeout_seconds'])


def prepared(case):
    record(case+'-prepared.json',remote(NODES[0],f"from pathlib import Path\nprint((Path({BASE!r})/{case!r}/'prepared.json').read_text())"))


def reference(case):
    guarded('reference-'+case,[SOURCE+'/target/release/b2-runner',BASE+'/'+case+'/model.json',BASE+'/'+case+'/reference'],32,1,P['pilot_reference_timeout_seconds'])


def deploy(case):
    # Fetch only one host's eight shards at a time. The whole 1% input would
    # expand to >20 GiB on the local workstation even though it compresses well.
    prep=json.loads((HERE/(case+'-prepared.json')).read_text())
    assert prep['compiled'] and prep['hosts']==len(NODES)
    shared=['b2-mpi','manifest.json','build.json','instance.bin','execution-plan.json']
    expected={name.removeprefix('mpi/'):value['sha256'] for name,value in prep['files'].items()
              if name.startswith('mpi/') and (name.removeprefix('mpi/') in shared
                  or name.removeprefix('mpi/').startswith('instance.rank-'))}
    def worker(i):
        names=shared+[f'instance.rank-{rank}.bin' for rank in range(i*8,(i+1)*8)]
        assert all(name in expected for name in names)
        fetch=f'''from pathlib import Path
import io,sys,tarfile,json
p=Path({BASE!r})/{case!r}/'mpi';names={names!r};buf=io.BytesIO()
assert sum((p/n).stat().st_size for n in names)<2*2**30
with tarfile.open(fileobj=buf,mode='w:gz',compresslevel=1) as t:
 for name in names:t.add(p/name,arcname=name,recursive=False)
sys.stdout.buffer.write(buf.getvalue())'''
        result=subprocess.run(['tsh','ssh','rock@'+NODES[0],shlex.join(['python3','-c',fetch])],
                              capture_output=True,timeout=600)
        assert result.returncode==0,result.stderr.decode()[-1000:]
        raw=result.stdout;assert len(raw)<256*2**20
        digest=hashlib.sha256(raw).hexdigest()
        code=f'''from pathlib import Path
import hashlib,io,json,os,sys,tarfile
b=Path({BASE!r});p=b/{case!r}/'mpi';assert not p.exists()
assert os.statvfs(b).f_bavail*os.statvfs(b).f_frsize>128*2**30
raw=sys.stdin.buffer.read(256*2**20);assert hashlib.sha256(raw).hexdigest()=={digest!r}
with tarfile.open(fileobj=io.BytesIO(raw)) as t:
 members=t.getmembers();assert all(m.isfile() and Path(m.name).name==m.name for m in members)
 assert {{m.name for m in members}}==set({names!r}) and sum(m.size for m in members)<2*2**30
 p.mkdir(parents=True);t.extractall(p,filter='data')
(p/'b2-mpi').chmod(0o755)
print(json.dumps({{'host':os.uname().nodename,'payload_sha256':{digest!r},'owned_ranks':list(range({i*8},{(i+1)*8})),'files':{{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in {names!r}}}}}))'''
        row=remote(NODES[i],code,raw,timeout=180)
        assert all(expected[n]==h for n,h in row['files'].items())
        print(json.dumps({'deployed':NODES[i],'ranks':row['owned_ranks']}),flush=True)
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(worker,range(1,len(NODES))))
    record(case+'-deployment.json',rows)
    for i,row in enumerate(rows,1):
        assert row['owned_ranks'] == list(range(i*8,(i+1)*8))
        assert all(expected[n] == h for n,h in row['files'].items())
        assert {int(n.split('-')[1].split('.')[0]) for n in row['files'] if n.startswith('instance.rank-')} == set(row['owned_ranks'])
    assert {int(n.split('-')[1].split('.')[0]) for n in expected if n.startswith('instance.rank-')} == set(range(len(NODES)*8))
    record(case+'-deployment-audit.json',dict(passed=True,hosts=len(NODES),ranks=len(NODES)*8,
        input_catalog_sha256=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest(),
        transport='Per-host bounded gzip bundles; every worker file hash checked against the prepared leader catalog.'))


def inventory(case, memory_gib=None, coordinator_gib=None, node_memory_gib=None):
    def worker(node):
        memory = node_memory_gib[node] if node_memory_gib is not None else coordinator_gib if node == NODES[0] else memory_gib
        code = f'''from pathlib import Path
import json,os,subprocess,time
b=Path({BASE!r});m={{k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in ['MemAvailable','MemTotal']}}
active=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True).strip()
cpus=[]
for cpu in {P['cpu_ids']!r}:
 p=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology';cpus.append([cpu,(p/'physical_package_id').read_text().strip(),(p/'core_id').read_text().strip()])
assert len({{(r[1],r[2]) for r in cpus}})==8
memory_gib={memory!r}
if memory_gib is not None:
 assert m['MemAvailable']>memory_gib*2**30+64*2**30
 assert not active,active
 assert os.statvfs(b).f_bavail*os.statvfs(b).f_frsize>128*2**30
print(json.dumps({{'host':os.uname().nodename,'utc_epoch':time.time(),'memory':m,'free_bytes':os.statvfs(b).f_bavail*os.statvfs(b).f_frsize,'active_mpi_services':active,'cpu_topology':cpus,'loadavg':Path('/proc/loadavg').read_text().strip(),'host_cpu_stat':Path('/proc/stat').read_text().splitlines()[0],'requested_memory_gib':memory_gib}}))'''
        return remote(node,code)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(worker,NODES))
    record(case+'-inventory.json',dict(passed=True,rows=rows))


def launch(case, hosts):
    gate_tests()
    assert json.loads((HERE/'engine-source-readback.json').read_text())['passed']
    assert json.loads((HERE/(case+'-deployment-audit.json')).read_text())['passed']
    assert json.loads((HERE/('prepare-'+case+'-status.json')).read_text())['guard']['returncode'] == 0
    small = case == pilot(hosts)
    assert small or case == major(hosts)
    if not small:
        assert json.loads((HERE/(pilot(hosts)+'-validation.json')).read_text())['passed']
        assert json.loads((HERE/(pilot(hosts)+'-resources.json')).read_text())['passed']
    caps=resource_caps(case,hosts)
    worker_memory,leader_memory=caps[1],caps[0]
    inventory(case,node_memory_gib=dict(zip(NODES,caps,strict=True)))
    sys.path.insert(0,str(HERE/'tools'))
    from mpi_teleport_launch import launch as execute
    overrides = {node:dict(volume='/data/brick2',remote_base=REAL if node==NODES[0] else '/data/brick2/atlas-capacity-86m-20261007',
        script=(REAL if node==NODES[0] else '/data/brick2/atlas-capacity-86m-20261007')+'/source-scale1pct-v1/tools/mpi_resource_guard.py')
        for node in SELECTION['data_nodes'] if node in NODES}
    # The guard script is available on workers from the prior immutable source.
    for node in overrides:
        if node != NODES[0]:
            overrides[node]['script'] = '/data/brick2/atlas-capacity-86m-20261007/source/tools/mpi_resource_guard.py'
    for i,node in enumerate(NODES):
        if node not in overrides:
            overrides[node]=dict(volume='/',remote_base=BASE,script=BASE+'/source/tools/mpi_resource_guard.py')
        overrides[node]['memory_mib']=caps[i]*1024

    command = 'exec env B2_MPI_MAX_LOCAL_EDGES='+str(P['local_edge_guard'])+' B2_THREAD_AFFINITY=required /usr/bin/time -v -o '+BASE+'/'+case+'/rank-${PMI_RANK}.time '+BASE+'/'+case+'/mpi/b2-mpi '+BASE+'/'+case+'/mpi/instance.bin '+BASE+'/'+case+'/result'
    try:
        execute(nodes=NODES,ips=IPS,ranks_per_node=8,remote_base=BASE,application=['sh','-c',command],
            output=HERE/(case+'-launch'),mpi_prefix=BASE+'/mpi',timeout=P['pilot_timeout_seconds'] if small else P['simulation_timeout_seconds'],
            login='root',guard_script=BASE+'/source/tools/mpi_resource_guard.py',guard_volume='/',guard_allow_root_volume=True,
            guard_memory_mib=worker_memory*1024,guard_cpu_percent=800,guard_cpu_count=8,guard_cpu_ids=P['cpu_ids'],
            guard_file_mib=P['file_cap_gib']*1024,guard_min_free_gib=128,guard_node_overrides=overrides)
    finally:
        f=HERE/(case+'-launch/launch.json')
        if f.exists():
            record(case+'-launch-result.json',json.loads(f.read_text()))


def validate(case, hosts):
    code = f'''from pathlib import Path
import json,sys,numpy as np
b=Path({BASE!r});p=b/{case!r};sys.path.insert(0,{SOURCE!r}+'/python')
from brian2_rust.results import load_results
m=json.loads((p/'model.json').read_text());a=load_results(m,p/'reference');z=load_results(m,p/'result');checks=[];maximum=0
for x,y in zip(a['populations'],z['populations'],strict=True):
 for k in ['spike_ticks','indices','counts','last_spikes']:checks.append(bool(np.array_equal(x[k],y[k])))
 for kind in ['states','trace','refractory']:
  for k,v in x[kind].items():
   w=y[kind][k];assert np.shape(v)==np.shape(w);error=float(np.max(np.abs(np.asarray(v,dtype=float)-np.asarray(w,dtype=float)),initial=0));maximum=max(maximum,error);checks.append(error<=1e-12)
checks.extend([a['synaptic_events']==z['synaptic_events'],z['synaptic_events']>0,z['metadata']['spike_count']>0])
r=json.loads((p/'result/mpi-runtime.json').read_text());prep=json.loads((p/'prepared.json').read_text());ranks={hosts*8}
assert r['ranks']==ranks and set(r['processor_names'])==set({NODES!r})
assert len(r['rank_cpu_ids'])==ranks
for i,node in enumerate({NODES!r}):
 assert r['processor_names'][8*i:8*i+8]==[node]*8
 assert sorted(r['rank_cpu_ids'][8*i:8*i+8])=={P['cpu_ids']!r}
assert all(r['rank_work'][3*i]>0 for i in range(ranks))
assert len(r['procedural_topology'])==ranks*ranks and {{t['construction'] for t in r['procedural_topology']}}=={{'target-owner-local'}}
assert len(checks)==10*ranks+3
for t in r['procedural_topology']:
 syn=m['definition']['synapses'][t['projection']];owner=prep['population_owners'][syn['target_population']]
 assert t['rank_stats'][2*owner]==t['rank_stats'][2*owner+1]==t['global_edges']
 assert all(v==0 for i,v in enumerate(t['rank_stats'][::2]) if i!=owner)
 assert all(v==0 for i,v in enumerate(t['rank_stats'][1::2]) if i!=owner)
print(json.dumps({{'passed':all(checks),'checks':len(checks),'max_state_absolute_difference':maximum,'spikes':z['metadata']['spike_count'],'synaptic_events':z['synaptic_events'],'hosts':{hosts},'ranks':ranks,'plan_sha256':r['plan_sha256'],'runtime_sha256':__import__('hashlib').sha256((p/'result/mpi-runtime.json').read_bytes()).hexdigest()}}))'''
    result=remote(NODES[0],code,python=PYTHON,timeout=240)
    assert result['passed']
    record(case+'-validation.json',result)


def collect(case):
    launched=json.loads((HERE/(case+'-launch-result.json')).read_text())
    assert launched['error'] is None and all(v==0 for v in launched['returncodes'].values())
    prefix=launched['resource_guard']['unit_prefix']
    def worker(i):
        roles=['proxy-'+str(i)]+(['controller'] if i==0 else [])
        code=f'''from pathlib import Path
import json,os,hashlib
b=Path({BASE!r});roles={roles!r};prefix={prefix!r}
guards={{role:json.loads((b/'guards'/(prefix+'-'+role+'.json')).read_text()) for role in roles}}
assert all(g.get('returncode')==0 and g['admitted'] and 'oom_kill 0' in g['after']['memory.events'] for g in guards.values())
times={{p.name:p.read_text() for p in (b/{case!r}).glob('rank-*.time')}};assert len(times)==8 and all('Exit status: 0' in t for t in times.values())
r={{'host':os.uname().nodename,'guards':guards,'rank_times':times}}
if {i==0!r}:
 f=b/{case!r}/'result/mpi-runtime.json';runtime=json.loads(f.read_text());topology=runtime.pop('procedural_topology')
 runtime['procedural_topology_summary']={{'projections':len(topology),'construction':sorted({{t['construction'] for t in topology}})}}
 summary=json.loads((b/{case!r}/'result/summary.json').read_text())
 secondary=summary['mpi'].pop('procedural_topology');assert len(secondary)==len(topology)
 summary['mpi']['procedural_topology_summary']=runtime['procedural_topology_summary']
 r.update(prepared=json.loads((b/{case!r}/'prepared.json').read_text()),summary=summary,runtime=runtime,runtime_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
print(json.dumps(r))'''
        return remote(NODES[i],code,timeout=180)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(worker,range(len(NODES))))
    expected_limits=resource_caps(case,len(NODES))
    for i,row in enumerate(rows):
        g=row['guards']['proxy-'+str(i)]
        assert int(g['before']['memory.max'])==expected_limits[i]*2**30
        assert g['before']['memory.swap.max']=='0' and g['after']['memory.swap.max']=='0'
        events=dict(line.split() for line in g['after']['memory.events'].splitlines())
        assert all(events[key]=='0' for key in ['oom','oom_kill','oom_group_kill'])
    record(case+'-resources.json',dict(passed=True,rows=rows))


def audit_stage():
    raw=(HERE/'audit.py').read_bytes();sha=hashlib.sha256(raw).hexdigest()
    code=f'''from pathlib import Path
import hashlib,json,sys
p=Path({BASE!r})/'audit-scale1pct-v2.py';assert not p.exists();raw=sys.stdin.buffer.read(65536)
assert hashlib.sha256(raw).hexdigest()=={sha!r};p.write_bytes(raw);print(json.dumps({{'passed':True,'sha256':{sha!r}}}))'''
    record('audit-script-stage.json',remote(NODES[0],code,raw))


def audit(case):
    assert json.loads((HERE/'audit-script-stage.json').read_text())['passed']
    assert json.loads((HERE/(case+'-resources.json')).read_text())['passed']
    guarded('audit-'+case,[PYTHON,BASE+'/audit-scale1pct-v2.py','--base',BASE,'--case',case],P['audit_memory_gib'],4,P['audit_timeout_seconds'])


def collect_audit(case):
    code=f'''from pathlib import Path
import json,hashlib
p=Path({BASE!r})/{case!r}/'audit.json';r=json.loads(p.read_text());topology=r['runtime'].pop('procedural_topology')
r['runtime']['procedural_topology_summary']={{'projections':len(topology),'construction':sorted({{t['construction'] for t in topology}})}}
secondary=r['metadata']['mpi'].pop('procedural_topology');assert len(secondary)==len(topology)
r['metadata']['mpi']['procedural_topology_summary']=r['runtime']['procedural_topology_summary']
r['full_audit_file_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();r['full_audit_file_bytes']=p.stat().st_size
print(json.dumps(r))'''
    record(case+'-audit.json',remote(NODES[0],code,timeout=240))


def monitor():
    base.monitor()


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','build','tests','status','readback','prepare','prepared','reference','deploy','launch','validate','collect','audit-stage','audit','collect-audit','inventory','monitor']);parser.add_argument('case',nargs='?');parser.add_argument('--hosts',type=int,default=30);args=parser.parse_args();configure(args.hosts)
    {'stage':stage,'build':build,'tests':tests,'status':lambda:status(args.case),'readback':readback,
     'prepare':lambda:prepare(args.case,args.hosts),'prepared':lambda:prepared(args.case),
     'reference':lambda:reference(args.case),'deploy':lambda:deploy(args.case),
     'launch':lambda:launch(args.case,args.hosts),'validate':lambda:validate(args.case,args.hosts),
     'collect':lambda:collect(args.case),'audit-stage':audit_stage,'audit':lambda:audit(args.case),
     'collect-audit':lambda:collect_audit(args.case),'inventory':lambda:inventory(args.case),
     'monitor':monitor}[args.action]()
