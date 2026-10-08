"""Launch the staged native primary once, after Rust terminal/output acceptance.

Missing prerequisites return readiness=false before remote access. Fresh remote
admission is mandatory; no simulation retry or smaller replacement is scheduled.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from mam_primary_resources import MODEL, PREFIX, audit as audit_rust_resources

LABEL = 'nest-mam-primary-v1-metastable-seed1729-100500ms'
BASE = '/atlas-home/0003/workspace/brian2-mpi-cluster-20260907'
PROJECT = BASE+'/'+LABEL
NODES = ['hk-prod-model-ae02-25', 'hk-prod-model-ae08-81', 'hk-prod-model-ae08-83',
         'hk-prod-model-ae07-71', 'hk-prod-model-ae08-82', 'hk-prod-model-ae08-84']
IPS = ['192.168.20.25', '192.168.30.81', '192.168.30.83', '192.168.30.71',
       '192.168.30.82', '192.168.30.84']
PARAMETERS = 'ee1a642c94b9d6e808627c54f78162e7f5d87f4140ba56496c5dfee3ed900d3e'
LAYOUT_SHA = '2e65fe7079d6cee31c1d40fc83b81def463a19b40b5c84a627d1f0c358317515'
CPUS = [c for start in [0,12,24,36,48,60,72,84] for c in range(start,start+4)]


def check(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    check(path.stat().st_size <= 32*2**20, 'oversized control input')
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rust_gate(evidence, resource_directory):
    required = [resource_directory/(n+'.json') for n in
                ['admission','launch','collected','runtime','report','input-sha256']]
    output_path = evidence/'primary-output-audit/full-report.json'
    guard_path = evidence/'primary-output-audit/full-guard.json'
    if not all(p.exists() for p in required+[output_path, guard_path]):
        return None
    control = {n: read(resource_directory/(n+'.json')) for n in
               ['admission','launch','collected','runtime']}
    decision = audit_rust_resources(**control)
    check(decision == read(resource_directory/'report.json'), 'Rust resource decision differs')
    expected = read(resource_directory/'input-sha256.json')
    check(set(expected) == set(control), 'Rust input hash coverage')
    check(all(sha(resource_directory/(n+'.json')) == h for n,h in expected.items()),
          'Rust resource inputs changed')
    output = read(output_path)
    check(output['schema'] == 'b2-mam-primary-output-audit-v1'
          and output['internal_output_audit_passed'] is True
          and output['terminal_resource_audit_passed'] is True
          and output['model_sha256'] == MODEL
          and output['retained_10500ms_prefix']['exact'] is True
          and output['retained_10500ms_prefix']['spikes'] == 633265154,
          'Rust full output/prefix prerequisite failed')
    guard = read(guard_path)
    events = dict(line.split() for line in guard['after']['memory.events'].splitlines())
    check(guard['admitted'] is True and type(guard.get('returncode')) is int
          and guard['returncode'] == 0 and not guard.get('error')
          and all(events[k] == '0' for k in ['max','oom','oom_kill','oom_group_kill']),
          'Rust output audit resource failure')
    return dict(resource_report_sha256=sha(resource_directory/'report.json'),
                output_report_sha256=sha(output_path), output_guard_sha256=sha(guard_path))


def validate_run_identity(label, seed):
    check(type(seed) is int and 0 < seed < 2**31, 'invalid native seed')
    check(isinstance(label, str) and re.fullmatch(r'[a-z0-9][a-z0-9-]{0,127}', label),
          'invalid native run label')
    check(f'-seed{seed}-100500ms' in label and label.endswith(f'-seed{seed}-100500ms'),
          'native label/seed mismatch')


def launch_options(output, *, label=LABEL, seed=1729):
    # Source project stays pinned to the tested primary producer; a new run
    # changes its RNG seed and output/receipt directories only.
    validate_run_identity(label, seed)
    runtime = BASE+'/nest-runtime-v1'
    app = ['env', 'PYTHONPATH='+runtime+'/site',
           'LD_LIBRARY_PATH='+runtime+'/lib:'+BASE+'/mpi/lib',
           'OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=4','PYNEST_QUIET=1',
           'PYTHONDONTWRITEBYTECODE=1','TMPDIR='+runtime+'/tmp','MPLCONFIGDIR='+runtime+'/tmp',
           '/usr/bin/python3',PROJECT+'/mam_nest_rank_affinity.py',
           '--layout',PROJECT+'/layout.json','--layout-sha256',LAYOUT_SHA,
           '--receipt-directory',BASE+'/affinity/'+label,'--',
           '/usr/bin/python3',PROJECT+'/mam_nest_reference.py',
           '--parameters',PROJECT+'/parameters.json','--parameters-sha256',PARAMETERS,
           '--output',BASE+'/runs/'+label,'--ranks','48','--threads','4','--seed',str(seed),
           '--duration-ms','100500','--max-duration-ms','100500','--chunk-ms','50',
           '--max-neurons','4200000','--max-edges','25000000000',
           '--max-spikes-per-rank','268435456','--max-chunk-spikes','2000000']
    return dict(nodes=NODES, ips=IPS, ranks_per_node=8, remote_base=BASE,
        application=app, output=output, mpi_prefix=BASE+'/mpi', timeout=54000,
        login='root', guard_script=PROJECT+'/mpi_resource_guard.py',
        guard_volume='/', guard_allow_root_volume=True, guard_memory_mib=262144,
        guard_cpu_percent=3200, guard_cpu_count=32, guard_cpu_ids=CPUS,
        guard_file_mib=3072, guard_min_free_gib=128)


def remote(node, code):
    # Fixed low-impact control affinity, separate from Rust primary CPUs.
    result = subprocess.run(['tsh','ssh','rock@'+node,
        shlex.join(['taskset','-c','8,9','python3','-c',code])],
        capture_output=True, check=True, timeout=45)
    check(len(result.stdout) <= 2*2**20, 'oversized preflight response')
    return json.loads(result.stdout)


def preflight_code(index, staged, selected_topology, mpi_sha, *, output_label=LABEL):
    check(isinstance(output_label, str)
          and re.fullmatch(r'[a-z0-9][a-z0-9-]{0,127}', output_label), 'invalid native output label')
    return f'''from pathlib import Path
import os,json,hashlib,subprocess,time,datetime
b=Path({BASE!r});p=Path({PROJECT!r});runtime=b/'nest-runtime-v1'
assert b.resolve().is_relative_to(Path('/home/rock')) and os.uname().nodename=={NODES[index]!r}
assert not (b/'runs'/{output_label!r}).exists() and not (b/'affinity'/{output_label!r}).exists()
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True);assert not units.strip(),units
mem={{k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines()) if k in ['MemTotal','MemAvailable']}}
free=os.statvfs(b).f_bavail*os.statvfs(b).f_frsize
assert mem['MemAvailable']>={576 if index==0 else 320}*2**30 and free>=192*2**30
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for n,row in {staged['source_catalog']!r}.items():assert (p/n).stat().st_size==row['bytes'] and sha(p/n)==row['sha256'],n
assert sha(p/'parameters.json')=={PARAMETERS!r}
assert sha(runtime/'catalog.json')=={staged['runtime_catalog_sha256']!r}
for n,row in json.loads((runtime/'catalog.json').read_text()).items():
 f=runtime/n;assert f.resolve().is_relative_to(runtime.resolve())
 if row['kind']=='symlink':assert f.is_symlink() and os.readlink(f)==row['target']
 else:assert not f.is_symlink() and f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
assert sha(b/'mpi/lib/libmpi.so.12.4.3')=={mpi_sha!r}
topology={{}}
for line in subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET,NODE,ONLINE'],text=True).splitlines():
 if line.startswith('#'):continue
 cpu,core,socket,numa,online=line.split(',');cpu=int(cpu)
 if cpu not in {CPUS!r}:continue
 q=Path('/sys/devices/system/cpu')/('cpu'+str(cpu));l3=[]
 for c in sorted((q/'cache').glob('index*')):
  if (c/'level').read_text().strip()=='3':l3.append(dict(id=(c/'id').read_text().strip(),shared_cpus=(c/'shared_cpu_list').read_text().strip()))
 topology[cpu]=dict(cpu=cpu,core=int(core),socket=int(socket),numa=int(numa),online=online=='Y',l3=l3)
assert [topology[c] for c in {CPUS!r}]=={selected_topology!r}
def ticks():return {{int(x[0][3:]):list(map(int,x[1:9])) for line in Path('/proc/stat').read_text().splitlines() if (x:=line.split())[0].startswith('cpu') and x[0][3:].isdigit()}}
a=ticks();time.sleep(1);z=ticks();busy={{}}
for c in {CPUS!r}:
 d=[y-x for x,y in zip(a[c],z[c])];assert sum(d)>0;busy[c]=100*(1-(d[3]+d[4])/sum(d))
assert max(busy.values())<50 and sum(busy.values())/32<25,busy
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),available_memory_bytes=mem['MemAvailable'],free_bytes=free,cpu_busy_percent=busy,active_own_units=[],source_catalog_verified=True,runtime_catalog_verified=True,mpi_verified=True,selected_topology_verified=True)))'''


def run(evidence, t7):
    prerequisite = rust_gate(evidence, t7/'artifacts/primary-terminal-resources-v1')
    if prerequisite is None:
        return dict(ready=False, launch_started=False,
                    reason='Rust terminal resources and complete output audit are required')
    layout_dir = evidence/'primary-native-layout'
    check(sha(layout_dir/'layout.json') == LAYOUT_SHA, 'staged layout changed')
    layout, staged = read(layout_dir/'layout.json'), read(layout_dir/'stage.json')
    check([h['host'] for h in layout['hosts']] == [r['host'] for r in staged] == NODES,
          'six-host staging coverage')
    inventory = read(layout_dir/'inventory.json')['nodes']
    check([r['host'] for r in inventory] == NODES, 'runtime inventory coverage')
    check(all(r['parameters_sha256'] == PARAMETERS and r['verified_runtime_entries'] == 2559
              and r['source_catalog']['layout.json']['sha256'] == LAYOUT_SHA for r in staged),
          'staged input identity')
    smoke = read(layout_dir/'smoke-report.json')
    check(smoke['all_rank_event_bytes_exact'] is True
          and smoke['all_32_workers_on_distinct_single_cpus'] is True, 'binding validation missing')
    volume = Path('/Volumes/T7')
    check(volume.is_mount() and t7.resolve().is_relative_to(volume.resolve()), 'T7 must be mounted')
    stat = os.statvfs(t7)
    check(stat.f_bavail*stat.f_frsize >= 128*2**30, 'T7 reserve')
    output = t7/'logs'/LABEL
    admission_path = layout_dir/'admission.json'
    check(not output.exists() and not admission_path.exists(), 'native experiment already admitted/started')
    # Explicitly verify the prior job is absent on node23 as well as on the six
    # native hosts. A status file alone is not proof that compute stopped.
    leader = remote('hk-prod-model-ae02-23', f'''import subprocess,json
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend',{PREFIX+'-*'!r}],text=True)
assert not units.strip(),units
print(json.dumps(dict(rust_primary_active_units=[])))''')
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(lambda i: remote(NODES[i], preflight_code(i, staged[i],
            layout['hosts'][i]['selected_topology'], inventory[i]['mpi_sha256'])), range(6)))
    check(time.monotonic()-start < 60 and [r['host'] for r in rows] == NODES, 'preflight stale/incomplete')
    options = launch_options(output)
    admission = dict(schema='b2-mam-native-primary-admission-v1', label=LABEL,
        prerequisite=prerequisite, rust_leader_terminal=leader, preflight=rows,
        layout_sha256=LAYOUT_SHA, parameters_sha256=PARAMETERS,
        experiment_budget=dict(runs=1, automatic_retry=False, wall_seconds=54000,
            collection_seconds=7200, analysis_stage_seconds=10800),
        launch_options={k:str(v) if isinstance(v,Path) else v for k,v in options.items()},
        admitted=True, scientific_acceptance=False, performance_cost_acceptance=False)
    with admission_path.open('x') as f:
        f.write(json.dumps(admission,indent=2)+'\n')
    (t7/'artifacts'/(LABEL+'-admission.json')).write_text(json.dumps(admission,indent=2)+'\n')
    print(json.dumps(dict(event='native_primary_admitted', label=LABEL,
                          nodes=NODES, wall_limit_seconds=54000)), flush=True)
    from mpi_teleport_launch import launch
    result = launch(**options)
    (layout_dir/'launch.json').write_text(json.dumps(result,indent=2)+'\n')
    check(result.get('error') is None
          and set(result['returncodes']) == {'controller'} | {'proxy-'+str(i) for i in range(6)}
          and all(type(v) is int and v == 0 for v in result['returncodes'].values()),
          'native primary failed; retained evidence requires investigation, no retry')
    return dict(ready=True, launch_started=True, launcher_terminal_success=True,
                label=LABEL, raw_resource_scientific_audits_required=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--t7', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.evidence, args.t7)
    print(json.dumps(report))
    raise SystemExit(0 if report['ready'] else 2)
