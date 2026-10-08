"""Prepare one full-size reserved replicate artifact, without simulation.

Use the archived primary emitter and compiler settings. Preparation is a finite
engineering stage; it does not admit confirmation runs or scientific thresholds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

NODE='hk-prod-model-ae02-23'
BASE=Path('/data/brick2/brian2-mpi-region-20260907')
REPLICATE=1750
TAG=f'mam-confirmation-artifact-v1-seed{REPLICATE}'
STAGE=BASE/TAG
OLD=BASE/'full32-n1-k1-spool-primary-v1-100500ms'
ARCHIVE=BASE/'mam-primary-run-v1-source.tar.gz'
ARCHIVE_SHA='2f07c81058df3d27a2e00e547740d5c89fdf8d4b235eb81a4c5035ea62583f7c'
MODEL_SHA='9526a75e00e7cd4c3457691610fce4072c2014c6e6c6b53ea8a62f588dcac6bd'
PARAMETERS_SHA='ee1a642c94b9d6e808627c54f78162e7f5d87f4140ba56496c5dfee3ed900d3e'
PLAN_SHA='e9ee9c6eb22a10be8529c5bdc0045abc4534e093892360697398651902da7a06'
PLACEMENT_SHA='1ec6465b13cecc9855ae9458dfb5f92ec4a054c8e654837eb48965f109dc46cc'
EXE_SHA='8e0f21edf5e44bb193b1f6152cf560f702e07750bc0a171699f69b4a048352a5'
PYTHON='/atlas-home/0003/workspace/brian2-lk-20260906/.venv/bin/python'
RUSTC='/atlas-home/0003/workspace/brian2-lk-20260906/.cargo/bin/rustc'
MPI='/atlas-home/0003/workspace/brian2-mpi-linux-20260907/mpi'
RUNNER=BASE/'paper-duration-v3-source/target/release/b2-runner'
LIMIT=1500
MODULES=['mam_prepare_confirmation_artifact.py','mam_reseed_model.py','mam_replicate_seeds.py','mpi_resource_guard.py']


def need(ok,msg):
    if not ok:raise ValueError(msg)


def configure(replicate):
    """Select an isolated artifact directory; this never admits a simulation."""
    global REPLICATE,TAG,STAGE
    need(type(replicate) is int and replicate in (1750,1751),'unsupported preparation replicate')
    REPLICATE=replicate
    TAG=f'mam-confirmation-artifact-v1-seed{replicate}'
    STAGE=BASE/TAG


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


def worker():
    import tarfile
    started=time.monotonic()
    need(sha(ARCHIVE)==ARCHIVE_SHA,'original emitter archive changed')
    frozen=STAGE/'frozen';frozen.mkdir()
    with tarfile.open(ARCHIVE) as t:
        members=t.getmembers()
        need(len(members)<1000 and sum(m.size for m in members)<8*2**20,'source archive budget')
        need(all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members),'source archive member type/path')
        t.extractall(frozen,filter='data')
    source_catalog={str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*') if p.is_file()}
    sys.path.insert(0,str(frozen/'python'))
    os.environ.update(B2_MAX_NEURONS='4200000',B2_MAX_INITIAL_VALUES='20000000',
        B2_MAX_IR_BYTES='536870912',B2_MAX_POPULATION_STEPS='1005000')
    from brian2_rust.distributed import write_mpi_project,compile_mpi_project,_verify_artifact
    from mam_reseed_model import reseed
    need(sha(OLD/'model.json')==MODEL_SHA and sha(OLD/'parameters.json')==PARAMETERS_SHA,'frozen model/parameters changed')
    need(sha(OLD/'mpi/execution-plan.json')==PLAN_SHA and sha(OLD/'placement.json')==PLACEMENT_SHA,'frozen placement/plan changed')
    old=json.loads((OLD/'model.json').read_text());parameters=json.loads((OLD/'parameters.json').read_text())
    need(len(old['definition']['populations'])==254 and len(old['definition']['synapses'])==8344
         and sum(p['count'] for p in old['definition']['populations'])==4129924
         and sum(s['topology']['edge_count'] for s in old['instance']['synapses'])==24126516728,'full geometry changed')
    new,delta=reseed(old,parameters,REPLICATE)
    write(STAGE/'random-input-audit.json',delta)
    print(json.dumps(dict(event='legacy_and_random_delta_passed',neurons=4129924,projections=8344)),flush=True)
    project=STAGE/'artifact';project.mkdir()
    write(project/'model.json',new)
    for n in ['parameters.json','placement.json']:
        with (project/n).open('xb') as f:f.write((OLD/n).read_bytes());f.flush();os.fsync(f.fileno())
    placement=json.loads((project/'placement.json').read_text())
    plan=write_mpi_project(new,project/'mpi',ranks=32,population_owners=placement['population_owners'],
        compact_populations=True,prebuild_shared_topology=True,compact_queue_indices=True,
        compact_spike_history=True,compact_spike_output=True,spike_spool_bytes=96*2**30,
        spike_spool_population_bytes=16*2**30,runner=RUNNER)
    old_plan=json.loads((OLD/'mpi/execution-plan.json').read_text());new_plan=plan.to_dict()
    # Round-trip tuples to the on-disk JSON form before comparison.
    new_plan=json.loads(json.dumps(new_plan))
    need(old_plan['instance_sha256']!=new_plan['instance_sha256'],'instance identity unchanged')
    need({k:v for k,v in old_plan.items() if k!='instance_sha256'}=={k:v for k,v in new_plan.items() if k!='instance_sha256'},'non-instance plan fields changed')
    manifest=_verify_artifact(project/'mpi')
    old_index=json.loads((OLD/'mpi/instance.bin').read_text());new_index=json.loads((project/'mpi/instance.bin').read_text())
    need(set(old_index['files'])==set(new_index['files']) and len(new_index['files'])==32,'rank shard coverage changed')
    need(all(old_index['files'][n]!=v for n,v in new_index['files'].items()),'a rank shard reused legacy bytes')
    prepare_seconds=time.monotonic()-started
    write(STAGE/'prepared.json',dict(model_sha256=sha(project/'model.json'),layers=new['protocol']['layers'],
        definition_and_run_exact=True,plan_only_instance_identity_changed=True,all_32_shards_changed=True,
        plan_sha256=manifest['plan_sha256'],random_input_audit_sha256=sha(STAGE/'random-input-audit.json'),
        emitter_archive_sha256=ARCHIVE_SHA,emitter_source_catalog=source_catalog,prepare_seconds=prepare_seconds,
        simulation_started=False))
    print(json.dumps(dict(event='full_artifact_prepared',seconds=prepare_seconds)),flush=True)
    need(prepare_seconds<800,'preparation phase exceeded budget')
    compile_started=time.monotonic()
    compile_mpi_project(project/'mpi',mpicc=MPI+'/bin/mpicc',rustc=RUSTC,opt_level=3,panic_strategy='abort')
    build=json.loads((project/'mpi/build.json').read_text());prior=json.loads((OLD/'mpi/build.json').read_text())
    need({k:v for k,v in build.items() if k!='executable_sha256'}=={k:v for k,v in prior.items() if k!='executable_sha256'},'compiler options/runtime changed')
    need(build['executable_sha256']==sha(project/'mpi/b2-mpi')!=EXE_SHA,'new executable identity')
    files={}
    for p in project.rglob('*'):
        if p.is_file():
            need(not p.is_symlink() and p.stat().st_size<1024*2**20,'artifact file limit')
            files[str(p.relative_to(project))]=dict(bytes=p.stat().st_size,sha256=sha(p))
            with p.open('rb') as f:os.fsync(f.fileno());os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    need(sum(x['bytes'] for x in files.values())<2*2**30,'artifact total budget')
    for p in [project/'mpi',project,STAGE]:
        fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    result=dict(schema='b2-mam-confirmation-artifact-v1',replicate=REPLICATE,artifact=str(project),
        prepared_sha256=sha(STAGE/'prepared.json'),random_input_audit_sha256=sha(STAGE/'random-input-audit.json'),
        source_model_sha256=MODEL_SHA,parameters_sha256=PARAMETERS_SHA,build=build,files=files,
        prepare_seconds=prepare_seconds,compile_and_readback_seconds=time.monotonic()-compile_started,
        wall_seconds=time.monotonic()-started,neural_runs=0,confirmation_outcomes_observed=0,
        launch_admitted=False,scientific_acceptance=False)
    need(result['wall_seconds']<LIMIT-10,'whole preparation time exceeded')
    write(STAGE/'pending.json',result);print(json.dumps(dict(event='artifact_compiled',wall_seconds=result['wall_seconds'])),flush=True)


def remote(code,data=None):
    from mam_collect_native_primary_raw import remote as call
    return call(NODE,code,data)


def run(output):
    need(not output.exists(),'single preparation attempt already exists')
    output.mkdir()
    bundle={n:(Path(__file__).parent/n).read_text() for n in MODULES}
    write(output/'intent.json',dict(replicate=REPLICATE,maximum_full_preparations=1,maximum_compiles=1,
        neural_simulations=0,worker_seconds=LIMIT,preparation_phase_max_seconds=800,
        memory_gib=16,cpu_cores=4,cpu_ids=[8,9,10,11],file_limit_mib=1024,
        total_artifact_gib=2,minimum_free_gib=1280,required_start_free_gib=1296,automatic_retry=False,
        source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in bundle.items()}))
    started=time.monotonic()
    try:
        admission=remote(f'''import os,json,time,signal,resource,subprocess,hashlib,datetime
from pathlib import Path
signal.alarm(25);resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(10,10))
b=Path({str(BASE)!r});assert b.is_dir() and Path('/data/brick2').is_mount() and b.stat().st_dev!=Path('/').stat().st_dev
assert not Path({str(STAGE)!r}).exists()
assert os.uname().nodename=={NODE!r}
assert not subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5).strip()
s=os.statvfs(b);free=s.f_bavail*s.f_frsize;assert free>=1296*2**30
m={{k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())}};assert m['MemAvailable']>=80*2**30
assert Path({str(OLD/'model.json')!r}).stat().st_size<512*2**20
assert hashlib.sha256(Path({str(ARCHIVE)!r}).read_bytes()).hexdigest()=={ARCHIVE_SHA!r}
for p in {[PYTHON,RUSTC,MPI+'/bin/mpicc',str(RUNNER)]!r}:assert os.access(p,os.X_OK)
def ticks():return {{int(x[0][3:]):list(map(int,x[1:9])) for l in Path('/proc/stat').read_text().splitlines() if (x:=l.split())[0].startswith('cpu') and x[0][3:].isdigit()}}
a=ticks();time.sleep(1);z=ticks();busy={{}}
for c in [8,9,10,11]:
 d=[y-x for x,y in zip(a[c],z[c])];assert sum(d)>0;busy[c]=100*(1-(d[3]+d[4])/sum(d))
assert max(busy.values())<50 and sum(busy.values())/4<25,busy
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=free,available_memory_bytes=m['MemAvailable'],cpu_busy_percent=busy,source_archive_sha256={ARCHIVE_SHA!r},active_own_units=[])))''')
        write(output/'admission.json',admission)
        staged=remote(f'''import sys,json,hashlib
from pathlib import Path
raw=json.loads(sys.stdin.read());b=Path({str(STAGE)!r});b.mkdir();(b/'tools').mkdir();(b/'tmp').mkdir()
for n,s in raw.items():(b/'tools'/n).open('x').write(s)
print(json.dumps({{n:hashlib.sha256((b/'tools'/n).read_bytes()).hexdigest() for n in raw}}))''',json.dumps(bundle).encode())
        need(staged=={n:hashlib.sha256(s.encode()).hexdigest() for n,s in bundle.items()},'staged code differs')
        write(output/'stage.json',staged);need(time.monotonic()-started<60,'fresh admission expired')
        app=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',
            'TMPDIR='+str(STAGE/'tmp'),'MPLCONFIGDIR='+str(STAGE/'tmp'),
            'RUSTUP_HOME=/atlas-home/0003/workspace/brian2-lk-20260906/.rustup',
            'CARGO_HOME=/atlas-home/0003/workspace/brian2-lk-20260906/.cargo',
            'LD_LIBRARY_PATH='+MPI+'/lib',PYTHON,str(STAGE/'tools/mam_prepare_confirmation_artifact.py'),'worker']
        if REPLICATE!=1750:app.extend(['--replicate',str(REPLICATE)])
        cmd=['systemd-run','--expand-environment=no','--quiet','--wait','--pipe','--collect',
            '--unit=b2mpi-'+TAG+'-build','--uid=rock','--service-type=exec',
            '--property=MemoryMax=16384M','--property=MemorySwapMax=0','--property=CPUQuota=400%',
            '--property=AllowedCPUs=8-11','--property=TasksMax=64','--property=RuntimeMaxSec='+str(LIMIT+5),
            '--property=TimeoutStopSec=5','--property=KillMode=control-group','--property=OOMPolicy=continue',
            '/usr/bin/python3',str(STAGE/'tools/mpi_resource_guard.py'),'--output',str(STAGE/'guard.json'),
            '--volume','/data/brick2','--memory-mib','16384','--cpu-percent','400','--file-mib','1024',
            '--min-free-gib','1280','--timeout',str(LIMIT),'--',*app]
        write(output/'command.json',dict(command=cmd))
        with (output/'build.log').open('x') as log:
            rc=subprocess.run(['tsh','ssh','root@'+NODE,shlex.join(cmd)],stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,timeout=LIMIT+25).returncode
        controls=remote(f'''import json,subprocess
from pathlib import Path
b=Path({str(STAGE)!r});names=['guard.json','prepared.json','random-input-audit.json','pending.json'];out={{}}
for n in names:
 p=b/n
 if p.exists():assert p.stat().st_size<2**20;out[n]=json.loads(p.read_text())
out['active_services']=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend',{('b2mpi-'+TAG+'-*')!r}],text=True,timeout=5).strip()
print(json.dumps(out))''')
        for n,v in controls.items():
            if n!='active_services':write(output/n,v)
        write(output/'terminal-state.json',dict(returncode=rc,active_services=controls['active_services']))
        need(rc==0 and not controls['active_services'],'build failed or still active')
        from mam_benchmark_terminal import guard
        g=controls['guard.json'];need(g['command']==app,'guard command identity')
        spec=dict(host=NODE,role='build',memory_bytes=16*2**30,pids_max=64,cpu_ids=[8,9,10,11],cpu_quota_cores=4,
            volume='/data/brick2',allow_root_volume=False,file_limit_bytes=1024*2**20,
            minimum_free_bytes=1280*2**30,reserved_host_memory_bytes=64*2**30)
        accounting=guard(g,spec,'b2mpi-'+TAG,LIMIT)
        pending=controls['pending.json']
        need(pending['prepared_sha256']==sha(output/'prepared.json') and pending['random_input_audit_sha256']==sha(output/'random-input-audit.json'),'worker control digests changed')
        report=dict(complete=True,artifact_prepared=True,replicate=REPLICATE,worker= pending,
            guard_accounting=accounting,neural_runs=0,launch_admitted=False,scientific_acceptance=False,
            evidence_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},
            elapsed_seconds=time.monotonic()-started)
        write(output/'complete.json',report);print(json.dumps(dict(complete=True,artifact=pending['artifact'],accounting=accounting)),flush=True)
    except BaseException as exc:
        write(output/'failure.json',dict(error=str(exc),elapsed_seconds=time.monotonic()-started,automatic_retry=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['worker','run']);p.add_argument('--output',type=Path)
    p.add_argument('--replicate',type=int,choices=[1750,1751],default=1750)
    a=p.parse_args();configure(a.replicate)
    if a.mode=='worker':worker()
    else:run(a.output)
