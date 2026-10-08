"""One guarded Rust benchmark raw audit, with publication after guard success."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile
import time

from mam_rust_benchmark_raw import BASE,RUN,CASE,LABEL,NODES,MODEL,PLAN,EXE_SHA,terminal_gate,require
from mam_benchmark_terminal import guard
from mam_launch_performance_tuning import PROTOCOL_SHA,write
from mam_launch_native_primary import read,sha
from mam_rust_benchmark_terminal import target_contract
from mam_launch_rust_recovery import CASE as RECOVERY_CASE,PROTOCOL_SHA as RECOVERY_PROTOCOL_SHA

RECOVERY_VALIDATION_SHA='e6517390c06a7c84f8f40e886567deb09462a09feeddaf2ff2618825ceae9cb5'
RECOVERY_VALIDATION_PATH='rust-timing-quantization-v3/validation.json'

def protocol_for(case_id):
    require(case_id in [CASE,RECOVERY_CASE],'unknown Rust audit target')
    return RECOVERY_PROTOCOL_SHA if case_id==RECOVERY_CASE else PROTOCOL_SHA

SOURCE=BASE/'rust-target-raw-v1-source'
OUTPUT=BASE/'rust-target-raw-v1'
GUARD=BASE/'guards/rust-target-raw-v1.json'
PREFIX='b2mpi-rust-target-raw-v1'
UNIT=PREFIX+'-audit'
PYTHON='/atlas-home/0003/workspace/brian2-lk-20260906/.venv/bin/python'
SECONDS=6600


def application(binding_sha):
    return ['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',
        'PYTHONPATH='+str(SOURCE/'tools')+':'+str(SOURCE/'python'),PYTHON,
        str(SOURCE/'tools/mam_rust_benchmark_raw.py'),'--case',str(SOURCE/'case'),
        '--binding',str(SOURCE/'binding.json'),'--binding-sha256',binding_sha,'--output',str(OUTPUT)]


def command(binding_sha):
    return ['systemd-run','--expand-environment=no','--quiet','--wait','--pipe','--collect',
        '--unit='+UNIT,'--uid=rock','--service-type=exec','--property=MemoryMax=16384M',
        '--property=MemorySwapMax=0','--property=CPUQuota=200%','--property=AllowedCPUs=8-9',
        '--property=TasksMax=64','--property=RuntimeMaxSec='+str(SECONDS+5),
        '--property=TimeoutStopSec=5','--property=KillMode=control-group','--property=OOMPolicy=continue',
        '/usr/bin/python3',str(SOURCE/'tools/mpi_resource_guard.py'),'--output',str(GUARD),
        '--volume','/data/brick2','--memory-mib','16384','--cpu-percent','200','--file-mib','512',
        '--min-free-gib','1280','--timeout',str(SECONDS),'--',*application(binding_sha)]


def prepare(root,case,package):
    """Reproduce terminal controls in a bounded temporary directory, no network."""
    names=['admission.json','launch.json','terminal/report.json']+['terminal/host-'+str(i)+'.json.gz' for i in range(4)]
    controls={}
    for name in names:
        p=case/name;require(p.is_file() and not p.is_symlink() and p.stat().st_size<=64*2**20,'bounded terminal input')
        controls[name]=p.read_bytes()
    require(package.stat().st_size<=2*2**20,'package control ceiling')
    controls['package.json']=package.read_bytes()
    require(sum(map(len,controls.values()))<=48*2**20,'combined terminal controls ceiling')
    contract=target_contract(json.loads(controls['admission.json']))
    binding=dict(schema='b2-mam-rust-raw-input-binding-v1',case_id=contract['case_id'],protocol_sha256=contract['protocol_sha256'],
                 files={n:hashlib.sha256(v).hexdigest() for n,v in controls.items()})
    raw=(json.dumps(binding,indent=2)+'\n').encode();binding_sha=hashlib.sha256(raw).hexdigest()
    with tempfile.TemporaryDirectory(prefix='mam-rust-raw-controls-',dir='/private/tmp') as tmp:
        d=Path(tmp)
        for name,data in controls.items():
            p=d/'case'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        (d/'binding.json').write_bytes(raw)
        _,_,_,terminal=terminal_gate(d/'case',d/'binding.json',binding_sha)
    files={'case/'+n:v for n,v in controls.items()};files['binding.json']=raw
    for parent in ['tools','python']:
        for p in sorted((root/parent).rglob('*.py')):
            if p.name.startswith('._'):continue
            require(not p.is_symlink() and p.stat().st_size<=16*2**20,'bounded ordinary source')
            files[str(p.relative_to(root))]=p.read_bytes()
    require(sum(map(len,files.values()))<=64*2**20,'expanded source/control ceiling')
    catalog={n:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in files.items()}
    files['catalog.json']=(json.dumps(catalog,indent=2)+'\n').encode()
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name,data in sorted(files.items()):
            info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
    payload=buffer.getvalue();require(len(payload)<=32*2**20,'compressed source/control ceiling')
    return payload,catalog,binding,binding_sha,terminal


def preflight_code():
    return f'''from pathlib import Path
import os,json,signal,resource,subprocess,time,datetime
signal.alarm(20);resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20));resource.setrlimit(resource.RLIMIT_CPU,(10,10))
assert os.uname().nodename=={NODES[0]!r}
b=Path({str(BASE)!r});assert Path('/data/brick2').is_mount() and b.resolve().is_relative_to(Path('/data/brick2'))
assert b.stat().st_dev!=Path('/').stat().st_dev
for p in {list(map(str,[SOURCE,OUTPUT,GUARD]))!r}:assert not Path(p).exists(),p
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5);assert not units.strip(),units
s=os.statvfs(b);free=s.f_bavail*s.f_frsize;assert free>1280*2**30+64*2**20
m={{k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines()) if k in ['MemTotal','MemAvailable']}};assert m['MemAvailable']>=80*2**30
def ticks():return {{int(x[0][3:]):list(map(int,x[1:9])) for line in Path('/proc/stat').read_text().splitlines() if (x:=line.split())[0].startswith('cpu') and x[0][3:].isdigit()}}
a=ticks();time.sleep(1);z=ticks();busy={{}}
for c in [8,9]:
 d=[y-x for x,y in zip(a[c],z[c])];assert sum(d)>0;busy[c]=100*(1-(d[3]+d[4])/sum(d))
assert max(busy.values())<50 and sum(busy.values())/2<25
assert Path({PYTHON!r}).is_file()
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=free,memory=m,cpu_busy_percent=busy,active_own_units=[])))'''


def stage_code(payload_sha,catalog):
    return f'''from pathlib import Path
import sys,io,tarfile,hashlib,json,resource,signal,os
signal.alarm(30);resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(20,20))
raw=sys.stdin.buffer.read(32*2**20+1);assert len(raw)<=32*2**20 and hashlib.sha256(raw).hexdigest()=={payload_sha!r}
p=Path({str(SOURCE)!r});p.mkdir()
with tarfile.open(fileobj=io.BytesIO(raw)) as t:
 members=t.getmembers();assert set(x.name for x in members)==set({catalog!r})|{{'catalog.json'}}
 assert all(x.isfile() for x in members) and sum(x.size for x in members)<=65*2**20
 t.extractall(p,filter='data')
actual={{}}
for name,item in {catalog!r}.items():
 q=p/name;assert not q.is_symlink() and q.resolve().is_relative_to(p.resolve()) and q.stat().st_size==item['bytes']
 with q.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 assert digest==item['sha256'];actual[name]=item
assert json.loads((p/'catalog.json').read_text())==actual
print(json.dumps(dict(source_archive_sha256={payload_sha!r},verified_files=len(actual))))'''


def collect_code(catalog):
    return f'''from pathlib import Path
import json,hashlib,base64,signal,resource,subprocess
signal.alarm(30);resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20));resource.setrlimit(resource.RLIMIT_CPU,(20,20))
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend',{UNIT+'.service'!r}],text=True,timeout=5);assert not units.strip(),units
p=Path({str(SOURCE)!r})
for name,item in {catalog!r}.items():
 q=p/name;assert not q.is_symlink() and q.resolve().is_relative_to(p.resolve()) and q.stat().st_size==item['bytes']
 with q.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
assert not (Path({str(OUTPUT)!r})/'failure.json').exists()
out={{}}
for name,path in {dict(pending=str(OUTPUT/'pending.json'),guard=str(GUARD),attempt=str(OUTPUT/'attempt.json'))!r}.items():
 q=Path(path);assert q.is_file() and not q.is_symlink() and q.stat().st_size<=2*2**20
 raw=q.read_bytes();out[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
print(json.dumps(out))'''


def remote(code,payload=None):
    p=subprocess.run(['tsh','ssh','rock@'+NODES[0],shlex.join(['taskset','-c','8,9','python3','-c',code])],
        input=payload,capture_output=True,timeout=45)
    require(p.returncode==0,'bounded remote control failed: '+p.stderr[:1000].decode(errors='replace'))
    require(len(p.stdout)<=12*2**20,'remote control response ceiling')
    return json.loads(p.stdout)


def publish(pending,g,controller,*,binding_sha,terminal,raw_source_sha,elapsed):
    require(controller['returncode']==0 and type(controller['returncode']) is int and controller['error'] is None
            and controller['command']==command(binding_sha),'raw controller did not finish successfully')
    require(g['command']==application(binding_sha),'guard command differs from admitted raw audit')
    spec=dict(host=NODES[0],role='audit',memory_bytes=16*2**30,pids_max=64,cpu_ids=[8,9],cpu_quota_cores=2,
        volume='/data/brick2',allow_root_volume=False,file_limit_bytes=512*2**20,
        minimum_free_bytes=1280*2**30,reserved_host_memory_bytes=64*2**30)
    audited=guard(g,spec,PREFIX,SECONDS)
    require(pending['schema']=='b2-mam-rust-target-raw-pending-v1' and pending['case_id']==terminal['case_id']
            and pending['label']==terminal['label'] and pending['protocol_sha256']==protocol_for(terminal['case_id'])
            and pending['binding_sha256']==binding_sha and pending['model_sha256']==MODEL
            and pending['plan_sha256']==PLAN and pending['executable_sha256']==EXE_SHA
            and pending['implementation_sha256']==raw_source_sha,'pending raw audit identity differs')
    require(pending['complete_binary_scan_passed'] is True and pending['terminal_resource_audit_passed'] is True
            and pending['raw_output_audit_passed'] is False and pending['audit_guard_passed'] is False
            and pending['pending_guard_acceptance'] is True,'pending report state differs')
    require(all(pending[k] is False for k in ['scientific_acceptance','nest_statistical_acceptance',
        'performance_cost_acceptance','reused_scientific_summaries','prior_raw_identity_assumed']),'unsupported acceptance/reuse')
    require(pending['spikes']==terminal['reported_spikes'] and pending['delivered_edges']==terminal['reported_synaptic_events']
            and pending['dump_bytes']=={k:terminal['output_bytes'][k] for k in ['results.bin','events.bin']},'raw/terminal totals differ')
    require(set(pending['dump_sha256'])=={'results.bin','events.bin'}
            and all(re.fullmatch('[0-9a-f]{64}',h) for h in pending['dump_sha256'].values()),'complete binary hash coverage')
    work=pending['work']
    require(all(work[k] is True for k in ['all_result_and_event_records_equal',
        'all_projection_edge_and_csr_ownership_exact','rank_neuron_and_spike_work_exact',
        'local_delivery_work_and_shared_aggregate_exact'])
        and work['shared_delivery_rank_histories_independently_verified'] is False,'full engineering scan scope differs')
    require(len(work['exact_local_edges'])==32 and sum(work['exact_local_edges'])==24126516728
            and max(work['exact_local_edges'])<=1024000000 and len(work['population_profile'])==254,
            'full geometry scan coverage differs')
    require(0<pending['elapsed_seconds']<=g['wall_seconds']+.02 and g['wall_seconds']<=elapsed<=7200,'raw stage budget exceeded')
    result=dict(pending,raw_output_audit_passed=True,audit_guard_passed=True,pending_guard_acceptance=False,
        elapsed_through_guard_collection_seconds=elapsed,raw_audit_resource_accounting=audited,automatic_retry=False)
    return result


def run(evidence,t7,case_id=CASE):
    protocol_sha=protocol_for(case_id)
    case=evidence/'performance-runs-v1'/case_id
    if not (case/'terminal/report.json').is_file():
        return dict(ready=False,raw_audit_started=False,reason='successful Rust terminal controls are required')
    started=time.monotonic();root=Path(__file__).resolve().parents[1]
    require(sha(evidence/'performance-protocol-v1/protocol.json')==PROTOCOL_SHA,'finite protocol changed')
    if case_id==RECOVERY_CASE:
        require(sha(evidence/'rust-recovery-protocol-v2/protocol.json')==protocol_sha,'recovery protocol changed')
        proof_path=evidence/RECOVERY_VALIDATION_PATH
        require(sha(proof_path)==RECOVERY_VALIDATION_SHA,'recovery audit preparation changed')
    else:
        proof_path=evidence/'rust-target-raw-preparation-v1/completion.json'
        require(sha(proof_path)=='da3fcd681f7e640908a1f254c3dfd94713db0bc56a62fd198830a85acb704479','raw payload preparation changed')
    proof=read(proof_path)
    for name,item in proof['files'].items():require(sha(root/name)==item['sha256'],'prepared raw input code/evidence changed')
    # The target has not run yet. Its terminal verifier now also accepts the
    # evidence-bound CPU relocation. Full binary readers remain frozen.
    revised_terminal_dependencies={} if case_id==RECOVERY_CASE else {
        'tools/mam_rust_benchmark_terminal.py':'ed36954dee3e8fa0df3324cddfce6c02e69989be25c1acc001192d049986ec93',
        'tools/mam_rust_cpu_placement.py':'0bbd21282f874787c2121b4201a2e281f763fce6d50ae47c8ed90649b0284043'}
    for name,digest in proof['unchanged_reader_and_control_dependencies'].items():
        require(sha(root/name)==revised_terminal_dependencies.get(name,digest),'full-reader dependency changed')
    for name,digest in revised_terminal_dependencies.items():
        require(sha(root/name)==digest,'CPU relocation verifier changed')
    payload,catalog,binding,binding_sha,terminal=prepare(root,case,evidence/'primary-run/package.json')
    require(binding['case_id']==case_id and binding['protocol_sha256']==protocol_sha,
            'requested case differs from terminal binding')
    volume=Path('/Volumes/T7');require(volume.is_mount() and t7.resolve().is_relative_to(volume.resolve()),'T7 required')
    s=os.statvfs(t7);require(s.f_bavail*s.f_frsize>=512*2**30,'T7 start reserve')
    local=case/'raw';backup=t7/'artifacts/performance-runs-v1'/case_id/'raw'
    require(not local.exists() and not backup.exists(),'raw audit already attempted; do not retry')
    local.mkdir();backup.mkdir(parents=True)
    intent=dict(schema='b2-mam-rust-raw-stage-intent-v1',protocol_sha256=protocol_sha,binding_sha256=binding_sha,
        raw_guard_seconds=SECONDS,total_stage_seconds=7200,attempts=1,automatic_retry=False)
    write(local/'intent.json',intent);write(local/'binding.json',binding)
    try:
        admission=remote(preflight_code());payload_sha=hashlib.sha256(payload).hexdigest()
        staged=remote(stage_code(payload_sha,catalog),payload)
        require(time.monotonic()-started<60,'fresh admission expired during preparation/staging')
        with (backup/'source.tar.gz').open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        require(sha(backup/'source.tar.gz')==payload_sha,'T7 source archive readback')
        admission.update(binding_sha256=binding_sha,source=staged,source_catalog=catalog,command=command(binding_sha),
            memory_mib=16384,cpu_ids=[8,9],minimum_free_gib=1280,raw_guard_seconds=SECONDS,automatic_retry=False)
        write(local/'admission.json',admission);write(backup/'admission.json',admission)
        rc=None;error=None;cmd=command(binding_sha)
        try:
            require(time.monotonic()-started<60,'fresh admission expired before raw service launch')
            with (backup/'controller.log').open('x') as log:
                result=subprocess.run(['tsh','ssh','root@'+NODES[0],shlex.join(cmd)],stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,timeout=SECONDS+45)
                rc=result.returncode
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        controller=dict(returncode=rc,error=error,command=cmd,observation_failure_is_not_remote_completion=True)
        write(local/'controller.json',controller);write(backup/'controller.json',controller)
        require(rc==0 and error is None,'controller unsuccessful; retain attempt and inspect same service, never restart')
        collected=remote(collect_code(catalog));values={}
        require(set(collected)=={'pending','guard','attempt'},'raw terminal file coverage')
        for name,item in collected.items():
            raw=base64.b64decode(item['base64'],validate=True)
            require(len(raw)==item['bytes']<=2*2**20 and hashlib.sha256(raw).hexdigest()==item['sha256'],'raw terminal transfer differs')
            values[name]=json.loads(raw)
            with (local/(name+'.json')).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        require(values['attempt']['binding_sha256']==binding_sha and values['attempt']['automatic_retry'] is False,'attempt identity')
        published=publish(values['pending'],values['guard'],controller,binding_sha=binding_sha,terminal=terminal,
            raw_source_sha=catalog['tools/mam_rust_benchmark_raw.py']['sha256'],elapsed=time.monotonic()-started)
        published['evidence_sha256']={n:sha(local/n) for n in ['binding.json','intent.json','admission.json','controller.json','pending.json','guard.json','attempt.json']}
        write(local/'report.json',published)
        for p in sorted(local.glob('*.json')):
            target=backup/p.name
            if target.exists():require(target.read_bytes()==p.read_bytes(),'backup control differs')
            else:
                with target.open('xb') as f:f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
                require(sha(target)==sha(p),'backup hash differs')
        require(time.monotonic()-started<=7200,'whole raw stage including archive exceeded')
        completion=dict(schema='b2-mam-performance-case-completion-v1',case_id=case_id,protocol_sha256=protocol_sha,
            terminal_resource_audit_passed=True,raw_output_audit_passed=True,scientific_acceptance=False,
            performance_cost_acceptance=False,total_stage_seconds=time.monotonic()-started,
            accounting_scope='Preparation through raw audit, terminal collection and verified compact archive; final completion receipt writes excluded.',
            input_sha256={n:sha(case/n) for n in
                ['admission.json','launch.json','terminal/report.json','raw/report.json','raw/intent.json']})
        write(case/'completion.json',completion);write(backup.parent/'completion.json',completion)
        return dict(ready=True,raw_audit_started=True,raw_output_audit_passed=True,scientific_acceptance=False)
    except BaseException as exc:
        write(local/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-started,
            automatic_retry=False,observation_failure_is_not_remote_completion=True))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--t7',type=Path,required=True)
    p.add_argument('--case-id',choices=[CASE,RECOVERY_CASE],default=CASE)
    a=p.parse_args();result=run(a.evidence,a.t7,a.case_id);print(json.dumps(result))
    raise SystemExit(0 if result['ready'] else 2)
