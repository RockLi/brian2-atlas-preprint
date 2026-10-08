"""One guarded full raw audit for the admitted confirmation replication."""
from mam_confirmation_profile import REPLICATE,selected,worker_environment,IDENTITY_SHA_1751
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

from mam_confirmation_raw import BASE,CASE,LABEL,NODES,terminal_gate,terminal_names,require
from mam_confirmation_terminal_sync import IDENTITY_SHA
from mam_confirmation_terminal import context_gate
from mam_launch_confirmation_run import read,sha,write
from mam_benchmark_terminal import guard
IDENTITY_SHA=selected(IDENTITY_SHA,IDENTITY_SHA_1751)

PREPARATION_SHA='a9f01fffb250e26a91d9d5a667fbbab948b04ac2564e8e7e305927a8cc506122'
REMOTE_TERMINAL_PREPARATION_SHA='4b09f22aed7e10d15c7387338c302ea5759d4b3ebd68d7411303a177abd74318'
REMOTE_RAW_WORKER_SHA='a38245d4ceaa4b285f7a77aa5bcb08232a18cd9e7896ad890a9a825a994b47f2'
READER_PROOF_SHA='e6517390c06a7c84f8f40e886567deb09462a09feeddaf2ff2618825ceae9cb5'
SOURCE=BASE/f'confirmation-raw-source-v1-seed{REPLICATE}'
OUTPUT=BASE/'confirmation-raw-audit'/LABEL
GUARD=BASE/f'guards/confirmation-raw-v1-seed{REPLICATE}.json'
PREFIX=f'b2mpi-confirm{REPLICATE}-raw-v1'
UNIT=PREFIX+'-audit'
PYTHON='/atlas-home/0003/workspace/brian2-lk-20260906/.venv/bin/python'
SECONDS=6600


def application(binding_sha):
    return ['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',
        *worker_environment(),'PYTHONPATH='+str(SOURCE/'tools')+':'+str(SOURCE/'python'),PYTHON,
        str(SOURCE/'tools/mam_confirmation_raw.py'),'--case',str(SOURCE/'case'),
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


def prepare(root,case):
    """Reproduce terminal controls in a bounded temporary directory, no network."""
    report=read(case/'terminal/report.json')
    names=terminal_names(report)
    recovered=report['schema']=='b2-mam-confirmation-remote-terminal-v1'
    if recovered:require(not (case/'launch.json').exists(),'mixed local receipt requires inspection')
    controls={}
    for name in names:
        p=case/name;require(p.is_file() and not p.is_symlink() and p.stat().st_size<=64*2**20,'bounded terminal input')
        controls[name]=p.read_bytes()
    require(sum(map(len,controls.values()))<=48*2**20,'combined terminal controls ceiling')
    contract=context_gate(json.loads(controls['identity.json']),json.loads(controls['protocol.json']),json.loads(controls['admission.json']))
    binding=dict(schema='b2-mam-confirmation-remote-raw-binding-v1' if recovered else 'b2-mam-confirmation-raw-binding-v1',identity_sha256=IDENTITY_SHA,case_id=contract['case_id'],protocol_sha256=contract['protocol_sha256'],
                 files={n:hashlib.sha256(v).hexdigest() for n,v in controls.items()})
    raw=(json.dumps(binding,indent=2)+'\n').encode();binding_sha=hashlib.sha256(raw).hexdigest()
    with tempfile.TemporaryDirectory(prefix='mam-confirmation-raw-controls-',dir='/private/tmp') as tmp:
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
q=Path({str(OUTPUT.parent)!r});q.mkdir(exist_ok=True);assert q.resolve()==q and q.is_relative_to(Path('/data/brick2'))
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


def publish(pending,g,controller,*,binding_sha,terminal,identity,protocol_sha,raw_source_sha,elapsed):
    require(controller['returncode']==0 and type(controller['returncode']) is int and controller['error'] is None
            and controller['command']==command(binding_sha),'raw controller did not finish successfully')
    require(g['command']==application(binding_sha),'guard command differs from admitted raw audit')
    spec=dict(host=NODES[0],role='audit',memory_bytes=16*2**30,pids_max=64,cpu_ids=[8,9],cpu_quota_cores=2,
        volume='/data/brick2',allow_root_volume=False,file_limit_bytes=512*2**20,
        minimum_free_bytes=1280*2**30,reserved_host_memory_bytes=64*2**30)
    audited=guard(g,spec,PREFIX,SECONDS)
    require(pending['schema']=='b2-mam-confirmation-raw-pending-v1'
            and pending['replicate']==REPLICATE and pending['identity_sha256']==IDENTITY_SHA and pending['case_id']==terminal['case_id']
            and pending['label']==terminal['label'] and pending['protocol_sha256']==protocol_sha
            and pending['binding_sha256']==binding_sha and pending['model_sha256']==identity['model_sha256']
            and pending['plan_sha256']==identity['plan_sha256'] and pending['executable_sha256']==identity['executable_sha256']
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


def verify_profile_source(root,proof,receipt_sha):
    # The sole late-bound receipt digest cannot hash itself. All other profile
    # bytes, including the actual admission and runtime source catalog, are fixed.
    text=(root/'tools/mam_confirmation_profile.py').read_text()
    line='RAW_PREPARATION_SHA_1751='+repr(receipt_sha)
    require(text.count(line)==1,'raw preparation receipt pin assignment differs')
    normalized=text.replace(line,'RAW_PREPARATION_SHA_1751=None',1).encode()
    require(hashlib.sha256(normalized).hexdigest()==proof['profile_with_receipt_pin_unset_sha256'],
            'confirmation profile changed outside the bound receipt digest')


def run(evidence,t7):
    case=evidence/CASE
    if not (case/'terminal/report.json').is_file():
        return dict(ready=False,raw_audit_started=False,reason='Successful confirmation terminal controls are required; no outputs read or network action.')
    started=time.monotonic();root=Path(__file__).resolve().parents[1]
    if REPLICATE==1750:
        proof=read(case/'audit-preparation-v1/report.json',PREPARATION_SHA)
        for name,digest in proof['source_sha256'].items():
            expected=REMOTE_RAW_WORKER_SHA if name=='tools/mam_confirmation_raw.py' else digest
            require(sha(root/name)==expected,'prepared 1750 audit source changed')
        recovery_proof=read(case/'remote-terminal-recovery-ready-v1/report.json',REMOTE_TERMINAL_PREPARATION_SHA)
        for name,digest in recovery_proof['source_sha256'].items():
            require(sha(root/name)==digest,'remote terminal recovery verifier changed')
    else:
        from mam_confirmation_profile import RAW_PREPARATION_SHA_1751
        require(RAW_PREPARATION_SHA_1751 is not None,'new raw source binding has not been pinned')
        proof=read(case/'audit-preparation-v2/report.json',RAW_PREPARATION_SHA_1751)
        require(proof['replicate']==REPLICATE and proof['identity_sha256']==IDENTITY_SHA,'new raw preparation identity')
        verify_profile_source(root,proof,RAW_PREPARATION_SHA_1751)
        for name,digest in proof['source_sha256'].items():
            require(sha(root/name)==digest,'prepared new audit source changed')
        require(read(case/'terminal/report.json')['schema']=='b2-mam-confirmation-terminal-v1',
                'new remote recovery needs its own live provenance and audited binding')
    readers=read(evidence/'rust-timing-quantization-v3/validation.json',READER_PROOF_SHA)
    for name,digest in readers['unchanged_reader_and_control_dependencies'].items():
        require(sha(root/name)==digest,'frozen full binary reader dependency changed')
    payload,catalog,binding,binding_sha,terminal=prepare(root,case)
    identity=read(case/'identity.json',IDENTITY_SHA)
    protocol_sha=binding['protocol_sha256']
    require(binding['case_id']==CASE and str(OUTPUT)==identity['raw_audit'],'requested run differs from bound identity')
    volume=Path('/Volumes/T7');require(volume.is_mount() and t7.resolve().is_relative_to(volume.resolve()),'T7 required')
    s=os.statvfs(t7);require(s.f_bavail*s.f_frsize>=512*2**30,'T7 start reserve')
    local=case/'raw';backup=t7/'artifacts'/CASE/'raw'
    require(not local.exists() and not backup.exists(),'raw audit already attempted; do not retry')
    local.mkdir();backup.mkdir(parents=True)
    intent=dict(schema='b2-mam-confirmation-raw-stage-intent-v1',replicate=REPLICATE,identity_sha256=IDENTITY_SHA,protocol_sha256=protocol_sha,binding_sha256=binding_sha,
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
            identity=identity,protocol_sha=protocol_sha,raw_source_sha=catalog['tools/mam_confirmation_raw.py']['sha256'],elapsed=time.monotonic()-started)
        published['evidence_sha256']={n:sha(local/n) for n in ['binding.json','intent.json','admission.json','controller.json','pending.json','guard.json','attempt.json']}
        write(local/'report.json',published)
        for p in sorted(local.glob('*.json')):
            target=backup/p.name
            if target.exists():require(target.read_bytes()==p.read_bytes(),'backup control differs')
            else:
                with target.open('xb') as f:f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
                require(sha(target)==sha(p),'backup hash differs')
        require(time.monotonic()-started<=7200,'whole raw stage including archive exceeded')
        completion=dict(schema='b2-mam-confirmation-engineering-completion-v1',case_id=CASE,replicate=REPLICATE,identity_sha256=IDENTITY_SHA,protocol_sha256=protocol_sha,
            terminal_resource_audit_passed=True,raw_output_audit_passed=True,scientific_acceptance=False,
            performance_cost_acceptance=False,total_stage_seconds=time.monotonic()-started,
            accounting_scope='Preparation through raw audit, terminal collection and verified compact archive; final completion receipt writes excluded.',
            input_sha256={n:sha(case/n) for n in
                terminal_names(read(case/'terminal/report.json'))[:5]+['raw/report.json','raw/intent.json']})
        write(case/'completion.json',completion);write(backup.parent/'completion.json',completion)
        return dict(ready=True,raw_audit_started=True,raw_output_audit_passed=True,scientific_acceptance=False)
    except BaseException as exc:
        write(local/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-started,
            automatic_retry=False,observation_failure_is_not_remote_completion=True))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--t7',type=Path,required=True)
    a=p.parse_args();result=run(a.evidence,a.t7);print(json.dumps(result))
    raise SystemExit(0 if result['ready'] else 2)
