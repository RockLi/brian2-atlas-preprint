"""One primary archive after clean full analysis; shared collection budget."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from mam_primary_archive import BASE, ARCHIVE, REPORT, CATALOG, MAX_BYTES, attempt_paths
from mam_primary_analysis_pipeline import check, read, sha
from mam_primary_resources import LABEL, NODES
from mam_launch_native_primary import rust_gate

SOURCE = BASE/'primary-archive-v1-source'
GUARD = BASE/'guards/primary-archive-v1.json'
REMOTE_BASE = Path('/atlas-home/0003/workspace/brian2-mpi-primary-20260909')


def remote(node, code, payload=None):
    result = subprocess.run(['tsh','ssh','rock@'+node,shlex.join(['taskset','-c','8,9','python3','-c',code])],
                            input=payload,capture_output=True,check=True,timeout=45)
    check(len(result.stdout)<8*2**20,'oversized archive control response')
    return json.loads(result.stdout)


def prerequisites(evidence,t7):
    resource = t7/'artifacts/primary-terminal-resources-v1'
    gate = rust_gate(evidence,resource)
    analysis = t7/'artifacts/primary-postrun-v1/report.json'
    timing = resource/'collection-controller.json'
    if gate is None or not analysis.exists() or not timing.exists(): return None
    report, previous = read(analysis),read(timing)
    check(read(resource/'admission.json')['experiment_budget']['collection_wall_limit_seconds']==7200,
          'original primary collection budget differs')
    check(report['analysis_complete'] is True
          and report['output_audit_sha256']==gate['output_report_sha256'], 'complete analysis missing')
    check(previous['schema']=='b2-mam-terminal-collection-controller-v1'
          and previous['collection_complete'] is True
          and previous['resource_report_sha256']==sha(resource/'report.json')
          and previous['shared_collection_budget_seconds']==7200, 'terminal collection timing differs')
    elapsed = previous['elapsed_seconds']
    check(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<7200,'invalid prior collection time')
    return dict(previous_collection_seconds=elapsed,analysis_report_sha256=sha(analysis),rust_gate=gate)


def recovery_gate(evidence, t7, gate, *, now=None):
    """Only the diagnosed pre-write v1 failure admits one corrected attempt."""
    prior = t7/'artifacts/primary-archive-v1'
    required = ['admission.json','controller.json','controller.log','guard.json','catalog.json']
    check(all((prior/n).is_file() for n in required), 'prior archive evidence missing')
    admission, controller, guard = (read(prior/n) for n in required[:2]+['guard.json'])
    check(admission['gate']['rust_gate'] == gate['rust_gate']
          and admission['gate']['analysis_report_sha256'] == gate['analysis_report_sha256']
          and admission['total_collection_budget_seconds'] == 7200
          and admission['attempts'] == 1 and admission['automatic_retry'] is False,
          'prior archive admission differs')
    check(controller['returncode'] == 1 and controller['error'] is None
          and guard['admitted'] is True and guard['returncode'] == 1
          and not guard.get('error') and 0 <= guard['wall_seconds'] < 5
          and 'ValueError: unsafe archive name' in (prior/'controller.log').read_text()
          and not (prior/'report.json').exists() and not (prior/'complete.json').exists(),
          'prior archive is not the diagnosed pre-write failure')
    events = dict(x.split() for x in guard['after']['memory.events'].splitlines())
    check(all(events[k] == '0' for k in ['max','oom','oom_kill','oom_group_kill']),
          'prior archive had resource failure')
    catalog = read(prior/'catalog.json')
    hidden = {n:r for n,r in catalog.items() if any(p.startswith('._') for p in Path(n).parts)}
    check(len(hidden) == 14 and all(r['bytes'] == 4096 and r['sha256'] ==
          '8b1b286f9aec72652bef8bc735ec640d9bbb1b111df990f0a1283cfa78b80b21'
          for r in hidden.values()), 'prior sidecar diagnosis differs')
    # Conservatively charge all elapsed wall time, including debugging, from
    # 60 s before the earliest original host receipt. Never reset the budget.
    now = datetime.now(timezone.utc) if now is None else now
    origin = min(datetime.fromisoformat(h['utc']) for h in admission['hosts'])
    check(origin.tzinfo is not None and now >= origin, 'invalid recovery clock')
    charge = (now-origin).total_seconds()+60
    previous = gate['previous_collection_seconds']+charge
    check(math.isfinite(previous) and previous < 7110, 'collection recovery budget exhausted')
    return dict(prior_attempt_sha256={n:sha(prior/n) for n in required},
                charged_since_prior_admission_seconds=charge,
                previous_collection_seconds=previous, maximum_attempts=2,
                automatic_retry=False)


def host_code(index,package,package_sha,paths=None,recovery=None):
    paths = attempt_paths() if paths is None else paths
    extra = ''
    if index == 0 and recovery is not None:
        old = attempt_paths(1)
        extra = f'''assert digest(Path({str(old['guard'])!r}))=={recovery['prior_attempt_sha256']['guard.json']!r}
assert digest(Path({str(old['catalog'])!r}))=={recovery['prior_attempt_sha256']['catalog.json']!r}
assert not Path({str(old['archive'])!r}).exists()
assert not Path({str(old['archive'])+'.pending'!r}).exists()
assert not Path({str(old['report'])!r}).exists()
'''
    return f'''from pathlib import Path
import os,json,hashlib,subprocess,time,datetime
b=Path({str(REMOTE_BASE)!r});host={NODES[index]!r};assert os.uname().nodename==host
assert b.resolve()==Path({str(BASE/'primary-host-v1')!r}) if {index}==0 else b.resolve().is_relative_to(Path('/home/rock'))
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True)
assert not units.strip(),units
mem={{k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}}
assert mem['MemAvailable']>32*2**30
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  released=0
  while block:=f.read(2**20):
   h.update(block)
   if f.tell()-released>=64*2**20:
    os.posix_fadvise(f.fileno(),released,f.tell()-released,os.POSIX_FADV_DONTNEED);released=f.tell()
  if f.tell()>released:os.posix_fadvise(f.fileno(),released,f.tell()-released,os.POSIX_FADV_DONTNEED)
 return h.hexdigest()
for name,row in {package['catalog']!r}.items():
 p=b/{LABEL!r}/name;assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],name
assert digest(b/'guard.py')=={package['guard_sha256']!r}
for name,h in {package['mpi_runtime']!r}.items():assert digest(b/'mpi'/name)==h,name
free=os.statvfs(b).f_bavail*os.statvfs(b).f_frsize
if {index}==0:
 assert Path('/data/brick2').is_mount() and free>1280*2**30+{MAX_BYTES}
 for p in {list(map(str,[paths['source'],paths['guard'],paths['archive'],paths['archive'].with_name(paths['archive'].name+'.pending'),paths['report'],paths['catalog']]))!r}:assert not Path(p).exists(),p
{extra}
def ticks():return {{int(x[0][3:]):list(map(int,x[1:9])) for line in Path('/proc/stat').read_text().splitlines() if (x:=line.split())[0].startswith('cpu') and x[0][3:].isdigit()}}
a=ticks();time.sleep(1);z=ticks();busy={{}}
for c in [8,9]:
 d=[y-x for x,y in zip(a[c],z[c])];assert sum(d)>0;busy[c]=100*(1-(d[3]+d[4])/sum(d))
assert max(busy.values())<50 and sum(busy.values())/2<25,busy
print(json.dumps(dict(host=host,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),package_catalog_sha256={package_sha!r},all_deployed_files_verified=True,verified_artifact_files={len(package['catalog'])},guard_verified=True,mpi_runtime_verified=True,free_bytes=free,available_memory_bytes=mem['MemAvailable'],cpu_busy_percent=busy)))'''


def guard_command(seconds,attempt_version=1):
    paths=attempt_paths(attempt_version)
    check(type(seconds) is int and 0<seconds<=7110,'archive timeout exceeds collection budget')
    return ['systemd-run','--expand-environment=no','--quiet','--wait','--pipe','--collect',
        '--unit='+paths['unit'],'--uid=rock','--service-type=exec',
        '--property=MemoryMax=8192M','--property=MemorySwapMax=0','--property=CPUQuota=200%',
        '--property=AllowedCPUs=8-9','--property=TasksMax=64','--property=RuntimeMaxSec='+str(seconds+5),
        '--property=TimeoutStopSec=5','--property=KillMode=control-group','--property=OOMPolicy=continue',
        '/usr/bin/python3',str(BASE/'primary-host-v1/guard.py'),'--output',str(paths['guard']),
        '--volume','/data/brick2','--memory-mib','8192','--cpu-percent','200',
        '--file-mib',str(MAX_BYTES//2**20),'--min-free-gib','1280','--timeout',str(seconds),
        '--','env','PYTHONDONTWRITEBYTECODE=1','TMPDIR='+str(BASE/'tmp'),
        '/atlas-home/0003/workspace/brian2-lk-20260906/.venv/bin/python',
        str(paths['source']/'tools/mam_primary_archive.py'),'--control',str(paths['source']/'control'),
        '--attempt-version',str(attempt_version)]


def run(evidence,t7,attempt_version=1):
    started = time.monotonic()
    paths=attempt_paths(attempt_version)
    gate = prerequisites(evidence,t7)
    if gate is None:return dict(ready=False,archive_started=False,reason='complete primary analysis and terminal collection timing required')
    recovery = recovery_gate(evidence,t7,gate) if attempt_version == 2 else None
    if recovery is not None:
        gate=dict(gate,previous_collection_seconds=recovery['previous_collection_seconds'])
    check(Path('/Volumes/T7').is_mount() and t7.resolve().is_relative_to(Path('/Volumes/T7').resolve()),'mounted T7 required')
    check(os.statvfs(t7).f_bavail*os.statvfs(t7).f_frsize>128*2**30,'T7 control reserve')
    destination = t7/'artifacts'/paths['directory']
    out = evidence/('primary-archive' if attempt_version == 1 else paths['directory'])
    check(not destination.exists() and not (out/'admission.json').exists(),'archive already attempted')
    package_path = evidence/'primary-run/package.json'
    package = read(package_path)
    fresh_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        hosts = list(pool.map(lambda i:remote(NODES[i],host_code(i,package,sha(package_path),paths,recovery)),range(4)))
    root = Path(__file__).resolve().parents[1]
    files = {'tools/'+n:root/'tools'/n for n in ['mam_primary_archive.py','mam_primary_resources.py',
        'mam_primary_analysis_pipeline.py','mam_launch_native_primary.py','mam_launch_primary_archive.py']}
    resources = t7/'artifacts/primary-terminal-resources-v1'
    for p in resources.glob('*.json'):files['control/resources/'+p.name]=p
    if recovery is not None:
        for name in recovery['prior_attempt_sha256']:
            files['control/prior-attempt/'+name]=t7/'artifacts/primary-archive-v1'/name
    for name in ['full-report.json','full-guard.json']:
        files['control/evidence/primary-output-audit/'+name]=evidence/'primary-output-audit'/name
    files['control/analysis-report.json']=t7/'artifacts/primary-postrun-v1/report.json'
    files['control/package.json']=package_path
    rawfiles = {n:p.read_bytes() for n,p in files.items()}
    rawfiles['control/postrun-hosts.json']=(json.dumps(hosts,indent=2)+'\n').encode()
    manifest = {n:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for n,raw in rawfiles.items()}
    rawfiles['catalog.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name,raw in sorted(rawfiles.items()):
            info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    blob=buffer.getvalue();digest=hashlib.sha256(blob).hexdigest()
    check(len(blob)<4*2**20 and sum(len(v) for v in rawfiles.values())<16*2**20,'archive source/control budget')
    remote(NODES[0],f'''from pathlib import Path
import sys,io,tarfile,json,hashlib
r=sys.stdin.buffer.read(4*2**20+1);assert len(r)=={len(blob)} and hashlib.sha256(r).hexdigest()=={digest!r}
p=Path({str(paths['source'])!r});p.mkdir()
with tarfile.open(fileobj=io.BytesIO(r)) as t:t.extractall(p,filter='data')
for name,row in json.loads((p/'catalog.json').read_text()).items():
 f=p/name;assert f.resolve().is_relative_to(p.resolve()) and not f.is_symlink() and f.stat().st_size==row['bytes']
 with f.open('rb') as s:assert hashlib.file_digest(s,'sha256').hexdigest()==row['sha256']
print(json.dumps(dict(source_staged=True)))''',blob)
    check(time.monotonic()-fresh_started<60,'archive resource admission expired; no retry')
    # Leave 90 s for bounded controller/guard termination and the one 45 s
    # control collection. Prior terminal collection is charged, not reset.
    seconds=math.floor(7200-gate['previous_collection_seconds']-(time.monotonic()-started)-90)
    check(seconds>0,'collection budget exhausted before archiving')
    cmd=guard_command(seconds,attempt_version)
    destination.mkdir();out.mkdir(parents=True,exist_ok=True)
    admission=dict(gate=gate,hosts=hosts,source_sha256=digest,archive_timeout_seconds=seconds,
                   total_collection_budget_seconds=7200,completion_reserve_seconds=90,
                   maximum_archive_bytes=MAX_BYTES,attempts=1,automatic_retry=False,
                   attempt_version=attempt_version,recovery=recovery)
    for directory in [destination,out]:(directory/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
    (destination/'source.tar.gz').write_bytes(blob)
    result,error=None,None
    try:
        with (destination/'controller.log').open('x') as log:
            result=subprocess.run(['tsh','ssh','root@'+NODES[0],shlex.join(cmd)],stdin=subprocess.DEVNULL,
                                  stdout=log,stderr=subprocess.STDOUT,timeout=seconds+30)
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    receipt=dict(returncode=result.returncode if result else None,error=error,command=cmd,
                 observation_failure_is_not_remote_completion=True)
    (destination/'controller.json').write_text(json.dumps(receipt,indent=2)+'\n')
    controls=remote(NODES[0],f'''from pathlib import Path
import json
out={{}}
for name,path in {dict(report=str(paths['report']),catalog=str(paths['catalog']),guard=str(paths['guard']))!r}.items():
 p=Path(path)
 if p.exists():
  assert p.stat().st_size<2*2**20;out[name]=p.read_text()
print(json.dumps(out))''')
    for name,raw in controls.items():
        for directory in [destination,out]:(directory/(name+'.json')).write_text(raw)
    elapsed=time.monotonic()-started+gate['previous_collection_seconds']
    check(error is None and result.returncode==0 and elapsed<7200,'archive controller/budget failed')
    guard=read(destination/'guard.json');events=dict(s.split() for s in guard['after']['memory.events'].splitlines())
    check(guard['admitted'] is True and guard['returncode']==0 and not guard.get('error')
          and all(events[k]=='0' for k in ['max','oom','oom_kill','oom_group_kill']),'archive resource guard failed')
    report=read(destination/'report.json')
    check(report['archive_verified'] is True and report['catalog_sha256']==sha(destination/'catalog.json'),'archive receipt/catalog differs')
    success=dict(ready=True,archive_verified=True,total_collection_seconds=elapsed,
                 archive=report['archive'],independent_device_backup=False,scientific_acceptance=False)
    for directory in [destination,out]:(directory/'complete.json').write_text(json.dumps(success,indent=2)+'\n')
    return success


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--t7',type=Path,required=True)
    parser.add_argument('--attempt-version',type=int,choices=(1,2),default=1)
    args=parser.parse_args();result=run(args.evidence,args.t7,args.attempt_version)
    print(json.dumps(result));raise SystemExit(0 if result['ready'] else 2)
