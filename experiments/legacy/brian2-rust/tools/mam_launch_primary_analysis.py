"""Launch one bounded post-primary analysis after successful resource collection.

Missing prerequisites exit before network access or output creation. This is
an actual analysis launcher, never a simulation launcher or retry mechanism.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from mam_primary_analysis_pipeline import BASE, OUTPUT, AUDIT_OUTPUT, TOTAL_SECONDS, resource_gate, check, sha, read

NODE = 'hk-prod-model-ae02-23'
SOURCE = BASE/'primary-postrun-v1-source'
NORMALIZATION = 'normalization/mam-official-analysis-neuron-sizes-v1.json'
GUARD = BASE/'guards/primary-postrun-v1.json'
PYTHON = '/atlas-home/0003/workspace/brian2-lk-20260906/.venv/bin/python'
UNIT = 'b2mpi-analysis-primary-postrun-v1'


def remote(code, *, payload=None):
    result = subprocess.run(['tsh', 'ssh', 'rock@'+NODE,
                             shlex.join(['taskset','-c','8,9','python3','-c',code])],
                            input=payload, capture_output=True, check=True, timeout=45)
    check(len(result.stdout) <= 16*2**20, 'oversized control response')
    return json.loads(result.stdout)


def bundle(root, resources, normalization, generated_data):
    files = {str(p.relative_to(root)): p for parent in ['tools', 'python']
             for p in (root/parent).rglob('*.py') if not p.name.startswith('._')}
    files.update({'resources/'+p.name: p for p in resources.glob('*.json')})
    files[NORMALIZATION] = normalization
    metadata = read(normalization)
    check(Path(metadata['generated_data_name']).name == metadata['generated_data_name'], 'unsafe normalization data name')
    check(sha(generated_data) == metadata['generated_data_sha256'] ==
          '8c66bb68d55cf2bff222c67716952a2b6260576d6ba0a3650cc150af49f7c2cb',
          'official normalization data changed')
    files['normalization/'+metadata['generated_data_name']] = generated_data
    catalog = {n: dict(bytes=p.stat().st_size, sha256=sha(p)) for n, p in files.items()}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, p in sorted(files.items()):
            archive.add(p, arcname=name, recursive=False)
        data = (json.dumps(catalog, indent=2)+'\n').encode()
        info = tarfile.TarInfo('catalog.json'); info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    data = buffer.getvalue()
    check(len(data) < 8*2**20 and sum(r['bytes'] for r in catalog.values()) < 32*2**20,
          'source package budget exceeded')
    return data, catalog


def preflight_code(guard_sha):
    return f'''from pathlib import Path
import os,json,subprocess,time,datetime,hashlib
b=Path({str(BASE)!r})
assert Path('/data/brick2').is_mount() and b.resolve().is_relative_to(Path('/data/brick2'))
assert os.uname().nodename=={NODE!r}
with (b/'primary-host-v1/guard.py').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={guard_sha!r}
for p in {list(map(str,[SOURCE,OUTPUT,AUDIT_OUTPUT,GUARD]))!r}:assert not Path(p).exists(),p
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True)
assert not units.strip(),units
mem={{k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}}
free=os.statvfs(b).f_bavail*os.statvfs(b).f_frsize
assert free>1280*2**30 and mem['MemAvailable']>80*2**30
def ticks():return {{int(x[0][3:]):list(map(int,x[1:9])) for line in Path('/proc/stat').read_text().splitlines() if (x:=line.split())[0].startswith('cpu') and x[0][3:].isdigit()}}
a=ticks();time.sleep(1);z=ticks();busy={{}}
for c in [8,9]:
 d=[y-x for x,y in zip(a[c],z[c])];assert sum(d)>0;busy[c]=100*(1-(d[3]+d[4])/sum(d))
assert max(busy.values())<50 and sum(busy.values())/2<25,busy
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=free,available_memory_bytes=mem['MemAvailable'],cpu_busy_percent=busy,active_own_units=[])))'''


def command():
    return ['systemd-run','--expand-environment=no','--quiet','--wait','--pipe','--collect',
            '--unit='+UNIT,'--uid=rock','--service-type=exec',
            '--property=MemoryMax=16384M','--property=MemorySwapMax=0',
            '--property=CPUQuota=200%','--property=AllowedCPUs=8-9',
            '--property=TasksMax=64','--property=RuntimeMaxSec=10805',
            '--property=TimeoutStopSec=5','--property=KillMode=control-group',
            '--property=OOMPolicy=continue','/usr/bin/python3',str(BASE/'primary-host-v1/guard.py'),
            '--output',str(GUARD),'--volume','/data/brick2','--memory-mib','16384',
            '--cpu-percent','200','--file-mib','512','--min-free-gib','1280',
            '--timeout',str(TOTAL_SECONDS),'--','env','OPENBLAS_NUM_THREADS=1',
            'OMP_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','TMPDIR='+str(BASE/'tmp'),
            'MPLCONFIGDIR='+str(BASE/'mpl'),'PYTHONPATH='+str(SOURCE/'tools')+':'+str(SOURCE/'python'),
            PYTHON,str(SOURCE/'tools/mam_primary_analysis_pipeline.py'),
            '--source',str(SOURCE),'--resources',str(SOURCE/'resources'),
            '--normalization',str(SOURCE/NORMALIZATION)]


def collect_code():
    files = {'full-report.json': str(AUDIT_OUTPUT), 'full-guard.json': str(GUARD),
             'report.json': str(OUTPUT/'report.json'), 'stages.jsonl': str(OUTPUT/'stages.jsonl')}
    for stage, report in [('activity','activity.json'),('cell','paper-cell-metrics.json'),
                          ('correlation','correlation.json'),('series','time-series.json')]:
        for name in [report, 'catalog.json']:
            files[stage+'/'+name] = str(OUTPUT/stage/name)
    return f'''from pathlib import Path
import base64,json,hashlib
out={{}};total=0
for name,path in {files!r}.items():
 p=Path(path)
 if not p.exists():continue
 assert p.stat().st_size<2*2**20,name
 r=p.read_bytes();total+=len(r);assert total<8*2**20
 out[name]=dict(bytes=len(r),sha256=hashlib.sha256(r).hexdigest(),base64=base64.b64encode(r).decode())
print(json.dumps(out))'''


def run(evidence, t7, normalization, generated_data):
    resources = t7/'artifacts/primary-terminal-resources-v1'
    prerequisite = resource_gate(resources)
    if prerequisite is None:
        return dict(ready=False, analysis_started=False, reason='successful primary terminal resources required')
    volume = Path('/Volumes/T7')
    check(volume.is_mount() and t7.resolve().is_relative_to(volume.resolve()), 'T7 must be mounted')
    stat = os.statvfs(t7)
    check(stat.f_bavail*stat.f_frsize > 128*2**30, 'T7 reserve below 128 GiB')
    destination = t7/'artifacts/primary-postrun-v1'
    local_evidence = evidence/'primary-postrun'
    check(not destination.exists() and not (local_evidence/'admission.json').exists(), 'analysis already attempted')
    check(not (evidence/'primary-output-audit/full-report.json').exists(), 'standalone output audit already attempted')
    root = Path(__file__).resolve().parents[1]
    data, catalog = bundle(root, resources, normalization, generated_data)
    digest = hashlib.sha256(data).hexdigest()
    started = time.monotonic()
    guard_sha = read(resources/'admission.json')['preflight'][0]['guard_sha256']
    admission = remote(preflight_code(guard_sha))
    # Source stage is bounded, checks every member, and is also the remote
    # exclusive attempt marker. Its success never permits automatic retry.
    staged = remote(f'''from pathlib import Path
import sys,io,tarfile,json,hashlib
r=sys.stdin.buffer.read(8*2**20+1)
assert len(r)=={len(data)} and hashlib.sha256(r).hexdigest()=={digest!r}
p=Path({str(SOURCE)!r});p.mkdir()
with tarfile.open(fileobj=io.BytesIO(r)) as t:t.extractall(p,filter='data')
for name,row in {catalog!r}.items():
 f=p/name;assert f.resolve().is_relative_to(p.resolve()) and not f.is_symlink()
 assert f.stat().st_size==row['bytes']
 with f.open('rb') as s:assert hashlib.file_digest(s,'sha256').hexdigest()==row['sha256']
print(json.dumps(dict(source_sha256={digest!r},verified_files={len(catalog)})))''', payload=data)
    check(time.monotonic()-started < 60, 'fresh admission expired during staging; do not retry')
    destination.mkdir()
    local_evidence.mkdir(parents=True, exist_ok=True)
    admission.update(resource_inputs=prerequisite, source=staged, shared_analysis_seconds=TOTAL_SECONDS,
                     memory_mib=16384, cpu_ids=[8,9], file_mib=512, automatic_retry=False, attempts=1)
    (destination/'admission.json').write_text(json.dumps(admission, indent=2)+'\n')
    (destination/'source.tar.gz').write_bytes(data)
    (local_evidence/'admission.json').write_text(json.dumps(admission, indent=2)+'\n')
    cmd = command()
    result, error = None, None
    try:
        with (destination/'controller.log').open('x') as log:
            result = subprocess.run(['tsh','ssh','root@'+NODE,shlex.join(cmd)],
                                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                    timeout=TOTAL_SECONDS+45)
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
    receipt = dict(returncode=result.returncode if result else None, error=error, command=cmd,
                   observation_failure_is_not_remote_completion=True)
    (destination/'controller.json').write_text(json.dumps(receipt, indent=2)+'\n')
    # One read-only collection attempt also preserves partial/failed reports.
    collected = remote(collect_code())
    for name, row in collected.items():
        raw = base64.b64decode(row['base64'], validate=True)
        check(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'control transfer changed')
        for parent in [destination, local_evidence]:
            path = parent/name; path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream: stream.write(raw)
    check(error is None and result.returncode == 0, 'analysis controller failed; inspect retained evidence')
    guard = read(destination/'full-guard.json')
    events = dict(line.split() for line in guard['after']['memory.events'].splitlines())
    check(guard['admitted'] is True and guard['returncode'] == 0 and not guard.get('error')
          and all(events[k] == '0' for k in ['max','oom','oom_kill','oom_group_kill']), 'analysis guard failed')
    check(read(destination/'report.json')['analysis_complete'] is True, 'incomplete pipeline')
    # Existing native gate consumes these canonical paths. The guard covers
    # the entire chain, so its accounting must not be called raw-audit-only.
    for name in ['full-report.json','full-guard.json']:
        with (evidence/'primary-output-audit'/name).open('xb') as stream:
            stream.write((destination/name).read_bytes())
    return dict(ready=True, analysis_complete=True, output=str(destination),
                scientific_acceptance=False, performance_cost_acceptance=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--t7', type=Path, required=True)
    parser.add_argument('--normalization', type=Path, required=True)
    parser.add_argument('--generated-data', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.evidence, args.t7, args.normalization, args.generated_data)
    print(json.dumps(result))
    raise SystemExit(0 if result['ready'] else 2)
