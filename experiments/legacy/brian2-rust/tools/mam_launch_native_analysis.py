"""Run native raw audit/publication and scientific summaries in one shared budget.

Two terminal resource guards are necessary: raw-audit acceptance must precede
science input publication. The second guard receives only the unused portion
of the same 10800-second allowance. References additionally require FC and lags.
No simulation, retry or scientific pass.
"""
import argparse
import base64
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

from mam_native_analysis_pipeline import BASE, OUTPUT, RAW, MAX_RAW, TOTAL_SECONDS, analysis_spec
from mam_collect_native_primary_raw import control_gate
from mam_native_primary_output_audit import collection_gate, PENDING, SUMMARY, PRIOR
from mam_primary_analysis_pipeline import check, read, sha
from mam_launch_primary_analysis import remote, NORMALIZATION, PYTHON

NODE='hk-prod-model-ae02-23'
SOURCE=BASE/'native-primary-postrun-v1-source'
RAW_GUARD=BASE/'guards/native-primary-raw-audit-v1.json'
SCIENCE_GUARD=BASE/'guards/native-primary-science-v1.json'
BUDGET=BASE/'native-primary-science-budget-v1.json'
MAX_ARCHIVE=32*2**20
MAX_EXPANDED=256*2**20
RESERVE_SECONDS=90


def analysis_paths(reference_seed):
    if reference_seed is None:
        return dict(source=SOURCE,raw=RAW,output=OUTPUT,raw_guard=RAW_GUARD,
                    science_guard=SCIENCE_GUARD,budget=BUDGET,
                    unit_prefix='b2mpi-native-primary-analysis-',
                    evidence_subdir=Path('primary-native-postrun/run'),
                    controls='native-primary-terminal-resources-v3',
                    outer_guard='native-primary-terminal-controls-guard-v3.json',
                    collection='native-primary-raw-collection-v1')
    from mam_launch_native_full_reference import CAMPAIGN
    spec=analysis_spec(reference_seed);stem=f'native-reference{reference_seed}'
    return dict(source=BASE/(stem+'-postrun-v1-source'),raw=spec['raw'],output=spec['output'],
                raw_guard=BASE/'guards'/(stem+'-raw-audit-v1.json'),
                science_guard=BASE/'guards'/(stem+'-science-v1.json'),
                budget=BASE/(stem+'-science-budget-v1.json'),
                unit_prefix='b2mpi-'+stem+'-analysis-',
                evidence_subdir=Path(CAMPAIGN)/f'seed{reference_seed}'/'analysis',
                controls=stem+'-terminal-controls-v1',outer_guard=stem+'-terminal-controls-guard-v1.json',
                collection=stem+'-raw-collection-v1')


# Existing independently audited reference artifacts; no reference recomputation.
INTERAREA_INPUTS={
    'matrices/matrices.json':('mam-official-interarea-matrices-v2/matrices.json',7318,'7f5cb2b711f37005ad6eefd3f04b7c9a479174aa5a1b8dc99c745c107a3487d3'),
    'matrices/matrices.npz':('mam-official-interarea-matrices-v2/matrices.npz',83334,'2bf63e2ab2d11a235f2cf95f0eeb1d38e37ffe659ce9439c7dee8b11e161aa55'),
    'fc-reference/report.json':('mam-original-synaptic-reconstruction-v2/report.json',12586,'5389f31778d3ccb75831035fad832782bb60789ce12025ec589cc563756c6e58'),
    'fc-reference/catalog.json':('mam-original-synaptic-reconstruction-v2/catalog.json',520,'eef2503e3b1bc24ea3e5ecca40896580d13b833295254794c3368fa2aaca2d26'),
    'fc-reference/reconstructed.npz':('mam-original-synaptic-reconstruction-v2/reconstructed.npz',24174436,'631f5e185da506e7adac02fd94b3474388c7971c300ea5c986474dc47fef559a'),
    'lag-reference/report.json':('mam-propagation-reference-audit-v1/report.json',4294,'fa6ccc141e8eac984c1a0e7dc0a58b7fbb23f0c232137c5e8d0df0603f09a58b'),
    'lag-reference/catalog.json':('mam-propagation-reference-audit-v1/catalog.json',519,'ec010aabbb2d4070f7d9195f3297008c2c226acc16b65225c9353385724ceba9'),
    'lag-reference/selected-curves.npz':('mam-propagation-reference-audit-v1/selected-curves.npz',8537,'afff99f6e01bac6c87cbbf2f121a10d728f439e491390ef795510273f297eb2a'),
}


def interarea_files(artifact_root):
    files={}
    for name,(relative,size,digest) in INTERAREA_INPUTS.items():
        path=artifact_root/relative
        check(path.is_file() and not path.is_symlink() and path.stat().st_size==size
              and sha(path)==digest,'pinned inter-area reference changed: '+name)
        files['interarea/'+name]=path
    return files


def remaining(start,cap,clock=time.monotonic):
    value=math.floor(TOTAL_SECONDS-(clock()-start)-RESERVE_SECONDS)
    check(value>0,'shared native analysis budget exhausted')
    return min(value,cap)


def prerequisites(t7,*,reference_seed=None):
    paths=analysis_paths(reference_seed)
    controls=t7/'artifacts'/paths['controls']
    outer=t7/'artifacts'/paths['outer_guard']
    collection=t7/'artifacts'/paths['collection']
    if not (collection/'report.json').exists():return None
    gate=control_gate(controls,outer,reference_seed=reference_seed)
    if gate is None:return None
    _,report=collection_gate(collection,gate['hosts'],gate['resource_report_sha256'],outer,reference_seed=reference_seed)
    return dict(controls=controls,outer=outer,collection=collection,
                resource_report_sha256=gate['resource_report_sha256'],collection_report_sha256=sha(collection/'report.json'))


def bundle(root,gate,normalization,generated_data,*,reference_seed=None):
    files={str(p.relative_to(root)):p for parent in ['tools','python'] for p in (root/parent).rglob('*.py')
           if not p.name.startswith('._')}
    for prefix,directory in [('controls',gate['controls']),('collection',gate['collection'])]:
        files.update({prefix+'/'+p.name:p for p in directory.glob('*.json') if not p.name.startswith('._')})
    files['controls-outer-guard.json']=gate['outer']
    files['prior/catalog.json']=root/'mpi-evidence/mam-long-observation/mam-native-long-metastable-backup-catalog.json'
    files['prior/summary.json']=root/'mpi-evidence/mam-native-long-observation'/PRIOR/'summary.json'
    if reference_seed is not None:
        spec=analysis_spec(reference_seed)
        folder,catalog_name={1730:('nest-reference-ensemble','nest-ensemble-backup-catalog.json'),
                             1731:('nest-reference-three-seeds','nest-ensemble-three-backup-catalog.json')}[reference_seed]
        files['prior/catalog.json']=root/'mpi-evidence'/folder/catalog_name
        files['prior/summary.json']=root/'mpi-evidence'/folder/(spec['prior']+'-audit/summary.json')
        check(sha(files['prior/catalog.json'])==spec['prior_catalog_sha256']
              and sha(files['prior/summary.json'])==spec['prior_summary_sha256'],'reference prefix pins changed')
        files.update(interarea_files(gate['controls'].parent))
    files[NORMALIZATION]=normalization;metadata=read(normalization)
    check(Path(metadata['generated_data_name']).name==metadata['generated_data_name'],'unsafe normalization member')
    check(sha(generated_data)==metadata['generated_data_sha256']==
          '8c66bb68d55cf2bff222c67716952a2b6260576d6ba0a3650cc150af49f7c2cb','normalization input changed')
    files['normalization/'+metadata['generated_data_name']]=generated_data
    check(len(files)<=2048 and all(p.is_file() and not p.is_symlink() for p in files.values()),'invalid/oversized source set')
    catalog={n:dict(bytes=p.stat().st_size,sha256=sha(p)) for n,p in files.items()}
    check(sum(r['bytes'] for r in catalog.values())<MAX_EXPANDED,'native control/source package too large')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as t:
        for n,p in sorted(files.items()):t.add(p,arcname=n,recursive=False)
        raw=(json.dumps(catalog,indent=2)+'\n').encode();check(len(raw)<2**20,'source catalog too large')
        info=tarfile.TarInfo('catalog.json');info.size=len(raw);t.addfile(info,io.BytesIO(raw))
    data=buffer.getvalue();check(len(data)<MAX_ARCHIVE,'native compressed source package too large')
    return data,catalog


def preflight_code(science=False,*,reference_seed=None):
    paths=analysis_paths(reference_seed)
    absent=[paths['output'],paths['science_guard'],paths['budget']] if science else [paths['source'],paths['output'],paths['raw_guard'],paths['science_guard'],paths['budget'],paths['raw']/PENDING,paths['raw']/SUMMARY,paths['raw']/'primary-raw-audit-attempt.json']
    return f'''from pathlib import Path
import os,json,subprocess,time,datetime
b=Path({str(BASE)!r});assert os.uname().nodename=={NODE!r}
assert Path('/data/brick2').is_mount() and b.resolve().is_relative_to(Path('/data/brick2'))
assert Path({str(paths['raw'])!r}).is_dir() and not Path({str(paths['raw'])!r}).is_symlink()
for path in {list(map(str,absent))!r}:assert not Path(path).exists(),path
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5);assert not units.strip(),units
mem={{k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}}
free=os.statvfs(b).f_bavail*os.statvfs(b).f_frsize
assert mem['MemAvailable']>80*2**30 and free>1536*2**30
def ticks():return {{int(x[0][3:]):list(map(int,x[1:9])) for line in Path('/proc/stat').read_text().splitlines() if (x:=line.split())[0].startswith('cpu') and x[0][3:].isdigit()}}
a=ticks();time.sleep(1);z=ticks();busy={{}}
for cpu in [8,9]:
 d=[y-x for x,y in zip(a[cpu],z[cpu])];assert sum(d)>0;busy[cpu]=100*(1-(d[3]+d[4])/sum(d))
assert max(busy.values())<50 and sum(busy.values())/2<25,busy
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=free,available_memory_bytes=mem['MemAvailable'],cpu_busy_percent=busy,active_own_units=[],science_phase={science!r})))'''


def stage_code(size,digest,*,reference_seed=None):
    paths=analysis_paths(reference_seed)
    return f'''from pathlib import Path
import sys,io,tarfile,json,hashlib,resource
resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30));resource.setrlimit(resource.RLIMIT_CPU,(60,60))
r=sys.stdin.buffer.read({MAX_ARCHIVE}+1);assert len(r)=={size} and hashlib.sha256(r).hexdigest()=={digest!r}
p=Path({str(paths['source'])!r});p.mkdir()
with tarfile.open(fileobj=io.BytesIO(r)) as t:
 members=t.getmembers();assert len(members)<=2049 and len({{m.name for m in members}})==len(members)
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
 assert sum(m.size for m in members)<{MAX_EXPANDED}+2**20
 t.extractall(p,filter='data')
c=json.loads((p/'catalog.json').read_text());assert set(c)|{{'catalog.json'}}=={{m.name for m in members}}
for name,row in c.items():
 f=p/name;assert f.resolve().is_relative_to(p.resolve()) and not f.is_symlink() and f.stat().st_size==row['bytes']
 with f.open('rb') as s:assert hashlib.file_digest(s,'sha256').hexdigest()==row['sha256']
print(json.dumps(dict(source_sha256={digest!r},verified_files=len(c))))'''


def command(phase,timeout,*,reference_seed=None):
    paths=analysis_paths(reference_seed)
    check(phase in ['raw','science'] and 0<timeout<=TOTAL_SECONDS,'invalid native analysis guard')
    guard=paths['raw_guard'] if phase=='raw' else paths['science_guard']
    file_mib=512 if phase=='raw' else MAX_RAW//2**20
    app=[PYTHON,str(paths['source']/'tools/mam_native_primary_output_audit.py'),'audit',
        '--root',str(paths['raw']),'--controls',str(paths['source']/'controls'),'--collection',str(paths['source']/'collection'),
        '--control-guard',str(paths['source']/'controls-outer-guard.json'),'--prior-catalog',str(paths['source']/'prior/catalog.json'),
        '--prior-summary',str(paths['source']/'prior/summary.json')] if phase=='raw' else [
        PYTHON,str(paths['source']/'tools/mam_native_analysis_pipeline.py'),'--source',str(paths['source']),
        '--normalization',str(paths['source']/NORMALIZATION),'--budget',str(paths['budget'])]
    if reference_seed is not None:app+=['--reference-seed',str(reference_seed)]
    return ['systemd-run','--expand-environment=no','--quiet','--wait','--pipe','--collect',
        '--unit='+paths['unit_prefix']+phase+'-v1','--uid=rock','--service-type=exec',
        '--property=MemoryMax=16384M','--property=MemorySwapMax=0','--property=CPUQuota=200%',
        '--property=AllowedCPUs=8-9','--property=TasksMax=64','--property=RuntimeMaxSec='+str(timeout+5),
        '--property=TimeoutStopSec=5','--property=KillMode=control-group','--property=OOMPolicy=continue',
        '/usr/bin/python3',str(paths['source']/'tools/mpi_resource_guard.py'),'--output',str(guard),
        '--volume','/data/brick2','--memory-mib','16384','--cpu-percent','200','--file-mib',str(file_mib),
        '--min-free-gib','1280','--timeout',str(timeout),'--','env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',
        'PYTHONDONTWRITEBYTECODE=1','TMPDIR='+str(BASE/'tmp'),'MPLCONFIGDIR='+str(BASE/'mpl'),
        'PYTHONPATH='+str(paths['source']/'tools')+':'+str(paths['source']/'python'),*app]


def collect_code(*,reference_seed=None):
    paths=analysis_paths(reference_seed)
    files={'raw-guard.json':paths['raw_guard'],'science-guard.json':paths['science_guard'],'pending.json':paths['raw']/PENDING,
           'summary.json':paths['raw']/SUMMARY,'report.json':paths['output']/'report.json','stages.jsonl':paths['output']/'stages.jsonl'}
    for phase,report in [('activity','activity.json'),('cell','paper-cell-metrics.json'),('correlation','correlation.json'),('series','time-series.json')]:
        for name in [report,'catalog.json']:files[phase+'/'+name]=paths['output']/phase/name
        files[phase+'.log']=paths['output']/(phase+'.log')
    if reference_seed is not None:
        for phase in ['fc','lags']:
            for name in [phase+'.json','catalog.json']:files[phase+'/'+name]=paths['output']/phase/name
            files[phase+'.log']=paths['output']/(phase+'.log')
    return f'''from pathlib import Path
import json,base64,hashlib
result={{}};total=0
for name,path in { {n:str(p) for n,p in files.items()}!r}.items():
 p=Path(path)
 if not p.exists():continue
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<2*2**20
 raw=p.read_bytes();total+=len(raw);assert total<8*2**20
 result[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
print(json.dumps(result))'''


def collect(destination,*,reference_seed=None):
    for name,row in remote(collect_code(reference_seed=reference_seed)).items():
        raw=base64.b64decode(row['base64'],validate=True)
        check(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'native control copy changed')
        p=destination/name;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():check(p.read_bytes()==raw,'previous terminal control changed')
        else:
            with p.open('xb') as f:f.write(raw)


def guard_ok(path,phase,timeout,*,reference_seed=None):
    paths=analysis_paths(reference_seed)
    g=read(path)
    check(g['admitted'] is True and type(g.get('returncode')) is int and g['returncode']==0 and not g.get('error'),
          'native analysis guard not terminal success')
    check(g['host']==NODE and g['uid']==1000 and g['cgroup']=='/system.slice/'+paths['unit_prefix']+phase+'-v1.service'
          and 0<g['wall_seconds']<=timeout,'native analysis guard identity/time differs')
    for section in ['before','after']:
        facts=g[section];events=dict(line.split() for line in facts['memory.events'].splitlines())
        check(all(events[k]=='0' for k in ['max','oom','oom_kill','oom_group_kill'])
              and facts['memory.max']==str(16*2**30) and facts['memory.swap.max']=='0'
              and facts['pids.max']=='64' and facts['cpuset.cpus.effective']=='8-9','native analysis limits/pressure')
        quota,period=map(int,facts['cpu.max'].split());check(period>0 and quota==2*period,'native analysis CPU quota')
    check(0<int(g['after']['memory.peak'])<=16*2**30,'native analysis memory peak exceeds limit')
    check(g['file_limit_bytes']==(512*2**20 if phase=='raw' else MAX_RAW) and g['data_volume']=='/data/brick2'
          and g['data_device']!=g['root_device'] and g['minimum_free_bytes']==1280*2**30
          and g['minimum_observed_free_bytes']>=1280*2**30,'native analysis storage quota/reserve')
    if reference_seed is not None:
        expected=command(phase,timeout,reference_seed=reference_seed)
        check(g['command']==expected[expected.index('--')+1:],'reference analysis guard command differs')
    return g


def guarded(phase,timeout,destination,*,reference_seed=None):
    cmd=command(phase,timeout,reference_seed=reference_seed);result=None;error=None
    try:
        with (destination/(phase+'-controller.log')).open('x') as log:
            result=subprocess.run(['tsh','ssh','root@'+NODE,shlex.join(cmd)],stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,timeout=timeout+45)
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    (destination/(phase+'-controller.json')).write_text(json.dumps(dict(command=cmd,
        returncode=result.returncode if result else None,error=error,observation_failure_is_not_remote_completion=True),indent=2)+'\n')
    collect(destination,reference_seed=reference_seed)
    check(error is None and result.returncode==0,'native analysis controller failed; no retry')
    return guard_ok(destination/(phase+'-guard.json'),phase,timeout,reference_seed=reference_seed)


def sync_evidence(destination,evidence,*,reference_seed=None):
    paths=analysis_paths(reference_seed)
    target=evidence/paths['evidence_subdir'];target.mkdir(parents=True,exist_ok=True)
    total=0
    for path in destination.rglob('*'):
        if not path.is_file() or path.name.startswith('._') or path.suffix not in ['.json','.jsonl','.log']:continue
        check(path.stat().st_size<2*2**20,'oversized local analysis evidence')
        raw=path.read_bytes();total+=len(raw);check(total<16*2**20,'local evidence budget exceeded')
        p=target/path.relative_to(destination);p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():check(p.read_bytes()==raw,'native analysis evidence changed')
        else:p.write_bytes(raw)


def publication_code(pending_sha,*,reference_seed=None):
    paths=analysis_paths(reference_seed)
    reference_argument='' if reference_seed is None else ',reference_seed='+str(reference_seed)
    code=f'''import resource,sys,json
resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30));resource.setrlimit(resource.RLIMIT_CPU,(30,30))
sys.path.insert(0,{str(paths['source']/'tools')!r})
from pathlib import Path
from mam_native_primary_output_audit import publish
print(json.dumps(publish(Path({str(paths['raw'])!r}),Path({str(paths['raw_guard'])!r}),{pending_sha!r}{reference_argument})))'''
    return f"import os,subprocess\nsubprocess.run([{PYTHON!r},'-c',{code!r}],check=True,timeout=40,env={{**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}})"


def finish(start,destination,evidence,*,reference_seed=None):
    paths=analysis_paths(reference_seed)
    pending_sha=sha(destination/'pending.json')
    publication=remote(publication_code(pending_sha,reference_seed=reference_seed))
    (destination/'publication.json').write_text(json.dumps(publication,indent=2)+'\n');collect(destination,reference_seed=reference_seed)
    check(publication['passed'] is True and publication['summary_sha256']==sha(destination/'summary.json'),'native summary publication differs')
    fresh=time.monotonic();admission2=remote(preflight_code(science=True,reference_seed=reference_seed))
    previous=time.monotonic()-start+RESERVE_SECONDS
    budget=dict(schema='b2-mam-native-analysis-budget-v1',automatic_retry=False,shared_analysis_budget_seconds=TOTAL_SECONDS,
        previous_analysis_seconds=previous,summary_sha256=publication['summary_sha256'],raw_guard_sha256=sha(destination/'raw-guard.json'))
    if reference_seed is not None:
        from mam_launch_native_full_reference import PROTOCOL_SHA
        budget.update(seed=reference_seed,label=analysis_spec(reference_seed)['label'],protocol_sha256=PROTOCOL_SHA)
    limit=math.floor(TOTAL_SECONDS-previous);check(limit>0,'no native scientific analysis budget remains')
    remote(f'''from pathlib import Path
import sys,json
raw=sys.stdin.buffer.read(2**20+1);assert len(raw)<2**20
Path({str(paths['budget'])!r}).open('xb').write(raw)
print(json.dumps(dict(budget_written=True)))''',payload=json.dumps(budget).encode())
    check(time.monotonic()-fresh<60,'science admission stale')
    (destination/'science-budget.json').write_text(json.dumps(budget,indent=2)+'\n')
    (destination/'science-admission.json').write_text(json.dumps(admission2,indent=2)+'\n')
    guarded('science',limit,destination,reference_seed=reference_seed)
    report=read(destination/'report.json')
    check(report['analysis_complete'] is True and report['total_accounted_seconds']<TOTAL_SECONDS
          and time.monotonic()-start<TOTAL_SECONDS,'native science incomplete/shared budget exceeded')
    if reference_seed is not None:
        required=['activity','cell','correlation','series','fc','lags']
        check(report['seed']==reference_seed and report['label']==budget['label']
              and report['protocol_sha256']==PROTOCOL_SHA and report['required_stages']==required
              and set(report['catalogs'])==set(required) and report['interarea_analysis_complete'] is True,
              'complete reference science report differs')
    result=dict(ready=True,analysis_complete=True,raw_summary_sha256=sha(destination/'summary.json'),
        elapsed_seconds=time.monotonic()-start,shared_analysis_seconds=TOTAL_SECONDS,
        scientific_acceptance=False,performance_cost_acceptance=False,output=str(destination))
    if reference_seed is not None:
        result.update(seed=reference_seed,label=budget['label'],protocol_sha256=PROTOCOL_SHA,
                      interarea_analysis_complete=True,required_stages=required)
    (destination/'controller-complete.json').write_text(json.dumps(result,indent=2)+'\n')
    sync_evidence(destination,evidence,reference_seed=reference_seed)
    return result


def run(evidence,t7,normalization,generated_data,prior_preparation_seconds=0,*,reference_seed=None):
    paths=analysis_paths(reference_seed)
    check(0<=prior_preparation_seconds<TOTAL_SECONDS-RESERVE_SECONDS,'invalid prior preparation debit')
    start=time.monotonic()-prior_preparation_seconds
    if reference_seed is not None:
        from mam_launch_native_full_reference import protocol_gate, PROTOCOL_SHA
        check(protocol_gate(evidence,reference_seed) is not None,'reference diagnostic protocol prerequisites missing')
    gate=prerequisites(t7,reference_seed=reference_seed)
    if gate is None:return dict(ready=False,analysis_started=False,reason='Native terminal resources and complete raw collection are required')
    check(Path('/Volumes/T7').is_mount() and t7.resolve().is_relative_to(Path('/Volumes/T7').resolve()),'mounted T7 required')
    check(os.statvfs(t7).f_bavail*os.statvfs(t7).f_frsize>128*2**30,'T7 reserve')
    destination=t7/'artifacts'/paths['output'].name
    check(not destination.exists() and not (evidence/paths['evidence_subdir']/'admission.json').exists(),
          'native analysis already attempted')
    data,catalog=bundle(Path(__file__).resolve().parents[1],gate,normalization,generated_data,reference_seed=reference_seed)
    admission_start=time.monotonic();admission=remote(preflight_code(reference_seed=reference_seed))
    stage=remote(stage_code(len(data),hashlib.sha256(data).hexdigest(),reference_seed=reference_seed),payload=data)
    check(time.monotonic()-admission_start<60,'native analysis staging admission stale')
    destination.mkdir()
    (destination/'source.tar.gz').write_bytes(data)
    admission.update(source=stage,resource_report_sha256=gate['resource_report_sha256'],collection_report_sha256=gate['collection_report_sha256'],
        prior_preparation_seconds=prior_preparation_seconds,
        shared_analysis_seconds=TOTAL_SECONDS,raw_file_limit_mib=512,science_scratch_file_limit_bytes=MAX_RAW,
        automatic_retry=False,attempts=1)
    if reference_seed is not None:
        admission.update(seed=reference_seed,label=analysis_spec(reference_seed)['label'],protocol_sha256=PROTOCOL_SHA,
                         required_stages=['activity','cell','correlation','series','fc','lags'])
    (destination/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
    try:
        raw_guard=guarded('raw',remaining(start,7200),destination,reference_seed=reference_seed)
        return finish(start,destination,evidence,reference_seed=reference_seed)
    except BaseException as exc:
        (destination/'failure.json').write_text(json.dumps(dict(error=str(exc),elapsed_seconds=time.monotonic()-start,automatic_retry=False),indent=2)+'\n')
        sync_evidence(destination,evidence,reference_seed=reference_seed)
        raise


def resume_publication(evidence,t7):
    destination=t7/'artifacts/native-primary-postrun-v1'
    failure_path=destination/'failure.json';failure=read(failure_path)
    check('import publish' in failure['error'], 'not a failed publication boundary')
    controller=read(destination/'raw-controller.json')
    check(controller['returncode']==0 and controller['error'] is None
          and controller['command']==command('raw',7200),'raw controller was not successful')
    guard_ok(destination/'raw-guard.json','raw',7200)
    pending=read(destination/'pending.json')
    check(pending['raw_output_audit_passed'] is True and pending['passed'] is False,
          'complete unpublished raw audit required')
    marker=destination/'resume-publication.json'
    check(not marker.exists() and not (destination/'publication.json').exists()
          and not (destination/'science-budget.json').exists(), 'publication recovery already attempted')
    charged=failure['elapsed_seconds']+max(0.,time.time()-failure_path.stat().st_mtime)
    check(0<charged<TOTAL_SECONDS-RESERVE_SECONDS,'analysis budget exhausted before publication recovery')
    start=time.monotonic()-charged
    admission=remote(preflight_code(science=True))
    with marker.open('x') as stream:
        json.dump(dict(failure_sha256=sha(failure_path),charged_previous_seconds=charged,
            raw_audit_repeated=False,automatic_retry=False,python=PYTHON,admission=admission),stream,indent=2)
    try:
        return finish(start,destination,evidence)
    except BaseException as exc:
        with (destination/'resume-failure.json').open('x') as stream:
            json.dump(dict(error=str(exc),elapsed_seconds=time.monotonic()-start,automatic_retry=False),stream,indent=2)
        sync_evidence(destination,evidence)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['evidence','t7','normalization','generated-data']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prior-preparation-seconds',type=int,default=0)
    p.add_argument('--resume-publication',action='store_true')
    p.add_argument('--reference-seed',type=int,choices=[1730,1731])
    a=p.parse_args()
    check(not (a.resume_publication and a.reference_seed is not None),'reference runs cannot reuse primary publication recovery')
    r=resume_publication(a.evidence,a.t7) if a.resume_publication else run(a.evidence,a.t7,a.normalization,a.generated_data,a.prior_preparation_seconds,reference_seed=a.reference_seed)
    print(json.dumps(r));raise SystemExit(0 if r['ready'] else 2)
