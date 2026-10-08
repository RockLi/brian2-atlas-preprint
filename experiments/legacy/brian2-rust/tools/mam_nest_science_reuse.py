"""Reuse descriptive NEST metrics only after exact full-event and artifact proof.

One read-only remote rehash, no simulation, scientific equivalence, or new seed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

BASE=Path('/data/brick2/brian2-mpi-region-20260907')
SOURCE=BASE/'native-primary-postrun-v1-source'
NODE='hk-prod-model-ae02-23'
NEW_COMPLETION='15f6ffc2bae7ec859d7b1b503b40c06cebde02d33397ff1011de6081ed8bd641'
OLD_PIPELINE='ebc2d8bc6501839ee75c6ee702c38cd961c07727a810ed9bf5b21a608db0ac70'
OLD_RAW='0264da967ef2960fec090f37f5046d21cb58641c10b0b6780f9372d599ede97e'
INTERAREA_COMPLETE='87529af8df49ab9c65ea2710c2285a8f777197184052b52ad34224cbf3d15546'
MAX_REHASH_BYTES=512*2**20


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<=8*2**20,'bounded regular JSON required')
    return json.loads(path.read_text())


def match_events(raw,activity,summary,identity,parameters):
    need(raw['raw_output_audit_passed'] is True and raw['duration_ms']==100500,'new full raw audit required')
    need(summary['passed'] and summary['raw_output_audit_passed'] and summary['audit_guard_passed'],'old raw audit required')
    need(activity['parameters_sha256']==summary['parameters_sha256']==parameters,'parameter identity')
    need(all(activity['simulation'][k]==identity[k] for k in ['seed','ranks','threads','duration_ms','dt_ms']),
         'simulation identity')
    previous={}
    for name,item in activity['result_files'].items():
        match=re.search(r'/rank(\d+)\.events\.bin$',name)
        need(match is not None,'old event path')
        rank=int(match[1]);need(rank not in previous,'duplicate old rank');previous[rank]=(name,item)
    rows=[r for host in raw['hosts'] for r in host['ranks']]
    need([r['rank'] for r in rows]==list(range(48)) and set(previous)==set(range(48)),'full rank coverage')
    result=[]
    for row in rows:
        name,old=previous[row['rank']]
        need(row['raw_event_stream_verified'] is True and row['event_bytes']==old['bytes']
             and row['event_sha256']==old['sha256'],'whole-event checksum identity')
        need(len(row['physical_50ms_bin_counts'])==2010,'whole physical window')
        result.append(dict(rank=row['rank'],bytes=row['event_bytes'],sha256=row['event_sha256'],old_path=name))
    need(sum(r['bytes'] for r in result)==summary['event_bytes']==52299346728,'whole byte count')
    histogram=[sum(r['physical_50ms_bin_counts'][i] for r in rows) for i in range(2010)]
    need(histogram==summary['physical_50ms_bin_counts'] and
         sum(r['terminal_tick_events'] for r in rows)==summary['terminal_tick_events'],'physical event counts')
    return result


def add_pin(manifest,path,digest,size=None):
    path=Path(path)
    need(path.is_absolute() and path.is_relative_to(BASE) and '..' not in path.parts,'pinned data-volume path')
    need(re.fullmatch('[0-9a-f]{64}',digest) is not None,'SHA-256 required')
    need(size is None or type(size) is int and 0<=size<=128*2**20,'bounded artifact size')
    key=str(path);row=dict(sha256=digest,bytes=size)
    if key in manifest:
        old=manifest[key];need(old['sha256']==digest and
            (old['bytes'] is None or size is None or old['bytes']==size),'conflicting artifact pin')
        if size is not None:old['bytes']=size
    else:manifest[key]=row


def prepare(evidence,art):
    from mam_launch_native_analysis import guard_ok
    case=evidence/'performance-runs-v1/nest-selected-target'
    need(sha(case/'completion.json')==NEW_COMPLETION,'new completion changed')
    completion=read(case/'completion.json')
    for name,digest in completion['input_sha256'].items():need(sha(case/name)==digest,'new accepted input changed')
    native=art/'native-primary-postrun-v1';inter=art/'mam-primary-interarea-v1'
    need(sha(native/'report.json')==OLD_PIPELINE and sha(native/'summary.json')==OLD_RAW,'old analysis pins changed')
    need(sha(inter/'complete.json')==INTERAREA_COMPLETE,'old interarea completion changed')
    pipeline=read(native/'report.json');summary=read(native/'summary.json');activity=read(native/'activity/activity.json')
    admission=read(case/'admission.json');raw=read(case/'raw/report.json')
    events=match_events(raw,activity,summary,admission['identity'],admission['parameters_sha256'])
    need(pipeline['analysis_complete'] and pipeline['total_accounted_seconds']<10800
         and pipeline['raw_summary_sha256']==OLD_RAW,'old analysis completion')
    local=evidence/'primary-native-postrun/run'
    for phase in ['raw','science']:guard_ok(local/(phase+'-guard.json'),phase,7200 if phase=='raw' else 10800)
    need(sha(local/'report.json')==OLD_PIPELINE and sha(local/'raw-guard.json')==pipeline['raw_guard_sha256'],
         'old guard/report binding')
    controller=read(local/'science-controller.json');science_guard=read(local/'science-guard.json')
    need(controller['returncode']==0 and controller['error'] is None,'old science controller')
    cmd=controller['command'];need(science_guard['command']==cmd[cmd.index('--')+1:],'old science command binding')
    ig=read(inter/'guard.json');ic=read(inter/'controller.json');icomp=read(inter/'complete.json')
    need(icomp['completed'] and sha(inter/'guard.json')==icomp['guard_sha256']
         and ic['returncode']==0 and ic['error'] is None and ig['admitted'] and ig['returncode']==0
         and not ig.get('error') and ig['host']==NODE and ig['uid']==1000
         and ig['cgroup']=='/system.slice/b2mpi-primary-interarea-v1.service','old interarea guard')
    need(ig['command']==ic['command'][ic['command'].index('--')+1:],'interarea command binding')
    for section in ['before','after']:
        facts=ig[section];need(facts['memory.max']==str(4*2**30) and facts['memory.swap.max']=='0'
          and facts['pids.max']=='64' and facts['cpuset.cpus.effective']=='8-9','interarea resource limits')
        need(all(int(line.split()[1])==0 for line in facts['memory.events'].splitlines()),'interarea memory pressure')
        quota,period=map(int,facts['cpu.max'].split());need(period>0 and quota==2*period,'interarea CPU limit')
    need(0<ig['wall_seconds']<360 and 0<int(ig['after']['memory.peak'])<4*2**30
       and ig['data_volume']=='/data/brick2' and ig['data_device']!=ig['root_device']
       and ig['minimum_observed_free_bytes']>=1280*2**30 and ig['file_limit_bytes']==64*2**20,'interarea budget')
    manifest={};reports={};catalogs={};output_pins={}
    phases=[('activity',native/'activity','activity.json',BASE/'native-primary-postrun-v1/activity'),
      ('cell',native/'cell','paper-cell-metrics.json',BASE/'native-primary-postrun-v1/cell'),
      ('correlation',native/'correlation','correlation.json',BASE/'native-primary-postrun-v1/correlation'),
      ('series',native/'series','time-series.json',BASE/'native-primary-postrun-v1/series'),
      ('fc',inter/'native-fc','fc.json',BASE/'mam-primary-interarea-v1/native-fc'),
      ('lags',inter/'native-lags','lags.json',BASE/'mam-primary-interarea-v1/native-lags')]
    for phase,folder,name,remote in phases:
        catalog=read(folder/'catalog.json');catalogs[phase]=sha(folder/'catalog.json')
        if phase in pipeline['catalogs']:need(catalogs[phase]==pipeline['catalogs'][phase],'old analysis catalog binding')
        add_pin(manifest,remote/'catalog.json',catalogs[phase],(folder/'catalog.json').stat().st_size)
        for filename,row in catalog.items():
            need(Path(filename).name==filename,'analysis catalog basename')
            add_pin(manifest,remote/filename,row['sha256'],row['bytes'])
            output_pins[str(remote/filename)]=row['sha256']
            if (folder/filename).exists():need(sha(folder/filename)==row['sha256'],'local analysis copy changed')
        need(name in catalog,'missing metric report');r=read(folder/name);reports[phase]=r
        need(r.get('scientific_equivalence') is False,'descriptive result scope')
        identity=r['simulation'] if phase=='activity' else r['identity']
        need(all(identity[k]==admission['identity'][k] for k in ['seed','ranks','threads','duration_ms','dt_ms']),
             'metric simulation identity')
        if phase!='activity':need(identity['simulator']=='NEST','metric simulator identity')
        for filename,digest in r['implementation_sha256'].items():
            need(Path(filename).name==filename,'implementation basename')
            parent=SOURCE/'python/brian2_rust' if filename in ['results.py','multi_area_analysis.py'] else SOURCE/'tools'
            add_pin(manifest,parent/filename,digest)
    old_event_pins={r['old_path']:r['sha256'] for r in events}
    for phase in ['cell','correlation','series']:
        sources=reports[phase]['source_sha256']
        need({p:d for p,d in sources.items() if p.endswith('.events.bin')}==old_event_pins,'complete metric raw-input identity')
        for path,digest in sources.items():
            if path.endswith('.events.bin'):continue
            if path in output_pins:need(output_pins[path]==digest,'metric baseline binding')
            add_pin(manifest,path,digest)
    series=reports['series'];fc=reports['fc'];lags=reports['lags']
    series_report=sha(native/'series/time-series.json');series_arrays=output_pins[str(BASE/'native-primary-postrun-v1/series/time-series.npz')]
    for r in [fc,lags]:
        need(r['source_sha256']['series_report']==series_report and r['source_sha256']['series_arrays']==series_arrays
             and r['observation_seconds']==100,'interarea series binding')
    need(lags['source_sha256']['fc_report']==sha(inter/'native-fc/fc.json'),'lag FC binding')
    need(activity['window']['start_tick']==5000 and activity['window']['end_tick']==1005000
         and activity['window']['spike_tick_offset']==0 and activity['window']['endpoint']=='[start,end)', 'physical rate convention')
    pins=read(inter/'pins.json')
    for path,row in pins.items():
        if '/primary-postrun-v1/' in path:continue  # Rust inputs are not reused here.
        add_pin(manifest,path,row['sha256'],row['bytes'])
    ref=BASE/'mam-primary-interarea-v1-source'
    bindings=[(fc,'matrices',ref/'matrices/matrices.npz'),(fc,'matrices_metadata',ref/'matrices/matrices.json'),
      (fc,'reference_report',ref/'fc-reference/report.json'),(fc,'reference_arrays',ref/'fc-reference/reconstructed.npz'),
      (lags,'reference_audit',ref/'lag-reference/report.json'),(lags,'matrices_metadata',ref/'matrices/matrices.json')]
    for r,key,path in bindings:need(manifest[str(path)]['sha256']==r['source_sha256'][key],'interarea reference binding')
    add_pin(manifest,SOURCE/'catalog.json',pipeline['source_catalog_sha256'])
    add_pin(manifest,BASE/'native-primary-postrun-v1/report.json',OLD_PIPELINE)
    for phase in ['raw','science']:
        add_pin(manifest,BASE/'guards'/f'native-primary-{phase}-audit-v1.json' if phase=='raw'
            else BASE/'guards/native-primary-science-v1.json',sha(local/(phase+'-guard.json')))
    add_pin(manifest,BASE/'guards/mam-primary-interarea-v1.json',sha(inter/'guard.json'))
    need(len(manifest)<=128,'bounded manifest')
    return dict(manifest=manifest,event_identity=events,metric_catalogs=catalogs,
        input_sha256=dict(new_completion=NEW_COMPLETION,new_raw=sha(case/'raw/report.json'),old_pipeline=OLD_PIPELINE,
          old_raw=OLD_RAW,old_interarea_complete=INTERAREA_COMPLETE,science_guard=sha(local/'science-guard.json')),
        mean_rate_hz=activity['mean_rate_hz'])


def remote_code(manifest):
    return f'''import os,json,signal,resource,hashlib,time,subprocess
from pathlib import Path
signal.alarm(55);resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20));resource.setrlimit(resource.RLIMIT_CPU,(30,30))
assert os.uname().nodename=={NODE!r} and sorted(os.sched_getaffinity(0))==[8,9]
assert not subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5).strip()
manifest={manifest!r};start=time.monotonic();total=0;result={{}}
for name,item in manifest.items():
 p=Path(name);assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(Path({str(BASE)!r}))
 with p.open('rb') as f:
  before=os.fstat(f.fileno());assert before.st_size<=128*2**20
  if item['bytes'] is not None:assert before.st_size==item['bytes'],name
  total+=before.st_size;assert total<={MAX_REHASH_BYTES}
  h=hashlib.sha256()
  while block:=f.read(2**20):h.update(block)
  after=os.fstat(f.fileno());assert (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
  os.posix_fadvise(f.fileno(),0,before.st_size,os.POSIX_FADV_DONTNEED)
 assert h.hexdigest()==item['sha256'],name
 result[name]=dict(bytes=before.st_size,sha256=h.hexdigest())
print(json.dumps(dict(host=os.uname().nodename,files=result,total_bytes=total,wall_seconds=time.monotonic()-start,
 peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
 address_space_limit_bytes=256*2**20,cpu_limit_seconds=30,wall_limit_seconds=55,active_own_units=[],neural_simulations=0)))'''


def run(evidence,t7):
    started=time.monotonic();plan=prepare(evidence,t7/'artifacts')
    out=evidence/'performance-runs-v1/nest-selected-target/science-reuse-v1';out.mkdir(exist_ok=False)
    def write(name,value):
        with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    write('intent.json',dict(**plan,maximum_remote_rehash_bytes=MAX_REHASH_BYTES,maximum_remote_seconds=55,
        maximum_stage_seconds=180,automatic_retry=False,implementation_sha256=sha(Path(__file__))))
    try:
        command=['tsh','ssh','rock@'+NODE,shlex.join(['taskset','-c','8,9','python3','-c',remote_code(plan['manifest'])])]
        with (out/'remote.json').open('xb') as stdout,(out/'remote.stderr').open('xb') as stderr:
            r=subprocess.run(command,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,timeout=60)
        need(r.returncode==0,'remote artifact rehash failed; retained without retry')
        remote=read(out/'remote.json');need(set(remote['files'])==set(plan['manifest']),'remote file coverage')
        for name,item in plan['manifest'].items():
            row=remote['files'][name];need(row['sha256']==item['sha256'] and
                (item['bytes'] is None or row['bytes']==item['bytes']),'remote artifact identity')
        need(remote['host']==NODE and remote['total_bytes']<=MAX_REHASH_BYTES and remote['wall_seconds']<=55
             and remote['cpu_seconds']<=30 and remote['peak_rss_kib']*1024<=256*2**20,'rehash resource budget')
        need(time.monotonic()-started<180,'reuse stage budget')
        write('report.json',dict(schema='b2-mam-nest-benchmark-science-reuse-v1',descriptive_metrics_reuse_accepted=True,
            six_metrics=['firing_rates','LvR','pairwise_correlation','PSD','FC','interarea_lags'],
            input_sha256=plan['input_sha256'],metric_catalogs=plan['metric_catalogs'],
            intent_sha256=sha(out/'intent.json'),remote_rehash_sha256=sha(out/'remote.json'),
            event_files_exact=48,event_bytes_exact=52299346728,mean_rate_hz=plan['mean_rate_hz'],
            wall_seconds=time.monotonic()-started,additional_independent_seed_samples=0,
            scientific_equivalence=False,paper_reproduction_accepted=False,performance_cost_acceptance=False,
            independent_full_raw_backup=False,scope='Existing descriptive metrics reused by exact complete event, parameter, artifact and analysis provenance identity; no recomputation or new independent sample.'))
        destination=t7/'artifacts/performance-runs-v1/nest-selected-target/science-reuse-v1';destination.mkdir()
        for p in out.iterdir():
            with (destination/p.name).open('xb') as f:f.write(p.read_bytes())
            need(sha(p)==sha(destination/p.name),'archive checksum')
        print(json.dumps(dict(descriptive_metrics_reuse_accepted=True,report_sha256=sha(out/'report.json'),t7_verified=True)))
    except BaseException as error:
        write('failure.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--t7',type=Path,required=True)
    args=parser.parse_args();run(args.evidence,args.t7)
