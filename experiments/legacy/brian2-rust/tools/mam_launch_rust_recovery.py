"""One four-node transport validation and one corrected full Rust target."""
import argparse,base64,concurrent.futures,hashlib,json,os,shlex,shutil,time
from pathlib import Path
import mam_launch_rust_performance as old
from mam_rust_cpu_placement import ALTERNATE_CPUS,revision,candidate_topologies
from mam_benchmark_terminal import guard
from mam_primary_resources import time_record
from mpi_teleport_launch import launch

CASE='rust32-target-v2'
LABEL='rust-mam-perf-v2-seed1729-100500ms'
SOURCE='mam-rust-performance-v2-source-20260911'
SMOKE='mam-rust-four-node-chain-v2'
PROTOCOL_SHA='0245bded278000947929c50bf3327979b6e4d9864eff2aa9edf38d75cf2f3803'
FIX_SHA='da528b59686480ea0e62003243ef64a74600b1f622dc9cfb0929594a7c4f4e10'
read,sha,check,write,remote=old.read,old.sha,old.check,old.write,old.remote

def prerequisites(evidence):
    protocol_path=evidence/'rust-recovery-protocol-v2/protocol.json'
    check(sha(protocol_path)==PROTOCOL_SHA,'recovery protocol changed')
    protocol=read(protocol_path)
    check(protocol['base_protocol_sha256']==old.PROTOCOL_SHA,'base protocol changed')
    check(protocol['maximum_corrected_target_launches']==protocol['maximum_transport_tests']==1
          and protocol['automatic_retry'] is False,'finite recovery required')
    failure_path=evidence/'performance-runs-v1/rust32-target/failed-launch-controls-v1/report.json'
    check(sha(failure_path)==protocol['failed_target_report_sha256'],'previous failure changed')
    failure=read(failure_path)
    check(failure['all_own_services_stopped'] is True and failure['output_directory_exists'] is False
          and failure['rank_failure_receipts']==32,'old target not confirmed stopped before output')
    for name,digest in failure['input_sha256'].items():
        check(sha(failure_path.parents[1]/name)==digest,'old failure input changed')
    fix=evidence/'rust-pmi-fd-fix-v1/completion.json';check(sha(fix)==FIX_SHA,'PMI fix proof changed')
    for name,item in read(fix)['files'].items():check(sha(evidence.parent/name)==item['sha256'],'PMI fix source changed')
    nest=old.nest_gate(evidence);check(nest is not None,'accepted NEST baseline required')
    package=evidence/'primary-run/package.json';check(sha(package)==old.PACKAGE_SHA,'frozen package changed')
    return protocol,nest,read(package)

def options(t7,*,smoke=False):
    opts=old.launch_options(t7/'logs'/(CASE+'-smoke' if smoke else CASE),cpu_ids=ALTERNATE_CPUS,source=SOURCE,label=LABEL)
    if smoke:
        app='exec env PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v -o '+shlex.quote(old.BASE+'/'+SMOKE+'/rank')+'"${PMI_RANK}.time" '
        app+=shlex.join(['/usr/bin/python3',old.BASE+'/'+SMOKE+'/wrapper.py',old.BASE+'/'+SMOKE+'/probe'])
        opts.update(application=['sh','-c',app],timeout=180,guard_memory_mib=1024,guard_file_mib=8)
    return opts

def backup(case,t7):
    dest=t7/'artifacts/performance-runs-v1'/CASE
    for p in case.rglob('*'):
        if not p.is_file() or '__pycache__' in str(p):continue
        check(p.stat().st_size<=8*2**20,'bounded recovery controls')
        target=dest/p.relative_to(case);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():check(sha(target)==sha(p),'archived recovery control changed')
        else:shutil.copyfile(p,target)
        check(sha(target)==sha(p),'recovery backup differs')

def stage_all(payload,catalog,source,out):
    def stage(index):
        row=remote(old.NODES[index],old.stage_code(index,payload,catalog,source=source))
        write(out/f'stage-{index}.json',row);return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:return list(pool.map(stage,range(4)))

def prepare(evidence,t7,protocol):
    case=evidence/'performance-runs-v1'/CASE;case.mkdir(exist_ok=False)
    start=time.monotonic();write(case/'intent.json',dict(protocol_sha256=PROTOCOL_SHA,maximum_target_launches=1,automatic_retry=False))
    root=evidence.parent
    files={n:(root/'tools'/n).read_bytes() for n in ['mam_rust_terminal_sync.py','mpi_resource_guard.py','mam_rust_guard23.py']}
    files['protocol.json']=(evidence/'rust-recovery-protocol-v2/protocol.json').read_bytes()
    check(hashlib.sha256(files['mam_rust_terminal_sync.py']).hexdigest()==protocol['wrapper_sha256'],'fixed wrapper identity')
    def pack(values):return ({n:base64.b64encode(v).decode() for n,v in values.items()},
                             {n:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in values.items()})
    payload,catalog=pack(files);source=case/'source';source.mkdir()
    staged=stage_all(payload,catalog,SOURCE,source)
    probe=remote(old.NODES[0],f'''from pathlib import Path
import hashlib,base64,json
p=Path({old.BRICK+'/rust-pmi-fd-fix-v1/probe'!r});assert p.is_file() and p.stat().st_size<65536
raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=={protocol['probe_executable_sha256']!r}
print(json.dumps(dict(base64=base64.b64encode(raw).decode())))''')
    wrapper=f'''import os,sys,json
from pathlib import Path
sys.path.insert(0,{old.BASE+'/'+SOURCE!r})
from mam_rust_terminal_sync import run_child
rank=int(os.environ['PMI_RANK']);size=int(os.environ['PMI_SIZE']);assert size==32 and 0<=rank<32
assert os.uname().nodename=={old.NODES!r}[rank//8]
assert sorted(os.sched_getaffinity(0))=={ALTERNATE_CPUS!r}
cpu={ALTERNATE_CPUS!r}[rank%8];os.sched_setaffinity(0,{{cpu}})
result=run_child(sys.argv[1:]);assert result.returncode==0
row=dict(rank=rank,host=os.uname().nodename,cpu=cpu,pmi_fd=int(os.environ['PMI_FD']),child_returncode=result.returncode)
with (Path({old.BASE+'/'+SMOKE!r})/f'rank{{rank}}.json').open('x') as f:json.dump(row,f)
print(json.dumps(dict(event='fixed_wrapper_child_complete',**row)),flush=True)
'''
    smoke_payload,smoke_catalog=pack({'probe':base64.b64decode(probe['base64']),'wrapper.py':wrapper.encode()})
    smoke_stage=case/'smoke-source';smoke_stage.mkdir()
    smoke_staged=stage_all(smoke_payload,smoke_catalog,SMOKE,smoke_stage)
    # The uploaded probe is an ordinary file until this bounded explicit mode change.
    for node in old.NODES:
        remote(node,f'''import os,json,hashlib
from pathlib import Path
p=Path({old.BASE+'/'+SMOKE+'/probe'!r});assert hashlib.sha256(p.read_bytes()).hexdigest()=={protocol['probe_executable_sha256']!r}
p.chmod(0o700);print(json.dumps(dict(probe_executable=True)))''')
    check(time.monotonic()-start<=300,'source preparation budget exhausted')
    result=dict(protocol_sha256=PROTOCOL_SHA,launcher_sha256=sha(Path(__file__)),source_catalog=catalog,source_staging=staged,
        smoke_catalog=smoke_catalog,smoke_staging=smoke_staged,elapsed_seconds=time.monotonic()-start)
    write(case/'prepared.json',result);backup(case,t7);return result

def preflight(case,evidence,package,prepared,out):
    topology=dict(preflight=[dict(selected_cpu_topology=t) for t in candidate_topologies(evidence)])
    rows,extra,errors,start=old.perform_preflight(out,package,prepared['source_catalog'],topology,
        read(evidence/'performance-protocol-v1/protocol.json'),cpu_ids=ALTERNATE_CPUS,source=SOURCE,label=LABEL)
    check(not errors,'recovery preflight failed; no retries')
    check(time.monotonic()-start<60,'stale recovery preflight')
    return rows,extra,start

def verify_smoke_sources(prepared):
    def verify(node):
        remote(node,f'''from pathlib import Path
import json,hashlib
b=Path({old.BASE+'/'+SMOKE!r})
for name,item in {prepared['smoke_catalog']!r}.items():
 p=b/name;assert not p.is_symlink() and p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
assert not list(b.glob('rank*.json')) and not list(b.glob('rank*.time'))
print(json.dumps(dict(verified=True)))''')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(verify,old.NODES))

def collect_smoke(case,t7,opts,result,prepared):
    check(not result['error'] and set(result['returncodes'])=={'controller','proxy-0','proxy-1','proxy-2','proxy-3'}
          and all(type(v) is int and v==0 for v in result['returncodes'].values())
          and 0<result['wall_seconds']<=180,'smoke launcher failed or exceeded budget')
    prefix=result['resource_guard']['unit_prefix'];out=case/'smoke';hosts=[]
    for index,node in enumerate(old.NODES):
        roles=(['controller'] if index==0 else [])+[f'proxy-{index}']
        row=remote(node,f'''from pathlib import Path
import json,subprocess,hashlib
b=Path({old.BASE!r});s=b/{SMOKE!r}
u=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend',{prefix+'-*'!r}],text=True,timeout=5);assert not u.strip(),u
for name,item in {prepared['smoke_catalog']!r}.items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==item['sha256']
rows=[]
for rank in {list(range(index*8,(index+1)*8))!r}:
 p=s/f'rank{{rank}}.json';assert p.stat().st_size<4096;r=json.loads(p.read_text())
 p=s/f'rank{{rank}}.time';assert p.stat().st_size<8192;r['time_record']=p.read_text();rows.append(r)
guards={{role:json.loads((b/'guards'/({prefix!r}+'-'+role+'.json')).read_text()) for role in {roles!r}}}
print(json.dumps(dict(host={node!r},active_units=[],ranks=rows,guards=guards)))''')
        hosts.append(row);write(out/f'host-{index}.json',row)
    audited=[]
    for index,host in enumerate(hosts):
        check(host['host']==old.NODES[index] and host['active_units']==[]
              and set(host['guards'])==set((['controller'] if index==0 else [])+[f'proxy-{index}']),
              'smoke host or service coverage')
        for role,g in host['guards'].items():
            spec=dict(host=host['host'],role=role,memory_bytes=1024*2**20,pids_max=64,cpu_ids=ALTERNATE_CPUS,
                cpu_quota_cores=8,volume='/data/brick2' if index==0 else '/',allow_root_volume=True,
                file_limit_bytes=8*2**20,minimum_free_bytes=(1280 if index==0 else 128)*2**30,reserved_host_memory_bytes=64*2**30)
            audited.append(guard(g,spec,prefix,180))
            check(g['wall_seconds']<=result['wall_seconds']+.02,'service exceeds launcher interval')
        for r in host['ranks']:
            check(r['host']==host['host'] and r['rank']//8==index and r['cpu']==ALTERNATE_CPUS[r['rank']%8]
                  and r['pmi_fd']>=3 and r['child_returncode']==0,'smoke rank identity')
            timing=time_record(r['time_record'])
            check(timing['elapsed_seconds']<=result['wall_seconds']+.04 and timing['peak_rss_bytes']<=1024*2**20,
                  'smoke rank time or memory exceeded')
    log=opts['output']/'controller.log';check(log.stat().st_size<2**20,'bounded smoke transcript')
    probes=[json.loads(line) for line in log.read_text().splitlines() if line.startswith('{"schema":"b2-mpi-cluster-probe-v1"')]
    check(len(probes)==1 and probes[0]['ok'] is True and probes[0]['ranks']==32 and probes[0]['sum']==496
          and probes[0]['hosts']==[n for n in old.NODES for _ in range(8)],'four-node collective result')
    check(not result['error'] and all(v==0 for v in result['returncodes'].values()),'smoke launcher failed')
    check(sorted(r['rank'] for h in hosts for r in h['ranks'])==list(range(32)),'full rank coverage')
    report=dict(passed=True,protocol_sha256=PROTOCOL_SHA,neural_simulations=0,ranks=32,guards=audited,
        probe=probes[0],controller_log_sha256=sha(log),automatic_retry=False,
        input_sha256={str(p.relative_to(case)):sha(p) for p in [case/'prepared.json',*sorted(out.glob('*.json'))]})
    write(out/'completion.json',report);backup(case,t7);return report

def run(evidence,t7,phase):
    protocol,nest,package=prerequisites(evidence)
    check(Path('/Volumes/T7').is_mount() and t7.resolve().is_relative_to(Path('/Volumes/T7')),'T7 required')
    s=os.statvfs(t7);check(s.f_bavail*s.f_frsize>=512*2**30,'T7 start reserve')
    if phase=='prepare':return prepare(evidence,t7,protocol)
    case=evidence/'performance-runs-v1'/CASE;prepared=read(case/'prepared.json')
    check(prepared['protocol_sha256']==PROTOCOL_SHA and prepared['launcher_sha256']==sha(Path(__file__)),
          'prepared recovery identity')
    source_paths=[evidence.parent/'tools'/n for n in ['mam_rust_terminal_sync.py','mpi_resource_guard.py','mam_rust_guard23.py']]
    source_paths.append(evidence/'rust-recovery-protocol-v2/protocol.json')
    check(prepared['source_catalog']=={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in source_paths},
          'prepared source catalog changed')
    old_sources=read(evidence/'performance-runs-v1/rust32-target/admission.json')['source_catalog']
    for name in ['mpi_resource_guard.py','mam_rust_guard23.py']:
        check(prepared['source_catalog'][name]==old_sources[name],'resource guard changed')
    for key,catalog_key in [('source_staging','source_catalog'),('smoke_staging','smoke_catalog')]:
        check(len(prepared[key])==4,'source staging coverage')
        for node,row in zip(old.NODES,prepared[key],strict=True):
            check(row['host']==node and row['source_catalog']==prepared[catalog_key],'source staging changed')
    smoke=phase=='smoke';opts=options(t7,smoke=smoke)
    check(not opts['output'].exists(),'one launch per recovery phase')
    out=case/('smoke' if smoke else 'target');out.mkdir(exist_ok=False)
    write(out/'intent.json',dict(protocol_sha256=PROTOCOL_SHA,automatic_retry=False,maximum_launches=1))
    try:
        if smoke:verify_smoke_sources(prepared)
        else:
            proof=read(case/'smoke/completion.json');check(proof['passed'] is True and proof['protocol_sha256']==PROTOCOL_SHA,'smoke prerequisite')
            for name,digest in proof['input_sha256'].items():check(sha(case/name)==digest,'smoke evidence changed')
            check(sha(t7/'logs'/(CASE+'-smoke')/'controller.log')==proof['controller_log_sha256'],'smoke transcript changed')
            check(not (case/'admission.json').exists() and not (case/'launch.json').exists(),'target already admitted')
        rows,extra,start=preflight(case,evidence,package,prepared,out)
        admission=dict(schema='b2-mam-rust-performance-admission-v1',case_id=CASE,label=LABEL,admitted=True,
            protocol_sha256=PROTOCOL_SHA,base_protocol_sha256=old.PROTOCOL_SHA,nest_prerequisite=nest,
            model_sha256=old.MODEL,plan_sha256=old.PLAN,executable_sha256=old.EXE_SHA,package_sha256=old.PACKAGE_SHA,
            source_catalog=prepared['source_catalog'],source_staging=prepared['source_staging'],
            cpu_placement_revision=revision(),preflight=rows,other_native_hosts=extra,
            identity=dict(ranks=32,threads=1,seed=1729,duration_ms=100500,dt_ms=.1),
            output=old.BRICK+'/runs/'+LABEL,receipt_directory=old.BASE+'/performance-receipts/'+LABEL,
            resources=dict(nodes=old.NODES,ranks_per_node=8,cpu_ids=ALTERNATE_CPUS,memory_bytes_per_service=opts['guard_memory_mib']*2**20,
                cpu_quota_cores_per_service=8,zero_swap=True,pids_max=64,file_limit_bytes=opts['guard_file_mib']*2**20,
                minimum_free_bytes_by_host={n:(1280 if i==0 else 128)*2**30 for i,n in enumerate(old.NODES)},
                total_final_output_bytes=128*2**30,spool_bytes=96*2**30,wall_seconds=opts['timeout']),
            launch_options={k:str(v) if isinstance(v,Path) else v for k,v in opts.items()},
            launcher_sha256=sha(Path(__file__)),scientific_acceptance=False,performance_cost_acceptance=False)
        if smoke:admission['transport_only']=True;admission['identity']=dict(ranks=32,neural_simulations=0)
        write(out/'admission.json',admission)
        if not smoke:write(case/'admission.json',admission)
        backup(case,t7);check(time.monotonic()-start<60,'preflight expired before recovery launch')
        print(json.dumps(dict(event='recovery_admitted',phase=phase,admission_sha256=sha(out/'admission.json'))),flush=True)
        result=launch(**opts);write(out/'launch.json',result)
        if smoke:return collect_smoke(case,t7,opts,result,prepared)
        write(case/'launch.json',result);backup(case,t7)
        return dict(launcher_terminal=True,scientific_acceptance=False,all_terminal_raw_science_audits_required=True)
    except BaseException as error:
        if (opts['output']/'launch.json').exists():
            write(out/'launch.json',read(opts['output']/'launch.json'))
            if not smoke:write(case/'launch.json',read(opts['output']/'launch.json'))
        write(out/'failure.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False))
        backup(case,t7);raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--t7',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','smoke','target'],required=True)
    args=parser.parse_args();print(json.dumps(run(args.evidence,args.t7,args.phase)))
