"""One-shot bounded collection of terminal controls; no raw-event transfer.

Admission must be frozen by the future finite experiment launcher. This command
does not admit or launch an experiment. Failed collections retain every file.
"""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from mam_benchmark_terminal import audit, require, parse_report
from mam_nest_benchmark import WORKLOAD_SHA256

CONTROL_CAP=64*2**20


def pinned(path, expected_sha, cap):
    require(path.is_file() and not path.is_symlink(), 'regular pinned input required')
    with path.open('rb') as f:raw=f.read(cap+1)
    require(len(raw)<=cap and hashlib.sha256(raw).hexdigest()==expected_sha, 'input digest or size differs')
    return json.loads(raw)


def launch_gate(admission, launch):
    require(admission['schema']=='b2-mam-benchmark-admission-v1' and admission['admitted'] is True
            and admission['workload_sha256']==WORKLOAD_SHA256, 'frozen benchmark admission required')
    hosts=admission['nodes'];roles={'controller'}|{'proxy-'+str(i) for i in range(len(hosts))}
    require(1<=len(hosts)<=16 and len(set(hosts))==len(hosts), 'bounded unique host set required')
    require(launch['schema']=='b2-teleport-hydra-launch-v0' and not launch.get('error')
            and launch['nodes']==hosts and launch['ranks_per_node']==admission['ranks_per_node']
            and set(launch['returncodes'])==roles
            and all(type(x) is int and x==0 for x in launch['returncodes'].values()), 'launcher not terminal success')
    prefix=launch['resource_guard']['unit_prefix']
    require(re.fullmatch('b2mpi-[0-9a-f]{12}',prefix) is not None, 'invalid guard prefix')
    require(len(admission['host_paths'])==len(hosts), 'host path coverage')
    return prefix


def host_code(admission, index, prefix, *, host_timeout_seconds=45):
    require(type(host_timeout_seconds) is int and 45<=host_timeout_seconds<=120,
            'host transfer deadline must be 45 to 120 seconds')
    host=admission['nodes'][index];paths=admission['host_paths'][index]
    roles=(['controller'] if index==0 else [])+['proxy-'+str(index)]
    ranks=list(range(index*admission['ranks_per_node'],(index+1)*admission['ranks_per_node']))
    require(len(ranks)<=32 and len(admission['nodes'])*len(ranks)<=256, 'bounded rank controls required')
    source_catalog=admission['source_catalog']
    require(all(Path(n).name==n for n in source_catalog), 'source names must be basenames')
    code=f'''from pathlib import Path
import gzip,hashlib,json,os,resource,signal,subprocess,sys
signal.alarm({host_timeout_seconds-5})
resource.setrlimit(resource.RLIMIT_AS,(768*2**20,768*2**20))
resource.setrlimit(resource.RLIMIT_CPU,(30,30))
host={host!r};paths={paths!r};prefix={prefix!r};roles={roles!r};ranks={ranks!r}
assert os.uname().nodename==host
base=Path(paths['base']);assert base.is_absolute() and base.resolve()==base
allowed=Path('/data/brick2') if host.endswith('-23') else Path('/home/rock')
assert base.is_relative_to(allowed)
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend',prefix+'-*'],text=True,timeout=5)
assert not units.strip(),units
total=0;source_files={{}}
def read(path,cap):
 global total
 assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(base)
 with path.open('rb') as f:
  raw=f.read(cap+1);assert len(raw)<=cap
  os.posix_fadvise(f.fileno(),0,len(raw),os.POSIX_FADV_DONTNEED)
 total+=len(raw);assert total<={CONTROL_CAP}
 source_files[str(path)]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 return raw
project=base/paths['project'];run=base/paths['run'];affinity=base/paths['affinity'];guard_dir=base/paths['guards']
expected={source_catalog!r};actual={{}}
for name,item in expected.items():
 raw=read(project/name,2**20);actual[name]=source_files[str(project/name)];assert actual[name]==item
assert hashlib.sha256(read(project/'parameters.json',8*2**20)).hexdigest()=={admission['parameters_sha256']!r}
runtime_sha=hashlib.sha256(read(base/paths['runtime_catalog'],8*2**20)).hexdigest()
assert runtime_sha=={admission['runtime_catalog_sha256']!r}
failures=[p.name for p in run.glob('rank*.failed.json')];assert not failures
reports={{}}
for rank in ranks:
 report_raw=read(run/f'rank{{rank}}.json',8*2**20).decode()
 done_raw=read(run/f'rank{{rank}}.done.json',16384).decode()
 bind=json.loads(read(affinity/f'rank{{rank}}.json',16384))
 assert bind['rank']==rank and bind['host']==host and bind['ranks']=={admission['identity']['ranks']!r}
 event=run/f'rank{{rank}}.events.bin'
 assert event.is_file() and not event.is_symlink() and event.resolve().is_relative_to(base)
 reports[str(rank)]=dict(report_json=report_raw,done_json=done_raw,
  initial_rank_cpu_ids=bind['initial_rank_cpu_ids'],observed_event_file_bytes=event.stat().st_size)
guards={{role:json.loads(read(guard_dir/(prefix+'-'+role+'.json'),2*2**20)) for role in roles}}
result=dict(host=host,active_own_units=[],failure_files=failures,source_catalog=actual,
 runtime_catalog_sha256=runtime_sha,guards=guards,ranks=reports,source_files=source_files,
 source_bytes=total,raw_events_read=False)
raw=json.dumps(result).encode();assert len(raw)<={CONTROL_CAP}
sys.stdout.buffer.write(gzip.compress(raw,compresslevel=1,mtime=0))
'''
    ast.parse(code)
    return code


def run(args):
    started=time.monotonic()
    admission=pinned(args.admission,args.admission_sha256,8*2**20)
    launch=pinned(args.launch,args.launch_sha256,16*2**20)
    parameters=pinned(args.parameters,admission['parameters_sha256'],8*2**20)
    prefix=launch_gate(admission,launch)  # No network or output before terminal validation.
    require(0<args.wall_seconds<=900, 'finite collection wall budget required')
    host_timeout=getattr(args,'host_timeout_seconds',45)
    require(type(host_timeout) is int and 45<=host_timeout<=120,
            'host transfer deadline must be 45 to 120 seconds')
    args.output.mkdir(exist_ok=False)
    collected=[];receipts=[]
    try:
        for index,host in enumerate(admission['nodes']):
            remaining=args.wall_seconds-(time.monotonic()-started)
            require(remaining>host_timeout,'insufficient budget for one bounded host read')
            code=host_code(admission,index,prefix,host_timeout_seconds=host_timeout)
            out=args.output/f'host-{index}.json.gz';err=args.output/f'host-{index}.stderr'
            with out.open('xb') as stdout,err.open('xb') as stderr:
                process=subprocess.run(['tsh','ssh','rock@'+host,shlex.join(['taskset','-c','8,9','python3','-c',code])],
                    stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,timeout=host_timeout)
            require(process.returncode==0,'host collection failed; retained, no retry')
            require(out.stat().st_size<=CONTROL_CAP,'compressed controls exceed cap')
            with gzip.open(out,'rb') as stream:raw=stream.read(CONTROL_CAP+1)
            require(len(raw)<=CONTROL_CAP,'expanded controls exceed cap')
            collected.append(json.loads(raw))
            receipts.append(dict(host=host,file=out.name,bytes=out.stat().st_size,
                                 sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
        result=audit(admission,launch,collected,parameters,parameters_sha256=admission['parameters_sha256'])
        elapsed=time.monotonic()-started;require(elapsed<=args.wall_seconds,'collection/audit wall exceeded')
        result.update(collection_seconds=elapsed,collection_files=receipts,
                      host_timeout_seconds=host_timeout,
                      input_sha256=dict(admission=args.admission_sha256,launch=args.launch_sha256,
                                        parameters=admission['parameters_sha256']),
                      implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      raw_events_collected=False,automatic_retry=False)
        with (args.output/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps(dict(terminal_resource_audit_passed=True,raw_output_audit_passed=False,collection_seconds=elapsed)))
        return result
    except BaseException as error:
        with (args.output/'failure.json').open('x') as f:
            json.dump(dict(error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started,
                           completed_hosts=len(collected),automatic_retry=False),f,indent=2)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['admission','launch','parameters','output']:parser.add_argument('--'+name,type=Path,required=True)
    for name in ['admission-sha256','launch-sha256']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--wall-seconds',type=int,required=True)
    parser.add_argument('--host-timeout-seconds',type=int,default=45)
    run(parser.parse_args())
