"""One bounded, read-only collection of the new Rust target's terminal controls."""
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

from mam_collect_benchmark_terminal import pinned
from mam_rust_benchmark_terminal import audit,require,target_contract
from mam_launch_rust_performance import BASE,BRICK,SOURCE,OLD,LABEL,CASE,NODES,PACKAGE_SHA
from mam_launch_performance_tuning import write,PROTOCOL_SHA

CAP=64*2**20


def launch_gate(admission,launch):
    contract=target_contract(admission)
    require(admission['schema']=='b2-mam-rust-performance-admission-v1' and admission['admitted'] is True
            and admission['protocol_sha256']==contract['protocol_sha256'] and admission['case_id']==contract['case_id']
            and admission['label']==contract['label'],'Rust target admission required')
    require(launch['schema']=='b2-teleport-hydra-launch-v0' and not launch.get('error')
            and launch['nodes']==NODES and launch['ranks_per_node']==8
            and set(launch['returncodes'])=={'controller','proxy-0','proxy-1','proxy-2','proxy-3'}
            and all(type(v) is int and v==0 for v in launch['returncodes'].values()),'Rust launcher not successful terminal')
    prefix=launch['resource_guard']['unit_prefix']
    require(re.fullmatch('b2mpi-[0-9a-f]{12}',prefix) is not None,'service prefix')
    return prefix


def host_code(admission,package,index,prefix):
    contract=target_contract(admission);source=contract['source'];label=contract['label']
    ranks=list(range(index*8,(index+1)*8));roles=(['controller'] if index==0 else [])+['proxy-'+str(index)]
    code=f'''from pathlib import Path
import os,json,gzip,hashlib,resource,signal,subprocess,sys
signal.alarm(40);resource.setrlimit(resource.RLIMIT_AS,(768*2**20,768*2**20));resource.setrlimit(resource.RLIMIT_CPU,(30,30))
assert os.uname().nodename=={NODES[index]!r}
b=Path({BASE!r});root=b.resolve();leader={index==0!r};prefix={prefix!r}
assert root==Path({BRICK!r}) if leader else root.is_relative_to(Path('/home/rock'))
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend',prefix+'-*'],text=True,timeout=5);assert not units.strip(),units
files={{}};total=0
def read(path,cap):
 global total
 assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
 with path.open('rb') as f:
  raw=f.read(cap+1);assert len(raw)<=cap
  os.posix_fadvise(f.fileno(),0,len(raw),os.POSIX_FADV_DONTNEED)
 total+=len(raw);assert total<={CAP}
 files[str(path)]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 return raw
def digest(path):
 assert path.is_file() and path.stat().st_size<=512*2**20
 with path.open('rb') as f:
  result=hashlib.file_digest(f,'sha256').hexdigest()
  os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 return result
actual={{}}
for name,item in {admission['source_catalog']!r}.items():
 path=b/{source!r}/name;read(path,65536);actual[name]=files[str(path)];assert actual[name]==item
project=b/{OLD!r}
for name,item in {package['catalog']!r}.items():
 path=project/name;assert not path.is_symlink() and path.resolve().is_relative_to(project.resolve())
 assert path.stat().st_size==item['bytes'] and digest(path)==item['sha256'],name
for name,expected in {package['mpi_runtime']!r}.items():
 path=b/'mpi'/name;assert path.resolve().is_relative_to((b/'mpi').resolve()) and digest(path)==expected
receipts=b/'performance-receipts'/{label!r};rank_ids={ranks!r}
failures=[p.name for p in receipts.glob('rank*.failed.json')];assert not failures
assert {{p.name for p in receipts.glob('rank*.started.json')}}=={{f'rank{{r}}.started.json' for r in rank_ids}}
assert {{p.name for p in receipts.glob('rank*.done.json')}}=={{f'rank{{r}}.done.json' for r in rank_ids}}
rows={{}}
for rank in rank_ids:
 rows[str(rank)]=dict(started_json=read(receipts/f'rank{{rank}}.started.json',65536).decode(),
   done_json=read(receipts/f'rank{{rank}}.done.json',65536).decode(),
   time_record=read(b/'metrics'/({label!r}+f'-rank{{rank}}.time'),65536).decode())
guards={{role:json.loads(read(b/'guards'/(prefix+'-'+role+'.json'),2**20)) for role in {roles!r}}}
output=None
if leader:
 run=b/'runs'/{label!r};assert not (run/'spike-spool').exists()
 sizes={{}}
 for name in ['results.bin','events.bin']:
  p=run/name;assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root);sizes[name]=p.stat().st_size
 output=dict(summary_json=read(run/'summary.json',32*2**20).decode(),runtime_json=read(run/'mpi-runtime.json',32*2**20).decode(),binary_file_bytes=sizes,spool_present=False)
result=dict(host=os.uname().nodename,active_own_units=[],failure_files=failures,source_catalog=actual,
 artifact_catalog_verified=True,mpi_runtime_verified=True,guards=guards,ranks=rows,leader_outputs=output,
 source_files=files,source_bytes=total,raw_binary_payloads_read=False)
raw=json.dumps(result,allow_nan=False).encode();assert len(raw)<={CAP}
sys.stdout.buffer.write(gzip.compress(raw,compresslevel=1,mtime=0))
'''
    ast.parse(code)
    return code


def run(args):
    started=time.monotonic()
    admission=pinned(args.admission,args.admission_sha256,8*2**20)
    launch=pinned(args.launch,args.launch_sha256,16*2**20)
    prefix=launch_gate(admission,launch)
    package=pinned(args.package,PACKAGE_SHA,2*2**20)
    require(0<args.wall_seconds<=900,'finite control collection budget required')
    args.output.mkdir(exist_ok=False)
    collected=[];receipts=[]
    try:
        for index,host in enumerate(NODES):
            require(args.wall_seconds-(time.monotonic()-started)>45,'collection budget exhausted before host read')
            code=host_code(admission,package,index,prefix)
            output=args.output/f'host-{index}.json.gz';stderr=args.output/f'host-{index}.stderr'
            with output.open('xb') as out,stderr.open('xb') as err:
                process=subprocess.run(['tsh','ssh','rock@'+host,shlex.join(['taskset','-c','8,9','python3','-c',code])],
                    stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=45)
            require(process.returncode==0,'Rust terminal host read failed; retained, no retry')
            require(output.stat().st_size<=CAP,'compressed terminal controls too large')
            with gzip.open(output,'rb') as stream:raw=stream.read(CAP+1)
            require(len(raw)<=CAP,'expanded terminal controls too large')
            collected.append(json.loads(raw))
            receipts.append(dict(host=host,file=output.name,bytes=output.stat().st_size,
                sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
        result=audit(admission,launch,collected)
        elapsed=time.monotonic()-started;require(elapsed<=args.wall_seconds,'terminal collection wall exceeded')
        result.update(collection_seconds=elapsed,collection_files=receipts,
            input_sha256=dict(admission=args.admission_sha256,launch=args.launch_sha256,package=PACKAGE_SHA),
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            verifier_sha256=hashlib.sha256(Path(__file__).with_name('mam_rust_benchmark_terminal.py').read_bytes()).hexdigest(),
            raw_binary_payloads_collected=False,automatic_retry=False)
        write(args.output/'report.json',result)
        print(json.dumps(dict(terminal_resource_audit_passed=True,metadata_sync_audit_passed=True,
                             raw_output_audit_passed=False,collection_seconds=elapsed)))
        return result
    except BaseException as error:
        write(args.output/'failure.json',dict(error_type=type(error).__name__,error=str(error),
            elapsed_seconds=time.monotonic()-started,completed_hosts=len(collected),automatic_retry=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['admission','launch','package','output']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['admission-sha256','launch-sha256']:p.add_argument('--'+name,required=True)
    p.add_argument('--wall-seconds',type=int,required=True)
    run(p.parse_args())
