"""Collect bounded native terminal controls onto T7; never read raw spike files.

Missing terminal launch evidence returns before network access or output
creation. One attempt, six host reads, no automatic retry. Bulk recording
collection must use the remainder of the same 7200-second collection budget.
"""
import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import zlib

from mam_native_primary_resources import audit, terminal_launch
from mam_launch_native_primary import LABEL, BASE, PROJECT, NODES, PARAMETERS, LAYOUT_SHA
from mam_primary_resources import require

CAP = 32*2**20


def read(path, cap=CAP):
    require(path.is_file() and not path.is_symlink(), 'regular control file required')
    with path.open('rb') as stream:raw=stream.read(cap+1)
    require(len(raw) <= cap, 'control file exceeds size cap')
    return raw


def host_code(index, prefix, source_catalog, *, run_label=LABEL):
    require(isinstance(run_label,str) and re.fullmatch(r'[a-z0-9][a-z0-9-]{0,127}',run_label),
            'invalid native output label')
    code = f'''from pathlib import Path
import os,json,hashlib,subprocess,resource
resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20))
resource.setrlimit(resource.RLIMIT_CPU,(30,30))
b=Path({BASE!r});host={NODES[index]!r};index={index};prefix={prefix!r}
assert os.uname().nodename==host and b.resolve().is_relative_to(Path('/home/rock'))
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend',prefix+'-*'],text=True,timeout=5)
assert not units.strip(),units
sources={{}};total=0
def read(relative,cap):
 global total
 p=b/relative;assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(b.resolve())
 with p.open('rb') as f:
  raw=f.read(cap+1);assert len(raw)<=cap,relative
  os.posix_fadvise(f.fileno(),0,len(raw),os.POSIX_FADV_DONTNEED)
 total+=len(raw);assert total<=32*2**20
 sources[relative]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 return raw
for n,expected in {source_catalog!r}.items():
 raw=read({LABEL!r}+'/'+n,2**20)
 assert sources[{LABEL!r}+'/'+n]==expected,n
raw=read({LABEL!r}+'/parameters.json',8*2**20)
assert hashlib.sha256(raw).hexdigest()=={PARAMETERS!r}
roles=(['controller'] if index==0 else [])+['proxy-'+str(index)]
guards={{r:json.loads(read('guards/'+prefix+'-'+r+'.json',2**20)) for r in roles}}
assert all(type(g.get('returncode')) is int and g['returncode']==0 and 'after' in g and not g.get('error') for g in guards.values())
reports={{str(r):json.loads(read('runs/'+{run_label!r}+'/rank'+str(r)+'.json',4*2**20)) for r in range(index*8,(index+1)*8)}}
affinity={{str(r):json.loads(read('affinity/'+{run_label!r}+'/rank'+str(r)+'.json',16384)) for r in range(index*8,(index+1)*8)}}
result=json.dumps(dict(host=host,guards=guards,rank_reports=reports,affinity=affinity,source_files=sources,active_own_units=[],source_bytes=total))
assert len(result.encode())<=32*2**20
import gzip,sys
sys.stdout.buffer.write(gzip.compress(result.encode(),compresslevel=1,mtime=0))
sys.stdout.buffer.flush()'''
    ast.parse(code)
    return code


def collect_host(index, prefix, catalog, directory, *, run_label=LABEL):
    # Stream stdout to a bounded local file instead of buffering all six SSH
    # responses. The remote stdlib-only reader has a 512 MiB address-space cap.
    output = directory/('host-'+str(index)+'.json.gz')
    error = directory/('host-'+str(index)+'.stderr')
    with output.open('xb') as out, error.open('xb') as err:
        result = subprocess.run(['tsh','ssh','rock@'+NODES[index],shlex.join([
            'taskset','-c','8,9','python3','-c',host_code(index,prefix,catalog,run_label=run_label)])],
            stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=45)
    require(result.returncode == 0, 'native control collection failed on '+NODES[index])
    require(output.stat().st_size <= CAP+1, 'native response exceeds control cap')
    decoded = decode_control(read(output,CAP+1))
    published = directory/('host-'+str(index)+'.json')
    with published.open('xb') as stream:stream.write(decoded)
    return published


def decode_control(raw):
    require(len(raw)<=CAP, 'compressed native response exceeds cap')
    decoder=zlib.decompressobj(16+zlib.MAX_WBITS)
    decoded=decoder.decompress(raw,CAP+1)
    require(len(decoded)<=CAP and decoder.eof and not decoder.unused_data
            and not decoder.unconsumed_tail, 'incomplete, trailing or oversized native response')
    json.loads(decoded)
    return decoded


def recovery_cost(directory, guard_path):
    """One investigated recovery; preserve failure and charge diagnosis time too."""
    require((directory is None)==(guard_path is None), 'both prior attempt and guard required')
    if directory is None:return None
    require(not (directory/'recovery.json').exists()
            and not (directory/'collection-controller.json').exists(), 'only one failed original attempt may be recovered')
    failure_raw=read(directory/'failure.json');guard_raw=read(guard_path)
    failure=json.loads(failure_raw);guard=json.loads(guard_raw);command=guard['command']
    require(guard['returncode']!=0 and guard['stop_reason'] is None
            and 0<guard['wall_seconds']<=180 and failure['automatic_retry'] is False
            and 0<failure['elapsed_seconds']<=600
            and Path(command[0]).name=='mam_collect_native_primary_resources.py'
            and '--prior-attempt' not in command and command.count('--output')==1
            and command[command.index('--output')+1]==str(directory), 'invalid original failed control attempt')
    diagnosis=max(0.,time.time()-guard_path.stat().st_mtime)
    charged=max(failure['elapsed_seconds'],guard['wall_seconds'])+diagnosis
    require(charged<7200-270, 'insufficient shared collection budget for recovery')
    return dict(prior_attempt=str(directory),prior_guard=str(guard_path),
        failure_sha256=hashlib.sha256(failure_raw).hexdigest(),
        guard_sha256=hashlib.sha256(guard_raw).hexdigest(),
        failed_attempt_seconds=max(failure['elapsed_seconds'],guard['wall_seconds']),
        diagnosis_seconds=diagnosis,charged_prior_seconds=charged,
        recovery_attempts=1,automatic_retry=False)


def run(evidence, output, parameter_path, prior_attempt=None, prior_guard=None,
        offline_from=None, offline_guard=None, *, reference_seed=None):
    started = time.monotonic()
    identity_options = {}
    run_label = LABEL
    run_directory = evidence/'primary-native-layout'
    reference_protocol_sha = None
    if reference_seed is not None:
        from mam_launch_native_full_reference import CAMPAIGN, PROTOCOL_SHA, label_for, protocol_gate
        run_label = label_for(reference_seed)
        require(all(value is None for value in [prior_attempt, prior_guard, offline_from, offline_guard]),
                'diagnostic references cannot reuse historical primary recovery')
        if protocol_gate(evidence, reference_seed) is None:
            return dict(ready=False, collection_started=False,
                        reason='Previous diagnostic reference has not completed its required audits')
        identity_options = dict(label=run_label, seed=reference_seed)
        run_directory = evidence/CAMPAIGN/f'seed{reference_seed}'
        reference_protocol_sha = PROTOCOL_SHA
    recovery=recovery_cost(prior_attempt,prior_guard)
    require((offline_from is None)==(offline_guard is None), 'both offline controls and guard required')
    if offline_from is not None:
        require(recovery is not None, 'offline audit requires failed-attempt budget provenance')
        old_guard=json.loads(read(offline_guard));old_failure=json.loads(read(offline_from/'failure.json'))
        old_recovery=json.loads(read(offline_from/'recovery.json'));old_command=old_guard['command']
        require(old_guard['returncode']!=0 and old_guard['stop_reason'] is None
                and 0<old_guard['wall_seconds']<=180 and old_failure['error']=='rank RSS snapshot differs'
                and old_recovery['prior_attempt']==str(prior_attempt)
                and old_recovery['prior_guard']==str(prior_guard)
                and Path(old_command[0]).name=='mam_collect_native_primary_resources.py'
                and '--offline-from' not in old_command and old_command.count('--output')==1
                and old_command[old_command.index('--output')+1]==str(offline_from),
                'offline controls are not the complete transport / failed RSS audit')
        recovery['offline_from']=str(offline_from)
        recovery['offline_guard']=str(offline_guard)
        recovery['offline_guard_sha256']=hashlib.sha256(read(offline_guard)).hexdigest()
        recovery['offline_failure_sha256']=hashlib.sha256(read(offline_from/'failure.json')).hexdigest()
    launch_path = run_directory/'launch.json'
    admission_path = run_directory/'admission.json'
    if not launch_path.exists() or not admission_path.exists():
        return dict(ready=False, collection_started=False,
                    reason='Native terminal launch and admission for the requested run are required')
    admission_raw, launch_raw = read(admission_path,2**20), read(launch_path,2**20)
    admission, launch = json.loads(admission_raw), json.loads(launch_raw)
    if reference_seed is not None:
        require(admission['seed'] == reference_seed and admission['campaign'] == CAMPAIGN
                and admission['protocol_sha256'] == reference_protocol_sha
                and admission['source_project'] == PROJECT
                and admission['source_project_is_retained_primary'] is True,
                'diagnostic reference admission or source project differs')
    prefix = terminal_launch(admission, launch, **identity_options)
    parameter_raw = read(parameter_path,8*2**20)
    layout_raw = read(evidence/'primary-native-layout/layout.json',2**20)
    require(hashlib.sha256(parameter_raw).hexdigest() == PARAMETERS
            and hashlib.sha256(layout_raw).hexdigest() == LAYOUT_SHA, 'pinned parameter/layout bytes changed')
    staged_raw = read(evidence/'primary-native-layout/stage.json',2**20)
    staged = json.loads(staged_raw)
    require([s['host'] for s in staged] == NODES, 'staging coverage differs')
    require(all(set(s['source_catalog']) == {'mam_nest_reference.py','mam_nest_rank_affinity.py',
            'mpi_resource_guard.py','layout.json','planned-budget.json'}
            and s['source_catalog'] == staged[0]['source_catalog']
            and s['parameters_sha256'] == PARAMETERS
            and s['source_catalog']['layout.json']['sha256'] == LAYOUT_SHA
            for s in staged), 'staged native identity differs')
    volume = Path('/Volumes/T7')
    require(volume.is_mount() and output.resolve().is_relative_to(volume.resolve()), 'native collection must use T7')
    require(os.statvfs(output.parent).f_bavail*os.statvfs(output.parent).f_frsize >= 128*2**30, 'T7 reserve')
    output.mkdir(parents=True,exist_ok=False)
    if recovery is not None:
        (output/'recovery.json').write_text(json.dumps(recovery,indent=2)+'\n')
    # Retain original admission bytes and any partial/failed collection. These
    # hash-pinned source controls allow a later auditor to reproduce the inputs.
    for name,raw in [('admission.json',admission_raw),('launch.json',launch_raw),
                     ('parameters.json',parameter_raw),('layout.json',layout_raw),('stage.json',staged_raw)]:
        if recovery is not None:
            require(read(prior_attempt/name,8*2**20)==raw, 'recovery changed original control identity')
        if offline_from is not None:
            require(read(offline_from/name,8*2**20)==raw, 'offline controls changed original identity')
        (output/name).write_bytes(raw)
    try:
        if offline_from is not None:
            paths=[]
            for i in range(6):
                name='host-'+str(i)+'.json'
                packed=read(offline_from/(name+'.gz'),CAP)
                decoded=decode_control(packed)
                require(decoded==read(offline_from/name,CAP), 'offline compressed/raw control mismatch')
                (output/(name+'.gz')).write_bytes(packed)
                (output/name).write_bytes(decoded);paths.append(output/name)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                futures=[pool.submit(collect_host,i,prefix,staged[i]['source_catalog'],output,
                                     run_label=run_label) for i in range(6)]
                paths=[];failures=[]
                for index,f in enumerate(futures):
                    try:paths.append(f.result())
                    except Exception as exc:failures.append(dict(host=NODES[index],error=str(exc)))
                require(not failures, 'Native control collection failures: '+json.dumps(failures))
        collected=[json.loads(read(p,CAP+1)) for p in paths]
        report=audit(admission,launch,collected,json.loads(parameter_raw),json.loads(layout_raw),
                     **identity_options)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        hashes={p.name:hashlib.sha256(read(p,CAP+1)).hexdigest() for p in paths}
        for name in ['admission.json','launch.json','parameters.json','layout.json','stage.json','report.json']:
            hashes[name]=hashlib.sha256(read(output/name)).hexdigest()
        (output/'input-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
        stage_elapsed=time.monotonic()-started
        require(stage_elapsed < 600, 'native control collection exceeded 600-second stage budget')
        elapsed=stage_elapsed+(recovery['charged_prior_seconds'] if recovery else 0.)
        require(elapsed<7200-90, 'native shared collection budget exhausted')
        result=dict(ready=True,collection_started=True,terminal_resource_audit_passed=True,
            raw_output_audit_passed=False,scientific_acceptance=False,elapsed_seconds=elapsed,
            stage_elapsed_seconds=stage_elapsed,
            recovery=recovery,
            shared_collection_budget_seconds=7200,remaining_bulk_collection_seconds=7200-elapsed,
            automatic_retry=False,accounting=report['accounting'])
        if reference_seed is not None:
            result.update(label=run_label, seed=reference_seed,
                          protocol_sha256=reference_protocol_sha, source_project=PROJECT)
        (output/'collection-controller.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps(dict(error=str(exc),
            elapsed_seconds=time.monotonic()-started,automatic_retry=False))+'\n')
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parameters',type=Path,required=True)
    parser.add_argument('--prior-attempt',type=Path)
    parser.add_argument('--prior-guard',type=Path)
    parser.add_argument('--offline-from',type=Path)
    parser.add_argument('--offline-guard',type=Path)
    parser.add_argument('--reference-seed',type=int,choices=[1730,1731],
                        help='Explicit frozen full diagnostic identity; default remains primary seed1729')
    args=parser.parse_args()
    result=run(args.evidence,args.output,args.parameters,args.prior_attempt,args.prior_guard,
               args.offline_from,args.offline_guard,reference_seed=args.reference_seed)
    print(json.dumps(result))
    raise SystemExit(0 if result['ready'] else 2)
