"""E1-small fresh-process cold measurement; independent revision, not yet run.

Only coordinator --allow-run may launch workers. Old evidence/helpers are read-only.
Current imports are stdlib-only so control-flow tests never import model packages.
"""
import argparse
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
VIEWS=[('atlas','cpu','atlas',False),('sj-layerwise','cpu','spikingjelly_frontier',False),
       ('snn-layerwise','cpu','snntorch_fp64',False),('spyx','jax','spyx',False),
       ('brainstate','jax','brainx_state',False),('sj-compile','cpu','spikingjelly_frontier',True),
       ('snn-compile','cpu','snntorch_fp64',True)]
SEEDS=(11,23,37,51,71)
LOCK=ROOT/'evidence/.e1-small-cold-r1.lock'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    content=json.dumps(value,indent=2,allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(content)


def progress(path,value):
    temporary=Path(path).with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


@contextmanager
def exclusive_lock(path,token,output):
    """Persistent inode + flock; never unlink a held/stale lock inode."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another E1-small cold coordinator holds the exclusive lock') from None
        stream.seek(0);stream.truncate()
        json.dump(dict(pid=os.getpid(),token=token,output=str(output),
                       started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),stream)
        stream.flush();os.fsync(stream.fileno())
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def check_worker_lease(token):
    record=json.loads(LOCK.read_text())
    if not token or record.get('token')!=token or record.get('pid')!=os.getppid():
        raise RuntimeError('Worker requires the active direct-parent coordinator lease')
    with LOCK.open('r+') as stream:
        try:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        raise RuntimeError('Coordinator lease is not actively locked')


def declared_order():
    return [dict(seed=seed,view=view[0],environment=view[1],engine=view[2],compiled=view[3])
            for i,seed in enumerate(SEEDS) for view in VIEWS[i:]+VIEWS[:i]]


def new_ledger():
    return [dict(**entry,status='not_launched',resource_qualification=False,
                 strict_ranking_eligible=False,
                 resource_reason='JAX host-config diagnostic; no verified one-core result'
                    if entry['environment']=='jax' else 'resource inheritance pending')
            for entry in declared_order()]


def abort_pending(records,reason,active=None):
    """Keep all 35 slots even when a coordinator exception prevents launch."""
    for i,row in enumerate(records):
        if row['status'] in ('not_launched','launching'):
            row.update(status='coordinator_failed_during_slot' if i==active else 'not_launched_coordinator_failure',
                       reason=reason)
    return records


def concurrent_phases():
    """Local coverage supplements frozen helpers without changing them.

    Existing queue scripts do not honor this new lock; this process scan is an
    additional barrier, not an atomic shared lock with every historical queue.
    """
    import psutil
    own=psutil.Process()
    excluded={own.pid,*[p.pid for p in own.parents()]}
    conflicts=[];uncertain=[]
    for process in psutil.process_iter(['pid','cmdline','uids']):
        if process.pid in excluded:
            continue
        try:
            uids=process.info.get('uids')
            if uids is not None and uids.real!=os.getuid():
                continue
            argv=process.info.get('cmdline')
            if argv is None:
                uncertain.append(process.pid);continue
            for arg in argv:
                path=Path(arg)
                if str(ROOT) in arg and (
                    path.name=='b2-train' or
                    (path.suffix=='.py' and path.name.startswith(('run_','qualify_','benchmark_','measure_','generate_','e5_oracle')))):
                    conflicts.append(dict(pid=process.pid,argv=argv));break
        except psutil.NoSuchProcess:
            continue
        except psutil.Error as error:
            uncertain.append(dict(pid=process.pid,error=type(error).__name__))
    if uncertain:
        raise RuntimeError('Same-user phase visibility is incomplete: '+json.dumps(uncertain))
    return conflicts


def require_idle():
    from run_recurrent_queue import assert_no_other_phase
    from run_a1_queue_v4 import a1_conflicts
    assert_no_other_phase()
    found=a1_conflicts()+concurrent_phases()
    if found:
        raise RuntimeError('Another evaluation phase is active: '+json.dumps(found))


def normalize(name):
    return re.sub(r'[-_.]+','-',name).lower()


def locked_versions(path):
    result={}
    for raw in Path(path).read_text().splitlines():
        line=raw.strip()
        if not line or line.startswith('#'):
            continue
        if '==' not in line or line.count('==')!=1:
            raise ValueError('Unsupported qualified lock line: '+line)
        name,version=line.split('==')
        key=normalize(name)
        if key in result:
            raise ValueError('Duplicate lock package '+key)
        result[key]=version
    return result


# Executed only in a formally allowed phase, not by module import or tests.
METADATA_PROBE=r'''
import hashlib,importlib.metadata as metadata,json,platform,re,sys
from pathlib import Path
norm=lambda n:re.sub(r"[-_.]+","-",n).lower()
distributions={};duplicates=[];files={}
for dist in metadata.distributions():
    name=dist.metadata.get("Name")
    if not name:raise RuntimeError("Distribution has no Name metadata")
    key=norm(name)
    if key in distributions:duplicates.append(key)
    distributions[key]=dist.version
    for entry in dist.files or []:
        if str(entry).endswith((".dist-info/METADATA",".dist-info/RECORD")):
            p=Path(dist.locate_file(entry)).resolve()
            files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
exe=Path(sys.executable).resolve()
files[str(exe)]=hashlib.sha256(exe.read_bytes()).hexdigest()
cfg=Path(sys.prefix)/"pyvenv.cfg"
if cfg.exists():files[str(cfg.resolve())]=hashlib.sha256(cfg.read_bytes()).hexdigest()
print(json.dumps(dict(python=sys.version,implementation=platform.python_implementation(),
 executable=sys.executable,resolved_executable=str(exe),prefix=sys.prefix,
 distributions=distributions,duplicates=duplicates,identity_files=files),sort_keys=True))
'''


def installed_preflight(oldfreeze,rows,output):
    expected_python=json.loads((ROOT/'environment/hardware.json').read_text())['python']
    raw_root=ROOT/'evidence/remote-benchmark-v1'
    raw_hashes={};raw_versions={'cpu':{},'jax':{}}
    for entry in declared_order():
        path=raw_root/f"{entry['view']}-seed-{entry['seed']}"/'result.json'
        raw=json.loads(path.read_text())
        if raw['status']!='completed':
            raise RuntimeError('Inherited raw worker did not complete: '+str(path))
        raw_hashes[str(path.relative_to(ROOT))]=digest(path)
        impl=raw.get('implementation',{})
        keys=() if entry['engine']=='atlas' else ('jax','optax') if entry['environment']=='jax' else ('torch',)
        for key in keys:
            if key not in impl:
                raise RuntimeError('Missing historical runtime version '+key)
            old=raw_versions[entry['environment']].setdefault(key,impl[key])
            if old!=impl[key]:
                raise RuntimeError('Historical runtime versions disagree across qualified slots')
    current={}
    for environment in ('cpu','jax'):
        interpreter=ROOT/f'environment/{environment}/bin/python'
        # -I strips user Python path/startup settings. It imports stdlib metadata
        # readers only; no torch/jax/brainstate/spyx model import occurs here.
        check=subprocess.run([str(interpreter),'-I','-c',METADATA_PROBE],
            cwd=ROOT,text=True,capture_output=True,check=False,timeout=60)
        (output/f'{environment}-metadata.stdout.txt').write_text(check.stdout)
        (output/f'{environment}-metadata.stderr.txt').write_text(check.stderr)
        if check.returncode:
            raise RuntimeError(f'{environment} installed-metadata probe failed: {check.returncode}')
        data=json.loads(check.stdout)
        if data['duplicates']:
            raise RuntimeError('Duplicate installed distributions: '+repr(data['duplicates']))
        expected=locked_versions(ROOT/f'environment/{environment}-lock.txt')
        if data['distributions']!=expected:
            missing={k:v for k,v in expected.items() if data['distributions'].get(k)!=v}
            extra={k:v for k,v in data['distributions'].items() if k not in expected}
            raise RuntimeError(f'{environment} installed versions differ from qualified lock: '+json.dumps(dict(missing_or_changed=missing,extra=extra)))
        if data['python']!=expected_python or data['implementation']!='CPython':
            raise RuntimeError(f'{environment} interpreter differs from historical hardware Python profile')
        for name,version in raw_versions[environment].items():
            if data['distributions'].get(normalize(name))!=version:
                raise RuntimeError(f'{environment}/{name} differs from actual historical qualified worker')
        data['historical_version_binding']=raw_versions[environment]
        data['historical_python_binding']='Full sys.version from old-freeze-bound environment/hardware.json; raw workers did not each record interpreter bytes'
        current[environment]=data
    report=dict(status='matched_historical_versions',environments=current,
        qualified_raw_sha256=raw_hashes,
        limitations=['Historical qualification did not record per-environment interpreter binary hashes or installed METADATA/RECORD hashes. This binds current versions to historical locks/raw versions and shared hardware Python profile, not historical byte-for-byte installed files.',
                     'Current interpreter and METADATA/RECORD files are frozen and rechecked before each slot; package code payload files beyond metadata are not individually hashed.',
                     'Metadata preflight occurs before the cold clock; OS filesystem/page caches are not claimed cold.'])
    save(output/'installed-preflight.json',report)
    return report


def installed_unchanged(preflight):
    for environment,data in preflight['environments'].items():
        for path,expected in data['identity_files'].items():
            if digest(path)!=expected:
                raise RuntimeError(f'{environment} installed identity changed before slot: {path}')


def cache_environment(folder):
    env=os.environ.copy()
    removed={}
    # These variables can silently redirect graph caches to shared/remote stores.
    for key in list(env):
        if ((key.startswith(('TORCH','TRITON','JAX','XLA')) and 'CACHE' in key)
                or key in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP')):
            removed[key]=env.pop(key)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        JAX_PLATFORM_NAME='cpu',JAX_ENABLE_X64='true',
        MPLCONFIGDIR=str(folder/'cache/matplotlib'),
        TORCHINDUCTOR_CACHE_DIR=str(folder/'cache/torch'),
        TRITON_CACHE_DIR=str(folder/'cache/triton'),
        JAX_COMPILATION_CACHE_DIR=str(folder/'cache/jax'),
        TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE='0',TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE='0',
        TORCHINDUCTOR_FX_GRAPH_CACHE='1',TMPDIR=str(folder/'tmp'))
    Path(env['TMPDIR']).mkdir()
    inventory={key:dict(path=env[key],exists=Path(env[key]).exists()) for key in
        ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','JAX_COMPILATION_CACHE_DIR')}
    if any(row['exists'] for row in inventory.values()):
        raise RuntimeError('Declared cold cache directory already exists')
    return env,dict(directories=inventory,cleared_environment_keys=sorted(removed),
        remote_cache_policy='Inherited Torch/Trition/JAX/XLA cache environment settings cleared; known Torch graph/autotune remote caches disabled',
        scope='Three declared persistent cache directories plus a fresh TMPDIR; fresh process in-memory caches. No claim that OS/page caches, compiler binaries, or every library-level cache are cold.')


def worker(a):
    check_worker_lease(a.phase_token)
    sys.path.insert(0,str(ROOT/'adapters'))
    case_path=ROOT/f'fixtures/e1-small/seed-{a.seed}.json'
    case=json.loads(case_path.read_text())
    if digest(case_path)!=a.array_sha:
        raise ValueError('Common array identity mismatch')
    if a.engine=='atlas':
        from atlas_adapter import AtlasAdapter
        adapter=AtlasAdapter(case,ROOT/'snapshot/brian2-rust',ROOT/'runtime/b2-train')
        value=adapter.call('train')
        if value['status']!='executed':
            save(a.output/'cold-result.json',dict(status=value['status'],failure=value));return 1
        loss=float(value['result']['loss'])
        finished=time.monotonic_ns()
        implementation=dict(profile='frozen Atlas full public execute including preflight',
                            returned_step=value['result']['state']['step'])
    else:
        from benchmark_competitor import torch_setup,jax_setup
        if a.engine in ('spyx','brainx_state'):
            step,reset,implementation=jax_setup(case,a.engine)
        else:
            step,reset,implementation=torch_setup(case,a.engine,a.compiled,True)
        if hasattr(step,'performance_mode'):
            step.performance_mode()
        loss=float(step())
        finished=time.monotonic_ns()
    checked=math.isfinite(loss) and abs(loss-a.expected_loss)<=1e-10+1e-8*abs(a.expected_loss)
    environment='jax' if a.engine in ('spyx','brainx_state') else 'cpu'
    inherited=json.loads((a.output.parent/'installed-preflight.json').read_text())['environments'][environment]
    version_match=sys.version==inherited['python'] and all(
        implementation.get(name)==version for name,version in inherited['historical_version_binding'].items()
        if a.engine!='atlas')
    status=('runtime_version_mismatch' if not version_match else 'numerical_divergence'
            if not math.isfinite(loss) else 'completed' if checked else 'first_loss_mismatch')
    save(a.output/'cold-result.json',dict(status=status,
        seed=a.seed,engine=a.engine,compiled=a.compiled,loss=loss if math.isfinite(loss) else None,
        loss_finite=math.isfinite(loss),nonfinite_loss_repr=repr(loss) if not math.isfinite(loss) else None,
        reference_loss=a.expected_loss,
        array_sha256=a.array_sha,launcher_start_ns=a.started_ns,first_update_complete_ns=finished,
        process_to_first_update_s=(finished-a.started_ns)/1e9,
        clock='same-host time.monotonic_ns across coordinator and fresh worker processes',
        numerical_gate='Matching frozen full-shape qualification and installed-version preflight inherited; this run additionally compares first loss only',
        current_python=sys.version,runtime_version_match=version_match,
        historical_runtime_versions=inherited['historical_version_binding'],
        resource_qualification=False if a.engine in ('spyx','brainx_state') else bool(a.resource_qualified),
        strict_ranking_eligible=False if a.engine in ('spyx','brainx_state') else bool(a.resource_qualified),
        resource_note='JAX host-config diagnostic, not verified one-core' if a.engine in ('spyx','brainx_state') else 'Inherited requested-thread resource gate; no affinity claim',
        implementation=implementation))
    return 0 if checked and version_match else 1


def cleanup_after_exception(output,records,active):
    """Best-effort cleanup of currently observable owned children; fail closed.

    The external phase supervisor remains responsible for an entire owned tree
    if the coordinator itself is killed. Unknown visibility is never success.
    """
    import psutil
    tracked={}
    errors=[]
    try:
        for process in psutil.Process().children(recursive=True):
            tracked[(process.pid,process.create_time())]=process
    except psutil.Error as error:
        errors.append(repr(error))
    if active is not None:
        row=records[active]
        terminal=output/f"{row['view']}-seed-{row['seed']}-terminal.json"
        if terminal.exists():
            raw=json.loads(terminal.read_text())
            row['partial_supervisor']=raw
            for identity in raw.get('process_identities',[]):
                try:
                    process=psutil.Process(identity['pid'])
                    if process.create_time()==identity['create_time']:
                        tracked[(process.pid,process.create_time())]=process
                except psutil.NoSuchProcess:
                    pass
                except psutil.Error as error:
                    errors.append(repr(error))
    for (pid,created),process in tracked.items():
        try:
            if process.is_running() and process.create_time()==created:
                process.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as error:
            errors.append(repr(error))
    psutil.wait_procs(list(tracked.values()),timeout=3)
    remaining=[]
    for (pid,created),process in tracked.items():
        try:
            if process.is_running() and process.create_time()==created and process.status()!=psutil.STATUS_ZOMBIE:
                remaining.append(dict(pid=pid,create_time=created))
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as error:
            errors.append(repr(error))
    return dict(observed_owned_tree_clean=not remaining and not errors,
                remaining=remaining,inspection_errors=errors,
                scope='Observable children plus identities from current slot terminal; not proof about unobserved reparented descendants')


def run_coordinator(a,records):
    from run_recurrent_queue import launch
    require_idle()
    evidence=ROOT/'evidence/e1-small-validation-full-r1.json'
    verified=json.loads(evidence.read_text())
    if verified['status']!='passed_full_evidence_checks' or not verified['raw_verified']:
        raise RuntimeError('Prior full evidence validation is required')
    rows={(r['view'],r['seed']):r for r in verified['rows']}
    expected={(x['view'],x['seed']) for x in declared_order()}
    if set(rows)!=expected or len(verified['rows'])!=35 or any(
        not r['numerical_qualification'] or r['effective_status']!='completed' for r in rows.values()):
        raise RuntimeError('All exact 35 prior numerical qualifications must be available')
    for row in records:
        reference=rows[row['view'],row['seed']]
        qualified=bool(reference.get('resource_qualification')) and row['environment']!='jax'
        row.update(resource_qualification=qualified,strict_ranking_eligible=qualified,
                   resource_reason='JAX host-config diagnostic; no verified one-core result'
                      if row['environment']=='jax' else reference.get('resource_qualification_reason','Inherited requested-thread gate'),
                   inherited_numerical_qualification=True,array_sha256=reference['array_sha256'])
    oldfreeze_path=ROOT/'evidence/remote-benchmark-v1/freeze.json'
    if digest(oldfreeze_path)!=verified['freeze_sha256']:
        raise RuntimeError('Verified benchmark freeze changed')
    oldfreeze=json.loads(oldfreeze_path.read_text())
    inherited={**oldfreeze['scripts'],
               **{'environment/'+name:h for name,h in oldfreeze['locks'].items()},
               'runtime/b2-train':oldfreeze['runtime_sha256'],
               'environment/hardware.json':oldfreeze['hardware'],
               'sources/snapshot-manifest.json':oldfreeze['source_manifest']}
    if any(digest(ROOT/rel)!=h for rel,h in inherited.items()):
        raise RuntimeError('Inherited qualified implementation changed')
    snapshot=json.loads((ROOT/'sources/snapshot-manifest.json').read_text())
    if any(digest(ROOT/'snapshot'/rel)!=h for rel,h in snapshot['source_hashes'].items()):
        raise RuntimeError('Qualified source snapshot changed')
    preflight=installed_preflight(oldfreeze,rows,a.output)
    files=['tools/measure_e1_small_cold_r1.py','tools/benchmark_e1.py','tools/benchmark_competitor.py',
           'tools/run_recurrent_queue.py','tools/generate_dense_large.py','tools/generate_recurrent_cases.py',
           'tools/run_a1_queue_v4.py','tools/run_qualification_followup_queue.py',
           'adapters/atlas_adapter.py','adapters/torch_adapter.py','adapters/jax_adapter.py','adapters/oracle.py',
           'runtime/b2-train','environment/cpu-lock.txt','environment/jax-lock.txt',
           'fixtures/e1-small/manifest.json','evidence/e1-small-validation-full-r1.json']
    hashes={**inherited,**{'snapshot/'+rel:h for rel,h in snapshot['source_hashes'].items()},
            **{rel:digest(ROOT/rel) for rel in files},**preflight['qualified_raw_sha256']}
    save(a.output/'freeze.json',dict(schema='E1-small-cold-r1',order=declared_order(),identities=hashes,
        original_benchmark_freeze=oldfreeze,preregistered_metric='fresh process launch through complete first update',
        revision_reason='Earlier cold fields used different boundaries; retained unchanged',
        per_slot_cap_s=360,per_view_cap_s=1800,
        resource_note='Each JAX row remains resource_qualification=false and strict_ranking_eligible=false',
        cache_policy='Fresh TorchInductor/Trition/JAX directories, fresh TMPDIR and process memory; not OS/page cache cold',
        workload='same E1-small B16 T128 128-128-10 FP64 arrays, one actual Adam update',
        status='execution_frozen',qualified_reference_sha256=digest(evidence),
        installed_preflight_sha256=digest(a.output/'installed-preflight.json')))
    spent={view[0]:0. for view in VIEWS}
    a.spent=spent
    for index,row in enumerate(records):
        a.active=index
        require_idle()
        if any(digest(ROOT/rel)!=h for rel,h in hashes.items()):
            raise RuntimeError('Frozen source changed')
        installed_unchanged(preflight)
        name,seed=row['view'],row['seed']
        reference=rows[name,seed]
        job=f'{name}-seed-{seed}'
        folder=a.output/job
        folder.mkdir()
        cap=min(360.,1800.-spent[name])
        if cap<=0:
            row['status']='aggregate_timeout_not_launched'
            progress(a.output/'progress.json',records)
            continue
        env,inventory=cache_environment(folder)
        save(folder/'initial-cache-inventory.json',inventory)
        command=[str(ROOT/f"environment/{row['environment']}/bin/python"),str(Path(__file__).resolve()),
            '--worker','--output',str(folder),'--engine',row['engine'],'--seed',str(seed),
            '--expected-loss',str(reference['raw_loss_trajectory'][0]),'--array-sha',reference['array_sha256'],
            '--phase-token',a.phase_token]
        if row['compiled']:
            command.append('--compiled')
        if row['resource_qualification']:
            command.append('--resource-qualified')
        row['status']='launching'
        progress(a.output/'progress.json',records)
        started=time.monotonic_ns()
        command+=['--started-ns',str(started)]
        launched=launch(job,command,cap,64*1024**3,a.output,folder,env,'cold_first_update')
        spent[name]+=launched['elapsed_s']
        row['supervisor']=launched
        path=folder/'cold-result.json'
        child=json.loads(path.read_text()) if path.exists() else None
        if child is not None:
            row.update(worker=child,worker_sha256=digest(path))
        if launched.get('remaining_owned_processes'):
            raise RuntimeError('Owned cleanup was not verified')
        if launched['termination_reason']!='exited':
            row['status']=launched['termination_reason']
        elif launched['exit_code']!=0:
            row['status']=child.get('status','worker_failed') if child else 'worker_failed_without_result'
        elif child is None:
            row['status']='worker_missing_result'
        else:
            row['status']=child['status']
        progress(a.output/'progress.json',records)
        a.active=None
    return spent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--allow-run',action='store_true')
    p.add_argument('--engine');p.add_argument('--seed',type=int);p.add_argument('--compiled',action='store_true')
    p.add_argument('--started-ns',type=int);p.add_argument('--expected-loss',type=float);p.add_argument('--array-sha')
    p.add_argument('--phase-token');p.add_argument('--resource-qualified',action='store_true')
    a=p.parse_args()
    if socket.gethostname()!='rock-mac-studio-1.local':
        raise RuntimeError('Remote host only')
    sys.path.insert(0,str(ROOT/'tools'))
    a.output=a.output.resolve()
    if a.worker:
        return worker(a)
    if not a.allow_run:
        raise RuntimeError('Prepared only; explicit coordinated --allow-run required')
    a.output.mkdir(parents=True,exist_ok=False)
    records=new_ledger()
    a.phase_token=uuid.uuid4().hex;a.active=None;a.spent={}
    terminal=dict(supervisor_completed=False,finite_slots=35,records=records,
        status='coordinator_failed',note='Independent validation required before any ranking; JAX resource gate false',
        owned_lock=str(LOCK),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    progress(a.output/'progress.json',records)
    try:
        with exclusive_lock(LOCK,a.phase_token,a.output):
            try:
                terminal['spent_seconds']=run_coordinator(a,records)
                terminal['supervisor_completed']=True
                terminal['status']='completed' if all(r['status']=='completed' for r in records) else 'incomplete'
            except BaseException as error:
                terminal.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
                abort_pending(records,str(error),a.active)
                try:
                    terminal['exception_cleanup']=cleanup_after_exception(a.output,records,a.active)
                except BaseException as cleanup_error:
                    terminal['exception_cleanup']=dict(observed_owned_tree_clean=False,error=repr(cleanup_error))
    except BaseException as error:
        terminal.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        abort_pending(records,str(error),a.active)
    finally:
        terminal['spent_seconds']=getattr(a,'spent',{})
        terminal['completed_slots']=sum(r['status']=='completed' for r in records)
        terminal['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        if len(records)!=35 or any(r['status'] in ('not_launched','launching') for r in records):
            abort_pending(records,'coordinator did not finish declared slots',a.active)
        progress(a.output/'progress.json',records)
        save(a.output/'terminal.json',terminal)
    return 0 if terminal['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
