"""Remote ARM64 build phase after the frozen extended follow-up closes.
No phase is started concurrently; no terminal result is interpreted as scientific
qualification merely because its process returned zero. The fixed stage spec
retains failures and every declared slot. This does not complete the evaluation.
"""
import argparse, datetime, fcntl, hashlib, json, os
from pathlib import Path
import shutil, socket, time, traceback
from run_recurrent_queue import launch
ROOT=Path(__file__).resolve().parents[1]
HOST='rock-mac-studio-1.local'
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,value):
    tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(p)
def active_evaluation_processes():
    import psutil
    own=psutil.Process()
    excluded={own.pid,*[p.pid for p in own.parents()]}
    names={p.name for p in (ROOT/'tools').glob('*.py')
        if p.name.startswith(('run_','qualify_','benchmark_','generate_','measure_','retry_'))}
    names.update(['capability_migration.py','frontier_capability.py','seal_a1_v4_fixture_receipt.py'])
    found=[]
    for p in psutil.process_iter():
        if p.pid in excluded:continue
        try:
            ids=p.uids()
            # Numerical workers are ordinary user processes. Setuid login
            # services may have our real UID but effective UID0; their ordinary
            # user descendants are independently enumerated below.
            if ids.real!=os.getuid() or ids.effective!=os.geteuid():continue
            argv=p.cmdline()
            if not argv:continue
            known=any(Path(a).name in names for a in argv)
            native=any(Path(a).name=='b2-train' for a in argv) and any(str(ROOT) in a for a in argv)
            in_private_tree=False
            if not known and not native:
                cwd=p.cwd()
                in_private_tree=cwd==str(ROOT) or cwd.startswith(str(ROOT)+os.sep)
            if known or native or in_private_tree:
                found.append(dict(pid=p.pid,create_time=p.create_time(),argv=argv))
        except (psutil.NoSuchProcess,psutil.ZombieProcess):
            continue
        except psutil.AccessDenied as error:
            raise RuntimeError('Process inspection denied; cannot prove serial execution: '+str(p.pid)) from error
    return found
def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--allow-run',action='store_true')
    a.add_argument('--spec',type=Path,required=True)
    args=a.parse_args()
    if not args.allow_run or socket.gethostname()!=HOST:
        a.error('Explicit execution on authorized remote host only')
    spec=json.loads(args.spec.read_text())
    if spec['status']!='frozen_preexecution' or spec['coordinator_sha256']!=digest(Path(__file__)):
        raise RuntimeError('Preexecution spec/coordinator binding missing')
    out=ROOT/spec['output']
    lockfile=(ROOT/'evidence/arm64-runtime-followup-r1.lock').open('a+')
    fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True,exist_ok=False)
    lockfile.seek(0);lockfile.truncate();lockfile.write(json.dumps({'pid':os.getpid(),'output':str(out)}));lockfile.flush()
    deadline=datetime.datetime.fromisoformat(spec['remote_budget_deadline_utc']).timestamp()
    records=[dict(name=stage['name'],status='not_launched',finite_slots=stage.get('finite_slots'),
        slot_plan=stage.get('slot_plan'),performance_run=stage['performance_run']) for stage in spec['stages']]
    active=None
    terminal={'status':'coordinator_stopped','complete_evaluation':False}
    execution_start=None
    def verify():
        if any(digest(ROOT/p)!=h for p,h in spec['identities'].items()):
            raise RuntimeError('A preexecution frozen identity changed')
    try:
        verify()
        save(out/'freeze.json',dict(spec=spec,spec_sha256=digest(args.spec),
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            pid=os.getpid(),role='serial bounded extension; no overall completion claim'))
        start=time.monotonic()
        last_wait_state=None
        prerequisites=[ROOT/p for p in spec['prerequisites']]
        while True:
            if time.time()>=deadline:
                terminal.update(status='remote_reservation_exhausted_before_start')
                return 2
            present={str(p.relative_to(ROOT)):p.exists() for p in prerequisites}
            ready=all(present.values())
            conflicts=active_evaluation_processes() if ready else []
            wait_state=dict(prerequisites=present,process_inspection_performed=ready,blocking_processes=conflicts)
            if wait_state!=last_wait_state:
                save(out/'barrier-wait.json',dict(wait_state,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    note='After prerequisites close, inspect matching real/effective user processes, including those whose cwd is in this private evaluation tree; setuid login services excluded, descendants still independently inspected'))
                last_wait_state=wait_state
            if all(present.values()) and not conflicts:
                # Closed prior phases may contain scientific failures. They are
                # retained, and each later scientific dependency has its own gate.
                save(out/'barrier.json',dict(status='released',waited_s=time.monotonic()-start,
                    prerequisites={str(p.relative_to(ROOT)):digest(p) for p in prerequisites}))
                break
            time.sleep(15)
        execution_start=time.monotonic()
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1',
            'PYTHONPATH':str(ROOT/'snapshot')+':'+str(ROOT/'snapshot/brian2-rust/python'),
            'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','MPLBACKEND':'Agg',
            'JAX_PLATFORM_NAME':'cpu','JAX_ENABLE_X64':'true'}
        for index,stage in enumerate(spec['stages']):
            verify()
            if active_evaluation_processes():raise RuntimeError('Another evaluation process is active')
            if shutil.disk_usage(ROOT).free<50*1024**3:raise RuntimeError('50 GiB free disk floor reached')
            if time.time()+stage['cap_s']>deadline:raise RuntimeError('Insufficient remaining 72h remote reservation')
            if time.monotonic()-execution_start+stage['cap_s']>spec['stage_execution_cap_s']:
                raise RuntimeError('Insufficient remaining declared stage budget')
            jobenv=env.copy()
            jobenv.update(TMPDIR=str(out/'tmp'/stage['name']),
                MPLCONFIGDIR=str(out/'mpl'/stage['name']),
                TORCHINDUCTOR_CACHE_DIR=str(out/'compiler-cache'/stage['name']/'torch'),
                JAX_COMPILATION_CACHE_DIR=str(out/'compiler-cache'/stage['name']/'jax'))
            Path(jobenv['TMPDIR']).mkdir(parents=True,exist_ok=False)
            command=[str(ROOT/'environment/cpu/bin/python'),*[str(ROOT/v[1:]) if v.startswith('@') else v for v in stage['arguments']]]
            print('START',stage['name'],flush=True)
            active=index
            row=records[index]
            row.update(status='launching',command=command)
            save(out/'progress.json',records)
            launched=launch(stage['name'],command,stage['cap_s'],64*1024**3,out,ROOT/stage['output'],jobenv,'bounded_followup')
            row.update(launched,status='supervision_completed',
                scientific_success='Inspect child evidence, never inferred from exit code')
            active=None
            save(out/'progress.json',records)
            print('END',stage['name'],row['exit_code'],row['termination_reason'],flush=True)
            # Also catches tracked native sessions that did not exit. Never
            # start a later phase when cleanup cannot be demonstrated.
            if active_evaluation_processes():raise RuntimeError('Post-stage cleanup not demonstrably complete')
        terminal.update(status='finite_extension_queue_exited')
        return 0
    except Exception as error:
        if active is not None:
            row=records[active]
            row.update(status='launch_or_supervision_error',error_type=type(error).__name__,error=str(error))
            receipt=out/(row['name']+'-terminal.json')
            if receipt.exists():
                row.update(supervisor_receipt=json.loads(receipt.read_text()),supervisor_receipt_sha256=digest(receipt))
        terminal.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        terminal.update(records=records,
            unexecuted_stages=[dict(r,reason=terminal['status']) for r in records if r['status']=='not_launched'],
            elapsed_execution_s=None if execution_start is None else time.monotonic()-execution_start)
        save(out/'terminal.json',terminal)
        fcntl.flock(lockfile,fcntl.LOCK_UN);lockfile.close()
if __name__=='__main__':raise SystemExit(main())
