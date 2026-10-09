"""Finite serialized ARM64 seeds23/37/51/71, resumable after reboot/login."""
from pathlib import Path
import argparse,datetime,fcntl,json,os,platform,socket,subprocess,sys,time,traceback
from run_extended_followup_r2 import active_evaluation_processes
from run_recurrent_queue import launch
from a1_remaining_evidence_r1 import read,sha,save,assemble,audit,summary

ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/'protocol/A1-execution-addendum-arm64-remaining-r1.json'
OUT=ROOT/'evidence/a1-arm64-remaining-r1'
LABEL='org.bettiai.atlas-a1-remaining-r1'

def verify(spec):
    assert spec['status']=='frozen_pretraining'
    assert spec['remaining_seeds']==[23,37,51,71] and spec['max_epochs']==10
    assert spec['coordinator_sha256']==sha(__file__)
    assert spec['implementation_sha256']==sha(ROOT/'tools/run_a1_mnist_arm64_remaining_r1.py')
    for rel,digest in spec['frozen_identities'].items():
        assert sha(ROOT/rel)==digest,'Frozen identity differs: '+rel
    manifest=read(ROOT/'fixtures/a1-arm64-long-r1/manifest.json')
    assert manifest['contract_sha256']==spec['original_contract_sha256']
    for seed in spec['remaining_seeds']:
        row=manifest['files'][f'seed-{seed}.npz']
        assert sha(ROOT/f'fixtures/a1-arm64-long-r1/seed-{seed}.npz')==row['sha256']

def finish(value):
    save(OUT/'terminal.json',value)
    # This finite job no longer needs to start on subsequent logins.
    agent=Path.home()/'Library/LaunchAgents'/f'{LABEL}.plist'
    if agent.exists():agent.unlink()

def main():
    p=argparse.ArgumentParser();p.add_argument('--allow-run',action='store_true');a=p.parse_args()
    if not a.allow_run or socket.gethostname()!='rock-mac-studio-1.local' or platform.machine()!='arm64':
        p.error('Explicit authorized ARM64 remote only')
    OUT.mkdir(exist_ok=True)
    if (OUT/'terminal.json').exists():return 0
    lock=(ROOT/'evidence/extended-followup-r1.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return 75
    spec=read(SPEC)
    deadline=datetime.datetime.fromisoformat(spec['remote_budget_deadline_utc']).timestamp()-900
    if time.time()>=deadline:
        finish(dict(status='global_window_exhausted',complete_evaluation=False,records=read(OUT/'progress.json') if (OUT/'progress.json').exists() else []));return 0
    if active_evaluation_processes():return 75
    records=[];guard=None
    try:
        verify(spec)
        if not (OUT/'freeze.json').exists():
            save(OUT/'freeze.json',dict(spec_sha256=sha(SPEC),files=spec['frozen_identities'],seeds=spec['remaining_seeds'],deadline=spec['remote_budget_deadline_utc']))
        else:assert read(OUT/'freeze.json')['spec_sha256']==sha(SPEC),'Queue contract changed across recovery'
        guard=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],cwd='/private/tmp')
        save(OUT/'live.json',dict(status='running',pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),caffeinate_pid=guard.pid))
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'snapshot')+':'+str(ROOT/'snapshot/brian2-rust/python'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
        for seed in spec['remaining_seeds']:
            if active_evaluation_processes():raise RuntimeError('Another evaluation job became active')
            verify(spec)
            folder=OUT/f'seed-{seed}';folder.mkdir(exist_ok=True)
            sessions=sorted(folder.glob('session-*'))
            prior=sessions[-1]/'worker' if sessions else None
            result=read(prior/'result.json') if prior and (prior/'result.json').exists() else None
            if result and result['status']=='completed':
                contract=read(prior.parent/'contract.json');checked=audit(ROOT,prior,contract)
                save(prior.parent/'validation.json',checked)
                record=dict(seed=seed,status='completed',result=str((prior/'result.json').relative_to(ROOT)),audit=checked)
                records.append(record);save(OUT/'progress.json',records);continue
            ledger=ROOT/'evidence/a1-arm64-long-r1-test-once'/f"atlas-eager-seed-{seed}-{spec['original_contract_sha256']}.json"
            if ledger.exists():
                records.append(dict(seed=seed,status='final_test_interrupted_no_retry',ledger=str(ledger.relative_to(ROOT))))
                save(OUT/'progress.json',records);continue
            if result:
                records.append(dict(seed=seed,status=result['status'],result=str((prior/'result.json').relative_to(ROOT))))
                save(OUT/'progress.json',records);continue
            remaining=deadline-time.time()-60
            if remaining<=60:
                records.append(dict(seed=seed,status='not_launched_global_window_exhausted'));save(OUT/'progress.json',records);continue
            session=folder/f'session-{len(sessions)+1:03d}';session.mkdir(exist_ok=False)
            contract={k:v for k,v in spec.items() if k!='frozen_identities'}
            if prior:
                recovery=assemble(ROOT,prior,session/'recovery-assembly')
                if recovery:contract['resume']=recovery
            save(session/'contract.json',contract)
            # Restart reconstruction preserves original sessions and checkpoints.
            save(OUT/'current.json',dict(seed=seed,session=str(session.relative_to(ROOT)),status='starting',resume=bool(contract.get('resume')),remaining_global_window_s=remaining))
            command=[str(ROOT/'environment/cpu/bin/python'),str(ROOT/'tools/run_a1_mnist_arm64_remaining_r1.py'),
                '--root',str(ROOT),'--contract',str(session/'contract.json'),'--allow-run','--engine','atlas',
                '--seed',str(seed),'--output',str(session/'worker'),'--q0-report',str(ROOT/'evidence/arm64-r2/q0/atlas/report.json'),'--wall-cap',str(remaining)]
            print('START',seed,session.name,'resume',bool(contract.get('resume')),flush=True)
            supervision=launch(f'atlas-seed-{seed}',command,remaining+30,64*1024**3,session,session/'worker',env,'a1_remaining')
            save(session/'supervision.json',supervision)
            if supervision.get('remaining_owned_processes') or active_evaluation_processes():raise RuntimeError('Owned process cleanup not proved')
            verify(spec)
            child=read(session/'worker/terminal.json')
            if child['status']=='completed':
                checked=audit(ROOT,session/'worker',contract);save(session/'validation.json',checked)
                record=dict(seed=seed,status='completed',result=str((session/'worker/result.json').relative_to(ROOT)),audit=checked)
            else:record=dict(seed=seed,status=child['status'],terminal=str((session/'worker/terminal.json').relative_to(ROOT)))
            records.append(record);save(OUT/'progress.json',records)
            print('END',seed,record['status'],flush=True)
        if len(records)==4 and all(r['status']=='completed' for r in records):
            save(OUT/'five-seed-quality.json',summary(ROOT,records))
            finish(dict(status='four_remaining_seeds_completed_and_audited',five_seed_quality_available=True,records=records,complete_evaluation=False))
        else:finish(dict(status='finite_remaining_queue_closed_with_incomplete_outcomes',records=records,complete_evaluation=False))
        save(OUT/'live.json',dict(status='closed',pid=os.getpid()))
        return 0
    except BaseException as error:
        finish(dict(status='queue_failed_preserved',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),records=records,complete_evaluation=False))
        save(OUT/'live.json',dict(status='failed',pid=os.getpid()))
        print(traceback.format_exc(),flush=True)
        return 0  # Do not retry a scientific/software failure through launchd.
    finally:
        if guard is not None:
            guard.terminate();guard.wait(timeout=5)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':raise SystemExit(main())
