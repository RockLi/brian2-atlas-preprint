"""Independent ARM64 Atlas long A1 profile; no per-seed30min limit."""
from pathlib import Path
import argparse,datetime,fcntl,json,os,socket,time,platform
from run_extended_followup_r2 import active_evaluation_processes,digest,save
from run_recurrent_queue import launch
from arm64_artifact_gate_r1 import gate_artifact
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/a1-queue-arm64-long-r1'
CONTRACT=ROOT/'protocol/A1-execution-addendum-arm64-long-r1.json'
SEEDS=[11,23,37,51,71]
def main():
 a=argparse.ArgumentParser();a.add_argument('--allow-run',action='store_true');args=a.parse_args()
 if not args.allow_run or socket.gethostname()!='rock-mac-studio-1.local' or platform.machine()!='arm64':a.error('Explicit authorized ARM64 remote only')
 lock=(ROOT/'evidence/extended-followup-r1.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if active_evaluation_processes():raise RuntimeError('Another evaluation job is active')
 prior=json.loads((ROOT/'evidence/extended-followup-arm64-r6/terminal.json').read_text())
 if prior['status']!='finite_extension_queue_exited':raise RuntimeError('Prior phase not closed')
 spec=json.loads(CONTRACT.read_text());worker=ROOT/'tools/run_a1_mnist_arm64_long_r1.py'
 if spec['status']!='frozen_pretraining' or spec['implementation_sha256']!=digest(worker):raise RuntimeError('Worker/contract identity mismatch')
 runner=gate_artifact(ROOT/'runtime/arm64-r2/b2-train','runner',spec['runtime']['sha256'])
 # Reserve15min at the unchanged global reservation boundary for cleanup/audit.
 deadline=datetime.datetime.fromisoformat(spec['remote_budget_deadline_utc']).timestamp()-900
 if deadline-time.time()<600:raise RuntimeError('Reservation exhausted')
 original=json.loads((ROOT/'evidence/a1-queue-arm64-r1/freeze.json').read_text())
 hashes=original['files'].copy()
 for rel in ['tools/run_a1_queue_arm64_long_r1.py','tools/run_a1_mnist_arm64_long_r1.py','tools/validate_a1_arm64_long_slots_r1.py','protocol/A1-execution-addendum-arm64-long-r1.json','evidence/extended-followup-arm64-r6/terminal.json']:
  hashes[rel]=digest(ROOT/rel)
 for rel,h in hashes.items():
  if digest(ROOT/rel)!=h:raise RuntimeError('Frozen identity changed: '+rel)
 OUT.mkdir(parents=True,exist_ok=False)
 save(OUT/'freeze.json',dict(schema='a1-arm64-long-queue-r1',files=hashes,order=SEEDS,finite_slots=5,runner=runner,per_seed_wall_cap_s=None,execution_deadline_utc=datetime.datetime.fromtimestamp(deadline,datetime.timezone.utc).isoformat(),remote_reservation_deadline_utc=spec['remote_budget_deadline_utc'],claim='Independent uncapped profile; no mixing with capped results',source='Explicit user removal of30min limit; same numerical settings'))
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'snapshot')+':'+str(ROOT/'snapshot/brian2-rust/python'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
 cpu=str(ROOT/'environment/cpu/bin/python')
 command=[cpu,str(worker),'--root',str(ROOT),'--contract',str(CONTRACT)]
 records=[];terminal={'status':'coordinator_stopped','complete_evaluation':False}
 def verify():
  if active_evaluation_processes():raise RuntimeError('Other evaluation job active')
  for rel,h in hashes.items():
   if digest(ROOT/rel)!=h:raise RuntimeError('Frozen identity changed: '+rel)
 try:
  verify()
  prep=launch('shared-arrays',command+['--prepare-shared','--wall-cap',str(deadline-time.time())],300,64*1024**3,OUT,ROOT/'fixtures/a1-arm64-long-r1',env,'shared_array_preparation')
  save(OUT/'preparation.json',prep)
  if prep['exit_code']!=0 or prep.get('remaining_owned_processes'):raise RuntimeError('Preparation incomplete')
  manifest=json.loads((ROOT/'fixtures/a1-arm64-long-r1/manifest.json').read_text())
  old=json.loads((ROOT/'fixtures/a1-arm64-r1/manifest.json').read_text())
  for seed in SEEDS:
   for key in ['weights_sha256','epoch_order_sha256']:
    if manifest['files'][f'seed-{seed}.npz'][key]!=old['files'][f'seed-{seed}.npz'][key]:raise RuntimeError('Common numerical arrays changed')
  save(OUT/'common-array-equivalence.json',dict(all5seeds_identical_weights_and_orders=True,original_manifest_sha256=digest(ROOT/'fixtures/a1-arm64-r1/manifest.json'),new_manifest_sha256=digest(ROOT/'fixtures/a1-arm64-long-r1/manifest.json')))
  import validate_a1_arm64_long_slots_r1 as audit
  for seed in SEEDS:
   verify();remaining=deadline-time.time()-60
   if remaining<60:
    records.append(dict(view='atlas',engine='atlas',compiled=False,seed=seed,status='not_launched_global_reservation_exhausted'));save(OUT/'progress.json',records);continue
   dest=OUT/f'atlas-seed-{seed}';q0=ROOT/'evidence/arm64-r2/q0/atlas/report.json'
   cmd=command+['--allow-run','--engine','atlas','--seed',str(seed),'--output',str(dest),'--q0-report',str(q0),'--wall-cap',str(remaining)]
   print('START',seed,'remaining_shared_budget_s',remaining,flush=True)
   row=launch(f'atlas-seed-{seed}',cmd,remaining+30,64*1024**3,OUT,dest,env,'a1_long_supervisor')
   row.update(view='atlas',engine='atlas',compiled=False,seed=seed,wall_cap_s=remaining)
   if row.get('remaining_owned_processes'):raise RuntimeError('Owned processes remain')
   child=json.loads((dest/'terminal.json').read_text());row['status']=child['status'];row['completed_epochs']=child['completed_epochs']
   records.append(row);save(OUT/'progress.json',records)
   audit.CAP=remaining
   checked=audit.validate_slot(ROOT,OUT,dict(view='atlas',engine='atlas',compiled=False,seed=seed),row,{'files':hashes},manifest,digest(q0),True)
   save(OUT/f'audit-seed-{seed}.json',checked)
   print('END',seed,row['status'],'epochs',row['completed_epochs'],'audit_errors',checked['errors'],flush=True)
   if checked['errors']:raise RuntimeError('Long profile per-slot evidence audit failed')
   if child['status']=='cleanup_not_verified_do_not_launch_next_case':raise RuntimeError('Cleanup not verified')
  terminal.update(status='finite_a1_long_queue_exited')
  return 0
 except Exception as error:
  terminal.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  terminal.update(records=records,finite_slots=5,unexecuted_seeds=[s for s in SEEDS if s not in [r['seed'] for r in records]])
  save(OUT/'terminal.json',terminal);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':raise SystemExit(main())

