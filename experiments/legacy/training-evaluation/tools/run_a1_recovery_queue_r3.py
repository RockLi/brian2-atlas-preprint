"""Remote restart recovery for Atlas seed11. Original interrupted evidence is immutable."""
from pathlib import Path
import datetime,fcntl,json,os,socket,sys,time,collections,math
from run_extended_followup_r2 import active_evaluation_processes,digest,save
from run_recurrent_queue import launch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/a1-arm64-resume-r3'
SPEC=ROOT/'protocol/A1-execution-addendum-arm64-resume-r3.json'
def audit_result(freeze):
 from validate_a1_arm64_long_slots_r1 import checkpoint_counter
 errors=[]
 def check(ok,msg):
  if not ok:errors.append(msg)
 contract=json.loads(SPEC.read_text());resume=contract['resume'];dest=OUT/'atlas-seed-11'
 new=json.loads((dest/'result.json').read_text()) if (dest/'result.json').exists() else json.loads((dest/'progress.json').read_text())
 old=json.loads((ROOT/resume['source_progress']).read_text())
 completed=old['epochs']+new.get('epochs',[])
 check(new.get('resume',{}).get('restored_step')==13633,'restore counter mismatch')
 check(new.get('prior_completed_epochs')==old['epochs'],'prior complete epoch metadata differs')
 check([e['epoch'] for e in completed]==list(range(1,len(completed)+1)),'complete epoch sequence differs')
 check(new.get('runtime_sha256')==contract['runtime']['sha256'],'runtime differs')
 for rel,h in freeze.items():check(digest(ROOT/rel)==h,'frozen identity differs: '+rel)
 logs={}
 for session,p in [('original',ROOT/resume['source_batches']),('resume',dest/'batches.jsonl')]:
  rows=[]
  for line in p.read_text().splitlines():
   try:rows.append(json.loads(line))
   except ValueError:check(False,'malformed batch log '+session)
  logs[session]=rows
 prefix_path=ROOT/resume['partial_prefix_batches']
 logs['resume']=[json.loads(line) for line in prefix_path.read_text().splitlines()]+logs['resume']
 best=None;bestcorrect=-1
 for e in completed:
  session='original' if e['epoch']<=7 else 'resume'
  for phase,n in [('train',55000),('validation',5000)]:
   rows=[x for x in logs[session] if x['phase']==phase and x['epoch']==e['epoch']]
   check([x['offset'] for x in rows]==list(range(0,n,32)),f"epoch{e['epoch']} {phase} offset denominator differs")
   check(sum(x['samples'] for x in rows)==n,f"epoch{e['epoch']} {phase} sample denominator differs")
   check(e[phase]['samples']==n,f"epoch{e['epoch']} recorded aggregate samples differ")
   check(sum(x['correct'] for x in rows)==e[phase]['correct'],f"epoch{e['epoch']} aggregate correct differs")
   check(math.isclose(sum(x['loss']*x['samples'] for x in rows)/n,e[phase]['loss'],rel_tol=1e-9,abs_tol=1e-9),f"epoch{e['epoch']} loss aggregate differs")
  if e['validation']['correct']>bestcorrect:
   best=e['epoch'];bestcorrect=e['validation']['correct']
   checkpoint=(ROOT/resume['source_directory'] if session=='original' else dest)/f'best-epoch-{best}.json'
   c=checkpoint_counter(checkpoint,'atlas')
   check(c['counters']==[best*1719],f'checkpoint epoch{best} Adam count differs')
   check(c['runtime_sha256']==contract['runtime']['sha256'],'checkpoint runtime differs')
 selection=new.get('selection')
 check(selection is not None and selection['epoch']==best and selection['validation_correct']==bestcorrect,'selected best complete epoch differs')
 if selection:check(digest(dest/selection['checkpoint'])==selection['sha256'],'selected checkpoint hash differs')
 if new.get('status')=='completed':
  check(len(completed)==10,'full10epoch result not complete')
  check(new.get('test_status')=='completed_once' and new.get('test',{}).get('samples')==10000,'complete final test missing')
  marker=json.loads((dest/'test-started.json').read_text())
  ledger=ROOT/'evidence/a1-arm64-long-r1-test-once'/f"atlas-eager-seed-11-{resume['original_contract_sha256']}.json"
  check(json.loads(ledger.read_text())==marker,'original once-only ledger ownership differs')
 return dict(status='evidence_consistent_recovered_session' if not errors else 'evidence_inconsistent',errors=errors,completed_epochs=len(completed),test=new.get('test'),best_validation_accuracy=bestcorrect/5000,test_score_scope='Recorded complete aggregate; no held-out labels or predictions reread',time_scope='Reboot-interrupted and recovered sessions, not uninterrupted time-to-quality',performance_ranking=False)
def main():
 import platform
 if socket.gethostname()!='rock-mac-studio-1.local' or platform.machine()!='arm64':raise RuntimeError('Authorized remote only')
 spec=json.loads(SPEC.read_text());freeze=spec['recovery_identities']
 lock=(ROOT/'evidence/extended-followup-r1.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if active_evaluation_processes():raise RuntimeError('Evaluation conflict')
 for rel,h in freeze.items():
  if digest(ROOT/rel)!=h:raise RuntimeError('Frozen recovery identity changed: '+rel)
 OUT.mkdir(parents=True,exist_ok=False);save(OUT/'freeze.json',dict(spec_sha256=digest(SPEC),files=freeze))
 deadline=datetime.datetime.fromisoformat(spec['remote_budget_deadline_utc']).timestamp()-900
 remaining=deadline-time.time()-60
 if remaining<=0:raise RuntimeError('Reservation exhausted')
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'snapshot')+':'+str(ROOT/'snapshot/brian2-rust/python'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
 cmd=[str(ROOT/'environment/cpu/bin/python'),str(ROOT/'tools/run_a1_mnist_arm64_resume_r3.py'),'--root',str(ROOT),'--contract',str(SPEC),'--allow-run','--engine','atlas','--seed','11','--output',str(OUT/'atlas-seed-11'),'--q0-report',str(ROOT/'evidence/arm64-r2/q0/atlas/report.json'),'--wall-cap',str(remaining)]
 row=launch('atlas-seed-11',cmd,remaining+30,64*1024**3,OUT,OUT/'atlas-seed-11',env,'resumed_A1')
 save(OUT/'supervision.json',row)
 if row.get('remaining_owned_processes') or active_evaluation_processes():raise RuntimeError('Cleanup not proved')
 result=audit_result(freeze);save(OUT/'validation.json',result)
 save(OUT/'terminal.json',dict(status='recovery_session_closed',supervision=row,validation_status=result['status'],errors=result['errors'],complete_evaluation=False))
 print(json.dumps(result),flush=True)
 return 0 if not result['errors'] else 1
if __name__=='__main__':raise SystemExit(main())

