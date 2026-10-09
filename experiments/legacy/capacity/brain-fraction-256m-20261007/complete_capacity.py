"""One-shot bounded completion, no retry or automatic timeout extension."""
from pathlib import Path
import json,subprocess,sys,time
HERE=Path(__file__).resolve().parent;CASE='capacity-256m-owner60'
def run(*args):
 print(json.dumps({'stage':list(args),'utc_epoch':time.time()}),flush=True)
 subprocess.run([sys.executable,str(HERE/'control.py'),*args],check=True,stdout=subprocess.DEVNULL)
def wait(stage,seconds):
 end=time.monotonic()+seconds
 while True:
  run('status',stage)
  j=json.loads((HERE/(stage+'-status.json')).read_text());g=j['guard']
  if g and 'returncode' in g:
   assert g['returncode']==0,j['log'];return
  assert time.monotonic()<end,stage+' observer deadline'
  time.sleep(35)
wait('prepare-'+CASE,1900)
run('prepared',CASE)
subprocess.run([sys.executable,str(HERE/'readback_source.py')],check=True,stdout=subprocess.DEVNULL)
run('deploy',CASE)
subprocess.run([sys.executable,str(HERE/'verify_deployment.py'),CASE,'--ranks-per-node','2'],check=True)
run('launch',CASE)
j=json.loads((HERE/(CASE+'-launch-result.json')).read_text());assert j['error'] is None and all(v==0 for v in j['returncodes'].values())
run('collect',CASE);run('audit',CASE);wait('audit-'+CASE,1300);run('collect-audit',CASE)
run('monitor')
f=sorted(HERE.glob('monitor-*.json'))[-1];rows=json.loads(f.read_text());assert len(rows)==30 and not any(r['active_guards'] for r in rows)
(HERE/'final-cleanup.json').write_text(json.dumps({'passed':True,'hosts':30,'snapshot':f.name,'active_worker_guards':0},indent=2)+'\n')
print(json.dumps({'capacity_completed_and_audited':True}),flush=True)
