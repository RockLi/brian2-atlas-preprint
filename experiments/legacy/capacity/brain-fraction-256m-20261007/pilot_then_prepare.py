"""One-shot bounded numerical pilot, then prepare the next capacity case."""
from pathlib import Path
import json,subprocess,sys,time
HERE=Path(__file__).resolve().parent
PILOT='pilot-owner60-n256';MAJOR='capacity-256m-owner60'
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
  time.sleep(20)
wait('budget-tests-n256',360)
run('prepare',PILOT);wait('prepare-'+PILOT,1900)
run('prepared',PILOT)
run('reference');wait('reference-'+PILOT,360)
run('deploy',PILOT)
subprocess.run([sys.executable,str(HERE/'verify_deployment.py'),PILOT,'--ranks-per-node','2'],check=True)
run('launch',PILOT)
run('validate',PILOT);run('collect',PILOT);run('proof')
run('prepare',MAJOR)
print(json.dumps({'pilot_passed':True,'major_preparation_started':True}),flush=True)
