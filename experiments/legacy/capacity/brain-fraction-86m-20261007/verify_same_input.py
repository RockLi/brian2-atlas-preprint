"""Compare recorded and read-back input hashes for the two large attempts."""
from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('capacity_control',HERE/'control.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
code=f'''from pathlib import Path
import hashlib,json
b=Path({c.BASE!r});old=b/'major-p12';new=b/'major-b128'
a=json.loads((old/'prepared.json').read_text());z=json.loads((new/'prepared.json').read_text())
names=['model.json','mpi/instance.bin','mpi/execution-plan.json']+[f'mpi/instance.rank-{{i}}.bin' for i in range(240)]
assert a['plan_sha256']==z['plan_sha256']
for name in names:
 assert a['files'][name]['sha256']==z['files'][name]['sha256']
 with (new/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==z['files'][name]['sha256']
print(json.dumps({{'passed':True,'input_files_verified':len(names),'plan_sha256':z['plan_sha256'],'model_sha256':z['files']['model.json']['sha256'],'neurons':z['neurons'],'connections':z['connections'],'note':'Inputs are byte-identical; no completed output from the first attempt is claimed.'}}))'''
c.record('large-input-identity.json',c.remote(c.NODES[0],code,timeout=120))
