"""Verify unchanged canonical model and the explicitly changed rank partition."""
from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('capacity_control',HERE/'control.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
code=f'''from pathlib import Path
import hashlib,json
b=Path({c.BASE!r});old=b/'major-p12';new=b/'major-r60'
a=json.loads((old/'prepared.json').read_text());z=json.loads((new/'prepared.json').read_text())
assert a['files']['model.json']['sha256']==z['files']['model.json']['sha256']
with (new/'model.json').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==z['files']['model.json']['sha256']
assert z['neurons']==86000000 and z['connections']==86000000000 and z['ranks']==60 and z['compiled']
assert {{p.name for p in (new/'mpi').glob('instance.rank-*.bin')}}=={{f'instance.rank-{{i}}.bin' for i in range(60)}}
print(json.dumps({{'passed':True,'canonical_model_byte_identical':True,'model_sha256':z['files']['model.json']['sha256'],'plan_sha256':z['plan_sha256'],'ranks':60,'rank_partition_changed':True,'neurons':z['neurons'],'connections':z['connections']}}))'''
c.record('r60-input-identity.json',c.remote(c.NODES[0],code,timeout=120))
