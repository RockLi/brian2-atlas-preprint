"""Read back all files in the unchanged isolated engine catalog."""
from pathlib import Path
import hashlib,json
import control
p=Path(__file__).resolve().parent
catalog=json.loads((control.OLD/'source-b128-identity.json').read_text())
raw=json.dumps(catalog['files']).encode()
code=f'''from pathlib import Path
import hashlib,json,sys
b=Path({control.c.BASE!r})/'source-b128';expected=json.load(sys.stdin);actual={{}}
for name,row in expected.items():
 p=b/name
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 assert h==row['sha256'] and p.stat().st_size==row['bytes'],name
 actual[name]={{'bytes':p.stat().st_size,'sha256':h}}
print(json.dumps({{'passed':True,'files_checked':len(actual),'files':actual}}))'''
result=control.c.remote(control.c.NODES[0],code,raw)
result['archive_sha256']=catalog['archive_sha256']
control.c.record('engine-source-readback.json',result)
