"""Stage a hash-bound admission guard without changing the frozen engine."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import xml.etree.ElementTree as ET
import control as c

HERE=Path(__file__).resolve().parent
policy=json.loads((HERE/'protocol-revision-v3.json').read_text())
test=ET.parse(HERE/'policy48-tests.xml').getroot().find('testsuite')
assert test is not None and int(test.attrib['tests'])==11
assert test.attrib['failures']==test.attrib['errors']=='0'
raw=(HERE/'guard-host-reserve48-v3.py').read_bytes()
sha=hashlib.sha256(raw).hexdigest();assert sha==policy['guard_sha256']
assert len(raw)<65536
c.configure(30)
code=f'''from pathlib import Path
import hashlib,json,sys
p=Path({c.BASE!r})/'guard-host-reserve48-v3.py'
raw=sys.stdin.buffer.read(65536);assert hashlib.sha256(raw).hexdigest()=={sha!r}
assert p.parent.is_dir()
if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()=={sha!r}
else:p.write_bytes(raw)
print(json.dumps({{'passed':True,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'path':str(p)}}))'''
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows=list(pool.map(lambda n:dict(host=n,**c.remote(n,code,raw)),c.NODES))
assert all(r['passed'] and r['sha256']==sha for r in rows)
(HERE/'policy48-stage.json').write_text(json.dumps(dict(passed=True,guard_sha256=sha,tests=11,rows=rows,status='staged_not_activated'),indent=2)+'\n')
print(json.dumps({'passed':True,'staged_hosts':len(rows),'tests':11,'guard_sha256':sha,'status':'staged_not_activated'}))
