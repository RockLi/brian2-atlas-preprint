"""Bounded 60-rank next-scale experiment; preserve prior capacity artifacts."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,importlib.util,json,shlex
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('capacity_base',HERE/'base_control.py')
c=importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
c.c.ENV.update(B2_MAX_NEURONS='256000000',B2_MAX_INITIAL_VALUES='1200000000',B2_MAX_IR_BYTES=str(16*2**30))
SOURCE=c.BASE+'/source-n256'
P=json.loads((HERE/'protocol.json').read_text())
PILOT='pilot-owner60-n256';MAJOR='capacity-256m-owner60'

def stage():
    raw=(HERE/'source.tar.gz').read_bytes();identity=json.loads((HERE/'source-identity.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==identity['archive_sha256']
    code=f'''from pathlib import Path
import hashlib,io,json,sys,tarfile
p=Path({SOURCE!r});assert not p.exists();raw=sys.stdin.buffer.read(16*2**20)
assert hashlib.sha256(raw).hexdigest()=={identity['archive_sha256']!r};p.mkdir()
with tarfile.open(fileobj=io.BytesIO(raw)) as t:
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(p,filter='data')
print(json.dumps({{'passed':True,'source_archive_sha256':{identity['archive_sha256']!r}}}))'''
    c.record('source-stage.json',c.remote(c.NODES[0],code,raw))

def build():
    assert json.loads((HERE/'source-stage.json').read_text())['passed']
    cargo='/atlas-home/0003/workspace/brian2-lk-20260906/.cargo/bin/cargo'
    command='cd '+shlex.quote(SOURCE)+' && '+shlex.join([cargo,'build','--offline','--release','--bin','b2-runner','-j','4'])+' && '+shlex.join([cargo,'test','--offline','resource_limits::tests','-j','4'])
    c.guarded('rust-budget-tests-n256-fixtures',['sh','-c',command],16384,4,1200)

def tests():
    assert json.loads((HERE/'rust-budget-tests-n256-fixtures-status.json').read_text())['guard']['returncode']==0
    c.guarded('budget-tests-n256',['env','B2_RUNNER='+SOURCE+'/target/release/b2-runner',c.c.PYTHON,'-m','pytest','-q',SOURCE+'/tests/test_resource_limits_corrected_n256.py','--junitxml='+c.BASE+'/budget-n256.xml','--basetemp='+c.BASE+'/tmp/budget-n256'],4096,2,300)

def gates(case):
    assert json.loads((HERE/'test-fixture-stage.json').read_text())['passed']
    assert json.loads((HERE/'rust-budget-tests-n256-fixtures-status.json').read_text())['guard']['returncode']==0
    assert json.loads((HERE/'budget-tests-n256-status.json').read_text())['guard']['returncode']==0
    for name in ['rust-budget-tests-n256-fixtures-status.json','budget-tests-n256-status.json']:
        assert SOURCE in ' '.join(json.loads((HERE/name).read_text())['guard']['command'])
    if case==MAJOR:
        assert json.loads((HERE/(PILOT+'-validation.json')).read_text())['passed']
        assert json.loads((HERE/(PILOT+'-resources.json')).read_text())['passed']
        assert json.loads((HERE/(PILOT+'-placement-proof.json')).read_text())['passed']

def prepare(case):
    gates(case);assert case in [PILOT,MAJOR]
    n=24000 if case==PILOT else 256000000
    c.guarded('prepare-'+case,[c.c.PYTHON,SOURCE+'/experiment/prepare-owner60.py','--source',SOURCE,'--output',c.BASE+'/'+case,'--neurons',str(n),'--ranks','60','--compile'],98304,4,1800)

def reference():
    c.guarded('reference-'+PILOT,[SOURCE+'/target/release/b2-runner',c.BASE+'/'+PILOT+'/model.json',c.BASE+'/'+PILOT+'/reference'],4096,1,300)

def launch(case):
    gates(case)
    assert json.loads((HERE/('prepare-'+case+'-status.json')).read_text())['guard']['returncode']==0
    c.launch(case,memory=8192 if case==PILOT else 262144,timeout=300 if case==PILOT else 1800,ranks_per_node=2,cpu_ids=[1,49],edge_limit=P['local_edge_guard'])

def proof():
    j=json.loads((HERE/(PILOT+'-validation.json')).read_text());r=j['runtime']
    assert j['passed'] and j['checks']==603 and j['max_state_absolute_difference']==0
    assert r['ranks']==60 and len(set(r['processor_names']))==30
    assert r['rank_cpu_ids']==[v for _ in range(30) for v in [1,49]]
    assert all(r['rank_work'][i*3]>0 for i in range(60))
    assert len(r['procedural_topology'])==3600 and {t['construction'] for t in r['procedural_topology']}=={'target-owner-local'}
    c.record(PILOT+'-placement-proof.json',{'passed':True,'ranks':60,'hosts':30,'construction':'target-owner-local','checks':603,'difference':0})

def read_prepared(case):
    code=f'''from pathlib import Path
import json
print((Path({c.BASE!r})/{case!r}/'prepared.json').read_text())'''
    c.record(case+'-prepared.json',c.remote(c.NODES[0],code))

def audit(case):
    raw=(HERE/'audit.py').read_bytes();h=hashlib.sha256(raw).hexdigest()
    code=f'''from pathlib import Path
import hashlib,json,sys
p=Path({c.BASE!r})/'audit-n256.py';assert not p.exists();raw=sys.stdin.buffer.read(65536)
assert hashlib.sha256(raw).hexdigest()=={h!r};p.write_bytes(raw);print(json.dumps({{'sha256':{h!r}}}))'''
    c.record('audit-script-identity.json',c.remote(c.NODES[0],code,raw))
    c.guarded('audit-'+case,[c.c.PYTHON,c.BASE+'/audit-n256.py','--base',c.BASE,'--case',case],131072,4,1200)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','build','tests','prepare','reference','status','deploy','launch','validate','proof','collect','prepared','audit','collect-audit','monitor']);p.add_argument('case',nargs='?',default=PILOT);a=p.parse_args()
    {'stage':stage,'build':build,'tests':tests,'prepare':lambda:prepare(a.case),'reference':reference,'status':lambda:c.status(a.case),'deploy':lambda:c.deploy(a.case,ranks_per_node=2),'launch':lambda:launch(a.case),'validate':lambda:c.validate_pilot(a.case,expected_ranks=60),'proof':proof,'collect':lambda:c.collect(a.case,ranks_per_node=2),'prepared':lambda:read_prepared(a.case),'audit':lambda:audit(a.case),'collect-audit':lambda:c.collect_audit(a.case),'monitor':c.monitor}[a.action]()
