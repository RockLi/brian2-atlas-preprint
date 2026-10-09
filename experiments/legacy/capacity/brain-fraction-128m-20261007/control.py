"""128M follow-up using the exact engine and tested 30-host ownership path."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'brain-fraction-86m-20261007'
spec=importlib.util.spec_from_file_location('capacity86',OLD/'control.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
c.HERE=HERE;c.c.HERE=HERE;c.c.ENV['B2_MAX_NEURONS']='128000000';c.c.ENV['B2_MAX_INITIAL_VALUES']='600000000'
CASE='owner30-128m-b600m'
P=json.loads((HERE/'protocol-v2.json').read_text())

def gates():
    assert json.loads((OLD/'pilot-owner30-validation.json').read_text())['passed']
    assert json.loads((OLD/'pilot-owner30-resources.json').read_text())['passed']
    assert json.loads((OLD/'major-owner30-audit.json').read_text())['passed']
    assert json.loads((OLD/'owner30-final-cleanup.json').read_text())['passed']

def stage():
    gates()
    raw=(HERE/'prepare-owner30-128m-b600m.py').read_bytes();h=hashlib.sha256(raw).hexdigest()
    assert h==P['script_sha256']
    code=f'''from pathlib import Path
import hashlib,json,sys,os,importlib.util
b=Path({c.BASE!r});p=b/'source-b128/experiment/prepare-owner30-128m-b600m.py';assert not p.exists()
raw=sys.stdin.buffer.read(65536);assert hashlib.sha256(raw).hexdigest()=={h!r};p.write_bytes(raw)
q=b/'source-b128/python/brian2_rust/resource_limits.py'
s=importlib.util.spec_from_file_location('caps',q);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
os.environ['B2_MAX_NEURONS']='128000000';assert m.neuron_budget()==128000000
os.environ['B2_MAX_NEURONS']='128000001'
try:m.neuron_budget()
except ValueError:pass
else:raise AssertionError('over-cap value admitted')
binary=b/'source-b128/target/release/b2-runner'
with binary.open('rb') as f:bh=hashlib.file_digest(f,'sha256').hexdigest()
assert bh=='b9dd123cbf6c7514be1207e4f271cee93a55409fc50d03980d2758858023f01e'
print(json.dumps({{'passed':True,'script_sha256':{h!r},'reference_binary_sha256':bh,'upper_bound_accepted':128000000,'over_bound_rejected':128000001}}))'''
    c.record('script-staging-v2.json',c.remote(c.NODES[0],code,raw))

def prepare():
    gates();assert json.loads((HERE/'script-staging-v2.json').read_text())['passed']
    source=c.BASE+'/source-b128'
    c.guarded('prepare-'+CASE,[c.c.PYTHON,source+'/experiment/prepare-owner30-128m-b600m.py','--source',source,'--output',c.BASE+'/'+CASE,'--neurons','128000000','--ranks','30','--compile'],65536,4,1800)

def launch():
    gates()
    assert json.loads((HERE/'engine-source-readback.json').read_text())['passed']
    prepared=json.loads((HERE/'prepared-readback.json').read_text())
    assert prepared['compiled'] and prepared['neurons']==P['neurons'] and prepared['connections']==P['connections']
    assert json.loads((HERE/('prepare-'+CASE+'-status.json')).read_text())['guard']['returncode']==0
    c.launch(CASE,memory=262144,timeout=1800,ranks_per_node=1,cpu_ids=[1],edge_limit=P['local_edge_guard'])

def audit():
    raw=(OLD/'audit.py').read_bytes();h=hashlib.sha256(raw).hexdigest()
    code=f'''from pathlib import Path
import hashlib,json
p=Path({c.BASE!r})/'audit.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=={h!r}
print(json.dumps({{'sha256':{h!r},'reused_unchanged':True}}))'''
    c.record('audit-script-identity.json',c.remote(c.NODES[0],code))
    c.guarded('audit-'+CASE,[c.c.PYTHON,c.BASE+'/audit.py','--base',c.BASE,'--case',CASE],131072,4,1200)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','prepare','status','deploy','launch','collect','audit','collect-audit','monitor']);p.add_argument('stage',nargs='?');a=p.parse_args()
    {'stage':stage,'prepare':prepare,'status':lambda:c.status(a.stage),'deploy':lambda:c.deploy(CASE,ranks_per_node=1),'launch':launch,'collect':lambda:c.collect(CASE,ranks_per_node=1),'audit':audit,'collect-audit':lambda:c.collect_audit(CASE),'monitor':c.monitor}[a.action]()
