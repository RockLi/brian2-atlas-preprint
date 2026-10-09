"""Thirty hosts, two ranks each; retain the numerically verified b128 kernel."""
from pathlib import Path
import argparse,importlib.util,json
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('capacity_control',HERE/'control.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
def prepare(case):
    assert case in ['pilot-r60','major-r60']
    if case=='major-r60':
        assert json.loads((HERE/'pilot-r60-validation.json').read_text())['passed']
        assert json.loads((HERE/'pilot-r60-resources.json').read_text())['passed']
    source=c.BASE+'/source-b128';n=24000 if case.startswith('pilot') else 86000000
    c.guarded('prepare-'+case,[c.c.PYTHON,source+'/experiment/prepare-p12.py','--source',source,'--output',c.BASE+'/'+case,'--neurons',str(n),'--ranks','60','--compile'],65536,4,1800)
def launch(case):
    assert json.loads((HERE/'batch-boundary-proof.json').read_text())['passed']
    if case=='major-r60':
        assert json.loads((HERE/'r60-input-identity.json').read_text())['passed']
        assert json.loads((HERE/'pilot-r60-validation.json').read_text())['passed']
        assert json.loads((HERE/'pilot-r60-resources.json').read_text())['passed']
    c.launch(case,memory=8192 if case.startswith('pilot') else 262144,ranks_per_node=2,cpu_ids=[1,49],edge_limit=1600000000)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','deploy','launch','validate','collect']);p.add_argument('case');a=p.parse_args()
    {'prepare':prepare,'deploy':lambda x:c.deploy(x,ranks_per_node=2),'launch':launch,'validate':lambda x:c.validate_pilot(x,expected_ranks=60),'collect':lambda x:c.collect(x,ranks_per_node=2)}[a.action](a.case)
