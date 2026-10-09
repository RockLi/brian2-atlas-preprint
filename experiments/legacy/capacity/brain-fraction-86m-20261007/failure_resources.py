"""Preserve all terminal resource evidence of a failed run without claiming success."""
from pathlib import Path
import argparse, concurrent.futures, importlib.util, json
HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(); p.add_argument('case'); a = p.parse_args()
spec = importlib.util.spec_from_file_location('capacity_control', HERE / 'control.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
launch = json.loads((HERE / (a.case + '-launch-result.json')).read_text())
assert launch['error'] is not None or any(v != 0 for v in launch['returncodes'].values())
prefix = launch['resource_guard']['unit_prefix']
def worker(i):
    roles = ['proxy-' + str(i)] + (['controller'] if i == 0 else [])
    code = f'''from pathlib import Path
import json,os
b=Path({c.BASE!r});roles={roles!r};prefix={prefix!r}
guards={{role:json.loads((b/'guards'/(prefix+'-'+role+'.json')).read_text()) if (b/'guards'/(prefix+'-'+role+'.json')).exists() else None for role in roles}}
active={{role:(Path('/sys/fs/cgroup')/g['cgroup'].lstrip('/')).exists() for role,g in guards.items() if g}}
times={{p.name:p.read_text() for p in (b/{a.case!r}).glob('rank-*.time')}}
r={{'host':os.uname().nodename,'guards':guards,'cgroups_still_present':active,'rank_times':times}}
if {i==0!r}:
 r['prepared']=json.loads((b/{a.case!r}/'prepared.json').read_text())
 r['completed_runtime_present']=(b/{a.case!r}/'result/mpi-runtime.json').exists()
print(json.dumps(r))'''
    return c.remote(c.NODES[i], code)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows = list(pool.map(worker, range(len(c.NODES))))
c.record(a.case + '-failure-resources.json', {'passed': False, 'launch': launch, 'rows': rows})
