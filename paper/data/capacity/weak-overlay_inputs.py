"""Reuse verified immutable input files in a fresh, hash-audited run directory."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import control_policy48 as c

HERE=Path(__file__).resolve().parent


def overlay(old_case,new_case,pilot=False):
    c.configure(30)
    prep=json.loads((HERE/(old_case+'-prepared.json')).read_text())
    assert prep['compiled'] and prep['hosts']==30 and prep['ranks']==240
    assert json.loads((HERE/(old_case+'-deployment-audit.json')).read_text())['passed']
    policy=json.loads((HERE/'protocol-revision-v3.json').read_text())
    shared=['b2-mpi','manifest.json','build.json','instance.bin','execution-plan.json']
    expected={n.removeprefix('mpi/'):v['sha256'] for n,v in prep['files'].items()
              if n.startswith('mpi/') and (n.removeprefix('mpi/') in shared or n.removeprefix('mpi/').startswith('instance.rank-'))}
    def worker(i):
        names=shared+[f'instance.rank-{r}.bin' for r in range(i*8,(i+1)*8)]
        links=['mpi']+(['model.json','prepared.json'] if i==0 else [])+(['reference'] if i==0 and pilot else [])
        code=f'''from pathlib import Path
import hashlib,json,os
b=Path({c.BASE!r});old=b/{old_case!r};new=b/{new_case!r}
assert old.is_dir() and not (new/'result').exists()
assert hashlib.sha256((b/'guard-host-reserve48-v3.py').read_bytes()).hexdigest()=={policy['guard_sha256']!r}
new.mkdir(exist_ok=True)
for name in {links!r}:
 target=old/name;p=new/name
 assert target.exists()
 if p.is_symlink():assert p.resolve()==target.resolve()
 else:assert not p.exists();p.symlink_to(target,target_is_directory=target.is_dir())
files={{}}
for name in {names!r}:
 with (new/'mpi'/name).open('rb') as f:files[name]=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps({{'host':os.uname().nodename,'files':files,'owned_ranks':list(range({i*8},{(i+1)*8})),'links':{links!r},'guard_sha256':{policy['guard_sha256']!r}}}))'''
        row=c.remote(c.NODES[i],code,timeout=180)
        assert all(row['files'][n]==expected[n] for n in names)
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(worker,range(30)))
    (HERE/(new_case+'-prepared.json')).write_bytes((HERE/(old_case+'-prepared.json')).read_bytes())
    old_status=json.loads((HERE/('prepare-'+old_case+'-status.json')).read_text())
    assert old_status['guard']['returncode']==0
    old_status['input_preparation_reused_from']=old_case
    (HERE/('prepare-'+new_case+'-status.json')).write_text(json.dumps(old_status,indent=2)+'\n')
    c.record(new_case+'-deployment.json',rows)
    c.record(new_case+'-deployment-audit.json',dict(passed=True,hosts=30,ranks=240,
        input_catalog_sha256=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest(),
        transport='Read-only per-node input symlinks; every shared and owned-rank file rehashed against the original prepared catalog.',reused_from=old_case))
    c.record(new_case+'-input-reuse.json',dict(passed=True,old_case=old_case,new_case=new_case,
        model_sha256=prep['files']['model.json']['sha256'],plan_sha256=prep['plan_sha256'],
        preparation_status_source='prepare-'+old_case+'-status.json',
        numerical_input_changes=False,numerical_engine_changes=False,rows=rows))
