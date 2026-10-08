"""Synthetic terminal controls only; no claimed 1750 neural outcomes."""
import ast,copy,hashlib,json,shlex,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import test_mam_rust_benchmark_terminal as previous
import mam_confirmation_terminal as terminal
import mam_collect_confirmation_terminal as collector
from mam_launch_confirmation_run import CASE,launch_options
from mam_confirmation_identity import NODES
from mam_rust_cpu_placement import ALTERNATE_CPUS

E=Path(__file__).resolve().parents[1]/'mpi-evidence'


@pytest.fixture
def controls():
    _,launch,hosts=previous.controls.__wrapped__()
    root=E/CASE
    v=json.loads((root/'identity.json').read_text());p=json.loads((root/'protocol.json').read_text())
    a=json.loads((root/'admission.json').read_text())
    app=launch_options(v,Path('/unused'))['application']
    launch['commands']['controller']=['tsh','ssh','root@'+NODES[0],shlex.join(['mpiexec','-n','32',*app])]
    output=hosts[0]['leader_outputs'];runtime=json.loads(output['runtime_json'])
    runtime.update(plan_sha256=v['plan_sha256'],rank_cpu_ids=ALTERNATE_CPUS*4)
    summary=json.loads(output['summary_json']);summary['mpi']=runtime
    output.update(summary_json=json.dumps(summary),runtime_json=json.dumps(runtime))
    for host in hosts:
        host['source_catalog']=a['source_catalog']
        for g in host['guards'].values():
            for when in ['before','after']:
                g[when]['cpuset.cpus.effective']=','.join(map(str,ALTERNATE_CPUS))
        for rank,item in host['ranks'].items():
            for key in ['started_json','done_json']:
                value=json.loads(item[key])
                value.update(replicate=1750,identity_sha256=terminal.IDENTITY_SHA,
                    model_sha256=v['model_sha256'],instance_sha256=v['instance_sha256'],
                    plan_sha256=v['plan_sha256'],executable_sha256=v['executable_sha256'],
                    wrapper_sha256=a['source_catalog']['mam_confirmation_terminal_sync.py']['sha256'])
                if key=='done_json' and rank=='0':
                    sync=value['leader_output_synchronization']
                    for name,k in [('summary.json','summary_json'),('mpi-runtime.json','runtime_json')]:
                        raw=output[k].encode();sync['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                item[key]=json.dumps(value)
    return v,p,a,launch,hosts


def test_all_terminal_checks_keep_raw_and_science_pending(controls):
    r=terminal.audit(*controls)
    assert r['replicate']==1750 and len(r['ranks'])==32 and len(r['guards'])==5
    assert r['terminal_resource_audit_passed'] and not r['raw_output_audit_passed']
    assert not r['scientific_acceptance'] and not r['performance_cost_acceptance']


@pytest.mark.parametrize('fault',['identity','budget','seed_receipt','failed_rank','oom','live','spool','output_bytes','old_plan'])
def test_bad_terminal_evidence_rejected(controls,fault):
    v,p,a,l,hosts=controls
    if fault=='identity':v['random_keys']['runtime_input']=1729
    elif fault=='budget':p['budgets']['simulation_seconds']+=1
    elif fault=='seed_receipt':
        q=hosts[1]['ranks']['8'];r=json.loads(q['done_json']);r['replicate']=1729;q['done_json']=json.dumps(r)
    elif fault=='failed_rank':l['returncodes']['proxy-3']=1
    elif fault=='oom':hosts[0]['guards']['proxy-0']['after']['memory.events']='max 0\noom 1\noom_kill 1\noom_group_kill 0\n'
    elif fault=='live':hosts[3]['active_own_units']=['running']
    elif fault=='spool':hosts[0]['leader_outputs']['spool_present']=True
    elif fault=='output_bytes':hosts[0]['leader_outputs']['binary_file_bytes']['results.bin']+=1
    else:
        q=hosts[0]['ranks']['0'];r=json.loads(q['done_json']);r['plan_sha256']=previous.PLAN;q['done_json']=json.dumps(r)
    with pytest.raises(ValueError):terminal.audit(*controls)


def test_collector_only_reads_owned_shards_and_new_paths(controls):
    v,p,a,l,_=controls
    prefix=collector.launch_gate(v,p,a,l)
    for i in range(4):
        code=collector.host_code(a,v,i,prefix);ast.parse(code)
        assert v['project'] in code and v['receipts'] in code and v['metrics'] in code
        assert 'performance-receipts' not in code
        # The literal owned catalog inside the generated reader has 8 rank files.
        catalogs=[n.value for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('mpi/instance.rank-')]
        assert set(catalogs)=={'mpi/instance.rank-'+str(r)+'.bin' for r in range(i*8,i*8+8)}
