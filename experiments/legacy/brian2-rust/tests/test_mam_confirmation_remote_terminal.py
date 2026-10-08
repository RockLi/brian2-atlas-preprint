"""Synthetic terminal extensions of a real live anchor; no neural outcomes."""
import ast,copy,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import test_mam_confirmation_terminal as previous
import mam_confirmation_remote_terminal as remote
import mam_confirmation_terminal as legacy
import mam_collect_confirmation_remote_terminal as collector
from mam_collect_confirmation_terminal import host_code

@pytest.fixture
def controls():
    v,p,a,l,hosts=previous.controls.__wrapped__()
    anchor=json.loads((previous.E/'confirmation-run-v1-seed1750'/remote.PROVENANCE_NAME).read_text())
    for initial,host in zip(anchor['remote_hosts'],hosts,strict=True):
        for unit in initial['units']:
            role=unit['role'];start=json.loads(unit['guard_text'])
            final=copy.deepcopy(start);after=copy.deepcopy(start['before'])
            counts=dict(x.split() for x in after['cpu.stat'].splitlines())
            for k in ['usage_usec','user_usec','system_usec']:counts[k]=str(int(counts[k])+1000000)
            after['cpu.stat']='\n'.join(k+' '+x for k,x in counts.items())
            after['memory.peak']=str(max(1,int(after['memory.current'])))
            final.update(returncode=0,after=after,wall_seconds=40010. if role=='controller' else 40000.,minimum_observed_free_bytes=start['data_free_bytes'])
            host['guards'][role]=final
    return v,p,a,anchor,hosts

def test_preserves_full_checks_and_reports_missing_transport(controls):
    result=remote.audit(*controls)
    assert result['terminal_resource_audit_passed'] and result['metadata_sync_audit_passed']
    assert not result['raw_output_audit_passed'] and not result['scientific_acceptance']
    assert not result['performance_cost_acceptance']
    assert len(result['ranks'])==32 and len(result['guards'])==5
    assert result['local_launcher_returncodes'] is None and result['local_launcher_wall_seconds'] is None
    assert result['remote_mpi_wall_seconds']==40010 and 'launch_wall_seconds' not in result
    assert 'remote controller' in result['accounting']['wall_scope']

@pytest.mark.parametrize('fault',['anchor','command','before','failed_guard','pending_guard','old_duration','wall_cap','oom','active','missing_rank','failed_rank','old_model','spool','duration','output_size'])
def test_bad_remote_evidence_is_rejected(controls,fault):
    v,p,a,anchor,hosts=controls;g=hosts[0]['guards']['controller']
    if fault=='anchor':anchor['simulation_restarted']=True
    elif fault=='command':g['command'][-1]+=' changed'
    elif fault=='before':g['before']['memory.current']='0'
    elif fault=='failed_guard':g['returncode']=1
    elif fault=='pending_guard':del g['returncode']
    elif fault=='old_duration':g['wall_seconds']=100
    elif fault=='wall_cap':g['wall_seconds']=64801
    elif fault=='oom':hosts[1]['guards']['proxy-1']['after']['memory.events']='max 0\noom 1\noom_kill 1\noom_group_kill 0'
    elif fault=='active':hosts[2]['active_own_units']=['still running']
    elif fault=='missing_rank':del hosts[3]['ranks']['31']
    elif fault in ['failed_rank','old_model']:
        item=hosts[3]['ranks']['31'];r=json.loads(item['done_json'])
        if fault=='failed_rank':r['child_returncode']=1
        else:r['model_sha256']='old'
        item['done_json']=json.dumps(r)
    elif fault=='spool':hosts[0]['leader_outputs']['spool_present']=True
    elif fault=='duration':
        item=hosts[0]['leader_outputs'];r=json.loads(item['summary_json']);r['final_time_seconds']=2;item['summary_json']=json.dumps(r)
    else:hosts[0]['leader_outputs']['binary_file_bytes']['results.bin']+=1
    with pytest.raises((ValueError,KeyError)):remote.audit(*controls)

def test_resource_rank_fsync_and_work_checks_unchanged():
    def core(module):
        tree=ast.parse(Path(module.__file__).read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
        begin=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='guard_rows' for x in n.targets))
        return [ast.dump(n,include_attributes=False) for n in fn.body[begin:-1]]
    assert core(remote)==core(legacy)
    assert collector.host_code is host_code

def test_collector_does_not_fabricate_launcher_control():
    text=Path(collector.__file__).read_text()
    assert "write(args.case/'launch.json'" not in text
    assert 'args.launch_sha256' not in text
    assert 'launch_gate(' not in text
    assert 'recovery=PROVENANCE_SHA' in text
