"""Primary catalog composition and orchestration boundaries; no new transfer benchmark."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_collect_native_primary_raw as m


def host(index):
    prefix='b2mpi-123456abcdef'
    guards={r:dict(cgroup='/system.slice/'+prefix+'-'+r+'.service')
            for r in (['controller'] if index==0 else [])+['proxy-'+str(index)]}
    sources={m.LABEL+'/'+n:dict(bytes=10,sha256='a'*64) for n in [
        'mam_nest_reference.py','mam_nest_rank_affinity.py','mpi_resource_guard.py','layout.json','planned-budget.json','parameters.json']}
    for role in guards:sources['guards/'+prefix+'-'+role+'.json']=dict(bytes=10,sha256='a'*64)
    ranks={}
    for r in range(index*8,(index+1)*8):
        sources['runs/'+m.LABEL+'/rank'+str(r)+'.json']=dict(bytes=10,sha256='a'*64)
        sources['affinity/'+m.LABEL+'/rank'+str(r)+'.json']=dict(bytes=10,sha256='a'*64)
        ranks[str(r)]=dict(event_bytes=2*2**30,event_sha256='b'*64)
    return dict(host=m.NODES[index],guards=guards,source_files=sources,rank_reports=ranks)


def progress(index):
    return {'runs/'+m.LABEL+'/rank'+str(r)+'.progress.jsonl':dict(bytes=4*2**20,sha256='c'*64)
            for r in range(index*8,(index+1)*8)}


def test_all_six_catalogs_admit_complete_recording_quota_without_allocating_files():
    total=0;files=0
    for i in range(6):
        c=m.make_catalog(i,host(i),progress(i));n,d=m.validate(c,m.MAX_FILE,m.MAX_HOST)
        assert sum(r['path'].endswith('.events.bin') for r in c['files'])==8
        assert [r['path'] for r in c['files']]==sorted(set(r['path'] for r in c['files']))
        assert len(d)==64
        total+=n;files+=len(c['files'])
    assert total < m.MAX_ALL and total > 96*2**30 and files==235


@pytest.mark.parametrize('fault',['missing-control','extra-control','missing-progress','oversized-progress',
    'wrong-host','oversized-event','bad-event-hash'])
def test_catalog_must_match_terminal_controls(fault):
    h=host(0);p=progress(0)
    if fault=='missing-control':h['source_files'].pop(next(iter(h['source_files'])))
    elif fault=='extra-control':h['source_files']['unrelated']=dict(bytes=1,sha256='d'*64)
    elif fault=='missing-progress':p.pop(next(iter(p)))
    elif fault=='oversized-progress':p[next(iter(p))]['bytes']+=1
    elif fault=='wrong-host':h['host']=m.NODES[1]
    elif fault=='oversized-event':h['rank_reports']['0']['event_bytes']+=1
    else:h['rank_reports']['0']['event_sha256']='wrong'
    with pytest.raises(ValueError):m.make_catalog(0,h,p)


def test_shared_budget_subtracts_prior_work_and_completion_reserve(monkeypatch):
    monkeypatch.setattr(m.time,'monotonic',lambda:1200.)
    assert m.remaining(1000.,100.)==900
    assert m.remaining(1000.,6300.)==610
    with pytest.raises(ValueError):m.remaining(1000.,6850.)


def test_all_probes_parse_and_do_not_hash_raw_payloads():
    for i in [None,*range(6)]:
        code=m.probe_code(i,{} if i is None else m.expected_sources(i,host(i)))
        ast.parse(code)
        assert "if not n.endswith('.events.bin')" in code
        assert 'RLIMIT_AS,(512*2**20,512*2**20)' in code
        assert 'raw_recording_bytes_read=0' in code
        if i is None:assert '/data/brick2' in code and str(m.MAX_ALL) in code


def test_sender_and_receiver_guard_commands_have_finite_limits():
    for receiver in [False,True]:
        base=m.BUILD if receiver else m.BASE
        c=m.guard_command(base,'fixture',610,['python3','fixture.py'],receiver)
        assert '--property=MemoryMax=4096M' in c and '--property=MemorySwapMax=0' in c
        assert '--property=CPUQuota=200%' in c and '--property=AllowedCPUs=8-9' in c
        assert '--property=RuntimeMaxSec=615' in c and c[c.index('--timeout')+1]=='610'
        assert c[c.index('--file-mib')+1]=='3072'
        assert c[c.index('--min-free-gib')+1]==('1280' if receiver else '128')
        assert ('--allow-root-volume' in c)==(not receiver)


def test_missing_controls_do_not_read_remote_or_create_output(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'remote',lambda *a,**k:pytest.fail('unexpected remote access'))
    out=tmp_path/'out'
    result=m.run(tmp_path/'absent',tmp_path/'guard',out)
    assert not result['ready'] and not result['collection_started'] and not out.exists()


def test_changed_control_hash_fails_before_remote(tmp_path,monkeypatch):
    names=['admission.json','launch.json','parameters.json','layout.json','stage.json','report.json',
           'input-sha256.json','collection-controller.json']+['host-'+str(i)+'.json' for i in range(6)]
    for name in names:(tmp_path/name).write_text('{}')
    guard=tmp_path/'outer.json';guard.write_text('{}')
    monkeypatch.setattr(m,'remote',lambda *a,**k:pytest.fail('unexpected remote access'))
    with pytest.raises(ValueError,match='hashes incomplete or changed'):m.run(tmp_path,guard,tmp_path/'output')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('recovered,reference_seed',[(False,None),(True,None),(False,1730)])
def test_complete_control_gate_reproduces_resource_decision_and_binds_outer_guard(tmp_path,monkeypatch,recovered,reference_seed):
    # Use the already-tested full-coverage synthetic terminal-control fixture.
    # Only the raw-byte pins change for this generated fixture, never production.
    from test_mam_native_primary_resources import inputs as terminal_inputs
    a,l,hosts,parameters,layout=terminal_inputs.__wrapped__()
    for i,h in enumerate(hosts):h['source_files']=host(i)['source_files']
    identity={}
    if reference_seed is not None:
        from mam_launch_native_full_reference import CAMPAIGN, PROTOCOL_SHA
        from mam_launch_native_primary import launch_options
        label,_,_,identity=m.reference_identity(reference_seed)
        a.update(label=label,seed=reference_seed,campaign=CAMPAIGN,protocol_sha256=PROTOCOL_SHA,
                 source_project=m.PROJECT,source_project_is_retained_primary=True)
        a['launch_options']=launch_options(Path(a['launch_options']['output']),**identity)
        a['launch_options']['output']=str(a['launch_options']['output'])
        for h in hosts:
            for row in h['rank_reports'].values():row['seed']=reference_seed
            h['source_files']={n.replace('/'+m.LABEL+'/', '/'+label+'/'):v for n,v in h['source_files'].items()}
    decision=m.audit(a,l,hosts,parameters,layout,**identity)
    values=dict(zip(['admission.json','launch.json','parameters.json','layout.json','stage.json','report.json'],
                    [a,l,parameters,layout,[],decision]))
    values.update({'host-'+str(i)+'.json':h for i,h in enumerate(hosts)})
    hashes={}
    for name,value in values.items():
        raw=json.dumps(value).encode();(tmp_path/name).write_bytes(raw);hashes[name]=hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(m,'PARAMETERS',hashes['parameters.json'])
    monkeypatch.setattr(m,'LAYOUT_SHA',hashes['layout.json'])
    (tmp_path/'input-sha256.json').write_text(json.dumps(hashes))
    timing=dict(ready=True,terminal_resource_audit_passed=True,
        shared_collection_budget_seconds=7200,elapsed_seconds=10.,automatic_retry=False)
    if reference_seed is not None:
        timing.update(label=label,seed=reference_seed,protocol_sha256=PROTOCOL_SHA,source_project=m.PROJECT)
    (tmp_path/'collection-controller.json').write_text(json.dumps(timing))
    guard=dict(returncode=0,stop_reason=None,sampled_rss_limit_bytes=1536*2**20,sampled_peak_rss_bytes=100*2**20,
        kernel_cpu_limit_seconds=120,kernel_file_limit_bytes=64*2**20,wall_seconds=12.,wall_limit_seconds=180,
        command=['mam_collect_native_primary_resources.py','--output',str(tmp_path)])
    if reference_seed is not None:guard['command']+=['--reference-seed',str(reference_seed)]
    gp=tmp_path/'outer.json';gp.write_text(json.dumps(guard))
    result=m.control_gate(tmp_path,gp,reference_seed=reference_seed)
    assert result['previous_seconds']==12. and result['resource_report_sha256']==hashes['report.json']
    assert len(result['hosts'])==6
    if reference_seed is not None:
        with pytest.raises(ValueError):m.control_gate(tmp_path,gp)
        guard['command'][-1]='1731';gp.write_text(json.dumps(guard))
        with pytest.raises(ValueError,match='outer guard'):m.control_gate(tmp_path,gp,reference_seed=reference_seed)
        guard['command'][-1]=str(reference_seed);gp.write_text(json.dumps(guard))
    if recovered:
        prior=tmp_path/'prior';prior.mkdir();prior_guard=prior/'guard.json'
        (prior/'failure.json').write_text(json.dumps(dict(elapsed_seconds=44.)))
        prior_guard.write_text(json.dumps(dict(returncode=1,stop_reason=None,wall_seconds=45.)))
        recovery=dict(prior_attempt=str(prior),prior_guard=str(prior_guard),recovery_attempts=1,automatic_retry=False,
            failed_attempt_seconds=45.,diagnosis_seconds=800.,charged_prior_seconds=845.,
            failure_sha256=hashlib.sha256((prior/'failure.json').read_bytes()).hexdigest(),
            guard_sha256=hashlib.sha256(prior_guard.read_bytes()).hexdigest())
        (tmp_path/'recovery.json').write_text(json.dumps(recovery))
        timing=dict(ready=True,terminal_resource_audit_passed=True,shared_collection_budget_seconds=7200,
            elapsed_seconds=855.,stage_elapsed_seconds=10.,automatic_retry=False,recovery=recovery)
        (tmp_path/'collection-controller.json').write_text(json.dumps(timing))
        guard['command']+=['--prior-attempt',str(prior),'--prior-guard',str(prior_guard)]
        gp.write_text(json.dumps(guard))
        assert m.control_gate(tmp_path,gp)['previous_seconds']==857.
        timing['elapsed_seconds']=10.;(tmp_path/'collection-controller.json').write_text(json.dumps(timing))
        with pytest.raises(ValueError,match='budget excludes'):m.control_gate(tmp_path,gp)
        timing['elapsed_seconds']=855.;(tmp_path/'collection-controller.json').write_text(json.dumps(timing))
    guard['command'][guard['command'].index('--output')+1]=str(tmp_path/'different-collection');gp.write_text(json.dumps(guard))
    with pytest.raises(ValueError,match='not bound'):m.control_gate(tmp_path,gp,reference_seed=reference_seed)


@pytest.mark.parametrize('seed',[1730,1731])
def test_reference_catalogs_keep_primary_source_and_distinct_run_paths(seed):
    label,tag,destination,identity=m.reference_identity(seed)
    assert identity==dict(label=label,seed=seed)
    assert label!=m.LABEL and tag!=m.TAG and destination==m.BUILD+'/'+label+'-audit'
    files=total=0
    for i in range(6):
        h=host(i)
        h['source_files']={n.replace('/'+m.LABEL+'/', '/'+label+'/'):v for n,v in h['source_files'].items()}
        p={n.replace('/'+m.LABEL+'/', '/'+label+'/'):v for n,v in progress(i).items()}
        c=m.make_catalog(i,h,p,run_label=label)
        size,_=m.validate(c,m.MAX_FILE,m.MAX_HOST);total+=size;files+=len(c['files'])
        paths={x['path'] for x in c['files']}
        assert m.LABEL+'/mam_nest_reference.py' in paths
        assert all('/'+m.LABEL+'/' not in n for n in paths)
        assert sum(n.startswith('runs/'+label+'/') for n in paths)==24
        code=m.probe_code(i,m.expected_sources(i,h,run_label=label),run_label=label)
        ast.parse(code)
        assert repr(label) in code and 'raw_recording_bytes_read=0' in code
        with pytest.raises(ValueError):m.make_catalog(i,h,p)
    assert files==235 and 96*2**30<total<m.MAX_ALL
    for receiver in [False,True]:
        base=m.BUILD if receiver else m.BASE
        cmd=m.guard_command(base,tag+'-node25-'+('receive' if receiver else 'send'),610,['python3','fixture.py'],receiver,tag=tag)
        assert base+'/'+tag+'-source/mpi_resource_guard.py' in cmd
        assert all(m.TAG+'-source' not in x for x in cmd)
        assert '--property=MemoryMax=4096M' in cmd and '--property=CPUQuota=200%' in cmd
        assert cmd[cmd.index('--min-free-gib')+1]==('1280' if receiver else '128')


@pytest.mark.parametrize('seed',[1729,1750,1754,True])
def test_unadmitted_reference_transfer_never_contacts_remote(tmp_path,monkeypatch,seed):
    monkeypatch.setattr(m,'remote',lambda *a,**k:pytest.fail('remote before seed admission'))
    with pytest.raises(ValueError):m.run(tmp_path/'absent',tmp_path/'guard',tmp_path/'output',reference_seed=seed)
    assert list(tmp_path.iterdir())==[]
