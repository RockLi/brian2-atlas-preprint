"""Bounded synthetic publication and source-package integration tests."""
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_launch_confirmation_raw as launch
from mam_confirmation_terminal import audit
from test_mam_confirmation_terminal import controls
from test_mam_confirmation_raw import bound


def test_missing_terminal_prerequisite_has_no_network_or_output(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'remote',lambda *a,**k:pytest.fail('unexpected network'))
    result=launch.run(tmp_path/'evidence',tmp_path/'t7')
    assert result['ready'] is False and result['raw_audit_started'] is False
    assert list(tmp_path.iterdir())==[]


def test_real_source_bundle_contains_verified_controls_and_bounded_remote_code(bound):
    root=Path(__file__).resolve().parents[1];case,binding,digest=bound
    payload,catalog,new_binding,new_sha,terminal=launch.prepare(root,case)
    assert terminal['terminal_resource_audit_passed'] and new_binding['files']==json.loads(binding.read_text())['files']
    with tarfile.open(fileobj=io.BytesIO(payload)) as t:
        assert set(t.getnames())==set(catalog)|{'catalog.json'}
        for name,item in catalog.items():
            raw=t.extractfile(name).read()
            assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        assert hashlib.sha256(t.extractfile('binding.json').read()).hexdigest()==new_sha
    assert 'python/brian2_rust/results.py' in catalog and 'tools/mpi_resource_guard.py' in catalog
    for code in [launch.preflight_code(),launch.stage_code(hashlib.sha256(payload).hexdigest(),catalog),launch.collect_code(catalog)]:
        ast.parse(code);assert len(code.encode())<100000
    cmd=launch.command(new_sha)
    for required in ['--property=MemoryMax=16384M','--property=MemorySwapMax=0','--property=CPUQuota=200%',
                     '--property=AllowedCPUs=8-9','--property=TasksMax=64','--property=RuntimeMaxSec=6605']:
        assert required in cmd
    assert cmd[cmd.index('--min-free-gib')+1]=='1280' and cmd[cmd.index('--timeout')+1]=='6600'
    assert cmd[cmd.index('--')+1:]==launch.application(new_sha)


@pytest.fixture
def publication(controls):
    terminal=audit(*controls);digest='b'*64;source='a'*64
    g=copy.deepcopy(controls[4][0]['guards']['proxy-0'])
    g.update(cgroup='/system.slice/'+launch.UNIT+'.service',command=launch.application(digest),
        root_volume_allowed=False,file_limit_bytes=512*2**20)
    for phase in ['before','after']:
        g[phase].update({'memory.max':str(16*2**30),'cpuset.cpus.effective':'8-9','cpu.max':'200000 100000'})
    pending=dict(schema='b2-mam-confirmation-raw-pending-v1',replicate=1750,identity_sha256=launch.IDENTITY_SHA,case_id=launch.CASE,label=launch.LABEL,
        protocol_sha256=controls[2]['protocol_sha256'],binding_sha256=digest,model_sha256=controls[0]['model_sha256'],
        plan_sha256=controls[0]['plan_sha256'],executable_sha256=controls[0]['executable_sha256'],implementation_sha256=source,
        complete_binary_scan_passed=True,terminal_resource_audit_passed=True,raw_output_audit_passed=False,
        audit_guard_passed=False,pending_guard_acceptance=True,scientific_acceptance=False,
        nest_statistical_acceptance=False,performance_cost_acceptance=False,reused_scientific_summaries=False,
        prior_raw_identity_assumed=False,spikes=32,delivered_edges=64,
        dump_bytes={'results.bin':256,'events.bin':256},dump_sha256={'results.bin':'c'*64,'events.bin':'d'*64},
        work=dict(all_result_and_event_records_equal=True,all_projection_edge_and_csr_ownership_exact=True,
            rank_neuron_and_spike_work_exact=True,local_delivery_work_and_shared_aggregate_exact=True,
            shared_delivery_rank_histories_independently_verified=False,
            exact_local_edges=[750000000]*31+[24126516728-750000000*31],population_profile=[{}]*254),
        elapsed_seconds=850.)
    controller=dict(returncode=0,error=None,command=launch.command(digest))
    kwargs=dict(binding_sha=digest,terminal=terminal,identity=controls[0],protocol_sha=controls[2]['protocol_sha256'],raw_source_sha=source,elapsed=1000.)
    return pending,g,controller,kwargs


def test_publication_requires_guard_and_keeps_science_unaccepted(publication):
    p,g,c,kw=publication;r=launch.publish(p,g,c,**kw)
    assert r['raw_output_audit_passed'] and r['audit_guard_passed'] and not r['pending_guard_acceptance']
    assert not r['scientific_acceptance'] and not r['performance_cost_acceptance']
    assert r['raw_audit_resource_accounting']['measured_cpu_seconds']==5
    assert not r['automatic_retry'] and r['elapsed_through_guard_collection_seconds']==1000.


@pytest.mark.parametrize('mutate',[
    lambda p,g,c,k:c.update(returncode=None,error='observation timeout'),
    lambda p,g,c,k:c['command'].append('different'),
    lambda p,g,c,k:g.update(command=['different']),
    lambda p,g,c,k:g['after'].update({'memory.events':'max 0\noom 1\noom_kill 0\noom_group_kill 0'}),
    lambda p,g,c,k:g.update(minimum_observed_free_bytes=1279*2**30),
    lambda p,g,c,k:p.update(complete_binary_scan_passed=False),
    lambda p,g,c,k:p.update(spikes=31),
    lambda p,g,c,k:p.update(replicate=1729),
    lambda p,g,c,k:p.update(identity_sha256='0'*64),
    lambda p,g,c,k:p.update(plan_sha256='0'*64),
    lambda p,g,c,k:p.update(implementation_sha256='0'*64),
    lambda p,g,c,k:p.update(scientific_acceptance=True),
    lambda p,g,c,k:p['work'].update(all_result_and_event_records_equal=False),
    lambda p,g,c,k:p['dump_sha256'].pop('events.bin'),
    lambda p,g,c,k:k.update(elapsed=7201.),
])
def test_failed_incomplete_or_mismatched_evidence_cannot_publish(publication,mutate):
    p,g,c,k=publication;mutate(p,g,c,k)
    with pytest.raises(ValueError):launch.publish(p,g,c,**k)
