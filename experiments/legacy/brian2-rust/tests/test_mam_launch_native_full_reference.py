"""Admission tests; fixtures are not simulations or scientific observations."""
import ast
import json
import hashlib
import shutil
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import mam_launch_native_full_reference as m


@pytest.mark.parametrize('seed',[1729,1750,1751,1754,1732,True,'1730'])
def test_unadmitted_seeds_refused_before_file_or_network_access(seed,tmp_path,monkeypatch):
    monkeypatch.setattr(m,'remote',lambda *a,**k:pytest.fail('network before seed admission'))
    with pytest.raises(ValueError): m.run(tmp_path,tmp_path/'t7',seed)
    assert list(tmp_path.iterdir())==[]


def test_real_frozen_protocol_pins_sources_and_keeps_confirmation_closed():
    evidence=ROOT/'mpi-evidence'
    p=m.protocol_gate(evidence,1730)
    assert p['condition']['duration_ms']==100500
    assert p['max_new_simulations']==2 and p['automatic_retry'] is False
    assert p['reserved_confirmation_seeds']==list(range(1750,1755))
    assert p['analysis']['equivalence_margins'] is None
    assert p['resources']['two_run_total_ceiling_seconds']==2*(54000+7200+10800)


def test_missing_first_completion_prevents_second_run_without_network(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'protocol_gate',lambda *a:None)
    monkeypatch.setattr(m,'remote',lambda *a,**k:pytest.fail('network before completion gate'))
    result=m.run(tmp_path,tmp_path/'t7',1731)
    assert not result['ready'] and not result['launch_started']
    assert list(tmp_path.iterdir())==[]


def test_node23_checks_all_simulations_and_campaign_storage_reserve():
    code=m.leader_preflight_code();ast.parse(code)
    assert "'b2mpi-*'" in code and 'free>=1792*2**30' in code
    assert "Path('/data/brick2').is_mount()" in code
    assert 'campaign_additional_allowance_bytes=512*2**30' in code
    assert 'runtime_reserve_bytes=1280*2**30' in code


@pytest.mark.parametrize('fault',[None,'core-only','missing-fc','missing-lags','interarea-incomplete'])
def test_second_reference_requires_all_six_scientific_summaries(tmp_path,fault):
    # Real pinned protocol inputs, synthetic completion records. This tests
    # structural admission only, never claims actual simulations completed.
    evidence=tmp_path/'brian2-rust/mpi-evidence';campaign=evidence/m.CAMPAIGN
    campaign.mkdir(parents=True)
    original=ROOT/'mpi-evidence'/m.CAMPAIGN/'protocol.json'
    shutil.copyfile(original,campaign/'protocol.json')
    for row in json.loads(original.read_text())['pinned_inputs']:
        dest=evidence.parent/row['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/row['path'],dest)
    run=campaign/'seed1730';run.mkdir()
    stages=['activity','cell','correlation','series','fc','lags']
    reports={
        'resources':dict(label=m.label_for(1730),terminal_resource_audit_passed=True),
        'raw':dict(label=m.label_for(1730),raw_output_audit_passed=True),
        'analysis':dict(label=m.label_for(1730),analysis_complete=True,seed=1730,protocol_sha256=m.PROTOCOL_SHA,
                        required_stages=stages,catalogs={n:'a'*64 for n in stages},interarea_analysis_complete=True),
    }
    analysis=reports['analysis']
    if fault=='core-only':analysis['required_stages']=stages[:4]
    elif fault=='missing-fc':del analysis['catalogs']['fc']
    elif fault=='missing-lags':del analysis['catalogs']['lags']
    elif fault=='interarea-incomplete':analysis['interarea_analysis_complete']=False
    refs={}
    for name,value in reports.items():
        raw=json.dumps(value).encode();(run/(name+'.json')).write_bytes(raw)
        refs[name]=dict(path=name+'.json',sha256=hashlib.sha256(raw).hexdigest())
    (run/'completion.json').write_text(json.dumps(dict(schema='b2-mam-full-native-diagnostic-completion-v1',
        label=m.label_for(1730),seed=1730,protocol_sha256=m.PROTOCOL_SHA,terminal_resource_audit_passed=True,
        raw_output_audit_passed=True,analysis_complete=True,reports=refs)))
    if fault is None:assert m.protocol_gate(evidence,1731) is not None
    else:
        with pytest.raises(ValueError,match='all six'):m.protocol_gate(evidence,1731)
