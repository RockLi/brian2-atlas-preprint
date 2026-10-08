"""Only new terminal-to-raw input gates; unchanged binary readers are not rerun."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_rust_benchmark_raw as raw
from mam_rust_benchmark_terminal import audit
from test_mam_rust_benchmark_terminal import controls


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bound(tmp_path,controls):
    case=tmp_path/'case';(case/'terminal').mkdir(parents=True)
    paths={}
    for name,value in [('admission.json',controls[0]),('launch.json',controls[1])]:
        p=case/name;p.write_text(json.dumps(value));paths[name]=sha(p)
    root=Path(__file__).resolve().parents[1]
    (case/'package.json').write_bytes((root/'mpi-evidence/primary-run/package.json').read_bytes())
    paths['package.json']=sha(case/'package.json')
    files=[]
    for index,host in enumerate(controls[2]):
        p=case/'terminal'/('host-'+str(index)+'.json.gz')
        p.write_bytes(gzip.compress(json.dumps(host).encode(),mtime=0))
        paths['terminal/'+p.name]=sha(p)
        files.append(dict(host=host['host'],file=p.name,bytes=p.stat().st_size,sha256=sha(p)))
    report=audit(*controls)
    report.update(input_sha256=dict(admission=paths['admission.json'],launch=paths['launch.json'],package=raw.PACKAGE_SHA),
        verifier_sha256=sha(root/'tools/mam_rust_benchmark_terminal.py'),collection_files=files,
        raw_binary_payloads_collected=False,automatic_retry=False)
    (case/'terminal/report.json').write_text(json.dumps(report));paths['terminal/report.json']=sha(case/'terminal/report.json')
    binding=tmp_path/'binding.json'
    binding.write_text(json.dumps(dict(schema='b2-mam-rust-raw-input-binding-v1',case_id=raw.CASE,
        protocol_sha256=raw.PROTOCOL_SHA,files=paths)))
    return case,binding,sha(binding)


def rewrite(path,modify):
    v=json.loads(path.read_text());modify(v);path.write_text(json.dumps(v))


def test_terminal_decision_is_reproduced_from_all_bound_hosts(bound):
    binding,package,hosts,report=raw.terminal_gate(*bound)
    assert len(hosts)==4 and len(report['ranks'])==32
    assert report['terminal_resource_audit_passed'] and not report['raw_output_audit_passed']
    assert binding['files']['package.json']==raw.PACKAGE_SHA and len(package['catalog'])==43


@pytest.mark.parametrize('name',['admission.json','launch.json','terminal/report.json','terminal/host-3.json.gz'])
def test_any_control_byte_change_rejected(bound,name):
    case,binding,digest=bound
    with (case/name).open('ab') as f:f.write(b' ')
    with pytest.raises(ValueError,match='hash'):raw.terminal_gate(*bound)


def test_missing_host_binding_is_not_partial_acceptance(bound):
    case,binding,digest=bound
    rewrite(binding,lambda b:b['files'].pop('terminal/host-3.json.gz'))
    with pytest.raises(ValueError,match='coverage'):raw.terminal_gate(case,binding,sha(binding))


def test_forged_success_report_cannot_replace_reproduced_accounting(bound):
    case,binding,digest=bound;p=case/'terminal/report.json'
    rewrite(p,lambda r:r['accounting'].update(measured_cgroup_core_hours=0))
    rewrite(binding,lambda b:b['files'].update({'terminal/report.json':sha(p)}))
    with pytest.raises(ValueError,match='reproducible'):raw.terminal_gate(case,binding,sha(binding))


def test_rebound_nonterminal_launch_still_fails_semantics(bound):
    case,binding,digest=bound;p=case/'launch.json';r=case/'terminal/report.json'
    rewrite(p,lambda v:v['returncodes'].update({'proxy-2':None}))
    rewrite(r,lambda v:v['input_sha256'].update(launch=sha(p)))
    rewrite(binding,lambda v:v['files'].update({'launch.json':sha(p),'terminal/report.json':sha(r)}))
    with pytest.raises(ValueError,match='terminal'):raw.terminal_gate(case,binding,sha(binding))


def test_wrong_verifier_cannot_be_silently_reused(bound):
    case,binding,digest=bound;p=case/'terminal/report.json'
    rewrite(p,lambda v:v.update(verifier_sha256='0'*64))
    rewrite(binding,lambda v:v['files'].update({'terminal/report.json':sha(p)}))
    with pytest.raises(ValueError,match='verifier'):raw.terminal_gate(case,binding,sha(binding))


def test_missing_binding_does_not_touch_run_output_or_transport(tmp_path,monkeypatch):
    output=tmp_path/'new-output'
    monkeypatch.setattr(raw.subprocess,'check_output',lambda *a,**k:pytest.fail('unexpected process observation'))
    with pytest.raises(ValueError,match='control'):raw.run(tmp_path,tmp_path/'missing','0'*64,output)
    assert not output.exists()


def test_wrong_binding_hash_is_rejected_before_opening_case(bound):
    case,binding,digest=bound
    with pytest.raises(ValueError,match='hash'):raw.terminal_gate(case,binding,'0'*64)
