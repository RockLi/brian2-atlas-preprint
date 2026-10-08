import ast,json,shlex,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import mam_launch_rust_recovery as recovery
import mam_rust_benchmark_terminal as terminal
import mam_collect_rust_benchmark_terminal as collector
import mam_rust_benchmark_raw as raw
import mam_launch_rust_benchmark_raw as launcher
from test_mam_rust_benchmark_terminal import controls,edit_done
from test_mam_rust_cpu_placement import alternate_controls
from test_mam_rust_benchmark_raw import bound,sha
from test_mam_launch_rust_benchmark_raw import publication


@pytest.fixture
def corrected_controls(controls):
    alternate_controls(controls);admission,launch,hosts=controls
    admission.update(case_id=recovery.CASE,label=recovery.LABEL,protocol_sha256=recovery.PROTOCOL_SHA,
                     base_protocol_sha256=recovery.old.PROTOCOL_SHA)
    admission['source_catalog']['protocol.json']['sha256']=recovery.PROTOCOL_SHA
    fixed='e66a7f95398696389f749fa8ab60646923b3585f4b9f214cde1afddc2e6f3bfb'
    admission['source_catalog']['mam_rust_terminal_sync.py']['sha256']=fixed
    app=recovery.options(Path('/unused'))['application'];admission['launch_options']['application']=app
    launch['commands']['controller'][-1]=shlex.join(['mpiexec','-n','32',*app])
    for host in hosts:
        for item in host['ranks'].values():
            for name in ['started_json','done_json']:
                row=json.loads(item[name]);row['wrapper_sha256']=fixed;item[name]=json.dumps(row)
    return controls


def test_corrected_target_has_exact_new_collection_paths(corrected_controls):
    admission,launch,hosts=corrected_controls
    report=terminal.audit(*corrected_controls)
    assert report['case_id']==recovery.CASE and report['label']==recovery.LABEL
    assert not report['scientific_acceptance'] and not report['performance_cost_acceptance']
    prefix=collector.launch_gate(admission,launch)
    for index in range(4):
        code=collector.host_code(admission,dict(catalog={},mpi_runtime={}),index,prefix)
        ast.parse(code)
        assert recovery.SOURCE in code and recovery.LABEL in code
        assert recovery.old.LABEL not in code


@pytest.mark.parametrize('mutate',[
    lambda c:c[0].update(label=recovery.old.LABEL),
    lambda c:c[0].update(protocol_sha256=recovery.old.PROTOCOL_SHA),
    lambda c:c[0].update(transport_only=True),
    lambda c:c[0].pop('cpu_placement_revision'),
    lambda c:c[0]['source_catalog']['mam_rust_terminal_sync.py'].update(sha256='0'*64),
])
def test_old_or_mixed_contract_cannot_pass_as_corrected_target(corrected_controls,mutate):
    mutate(corrected_controls)
    with pytest.raises(ValueError):terminal.audit(*corrected_controls)


@pytest.fixture
def corrected_bound(tmp_path,corrected_controls):
    case,binding,digest=bound.__wrapped__(tmp_path,corrected_controls)
    row=json.loads(binding.read_text());row.update(case_id=recovery.CASE,protocol_sha256=recovery.PROTOCOL_SHA)
    binding.write_text(json.dumps(row))
    return case,binding,sha(binding)


def test_new_raw_binding_reproduces_all_terminal_records(corrected_bound):
    binding,package,hosts,report=raw.terminal_gate(*corrected_bound)
    assert binding['case_id']==report['case_id']==recovery.CASE and len(hosts)==4
    assert not report['raw_output_audit_passed']


def test_full_source_package_uses_new_target_binding(corrected_bound):
    case,_,_=corrected_bound
    payload,catalog,binding,digest,report=launcher.prepare(ROOT,case,case/'package.json')
    assert binding['case_id']==recovery.CASE and binding['protocol_sha256']==recovery.PROTOCOL_SHA
    assert report['label']==recovery.LABEL and len(payload)<32*2**20
    assert 'tools/mam_launch_rust_recovery.py' in catalog


def test_relabeling_binding_without_matching_admission_fails(corrected_bound):
    case,binding,_=corrected_bound
    row=json.loads(binding.read_text());row.update(case_id=recovery.old.CASE,protocol_sha256=recovery.old.PROTOCOL_SHA)
    binding.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='admitted target'):raw.terminal_gate(case,binding,sha(binding))


def test_new_raw_publication_remains_engineering_only(publication):
    p,g,c,kw=publication
    p.update(case_id=recovery.CASE,label=recovery.LABEL,protocol_sha256=recovery.PROTOCOL_SHA)
    kw['terminal'].update(case_id=recovery.CASE,label=recovery.LABEL)
    report=launcher.publish(p,g,c,**kw)
    assert report['raw_output_audit_passed'] and not report['scientific_acceptance']
    assert not report['performance_cost_acceptance']


def test_running_target_does_not_start_raw_audit(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher,'remote',lambda *a,**k:pytest.fail('unexpected remote'))
    result=launcher.run(tmp_path,tmp_path/'t7',recovery.CASE)
    assert not result['ready'] and not result['raw_audit_started'] and list(tmp_path.iterdir())==[]
