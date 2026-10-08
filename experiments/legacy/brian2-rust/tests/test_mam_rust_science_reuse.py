import ast,copy,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import mam_rust_science_reuse as reuse


@pytest.fixture(scope='module')
def old_plan():
    return reuse.old_artifacts(ROOT/'mpi-evidence',Path('/atlas-storage/0002/brian2-mpi-20260908/artifacts'))


@pytest.fixture
def raw_pair(old_plan):
    old=copy.deepcopy(old_plan['old_raw'])
    new=dict(raw_output_audit_passed=True,audit_guard_passed=True,complete_binary_scan_passed=True,
        model_sha256=old['model_sha256'],plan_sha256=old['plan_sha256'],executable_sha256=old['executable_sha256'],
        dump_sha256=copy.deepcopy(old['dump_sha256']),
        dump_bytes={'results.bin':old['dump_bytes']['results_bytes'],'events.bin':old['dump_bytes']['events_bytes']},
        spikes=old['spikes'],delivered_edges=old['delivered_edges'])
    return new,old


def test_complete_raw_files_and_model_allow_identity_check(raw_pair):
    assert reuse.match_raw(*raw_pair)


@pytest.mark.parametrize('change',[
    lambda r:r['dump_sha256'].update({'events.bin':'0'*64}),
    lambda r:r['dump_sha256'].update({'results.bin':'0'*64}),
    lambda r:r['dump_bytes'].update({'events.bin':8}),
    lambda r:r.update(spikes=r['spikes']-1),
    lambda r:r.update(delivered_edges=r['delivered_edges']-1),
])
def test_any_raw_difference_requires_fresh_analysis(raw_pair,change):
    change(raw_pair[0]);assert reuse.match_raw(*raw_pair) is False


@pytest.mark.parametrize('change',[
    lambda r:r.update(raw_output_audit_passed=False),
    lambda r:r.update(audit_guard_passed=False),
    lambda r:r.update(complete_binary_scan_passed=False),
    lambda r:r.update(model_sha256='0'*64),
    lambda r:r.update(plan_sha256='0'*64),
    lambda r:r['dump_sha256'].pop('results.bin'),
])
def test_unaccepted_or_wrong_model_cannot_be_reused(raw_pair,change):
    change(raw_pair[0])
    with pytest.raises(ValueError):reuse.match_raw(*raw_pair)


def test_old_six_metrics_have_complete_bounded_provenance(old_plan):
    assert set(old_plan['metric_catalogs'])=={'activity','cell','correlation','series','fc','lags'}
    assert len(old_plan['manifest'])==67
    assert sum(x['bytes'] or 0 for x in old_plan['manifest'].values())<reuse.MAX_REHASH_BYTES
    assert all(not p.endswith('/results.bin') and not p.endswith('/events.bin') for p in old_plan['manifest'])
    assert any(p.endswith('/rust-lags/lags.npz') for p in old_plan['manifest'])
    ast.parse(reuse.remote_code(old_plan['manifest']))


def test_live_target_does_not_read_old_metrics_or_start_rehash(tmp_path,monkeypatch):
    monkeypatch.setattr(reuse,'old_artifacts',lambda *a:pytest.fail('old artifacts before new raw gate'))
    monkeypatch.setattr(reuse.subprocess,'run',lambda *a,**k:pytest.fail('remote before new raw gate'))
    result=reuse.run(tmp_path,tmp_path/'t7')
    assert not result['ready'] and not result['remote_rehash_started']
    assert not result['descriptive_metrics_reuse_accepted'] and list(tmp_path.iterdir())==[]


def test_mismatch_selects_fresh_analysis_without_remote_or_output(tmp_path,monkeypatch,raw_pair,old_plan):
    raw_pair[0]['dump_sha256']['events.bin']='0'*64
    monkeypatch.setattr(reuse,'accepted_new_raw',lambda *a:raw_pair[0])
    monkeypatch.setattr(reuse,'old_artifacts',lambda *a:copy.deepcopy(old_plan))
    monkeypatch.setattr(reuse.subprocess,'run',lambda *a,**k:pytest.fail('remote on mismatch'))
    result=reuse.run(tmp_path,tmp_path/'t7','a'*64)
    assert result['fresh_analysis_required'] and not result['descriptive_metrics_reuse_accepted']
    assert not result['remote_rehash_started'] and list(tmp_path.iterdir())==[]


def test_completion_requires_explicit_exact_pin_before_reading_outputs(tmp_path):
    case=tmp_path/'performance-runs-v1'/reuse.CASE;case.mkdir(parents=True)
    (case/'completion.json').write_text('{}')
    with pytest.raises(ValueError,match='pin'):reuse.accepted_new_raw(tmp_path,None)


def test_new_raw_audit_must_use_approved_verifier_and_readers():
    from mam_launch_rust_benchmark_raw import RECOVERY_VALIDATION_PATH
    proof=json.loads((ROOT/'mpi-evidence'/RECOVERY_VALIDATION_PATH).read_text())
    catalog={name:item for name,item in proof['files'].items() if name.startswith('tools/')}
    catalog.update({name:dict(sha256=digest) for name,digest in proof['unchanged_reader_and_control_dependencies'].items()})
    reuse.validate_raw_source_catalog(ROOT/'mpi-evidence',catalog)
    catalog['python/brian2_rust/results.py']['sha256']='0'*64
    with pytest.raises(ValueError,match='unapproved'):reuse.validate_raw_source_catalog(ROOT/'mpi-evidence',catalog)
