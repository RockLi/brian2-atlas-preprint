"""Offline completion-link tests; synthetic records are not simulation evidence."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import mam_complete_native_reference as m


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    seed = 1730
    identity = dict(label=m.label_for(seed), seed=seed, protocol_sha256=m.PROTOCOL_SHA)
    gate = dict(resource_report_sha256='resource-digest', collection_report_sha256='collection-digest')
    calls = []
    # Production guard validation has its own tests. Here isolate cross-report
    # consistency; no synthetic guard is passed off as actual resource proof.
    monkeypatch.setattr(m, 'guard_ok', lambda path, phase, timeout, **kw: calls.append((phase, timeout, kw)))
    for phase, timeout in [('raw', 7200), ('science', 9900)]:
        write(tmp_path/(phase+'-guard.json'), dict(synthetic_fixture=True))
        write(tmp_path/(phase+'-controller.json'), dict(returncode=0, error=None,
            command=m.command(phase, timeout, reference_seed=seed)))
    raw = dict(label=identity['label'], seed=seed, passed=True,
               raw_output_audit_passed=True, audit_guard_passed=True,
               terminal_resource_audit_passed=True, all_first_2500ms_event_prefixes_exact=True,
               audit_guard_sha256=m.sha(tmp_path/'raw-guard.json'), **gate)
    write(tmp_path/'summary.json', raw)
    budget = dict(**identity, summary_sha256=m.sha(tmp_path/'summary.json'),
                  raw_guard_sha256=m.sha(tmp_path/'raw-guard.json'),
                  shared_analysis_budget_seconds=10800, previous_analysis_seconds=900,
                  automatic_retry=False)
    write(tmp_path/'science-budget.json', budget)
    write(tmp_path/'admission.json', dict(**identity, **gate))
    catalogs = {}
    for stage, name in m.REPORTS.items():
        p = tmp_path/stage/name
        write(p, dict(synthetic_fixture=True, stage=stage))
        write(p.parent/'catalog.json', {name:dict(bytes=p.stat().st_size, sha256=m.sha(p))})
        catalogs[stage] = m.sha(p.parent/'catalog.json')
    common = dict(**identity, analysis_complete=True, scientific_acceptance=False,
                  performance_cost_acceptance=False, required_stages=m.STAGES,
                  interarea_analysis_complete=True, raw_summary_sha256=budget['summary_sha256'])
    rows = []
    for stage in m.STAGES:
        rows.extend([dict(stage=stage, state='started', timeout_seconds=100),
                     dict(stage=stage, state='complete', seconds=10)])
    write(tmp_path/'report.json', dict(**common, catalogs=catalogs, stages=rows,
          raw_guard_sha256=budget['raw_guard_sha256'], shared_budget_seconds=10800,
          previous_analysis_seconds=900, science_seconds=60, total_accounted_seconds=960))
    write(tmp_path/'controller-complete.json', dict(**common, ready=True,
          shared_analysis_seconds=10800, elapsed_seconds=980))
    return tmp_path, gate, calls


def test_complete_linkage_checks_both_guards(evidence):
    root, gate, calls = evidence
    required = m.validate_analysis(root, gate, 1730)
    assert 'lags/lags.json' in required and 'science-guard.json' in required
    assert calls == [('raw', 7200, dict(reference_seed=1730)),
                     ('science', 9900, dict(reference_seed=1730))]
    assert not (root/'completion.json').exists()


@pytest.mark.parametrize('file,field,value', [
    ('report.json', 'seed', 1731),
    ('report.json', 'required_stages', m.STAGES[:4]),
    ('report.json', 'interarea_analysis_complete', False),
    ('report.json', 'scientific_acceptance', True),
    ('report.json', 'raw_summary_sha256', 'changed'),
    ('report.json', 'total_accounted_seconds', 10801),
    ('summary.json', 'all_first_2500ms_event_prefixes_exact', False),
    ('summary.json', 'collection_report_sha256', 'other-collection'),
    ('science-controller.json', 'returncode', 1),
    ('science-controller.json', 'error', 'observation timeout'),
    ('science-budget.json', 'automatic_retry', True),
])
def test_failed_or_cross_run_record_cannot_complete(evidence, file, field, value):
    root, gate, _ = evidence
    data = m.read(root/file); data[field] = value; write(root/file, data)
    with pytest.raises(ValueError): m.validate_analysis(root, gate, 1730)


def test_changed_stage_report_cannot_complete(evidence):
    root, gate, _ = evidence
    write(root/'fc/fc.json', dict(replaced=True))
    with pytest.raises(ValueError, match='stage report'): m.validate_analysis(root, gate, 1730)


def test_missing_final_stage_cannot_complete(evidence):
    root, gate, _ = evidence
    report = m.read(root/'report.json'); report['stages'] = report['stages'][:-1]
    write(root/'report.json', report)
    with pytest.raises(ValueError, match='ledger'): m.validate_analysis(root, gate, 1730)


def test_wrong_guard_command_cannot_complete(evidence):
    root, gate, _ = evidence
    p = root/'science-controller.json'; report = m.read(p)
    report['command'][-1] = '1731'; write(p, report)
    with pytest.raises(ValueError, match='controller'): m.validate_analysis(root, gate, 1730)


def test_running_analysis_returns_before_gates_or_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'prerequisites', lambda *a, **k: pytest.fail('premature terminal audit'))
    monkeypatch.setattr(m, 'protocol_gate', lambda *a, **k: pytest.fail('premature protocol read'))
    result = m.run(tmp_path/'evidence', tmp_path/'t7', 1730)
    assert not result['ready'] and not result['completion_published']
    assert not list(tmp_path.iterdir())


def test_reserved_seed_never_creates_completion(tmp_path):
    with pytest.raises(ValueError): m.run(tmp_path, tmp_path, 1750)
    assert not list(tmp_path.iterdir())
