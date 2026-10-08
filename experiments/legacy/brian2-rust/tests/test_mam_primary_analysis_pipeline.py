"""Test orchestration failure boundaries without reading or running a model."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'tools'))
    import mam_primary_analysis_pipeline as pipeline
    import mam_launch_primary_analysis as launch
    return pipeline, launch


def test_missing_terminal_never_reads_raw_or_contacts_remote(modules, monkeypatch, tmp_path):
    _, launch = modules
    monkeypatch.setattr(launch, 'remote', lambda *a, **k: pytest.fail('remote before terminal'))
    monkeypatch.setattr(launch, 'bundle', lambda *a, **k: pytest.fail('source stage before terminal'))
    result = launch.run(tmp_path/'evidence', tmp_path/'t7', tmp_path/'norm', tmp_path/'generated')
    assert result['ready'] is False and result['analysis_started'] is False
    assert not list(tmp_path.iterdir())


def test_one_deadline_includes_validation_and_reduces_next_timeout(modules, tmp_path):
    pipeline, _ = modules
    now = [0.]
    timeouts = []
    def runner(command, **kwargs):
        timeouts.append(kwargs['timeout']);now[0] += 6000 if len(timeouts)==1 else 100
        return SimpleNamespace(returncode=0)
    def validate(name):
        now[0] += 200
    rows = pipeline.execute_sequence([('audit',7200,['first']), ('activity',6000,['second'])],
        tmp_path, started=0, clock=lambda: now[0], runner=runner, validate=validate)
    assert timeouts == [7200,4600]
    assert [r['state'] for r in rows] == ['started','complete','started','complete']


@pytest.mark.parametrize('failure', ['child', 'validation', 'timeout'])
def test_failure_stops_chain_and_retains_attempt_marker(modules, tmp_path, failure):
    pipeline, _ = modules
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        if failure == 'child': raise subprocess.CalledProcessError(3, command)
        if failure == 'timeout': raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        return SimpleNamespace(returncode=0)
    def validate(name):
        raise ValueError('raw audit did not pass')
    with pytest.raises((ValueError,subprocess.SubprocessError)):
        pipeline.execute_sequence([('audit',10,['first']), ('activity',10,['forbidden'])],
            tmp_path, started=0, clock=lambda: 0, runner=runner, validate=validate)
    assert calls == [['first']]
    rows = list(map(json.loads,(tmp_path/'stages.jsonl').read_text().splitlines()))
    assert rows[-1]['state'] == 'failed'
    with pytest.raises(FileExistsError):
        pipeline.execute_sequence([],tmp_path,started=0)


def test_expired_budget_never_starts_child(modules, tmp_path):
    pipeline, _ = modules
    with pytest.raises(TimeoutError,match='shared analysis'):
        pipeline.execute_sequence([('audit',7200,['forbidden'])],tmp_path,started=0,
            clock=lambda:10801,runner=lambda *a,**k:pytest.fail('child after deadline'))
    assert json.loads((tmp_path/'stages.jsonl').read_text())['state']=='budget_exhausted'


def test_actual_subprocess_failure_does_not_run_later_stage(modules, tmp_path):
    pipeline, _ = modules
    import time
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.execute_sequence([
            ('success',5,[sys.executable,'-c','print("first stage completed")']),
            ('failure',5,[sys.executable,'-c','raise SystemExit(7)']),
            ('forbidden',5,[sys.executable,'-c','raise AssertionError("must never run")'])],
            tmp_path,started=time.monotonic())
    assert 'first stage completed' in (tmp_path/'success.log').read_text()
    assert not (tmp_path/'forbidden.log').exists()


def test_generated_remote_contract_and_shared_outer_guard(modules):
    pipeline, launch = modules
    ast.parse(launch.preflight_code('0'*64));ast.parse(launch.collect_code())
    command = launch.command()
    assert '--property=AllowedCPUs=8-9' in command
    assert '--property=MemoryMax=16384M' in command
    assert command[command.index('--timeout')+1] == str(pipeline.TOTAL_SECONDS)
    assert '--property=RuntimeMaxSec=10805' in command
    assert all('mpiexec' not in value and 'b2-mpi ' not in value for value in command)
    plan = pipeline.stages(launch.SOURCE,launch.SOURCE/'resources',launch.SOURCE/launch.NORMALIZATION)
    assert [r[0] for r in plan] == ['output-audit','activity','cell','correlation','series']
    assert sum(r[1] for r in plan) > pipeline.TOTAL_SECONDS  # caps are NOT fresh total budgets


def test_resource_gate_reproduces_and_hashes_inputs(modules, tmp_path):
    pipeline, _ = modules
    spec = importlib.util.spec_from_file_location('resource_fixture',ROOT/'tests/test_mam_primary_resources.py')
    fixture = importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    admission,launch,collected,runtime = fixture.records.__wrapped__()
    inputs = dict(admission=admission,launch=launch,collected=collected,runtime=runtime)
    for name,value in inputs.items(): (tmp_path/(name+'.json')).write_text(json.dumps(value))
    (tmp_path/'report.json').write_text(json.dumps(pipeline.audit_resources(**inputs)))
    (tmp_path/'input-sha256.json').write_text(json.dumps({n:pipeline.sha(tmp_path/(n+'.json')) for n in inputs}))
    assert len(pipeline.resource_gate(tmp_path)) == 6
    with (tmp_path/'runtime.json').open('a') as stream: stream.write('\n')
    with pytest.raises(ValueError,match='input changed'): pipeline.resource_gate(tmp_path)


def test_activity_must_match_the_raw_audit_hashes(modules, tmp_path, monkeypatch):
    pipeline, _ = modules
    output = tmp_path/'output';(output/'activity').mkdir(parents=True)
    monkeypatch.setattr(pipeline,'OUTPUT',output)
    monkeypatch.setattr(pipeline,'AUDIT_OUTPUT',tmp_path/'raw.json')
    baseline = dict(model_sha256=pipeline.MODEL, result_sha256={'results.bin':'different'},
                    window=dict(seconds=100.,spike_tick_offset=1),bounded_memory=True)
    (tmp_path/'raw.json').write_text(json.dumps(dict(dump_sha256={'results.bin':'audited'})))
    (output/'activity/activity.json').write_text(json.dumps(baseline))
    for name in ['activity-arrays.npz','activity-overview.png','spike-rasters.png']:
        (output/'activity'/name).write_bytes(b'fixture')
    catalog = {p.name:dict(bytes=p.stat().st_size,sha256=pipeline.sha(p)) for p in (output/'activity').iterdir()}
    (output/'activity/catalog.json').write_text(json.dumps(catalog))
    with pytest.raises(ValueError,match='not bound to raw audit'): pipeline.validate_output('activity')
