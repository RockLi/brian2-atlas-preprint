"""Reject incomplete/failed primary evidence before producing cost accounting."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('resources', ROOT/'tools/mam_primary_resources.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def records():
    # Genuine admission and live identity/limits; terminal data below is synthetic.
    evidence = ROOT/'mpi-evidence/primary-run'
    admission = json.loads((evidence/'admission.json').read_text())
    snapshot = json.loads((evidence/'snapshot-20260909T164350Z.json').read_text())
    nodes = []
    for i, source in enumerate(snapshot):
        guards = {}
        for original in source['guards']:
            g = copy.deepcopy(original)
            role = g['cgroup'].split(m.PREFIX + '-')[1].removesuffix('.service')
            g.update(returncode=0, wall_seconds=3600.,
                     minimum_observed_free_bytes=200*m.GIB)
            g['after'] = dict(g['before'], **g.pop('live'))
            g['after']['cpu.stat'] = 'usage_usec 3601000000\nuser_usec 3000000000\nsystem_usec 601000000'
            g['before']['cpu.stat'] = 'usage_usec 1000000\nuser_usec 600000\nsystem_usec 400000'
            guards[role] = g
        times = {str(r): '\n'.join([
            'User time (seconds): 80.00', 'System time (seconds): 20.00',
            'Elapsed (wall clock) time (h:mm:ss or m:ss): 1:00:00',
            'Maximum resident set size (kbytes): 1048576', 'Exit status: 0', ''])
            for r in range(i*8, (i+1)*8)}
        nodes.append(dict(host=source['host'], guards=guards, rank_time=times))
    launch = dict(error=None, returncodes={r: 0 for n in nodes for r in n['guards']},
                  nodes=m.NODES, ranks_per_node=8, resource_guard={'unit_prefix': m.PREFIX},
                  wall_seconds=3600.)
    runtime = dict(ranks=32, plan_sha256=m.PLAN, exchange_calls_per_rank=1005000,
                   processor_names=[n for n in m.NODES for _ in range(8)],
                   rank_cpu_ids=list(reversed(m.CPUS))*4)
    return admission, launch, nodes, runtime


def test_accounting_uses_counter_deltas_and_preserves_scope(records):
    report = m.audit(*records)
    cost = report['accounting']
    assert cost['participating_node_hours'] == 4
    assert cost['worker_allowed_capacity_core_hours'] == 32
    assert cost['measured_cgroup_core_hours_including_controller'] == 5
    assert cost['measured_rank_user_system_core_hours'] == pytest.approx(3200/3600)
    assert cost['exclusive_allocated_node_hours'] is cost['monetary_cost'] is None
    assert len(report['guards']) == 5 and len(report['ranks']) == 32
    assert report['terminal_resource_audit_passed']
    assert not any(report[k] for k in ['raw_output_audit_passed', 'scientific_acceptance',
                                      'performance_cost_acceptance'])


@pytest.mark.parametrize('fault', [
    'live-launch', 'failed-launch', 'missing-proxy-exit', 'wrong-job',
    'missing-host', 'duplicate-host', 'missing-guard', 'stale-guard', 'live-guard',
    'failed-guard', 'missing-after', 'oom', 'max-pressure', 'cpu-limit', 'memory-limit',
    'swap', 'disk-low', 'root-on-23', 'counter-regression', 'nan-wall', 'wall-overrun',
    'missing-rank', 'failed-rank', 'truncated-rank', 'duplicate-time-field',
    'wrong-runtime-host', 'duplicate-cpu', 'short-duration', 'wrong-model',
    'bad-original-admission'])
def test_corrupted_or_incomplete_records_cannot_pass(records, fault):
    admission, launch, nodes, runtime = records
    g = nodes[0]['guards']['proxy-0']
    if fault == 'live-launch': launch['returncodes']['controller'] = None
    elif fault == 'failed-launch': launch['returncodes']['proxy-3'] = 1
    elif fault == 'missing-proxy-exit': del launch['returncodes']['proxy-3']
    elif fault == 'wrong-job': launch['resource_guard']['unit_prefix'] += '-stale'
    elif fault == 'missing-host': nodes.pop()
    elif fault == 'duplicate-host': nodes[3] = copy.deepcopy(nodes[2])
    elif fault == 'missing-guard': del nodes[0]['guards']['controller']
    elif fault == 'stale-guard': g['cgroup'] = g['cgroup'].replace(m.PREFIX, 'b2mpi-old')
    elif fault == 'live-guard': del g['returncode']
    elif fault == 'failed-guard': g['returncode'] = 137
    elif fault == 'missing-after': del g['after']
    elif fault == 'oom': g['after']['memory.events'] = g['after']['memory.events'].replace('oom 0', 'oom 1')
    elif fault == 'max-pressure': g['after']['memory.events'] = g['after']['memory.events'].replace('max 0', 'max 1')
    elif fault == 'cpu-limit': g['after']['cpu.max'] = 'max 100000'
    elif fault == 'memory-limit': g['after']['memory.max'] = str(512*m.GIB)
    elif fault == 'swap': g['after']['memory.swap.max'] = 'max'
    elif fault == 'disk-low': g['minimum_observed_free_bytes'] = 128*m.GIB-1
    elif fault == 'root-on-23': g['data_device'] = g['root_device']
    elif fault == 'counter-regression': g['after']['cpu.stat'] = 'usage_usec 0\nuser_usec 0\nsystem_usec 0'
    elif fault == 'nan-wall': launch['wall_seconds'] = float('nan')
    elif fault == 'wall-overrun': g['wall_seconds'] = 64801
    elif fault == 'missing-rank': del nodes[3]['rank_time']['31']
    elif fault == 'failed-rank': nodes[3]['rank_time']['31'] = nodes[3]['rank_time']['31'].replace('status: 0', 'status: 1')
    elif fault == 'truncated-rank': nodes[3]['rank_time']['31'] = 'User time (seconds): 80.00'
    elif fault == 'duplicate-time-field': nodes[3]['rank_time']['31'] += 'Exit status: 0\n'
    elif fault == 'wrong-runtime-host': runtime['processor_names'][31] = m.NODES[0]
    elif fault == 'duplicate-cpu': runtime['rank_cpu_ids'][31] = runtime['rank_cpu_ids'][30]
    elif fault == 'short-duration': runtime['exchange_calls_per_rank'] = 105000
    elif fault == 'wrong-model': admission['model_sha256'] = '0'*64
    elif fault == 'bad-original-admission': admission['preflight'][0]['memory']['MemAvailable'] = 500*m.GIB
    with pytest.raises((ValueError, KeyError)):
        m.audit(admission, launch, nodes, runtime)


def test_genuine_live_snapshot_is_not_terminal_evidence(records):
    live = json.loads((ROOT/'mpi-evidence/primary-run/snapshot-20260909T164350Z.json').read_text())
    records[2][0]['guards']['proxy-0'] = next(g for g in live[0]['guards'] if '-proxy-0.' in g['cgroup'])
    with pytest.raises(ValueError, match='not terminal'):
        m.audit(*records)


@pytest.mark.parametrize('elapsed,seconds', [('1:02.50', 62.5), ('12:34:56', 45296)])
def test_gnu_elapsed_formats(records, elapsed, seconds):
    text = records[2][0]['rank_time']['0'].replace('1:00:00', elapsed)
    assert m.time_record(text)['elapsed_seconds'] == seconds


def test_cli_failure_never_creates_report(records, tmp_path, monkeypatch):
    import sys
    args = ['audit']
    for name, value in zip(['admission', 'launch', 'collected', 'runtime'], records):
        if name == 'launch': value['returncodes']['controller'] = None
        path = tmp_path/(name + '.json')
        path.write_text(json.dumps(value))
        args += ['--' + name, str(path)]
    output = tmp_path/'report.json'
    monkeypatch.setattr(sys, 'argv', args + ['--output', str(output)])
    with pytest.raises(ValueError, match='not terminal'):
        m.main()
    assert not output.exists()
