"""Lifecycle tests with prescribed events, never a scientific NEST simulation."""
import ast
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import mam_benchmark_event_io as event_io
import mam_benchmark_recording as recording
import mam_nest_benchmark as producer


class TraceIO:
    fail = None
    def sync_data(self, fd):
        if self.fail == 'sync':
            raise OSError('injected data sync failure')
    def release(self, fd, offset, length):
        pass
    def sync_directory(self, path):
        if self.fail == 'directory':
            raise OSError('injected directory sync failure')


@pytest.fixture(autouse=True)
def io_backend(monkeypatch):
    monkeypatch.setattr(event_io, 'LinuxIO', TraceIO)
    monkeypatch.setattr(recording, 'LinuxIO', TraceIO)
    monkeypatch.setattr(TraceIO, 'fail', None)


class Recorder:
    def __init__(self):
        self.events = dict(times=[], senders=[])
        self.extractions = self.resets = 0
        self.fail_reset = False
    def get(self, key):
        if key == 'n_events':
            return len(self.events['times'])
        assert key == 'events'
        self.extractions += 1
        return self.events
    def set(self, **values):
        assert values == dict(n_events=0)
        self.resets += 1
        if not self.fail_reset:
            self.events = dict(times=[], senders=[])


class FixedEvents:
    def __init__(self, fixtures):
        self.recorder = Recorder()
        self.fixtures = iter(fixtures)
        self.durations = []
    def Simulate(self, duration):
        self.durations.append(duration)
        times, senders = next(self.fixtures)
        self.recorder.events = dict(times=times, senders=senders)


def args_for(tmp_path, **updates):
    args = dict(output=tmp_path, duration_ms=.3, chunk_ms=.2,
                max_spikes_per_rank=8, max_chunk_spikes=8,
                parameters=tmp_path/'unused.json', parameters_sha256='fixed',
                max_neurons=3, max_edges=1, ranks=1, threads=1, seed=1729)
    args.update(updates)
    return SimpleNamespace(**args)


def execute(nest, args):
    return recording.record_chunks(nest, nest.recorder, args, 0, 3, lambda: dict(rss_kib=1))


def test_chunk_tail_exact_native_ticks_ids_and_terminal_receipt(tmp_path):
    nest = FixedEvents([([.1, .2], [1, 3]), ([.3], [2])])
    result = execute(nest, args_for(tmp_path))
    expected = struct.pack('<IIIIII', 1, 0, 2, 2, 3, 1)
    assert (tmp_path/'rank0.events.bin').read_bytes() == expected
    assert result['event_sha256'] == hashlib.sha256(expected).hexdigest()
    assert result['local_spikes'] == 3 and result['event_readback_verified']
    assert result['durable_event_receipt']['durable_bytes'] == 24
    assert nest.durations == pytest.approx([.2, .1])
    assert nest.recorder.resets == nest.recorder.extractions == 2
    assert [x['spikes'] for x in result['chunks']] == [2, 1]
    assert all(x['monitor_progress_seconds'] >= 0 for x in result['chunks'])
    progress = [json.loads(x) for x in (tmp_path/'rank0.progress.jsonl').read_text().splitlines()]
    assert [x['event_bytes'] for x in progress] == [16, 24]
    assert all(x['durable_event_bytes'] == 0 for x in progress)  # no false chunk durability


def test_empty_chunk_is_recorded_and_verified(tmp_path):
    result = execute(FixedEvents([([], [])]), args_for(tmp_path, duration_ms=.2))
    assert result['local_spikes'] == 0 and result['event_readback_verified']
    assert result['event_sha256'] == hashlib.sha256(b'').hexdigest()


@pytest.mark.parametrize('limit', ['chunk', 'total'])
def test_budget_refuses_before_extracting_over_budget_events(tmp_path, limit):
    nest = FixedEvents([([.1], [1]), ([.3, .3], [2, 3])])
    kwargs = {'max_chunk_spikes': 1} if limit == 'chunk' else {'max_spikes_per_rank': 2}
    with pytest.raises(RuntimeError, match='before array extraction'):
        execute(nest, args_for(tmp_path, **kwargs))
    assert nest.recorder.extractions == nest.recorder.resets == 1
    assert (tmp_path/'rank0.events.bin').read_bytes() == struct.pack('<II', 1, 0)


@pytest.mark.parametrize('times,senders', [([-.1], [1]), ([.3], [1]), ([.1], [0]), ([.1], [4])])
def test_invalid_event_refused_before_write(tmp_path, times, senders):
    with pytest.raises(ValueError, match='Invalid recorded event'):
        execute(FixedEvents([(times, senders)]), args_for(tmp_path, duration_ms=.2))
    assert (tmp_path/'rank0.events.bin').read_bytes() == b''


def test_recorder_reset_failure_retains_written_prefix(tmp_path):
    nest = FixedEvents([([.1], [1])])
    nest.recorder.fail_reset = True
    with pytest.raises(RuntimeError, match='did not reset'):
        execute(nest, args_for(tmp_path, duration_ms=.2))
    assert (tmp_path/'rank0.events.bin').read_bytes() == struct.pack('<II', 1, 0)


def test_independent_readback_detects_corruption(tmp_path):
    path = tmp_path/'events.bin'
    path.write_bytes(b'12345678')
    with pytest.raises(OSError, match='content mismatch'):
        recording.verify_event_file(path, 8, hashlib.sha256(b'87654321').hexdigest())


def test_sync_failure_cannot_return_recording_success(tmp_path, monkeypatch):
    monkeypatch.setattr(TraceIO, 'fail', 'sync')
    with pytest.raises(OSError, match='injected data sync'):
        execute(FixedEvents([([.1], [1])]), args_for(tmp_path, duration_ms=.2))
    assert (tmp_path/'rank0.events.bin').read_bytes() == struct.pack('<II', 1, 0)


def test_pinned_model_construction_and_parameter_checks_unchanged():
    old = (ROOT/'tools/mam_nest_reference.py').read_text()
    new = (ROOT/'tools/mam_nest_benchmark.py').read_text()
    assert hashlib.sha256(old.encode()).hexdigest() == '7c7ef162e2d448dcbf5db3dc262000188e23ada0ba31d7009bb7ef5826ab9dc2'
    start = '    groups, populations, offset = [], [], 0'
    end = "    max_delay_ms = nest.GetKernelStatus('max_delay')"
    assert old[old.index(start):old.index(end)+len(end)] == new[new.index(start):new.index(end)+len(end)]
    kernel_start, kernel_end = '    nest.ResetKernel()', '    rank = nest.Rank()'
    assert old[old.index(kernel_start):old.index(kernel_end)] == new[new.index(kernel_start):new.index(kernel_end)]
    trees = [ast.parse(s) for s in [old, new]]
    for name in ['virtual_process_capacity', 'memory_snapshot', 'read_parameters']:
        functions = [next(x for x in t.body if isinstance(x, ast.FunctionDef) and x.name == name) for t in trees]
        assert ast.dump(functions[0]) == ast.dump(functions[1])


class Cells:
    def __init__(self, ids):
        self.ids = ids
    def __getitem__(self, key):
        return Cells(self.ids[key])
    def tolist(self):
        return self.ids
    def set(self, **values):
        pass


class FakeNest(FixedEvents):
    __version__ = 'FIXED-EVENT-LIFECYCLE-TEST'
    __file__ = 'no-real-nest-module'
    def __init__(self):
        super().__init__([([.1, .2], [1, 3]), ([.3], [2])])
        self.next_id = 1
        self.connections = 0
        self.calls = []
        self.random = SimpleNamespace(normal=lambda **kw: kw)
        self.math = SimpleNamespace(redraw=lambda value, **kw: value)
    def ResetKernel(self):
        self.calls.append('reset')
    def SetKernelStatus(self, value):
        self.calls.append(('kernel', value))
    def NumProcesses(self):
        return 1
    def Rank(self):
        return 0
    def Create(self, model, n=1, params=None):
        self.calls.append(('create', model, n, params))
        if model == 'spike_recorder':
            return self.recorder
        cells = Cells(list(range(self.next_id, self.next_id+n)))
        self.next_id += n
        return cells
    def Connect(self, source, target, conn_spec=None, syn_spec=None):
        if conn_spec:
            self.connections += conn_spec['N']
        else:
            self.connections += len(source.ids) if isinstance(target, Recorder) else len(target.ids)
        self.calls.append(('connect', conn_spec, syn_spec))
    def GetKernelStatus(self, name):
        return self.connections if name == 'num_connections' else .1


@pytest.fixture
def synthetic_producer(monkeypatch):
    nest = FakeNest()
    p = dict(total_neurons=3, total_recurrent_synapses=1, N_scaling=1, K_scaling=1,
             populations=[dict(count=3, name='fixed', dc_pA=0, external_indegree=1, external_weight_pA=1)],
             projections=[dict(source=0, target=0, count=1, weight_mean_pA=1, weight_sd_pA=.1,
                               delay_mean_ms=1, delay_sd_ms=.1, excitatory=True)],
             params=dict(neuron_params=dict(single_neuron_dict={}, V0_mean=-60, V0_sd=1)))
    monkeypatch.setitem(sys.modules, 'nest', nest)
    monkeypatch.setattr(producer, 'read_parameters', lambda *a: p)
    monkeypatch.setattr(producer, 'memory_snapshot', lambda: dict(rss_kib=1))
    monkeypatch.setattr(producer.os, 'sched_getaffinity', lambda pid: {0}, raising=False)
    return nest


def test_full_producer_lifecycle_reports_only_after_persistence(tmp_path, synthetic_producer, capsys):
    producer.run(args_for(tmp_path))
    report = json.loads((tmp_path/'rank0.json').read_text())
    events = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    terminal = events[-1]
    assert terminal['event'] == 'rank_complete'
    assert json.loads((tmp_path/'rank0.done.json').read_text()) == terminal
    assert terminal['report']['sha256'] == hashlib.sha256((tmp_path/'rank0.json').read_bytes()).hexdigest()
    assert terminal['rank_wall_through_report_seconds'] >= report['rank_pre_report_wall_seconds']
    assert report['local_total_connections'] == 7 and report['local_spikes'] == 3
    assert not report['auxiliary_state_output_complete']
    assert not report['state_format_matching_required']
    assert not report['all_rank_output_acceptance']
    assert report['workload_sha256'] == hashlib.sha256((ROOT/'tools/mam_benchmark_workload_v1.json').read_bytes()).hexdigest()
    assert report['schema'] == 'b2-native-nest-mam-benchmark-events-v2'
    assert [x[1] for x in synthetic_producer.calls if isinstance(x, tuple) and x[0] == 'create'] == ['iaf_psc_exp', 'poisson_generator', 'spike_recorder']


def test_report_failure_never_emits_success(tmp_path, synthetic_producer, monkeypatch, capsys):
    original = producer.durable_json
    def fail_report(path, value):
        if path.name == 'rank0.json':
            raise OSError('injected report failure')
        return original(path, value)
    monkeypatch.setattr(producer, 'durable_json', fail_report)
    with pytest.raises(OSError, match='report failure'):
        producer.run(args_for(tmp_path))
    events = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert events[-1]['event'] == 'rank_failed'
    assert all(x.get('event') != 'rank_complete' for x in events)
    assert json.loads((tmp_path/'rank0.failed.json').read_text())['error_type'] == 'OSError'


def test_done_sidecar_failure_never_emits_success(tmp_path, synthetic_producer, monkeypatch, capsys):
    original = producer.durable_json
    def fail_done(path, value):
        if path.name == 'rank0.done.json':
            raise OSError('injected done sidecar failure')
        return original(path, value)
    monkeypatch.setattr(producer, 'durable_json', fail_done)
    with pytest.raises(OSError, match='done sidecar failure'):
        producer.run(args_for(tmp_path))
    events = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert events[-1]['event'] == 'rank_failed'
    assert all(x.get('event') != 'rank_complete' for x in events)
    assert (tmp_path/'rank0.json').exists()


def test_existing_rank_claim_cannot_overwrite_or_claim_failure_ownership(tmp_path, synthetic_producer):
    (tmp_path/'rank0.started.json').write_text('original')
    with pytest.raises(FileExistsError):
        producer.run(args_for(tmp_path))
    assert (tmp_path/'rank0.started.json').read_text() == 'original'
    assert not (tmp_path/'rank0.failed.json').exists()
    assert not any(isinstance(x, tuple) and x[0] == 'create' for x in synthetic_producer.calls)


@pytest.mark.parametrize('count', [0, (512*2**30)//8+1])
def test_invalid_writer_budget_rejected_before_kernel(tmp_path, synthetic_producer, count):
    with pytest.raises(ValueError, match='before kernel creation'):
        producer.run(args_for(tmp_path, max_spikes_per_rank=count))
    assert synthetic_producer.calls == []
    assert list(tmp_path.iterdir()) == []
