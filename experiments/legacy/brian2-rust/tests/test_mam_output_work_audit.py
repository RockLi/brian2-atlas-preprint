import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('work', Path(__file__).resolve().parents[1]/'tools/mam_output_work_audit.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def inputs():
    # 65 shared cells across 32 ranks exercises unequal, exact integer cuts.
    sizes = [3, 65]
    definitions = [dict(name='local', count=3), dict(name='shared', count=65)]
    populations = []
    for n in sizes:
        indices = np.repeat(np.arange(n), np.arange(n) % 3 + 1).astype('<u4')
        ticks = np.arange(len(indices), dtype='<u4')
        populations.append(dict(counts=np.bincount(indices).astype('<i8'),
            spike_ticks=ticks, indices=indices, event_monitors={},
            event_streams={'spike': dict(ticks=ticks.copy(), indices=indices.copy())}))
    model = dict(definition=dict(populations=definitions, synapses=[
        dict(target_population=0, source_count=65), dict(target_population=1, source_count=3)]),
        instance=dict(synapses=[dict(topology=dict(kind='fixed_total', edge_count=17)),
                               dict(topology=dict(kind='fixed_total', edge_count=31))]))
    hist = [0]+[1]*31
    counts = [0]*64
    counts[10:12] = [17, 17]
    offsets = [0]*32
    offsets[5] = 66*8
    shared_stats = [v for r in range(32) for v in [hist[r], 31*(r+1)//32-31*r//32]]
    work = []
    for r in range(32):
        lo, hi = 65*r//32, 65*(r+1)//32
        work += [hi-lo+(3 if r == 5 else 0),
                 int(populations[1]['counts'][lo:hi].sum())+(6 if r == 5 else 0),
                 (101 if r == 5 else 0)+(202 if r == 31 else 0)]
    runtime = dict(rank_work=work, procedural_topology=[
        dict(projection=0, global_edges=17, construction='target-owner-local',
             rank_stats=counts, rank_csr_offset_bytes=offsets),
        dict(projection=1, global_edges=31, construction='distributed-draw-ranges',
             rank_stats=shared_stats, rank_csr_offset_bytes=[0]+[32]*31)])
    loaded = dict(populations=populations, synapses=[dict(events=101), dict(events=202)],
                  metadata=dict(spike_count=sum(len(p['indices']) for p in populations), synaptic_events=303))
    return model, loaded, runtime, [5, None], {1: hist}


@pytest.mark.parametrize('block', [1, 7, 131072])
def test_complete_cross_dump_and_unequal_shared_shard_accounting(inputs, block):
    released = []
    def release(a):
        assert a.ndim == 1 and len(a) <= block
        released.append(len(a))
    result = m.audit_work(*inputs, release=release, event_release=release, block=block)
    assert result['all_result_and_event_records_equal'] and released
    assert result['rank_neuron_and_spike_work_exact']
    assert result['exact_local_edges'] == [0]+[18 if r == 5 else 1 for r in range(1, 32)]
    assert result['shared_deliveries_by_rank'] == [0]*31+[202]
    assert not result['shared_delivery_rank_histories_independently_verified']


@pytest.mark.parametrize('fault', ['late-event', 'late-id', 'event-length', 'dtype',
    'count', 'negative-count', 'neuron-work', 'spike-work', 'delivery-work',
    'shared-deliveries', 'missing-rank', 'owner', 'histogram', 'missing-histogram',
    'extra-histogram', 'edge-owner', 'draw-range', 'csr', 'projection-id',
    'projection-total', 'construction', 'missing-projection', 'global-total'])
def test_reject_corrupt_complete_output_or_ownership(inputs, fault):
    model, loaded, runtime, owners, hist = inputs
    p = loaded['populations'][1]
    a = runtime['procedural_topology'][1]
    if fault == 'late-event': p['event_streams']['spike']['ticks'][-1] += 1
    elif fault == 'late-id': p['event_streams']['spike']['indices'][-1] -= 1
    elif fault == 'event-length': p['event_streams']['spike']['ticks'] = p['spike_ticks'][:-1]
    elif fault == 'dtype': p['event_streams']['spike']['ticks'] = p['spike_ticks'].astype('<i8')
    elif fault == 'count': p['counts'][-1] += 1
    elif fault == 'negative-count': p['counts'][-1] = -1
    elif fault == 'neuron-work': runtime['rank_work'][-3] += 1
    elif fault == 'spike-work': runtime['rank_work'][-2] += 1
    elif fault == 'delivery-work': runtime['rank_work'][5*3+2] -= 1
    elif fault == 'shared-deliveries': runtime['rank_work'][-1] -= 1
    elif fault == 'missing-rank': runtime['rank_work'].pop()
    elif fault == 'owner': owners[0] = 32
    elif fault == 'histogram': hist[1][1] += 1
    elif fault == 'missing-histogram': hist.clear()
    elif fault == 'extra-histogram': hist[0] = [0]*32
    elif fault == 'edge-owner': a['rank_stats'][2] -= 1
    elif fault == 'draw-range': a['rank_stats'][3] -= 1
    elif fault == 'csr': a['rank_csr_offset_bytes'][0] = 32
    elif fault == 'projection-id': a['projection'] = 0
    elif fault == 'projection-total': a['global_edges'] += 1
    elif fault == 'construction': a['construction'] = 'replicated'
    elif fault == 'missing-projection': runtime['procedural_topology'].pop()
    elif fault == 'global-total': loaded['metadata']['synaptic_events'] -= 1
    with pytest.raises(ValueError):
        m.audit_work(*inputs, release=lambda a: None, event_release=lambda a: None, block=7)


def test_mismatch_releases_both_mappings(inputs):
    p = inputs[1]['populations'][0]
    p['event_streams']['spike']['ticks'][0] += 1
    released = []
    with pytest.raises(ValueError, match='cross-dump'):
        m.audit_work(*inputs, release=lambda a: released.append('result'),
                     event_release=lambda a: released.append('event'), block=1)
    assert released == ['result', 'event']


def test_candidate_table_selection_follows_current_owners(inputs):
    rows = [dict(q=0, population=0, counts=[17]+[0]*31),
            dict(q=1, population=1, counts=inputs[4][1])]
    selected = m.select_shared_counts(rows, inputs[0], inputs[3])
    assert selected == inputs[4] and list(selected) == [1]
    for changed in [rows[:1], rows[:1]+rows[:1], [dict(rows[1], population=0)]]:
        with pytest.raises(ValueError):
            m.select_shared_counts(changed, inputs[0], inputs[3])


def test_all_late_records_are_checked_with_bounded_temporary_memory(inputs):
    import tracemalloc
    model, loaded, runtime, owners, hist = inputs
    p = loaded['populations'][0]
    n = 300001
    data = bytearray(n*8+1)
    events = np.ndarray((n,), dtype=[('tick', '<u4'), ('id', '<u4')], buffer=data, offset=1)
    events['tick'] = np.arange(n, dtype='<u4')
    events['id'] = np.arange(n, dtype='<u4') % 3
    p['spike_ticks'], p['indices'] = events['tick'], events['id']
    p['counts'] = np.bincount(p['indices']).astype('<i8')
    p['event_streams']['spike'] = dict(ticks=events['tick'].copy(), indices=events['id'].copy())
    runtime['rank_work'][5*3+1] += n-6
    loaded['metadata']['spike_count'] += n-6
    assert not p['spike_ticks'].flags.aligned
    tracemalloc.start()
    try:
        m.audit_work(*inputs, release=lambda a: None, event_release=lambda a: None, block=1024)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 256*1024 # A full unaligned-column copy would exceed 1 MiB.
    p['event_streams']['spike']['indices'][-1] += 1
    with pytest.raises(ValueError, match='cross-dump'):
        m.audit_work(*inputs, release=lambda a: None, event_release=lambda a: None, block=1024)
