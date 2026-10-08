"""Bounded cross-dump equality and independent MAM ownership/work accounting.

Inputs must first pass the production binary reader. Shared-population delivery
work is checked in aggregate: these files cannot reconstruct its rank histories.
"""
import numpy as np


def check(ok, message):
    if not ok:
        raise ValueError(message)


def select_shared_counts(rows, model, owners):
    """Select the current shared projections from a pinned candidate table.

    Historical exploration measured several possible shared populations. Only
    current plan owners choose the required entries; duplicate/missing entries
    and mismatched projection targets remain errors.
    """
    definitions = model['definition']['synapses']
    check(len(rows) <= len(definitions) <= 8344, 'candidate histogram size')
    by_projection = {}
    for row in rows:
        q = row['q']
        check(type(q) is int and 0 <= q < len(definitions)
              and q not in by_projection, 'duplicate/invalid candidate projection')
        check(row['population'] == definitions[q]['target_population'], 'candidate target mismatch')
        by_projection[q] = row['counts']
    required = {q for q, d in enumerate(definitions) if owners[d['target_population']] is None}
    check(required <= set(by_projection), 'required shared histogram missing')
    return {q: by_projection[q] for q in sorted(required)}


def audit_work(model, loaded, runtime, owners, shared_counts, *, release,
               event_release, ranks=32, block=131072):
    check(type(ranks) is int and 1 <= ranks <= 32
          and type(block) is int and 1 <= block <= 131072, 'invalid audit bounds')
    definitions = model['definition']['populations']
    check(1 <= len(definitions) <= 254 and len(owners) == len(definitions)
          and len(loaded['populations']) == len(definitions), 'population count')
    check(all(o is None or type(o) is int and 0 <= o < ranks for o in owners), 'owner range')
    expected = [[0, 0, 0] for _ in range(ranks)]
    profiles = []
    for p, (definition, pop, owner) in enumerate(zip(definitions, loaded['populations'], owners, strict=True)):
        n = definition['count']
        check(type(n) is int and 0 < n <= 4200000, 'population size')
        check(pop['counts'].shape == (n,) and pop['counts'].dtype == np.dtype('<i8'),
              'validated cell counts missing')
        check(set(pop['event_streams']) == {'spike'} and not pop['event_monitors'],
              'requires frozen spike-only monitors')
        stream = pop['event_streams']['spike']
        spikes = len(pop['spike_ticks'])
        check(spikes == len(pop['indices']), 'result event shape')
        for left, right in [(pop['spike_ticks'], stream['ticks']), (pop['indices'], stream['indices'])]:
            check(left.shape == right.shape and left.ndim == 1
                  and left.dtype == right.dtype and left.dtype in [np.dtype('<i8'), np.dtype('<u4')],
                  'cross-dump event shape/type')
            for start in range(0, len(left), block):
                a, b = left[start:start+block], right[start:start+block]
                try:
                    check(np.array_equal(a, b), f'population {p} cross-dump event mismatch')
                finally:
                    release(a)
                    event_release(b)
        # The reader independently reconstructed these per-cell counts from all
        # events. Sum their exact shard ranges instead of re-scanning every spike
        # once per rank or allocating a whole-population event mask.
        def count(lo, hi):
            total = 0
            for start in range(lo, hi, block):
                values = pop['counts'][start:min(hi, start+block)]
                try:
                    check(not np.any(values < 0), 'negative cell count')
                    total += int(values.sum(dtype=np.int64))
                finally:
                    release(values)
            return total
        total = 0
        for rank in range(ranks):
            if owner is None:
                lo, hi = n*rank//ranks, n*(rank+1)//ranks
            elif rank == owner:
                lo, hi = 0, n
            else:
                continue
            c = count(lo, hi)
            expected[rank][0] += hi-lo
            expected[rank][1] += c
            total += c
        check(total == spikes, 'per-cell count total differs')
        profiles.append(dict(population=p, name=definition['name'], owner=owner,
                             neurons=n, spikes=spikes, edges=0, delivered_edges=0,
                             csr_bytes=0, max_projection_edges=0))
    definitions = model['definition']['synapses']
    instances = model['instance']['synapses']
    actuals = runtime['procedural_topology']
    check(len(definitions) == len(instances) == len(actuals) == len(loaded['synapses'])
          and len(definitions) <= 8344, 'projection count')
    required_shared = {q for q, d in enumerate(definitions) if owners[d['target_population']] is None}
    check(set(shared_counts) == required_shared, 'shared projection histogram coverage')
    local_edges = [0]*ranks
    for q, (d, instance, actual, syn) in enumerate(zip(definitions, instances, actuals, loaded['synapses'], strict=True)):
        owner = owners[d['target_population']]
        edges = instance['topology']['edge_count']
        check(instance['topology']['kind'] == 'fixed_total' and type(edges) is int and edges >= 0,
              'requires fixed-total topology')
        check(actual['projection'] == q and actual['global_edges'] == edges, 'projection identity/count')
        if owner is None:
            check(actual['construction'] == 'distributed-draw-ranges', 'shared construction')
            histogram = shared_counts[q]
            check(len(histogram) == ranks and all(type(v) is int and v >= 0 for v in histogram)
                  and sum(histogram) == edges, 'shared edge histogram')
            counts = [v for r in range(ranks) for v in
                      [histogram[r], edges*(r+1)//ranks-edges*r//ranks]]
            offsets = [(d['source_count']+1)*8 if v else 0 for v in histogram]
        else:
            check(actual['construction'] == 'target-owner-local', 'local construction')
            counts, offsets = [0]*(2*ranks), [0]*ranks
            counts[owner*2:owner*2+2] = [edges, edges]
            offsets[owner] = (d['source_count']+1)*8
            expected[owner][2] += syn['events']
        check(actual['rank_stats'] == counts, f'projection {q} edge ownership')
        check(actual['rank_csr_offset_bytes'] == offsets, f'projection {q} CSR ownership')
        for rank in range(ranks):
            local_edges[rank] += counts[rank*2]
        row = profiles[d['target_population']]
        row['edges'] += edges
        row['delivered_edges'] += syn['events']
        row['csr_bytes'] += sum(offsets)
        row['max_projection_edges'] = max(row['max_projection_edges'], edges)
    work = runtime['rank_work']
    check(len(work) == ranks*3 and all(type(v) is int and v >= 0 for v in work), 'rank work shape/type')
    shared_deliveries = []
    for rank in range(ranks):
        row = work[rank*3:(rank+1)*3]
        check(row[:2] == expected[rank][:2] and row[2] >= expected[rank][2], 'rank work mismatch')
        shared_deliveries.append(row[2]-expected[rank][2])
    check(sum(shared_deliveries) == sum(p['delivered_edges'] for p in profiles if p['owner'] is None),
          'shared delivery total mismatch')
    summary = loaded['metadata']
    check(sum(p['spikes'] for p in profiles) == summary['spike_count']
          and sum(p['delivered_edges'] for p in profiles) == summary['synaptic_events'], 'global work totals')
    return dict(all_result_and_event_records_equal=True, all_projection_edge_and_csr_ownership_exact=True,
                rank_neuron_and_spike_work_exact=True, local_delivery_work_and_shared_aggregate_exact=True,
                shared_delivery_rank_histories_independently_verified=False,
                shared_deliveries_by_rank=shared_deliveries, exact_local_edges=local_edges,
                population_profile=profiles, maximum_block_elements=block)
