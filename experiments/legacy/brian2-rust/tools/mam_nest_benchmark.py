"""Versioned benchmark producer: durable events and explicit terminal timing.

Not admitted for a performance run until all-rank receipts and a finite resource protocol.
Model construction is retained from the pinned native reference producer.

Uses the official NEST 3 connection rule, distributions and Poisson generators.
Initial voltages and graph draws use NEST's RNG: they are an independent
realization, not the Rust realization. Construction order is all populations,
then canonical exported projections, then external input/recording devices.
Scientific equivalence and performance advantages require separate comparisons.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from mam_benchmark_recording import record_chunks, durable_json
import mam_benchmark_recording as benchmark_recording
import mam_benchmark_event_io as benchmark_event_io

DT = .1
EVENT_DTYPE = np.dtype([('tick', '<u4'), ('cell', '<u4')])
# Pinned NEST 3.10 source: NUM_BITS_LCID=27, MAX_LCID=2**27-1;
# ConnectionManager rejects a count above MAX_LCID-1 for one VP/model.
# This necessary aggregate check does not guarantee balanced local counts.
STATIC_CONNECTIONS_PER_VP_LIMIT = 134217726
WORKLOAD_SHA256 = 'da61b893f2132c9e737462a4607ed1bd354235c6c7d1e510fa74fd6326231962'


def virtual_process_capacity(p, ranks, threads):
    total = p['total_recurrent_synapses'] + 2*p['total_neurons']
    minimum = (total + STATIC_CONNECTIONS_PER_VP_LIMIT-1) // STATIC_CONNECTIONS_PER_VP_LIMIT
    if ranks*threads < minimum:
        raise ValueError(f'Pinned NEST connection index capacity exceeded before kernel creation: '
                         f'{total} static connections need at least {minimum} virtual processes; '
                         f'{ranks*threads} requested')
    return dict(static_connections=total, virtual_processes=ranks*threads,
                minimum_virtual_processes=minimum, per_vp_limit=STATIC_CONNECTIONS_PER_VP_LIMIT,
                scope='Necessary aggregate index capacity only; local imbalance and memory need separate admission.')


def memory_snapshot():
    """Linux rank-local RSS; the external cgroup guard measures shared pages."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
    return dict(rss_kib=int(status['VmRSS'].split()[0]),
                peak_rss_kib=usage.ru_maxrss, user_seconds=usage.ru_utime, system_seconds=usage.ru_stime)


def read_parameters(path, expected_sha, max_neurons, max_edges):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Parameter SHA256 mismatch')
    p = json.loads(raw)
    assert p['schema'] == 'b2-official-mam-parameters-v1'
    assert p['source_commit'] == '0a658be40bef3249cbe452f38809edf7d2f524ba'
    assert p['nest_parameter_path'] == 3
    populations, projections = p['populations'], p['projections']
    assert len(populations) == 254 and len({x['area'] for x in populations}) == 32
    assert all(type(x['count']) is int and x['count'] > 0 for x in populations + projections)
    n, e = sum(x['count'] for x in populations), sum(x['count'] for x in projections)
    assert (n, e) == (p['total_neurons'], p['total_recurrent_synapses'])
    if n > max_neurons or e > max_edges:
        raise ValueError('Explicit native NEST preparation budget exceeded before kernel creation')
    assert p['params']['input_params']['poisson_input']
    assert p['params']['input_params']['rate_ext'] == 10.
    for x in projections:
        assert 0 <= x['source'] < len(populations) and 0 <= x['target'] < len(populations)
        assert np.isfinite([x[k] for k in ['weight_mean_pA', 'weight_sd_pA', 'delay_mean_ms', 'delay_sd_ms']]).all()
        assert x['weight_sd_pA'] > 0 and x['delay_sd_ms'] > 0 and x['delay_mean_ms'] > 0
    return p


def _run(args, lifecycle):
    # Reject an unsupported IO platform or byte ceiling before expensive graph work.
    benchmark_event_io.LinuxIO()
    if (type(args.max_spikes_per_rank) is not int
            or not 0 < args.max_spikes_per_rank <= (512*2**30)//8):
        raise ValueError('Event byte ceiling must fit the durable writer before kernel creation')
    workload_path = Path(__file__).with_name('mam_benchmark_workload_v1.json')
    workload_raw = workload_path.read_bytes()
    if hashlib.sha256(workload_raw).hexdigest() != WORKLOAD_SHA256:
        raise ValueError('Scientific workload contract mismatch before kernel creation')
    workload = json.loads(workload_raw)
    p = read_parameters(args.parameters, args.parameters_sha256, args.max_neurons, args.max_edges)
    capacity = virtual_process_capacity(p, args.ranks, args.threads)
    import nest

    nest.ResetKernel()
    nest.SetKernelStatus(dict(resolution=DT, local_num_threads=args.threads, rng_seed=args.seed))
    assert nest.NumProcesses() == args.ranks
    rank = nest.Rank()
    lifecycle['rank'] = rank
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / f'rank{rank}.json'
    event_path = args.output / f'rank{rank}.events.bin'
    assert not report_path.exists() and not event_path.exists()
    start_path = args.output / f'rank{rank}.started.json'
    # Claim an unused rank output before constructing any neurons.
    with start_path.open('x') as claim:
        lifecycle['owns_output'] = True
        claim.write(json.dumps(dict(rank=rank, pid=os.getpid(), status='started'))+'\n')
        claim.flush()
        os.fsync(claim.fileno())
    started = time.perf_counter()
    groups, populations, offset = [], [], 0
    for q in p['populations']:
        cells = nest.Create('iaf_psc_exp', q['count'],
                            params=p['params']['neuron_params']['single_neuron_dict'])
        cells.set(I_e=q['dc_pA'])
        cells.V_m = nest.random.normal(mean=p['params']['neuron_params']['V0_mean'],
                                       std=p['params']['neuron_params']['V0_sd'])
        # Read just the endpoints instead of materializing a global ID list
        # twice on each rank for every population.
        assert cells[:1].tolist()[0] == offset + 1
        assert cells[q['count']-1:].tolist()[0] == offset + q['count']
        populations.append(dict(name=q['name'], count=q['count'], first_gid=offset+1,
                                cell_start=offset, cell_end=offset+q['count']))
        offset += q['count']
        groups.append(cells)
    neurons_ready = time.perf_counter()
    phase_memory = dict(neurons_created=memory_snapshot())
    ledger = []
    previous = nest.GetKernelStatus('num_connections')
    assert previous == 0
    for i, q in enumerate(p['projections']):
        weight = nest.math.redraw(nest.random.normal(mean=q['weight_mean_pA'], std=q['weight_sd_pA']),
                                 min=0. if q['excitatory'] else -np.inf,
                                 max=np.inf if q['excitatory'] else 0.)
        delay = nest.math.redraw(nest.random.normal(mean=q['delay_mean_ms'], std=q['delay_sd_ms']),
                                min=DT, max=np.inf)
        nest.Connect(groups[q['source']], groups[q['target']],
                     dict(rule='fixed_total_number', N=q['count']),
                     dict(synapse_model='static_synapse', weight=weight, delay=delay))
        current = nest.GetKernelStatus('num_connections')  # Local count, summed offline.
        ledger.append(current-previous)
        previous = current
        if i % 500 == 0:
            print(json.dumps(dict(rank=rank, projection=i, local_recurrent_edges=previous)), flush=True)
    recurrent_ready = time.perf_counter()
    phase_memory['recurrent_connected'] = memory_snapshot()
    external = []
    for q, cells in zip(p['populations'], groups, strict=True):
        pg = nest.Create('poisson_generator', params=dict(rate=q['external_indegree'] * 10.))
        # Official static external connection uses the default 1 ms delay;
        # make it explicit to bind this reference contract to its audited value.
        nest.Connect(pg, cells, syn_spec=dict(weight=q['external_weight_pA'], delay=1.))
        external.append(pg)
    with_external = nest.GetKernelStatus('num_connections')
    recorder = nest.Create('spike_recorder', params=dict(record_to='memory'))
    for cells in groups:
        nest.Connect(cells, recorder)
    with_recorder = nest.GetKernelStatus('num_connections')
    devices_ready = time.perf_counter()
    phase_memory['devices_connected'] = memory_snapshot()
    min_delay_ms = nest.GetKernelStatus('min_delay')
    max_delay_ms = nest.GetKernelStatus('max_delay')
    recording = record_chunks(nest, recorder, args, rank, offset, memory_snapshot)
    chunks = recording['chunks']
    total_spikes = recording['local_spikes']
    phase_memory.update(recording['phase_memory'])
    completed = time.perf_counter()
    result = dict(schema='b2-native-nest-mam-benchmark-events-v2', host=os.uname().nodename, rank=rank,
        ranks=nest.NumProcesses(), threads=args.threads, seed=args.seed, nest_version=nest.__version__,
        nest_module=nest.__file__, parameters_sha256=args.parameters_sha256,
        N_scaling=p['N_scaling'], K_scaling=p['K_scaling'], dt_ms=DT, duration_ms=args.duration_ms,
        populations=populations, local_recurrent_edges=previous, projection_local_counts=ledger,
        local_external_connections=with_external-previous,
        local_recording_connections=with_recorder-with_external, local_total_connections=with_recorder,
        local_spikes=total_spikes, event_bytes=event_path.stat().st_size,
        event_sha256=recording['event_sha256'],
        event_format='Little-endian uint32 physical tick, uint32 zero-based global cell; 8 bytes/event; native within-chunk order.',
        chunks=chunks, wall_seconds=completed-started,
        neuron_seconds=neurons_ready-started, recurrent_seconds=recurrent_ready-neurons_ready,
        device_seconds=devices_ready-recurrent_ready,
        simulation_seconds=sum(x['simulation_seconds'] for x in chunks),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        phase_memory=phase_memory, min_delay_ms=min_delay_ms, max_delay_ms=max_delay_ms,
        connection_index_capacity=capacity,
        allowed_cpus=sorted(os.sched_getaffinity(0)),
        scope='Versioned durable scientific-event producer; all-rank audit and finite performance admission are pending. Not scientific or performance acceptance.')
    assert result['event_bytes'] == total_spikes * EVENT_DTYPE.itemsize
    result.update({key: value for key, value in recording.items()
                   if key not in ['chunks', 'phase_memory']})
    result['rank_pre_report_wall_seconds'] = time.perf_counter()-lifecycle['started']
    result['wall_seconds_scope'] = 'Construction start through recording verification; excludes parameter/kernel setup and report persistence'
    result['producer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['source_sha256'] = {Path(module.__file__).name:
                              hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                              for module in [benchmark_recording, benchmark_event_io]}
    result['workload_schema'] = workload['schema']
    result['workload_sha256'] = WORKLOAD_SHA256
    result['auxiliary_state_output_complete'] = False
    result['state_format_matching_required'] = workload['state_format_matching_required']
    result['all_rank_output_acceptance'] = False
    receipt = durable_json(report_path, result)
    # This time includes report persistence. The external controller must also
    # require successful exits from every rank; this marker alone is insufficient.
    terminal = dict(event='rank_complete', rank=rank,
                    rank_wall_through_report_seconds=time.perf_counter()-lifecycle['started'],
                    report=receipt, event_readback_verified=True,
                    scope='Time through report persistence; excludes this sidecar/stdout/process exit. External all-rank job wall is required.')
    durable_json(args.output / f'rank{rank}.done.json', terminal)
    print(json.dumps(terminal), flush=True)


def run(args):
    lifecycle = dict(started=time.perf_counter(), owns_output=False)
    try:
        _run(args, lifecycle)
    except BaseException as error:
        failure = dict(event='rank_failed', rank=lifecycle.get('rank'),
                       error_type=type(error).__name__, error=str(error)[:2000],
                       elapsed_seconds=time.perf_counter()-lifecycle['started'])
        if lifecycle['owns_output']:
            try:
                durable_json(args.output / f"rank{lifecycle['rank']}.failed.json", failure)
            except BaseException as receipt_error:
                failure['failure_receipt_error'] = str(receipt_error)[:1000]
        print(json.dumps(failure), flush=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--parameters-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--duration-ms', type=float, default=100.)
    parser.add_argument('--max-duration-ms', type=int, choices=[10500,100500], default=10500,
                        help='Explicit primary-duration admission; external resource and recording budgets remain required.')
    parser.add_argument('--chunk-ms', type=float, default=100.)
    parser.add_argument('--max-neurons', type=int, default=100000)
    parser.add_argument('--max-edges', type=int, default=10000000)
    parser.add_argument('--max-chunk-spikes', type=int, default=2000000)
    parser.add_argument('--max-spikes-per-rank', type=int, default=2000000)
    args = parser.parse_args()
    if any(x <= 0 for x in [args.ranks,args.threads,args.seed,args.max_neurons,args.max_edges,args.max_chunk_spikes,args.max_spikes_per_rank]):
        parser.error('All counts/budgets must be positive')
    if not 0 < args.chunk_ms <= args.duration_ms <= args.max_duration_ms or not all(
        np.isclose(x/DT, round(x/DT), rtol=0, atol=1e-9) for x in [args.duration_ms,args.chunk_ms]):
        parser.error(f'Duration and chunks must be grid aligned; admitted duration limit is {args.max_duration_ms} ms')
    run(args)
