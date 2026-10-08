"""Check an MPI NEST build against the archived deterministic six-cell fixture.

Every rank writes only its local observations. The offline comparison requires
complete, nonduplicated ownership and samples before comparing the physical
time grid, spike events and voltages with the previously validated NEST wheel.
This is a build/partition check, not full-model statistical reproduction.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

import mam_nest_fixed_input as fixture


def run(output, ranks, threads, cell_stride=1):
    import nest

    nest.ResetKernel()
    nest.SetKernelStatus(dict(resolution=fixture.DT_MS,
                             local_num_threads=threads, rng_seed=1729))
    assert nest.NumProcesses() == ranks
    rank = nest.Rank()
    started = time.perf_counter()
    all_cells = nest.Create('iaf_psc_exp', len(fixture.INITIAL_MV)*cell_stride, params=fixture.CELL)
    # Space the six observed neurons across hosts in a high-rank layout.
    # Unselected padding neurons have no input or connections.
    cell_gids = np.asarray(all_cells.tolist()[::cell_stride], dtype=np.int64)
    cells = nest.NodeCollection(cell_gids.tolist())
    # NEST 3.10's MPI setter rejects the dictionary-of-vectors form used by
    # the serial wheel. Every rank executes the same scalar assignments.
    for i, (voltage, dc) in enumerate(zip(fixture.INITIAL_MV, fixture.DC_PA, strict=True)):
        cells[i:i + 1].set(V_m=voltage, I_e=dc)
    inputs = [nest.Create('spike_generator', params=dict(spike_times=times))
              for times in [fixture.EX_TIMES_MS, fixture.IN_TIMES_MS]]
    for source, target, weight, delay in fixture.EDGES:
        nest.Connect(inputs[source], cells[target:target + 1],
                     syn_spec=dict(weight=weight, delay=delay))
    for source, target, weight, delay in fixture.RECURRENT_EDGES:
        nest.Connect(cells[source:source + 1], cells[target:target + 1],
                     syn_spec=dict(weight=weight, delay=delay))
    meter = nest.Create('multimeter', params=dict(interval=fixture.DT_MS,
        record_from=['V_m', 'I_syn_ex', 'I_syn_in']))
    recorder = nest.Create('spike_recorder')
    nest.Connect(meter, cells)
    nest.Connect(cells, recorder)
    prepared = time.perf_counter()
    nest.Simulate(2500.)
    finished = time.perf_counter()
    local = nest.GetLocalNodeCollection(cells)
    events = meter.get('events')
    spikes = recorder.get('events')
    def cell_indices(gids):
        gids = np.asarray(gids, dtype=np.int64)
        indices = np.searchsorted(cell_gids, gids)
        assert np.all(indices < len(cell_gids))
        assert np.array_equal(cell_gids[indices], gids)
        return indices
    output.mkdir(parents=True, exist_ok=True)
    target = output / f'rank{rank}.npz'
    with target.open('xb') as f:
        np.savez(f,
            sample_cells=cell_indices(events['senders']),
            sample_ticks=np.rint(np.asarray(events['times']) / fixture.DT_MS).astype(np.int64),
            voltage_mv=np.asarray(events['V_m']),
            current_pa=np.asarray(events['I_syn_ex']) + events['I_syn_in'],
            spike_ticks=np.rint(np.asarray(spikes['times']) / fixture.DT_MS).astype(np.int64),
            spike_cells=cell_indices(spikes['senders']),
            local_cells=cell_indices(local.tolist()),
            final_voltage_mv=np.atleast_1d(local.get('V_m')) if len(local) else np.empty(0))
    provenance = dict(version=nest.__version__, module=nest.__file__,
        host=os.uname().nodename, pid=os.getpid(), allowed_cpus=sorted(os.sched_getaffinity(0)),
        rank=rank, ranks=nest.NumProcesses(), threads=threads,
        local_cells=local.tolist(), cell_stride=cell_stride, observed_cell_gids=cell_gids.tolist(), duration_ms=2500., dt_ms=fixture.DT_MS,
        construction_seconds=prepared-started, simulation_seconds=finished-prepared)
    (output / f'rank{rank}.json').write_text(json.dumps(provenance, indent=2) + '\n')


def compare(output, reference, ranks, threads):
    with np.load(reference) as f:
        expected = dict(f)
    data = []
    provenance = []
    for rank in range(ranks):
        with np.load(output / f'rank{rank}.npz') as f:
            data.append(dict(f))
        p = json.loads((output / f'rank{rank}.json').read_text())
        assert (p['rank'], p['ranks'], p['threads']) == (rank, ranks, threads)
        provenance.append(p)
    observed = {k: np.concatenate([d[k] for d in data]) for k in data[0]}
    n = len(fixture.INITIAL_MV)
    local_order = np.argsort(observed['local_cells'])
    assert np.array_equal(observed['local_cells'][local_order], np.arange(n))
    order = np.lexsort((observed['sample_cells'], observed['sample_ticks']))
    ticks = np.rint(expected['times_ms'] / fixture.DT_MS).astype(np.int64)
    assert np.array_equal(observed['sample_ticks'][order], np.repeat(ticks, n))
    assert np.array_equal(observed['sample_cells'][order], np.tile(np.arange(n), len(ticks)))

    def sorted_spikes(ticks, cells):
        order = np.lexsort((cells, ticks))
        return np.column_stack([ticks, cells])[order]

    wanted = sorted_spikes(np.rint(expected['spike_times_ms'] / fixture.DT_MS).astype(np.int64),
                           expected['spike_cells'])
    actual = sorted_spikes(observed['spike_ticks'], observed['spike_cells'])
    errors = {}
    for key in ['voltage_mv', 'current_pa']:
        errors[key] = float(np.max(np.abs(observed[key][order].reshape(-1, n) - expected[key])))
    errors['final_voltage_mv'] = float(np.max(np.abs(
        observed['final_voltage_mv'][local_order] - expected['final_voltage_mv'])))
    limits = dict(voltage_mv=1e-9, current_pa=1e-8, final_voltage_mv=1e-9)
    passed = bool(np.array_equal(wanted, actual) and all(
        np.isfinite(errors[k]) and errors[k] < limits[k] for k in limits))
    result = dict(passed=passed, ranks=ranks, threads=threads, neurons=n,
        spikes=len(actual), spikes_exact=bool(np.array_equal(wanted, actual)),
        samples=len(order), max_absolute_errors=errors, numerical_limits=limits,
        provenance=provenance,
        scope='Deterministic six-cell NEST build and MPI/OpenMP partition check; not MAM reproduction or performance evidence.')
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'provenance'}, indent=2))
    if not passed:
        raise SystemExit('NEST MPI reference fixture mismatch; raw observations retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'compare'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--ranks', type=int, required=True)
    parser.add_argument('--threads', type=int, choices=[1, 2, 4], default=1)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--cell-stride', type=int, choices=[1,8], default=1)
    args = parser.parse_args()
    if not 1 <= args.ranks <= 48 or args.ranks * args.threads > 192:
        parser.error('This bounded fixture supports at most 48 ranks and 192 execution threads')
    if args.mode == 'run':
        run(args.output, args.ranks, args.threads, args.cell_stride)
    else:
        if args.reference is None:
            parser.error('compare requires --reference')
        compare(args.output, args.reference, args.ranks, args.threads)
