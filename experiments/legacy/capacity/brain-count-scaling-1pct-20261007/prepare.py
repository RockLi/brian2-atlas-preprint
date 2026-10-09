"""Repeated three-host blocks for a controlled synthetic weak-scaling study."""
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def layout(hosts, pilot=False):
    assert hosts in [3, 6, 12, 24, 30]
    blocks = hosts // 3
    block_n = 24_000 if pilot else 86_000_000
    e_n, i_n = block_n * 4 // 5, block_n // 5
    e_sizes = [e_n // 19 + (i < e_n % 19) for i in range(19)]
    i_sizes = [i_n // 5] * 5
    sizes = e_sizes * blocks + i_sizes * blocks
    owners = []
    for block in range(blocks):
        for k in range(19):
            host, slot = (0, k) if k < 6 else (1, k-6) if k < 12 else (2, k-12)
            owners.append((block*3+host)*8+slot)
    for block in range(blocks):
        for k in range(5):
            host, slot = (0, 6+k) if k < 2 else (1, 6+k-2) if k < 4 else (2, 7)
            owners.append((block*3+host)*8+slot)
    assert len(sizes) == hosts*8 and set(owners) == set(range(hosts*8))
    assert sum(sizes) == blocks*block_n
    assert sum(sizes[:19*blocks])*5 == sum(sizes)*4
    return sizes, owners, 19*blocks


def edge_counts(sizes, degree=1000):
    total = sum(sizes)
    numerators = [x*y*degree for x in sizes for y in sizes]
    counts = [x//total for x in numerators]
    remainder = total*degree-sum(counts)
    for i in sorted(range(len(counts)), key=lambda i: (-(numerators[i] % total), i))[:remainder]:
        counts[i] += 1
    assert sum(counts) == total*degree and max(counts) <= 2**31-1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hosts', type=int, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    sizes, creation_owners, e_pools = layout(args.hosts, args.pilot)
    neurons, pools, ranks = sum(sizes), len(sizes), args.hosts*8
    counts = edge_counts(sizes)
    incoming = [0]*ranks
    for target, owner in enumerate(creation_owners):
        incoming[owner] = sum(counts[source*pools+target] for source in range(pools))
    assert min(incoming) > 0 and max(incoming) <= 4_290_000_000
    os.environ.update(B2_MAX_NEURONS='860000000', B2_MAX_INITIAL_VALUES='4000000000',
                      B2_MAX_IR_BYTES=str(64*2**30))
    sys.path.insert(0, str(args.source/'python'))
    import brian2 as b
    import numpy as np
    import brian2_rust
    from brian2_rust.export import lower_network
    from brian2_rust.encoded_array import packed_export
    from brian2_rust.protocol import _canonical_chunks
    from brian2_rust.distributed import write_mpi_project, compile_mpi_project
    started = time.monotonic()
    args.output.mkdir(exist_ok=False, parents=True)
    b.set_device('rust_standalone', runner=args.source/'target/release/b2-runner')
    clock = b.Clock(dt=.1*b.ms)
    groups, objects = [], []
    for k, n in enumerate(sizes):
        group = b.NeuronGroup(n,
            'dv/dt=(drive-v+current)/(20*ms):1 (unless refractory)\ndcurrent/dt=-current/(5*ms):1',
            threshold='v>1', reset='v=0', refractory=2*b.ms, method='euler', clock=clock,
            namespace={'drive':1.05}, name=f'capacity_{k}')
        group.v = (np.arange(n, dtype=np.int64) % 1000)*.00095
        groups.append(group)
        objects.extend([group, b.SpikeMonitor(group), b.StateMonitor(group, ['v','current'], record=[0,n-1])])
    for source, sg in enumerate(groups):
        for target, tg in enumerate(groups):
            syn = b.Synapses(sg, tg, 'w:1 (constant)', on_pre='current_post += w',
                             clock=clock, name=f'capacity_projection_{source}_{target}')
            low, high = (.0027,.0033) if source < e_pools else (-.0132,-.0108)
            brian2_rust.connect_fixed_total(syn, counts[source*pools+target],
                seed=2026100700+source*pools+target,
                initializers={'w':brian2_rust.Uniform(low,high)},
                delay_initializer=brian2_rust.ClippedNormal(1.5*b.ms,.25*b.ms,
                                                            minimum=.1*b.ms,maximum=3*b.ms))
            objects.append(syn)
    lower_started = time.monotonic()
    with packed_export():
        model = lower_network(b.Network(*objects), 100*b.ms, rng_seed=20261007)
    lower_done = time.monotonic()
    with (args.output/'model.json').open('wb') as f:
        for chunk in _canonical_chunks(model):
            f.write(chunk)
    canonical_owners = [creation_owners[int(p['name'].rsplit('_',1)[1])]
                        for p in model['definition']['populations']]
    print(json.dumps({'event':'model_exported','neurons':neurons,'edges':sum(counts),
                      'seconds':time.monotonic()-started}), flush=True)
    project_started = time.monotonic()
    profile = cProfile.Profile() if args.pilot else None
    if profile:
        profile.enable()
    try:
        plan = write_mpi_project(model, args.output/'mpi', ranks=ranks,
            runner=args.source/'target/release/b2-runner', population_owners=tuple(canonical_owners),
            compact_populations=True, compact_queue_indices=True,
            compact_spike_history=True, compact_spike_output=True)
    finally:
        if profile:
            profile.disable()
            profile.dump_stats(args.output/'project-preparation.pstats')
    project_done = time.monotonic()
    print(json.dumps({'event':'mpi_project_written','seconds':project_done-project_started}), flush=True)
    report = dict(schema='atlas-weak-scaling-preparation-v1', neurons=neurons,
        connections=sum(counts), mean_indegree=1000, hosts=args.hosts, ranks=ranks,
        populations=pools, projections=pools*pools, excitatory_populations=e_pools,
        population_sizes=sizes, population_owners=canonical_owners,
        creation_population_owners=creation_owners, local_edges_minmax=[min(incoming),max(incoming)],
        seed=20261007, dt_ms=.1, duration_ms=100, precision='reference-f64',
        neuron_count_fraction_of_86B_percent=neurons/86e9*100,
        scientific_scope='Connected synthetic E/I count-normalized weak scaling; not anatomical brain simulation.',
        connection_semantics='Fixed-total uniform random with replacement; multapses and autapses permitted.',
        state_observation='Two neurons per population; complete spike histories retained.',
        layers=model['protocol']['layers'], plan_sha256=plan.sha256,
        prepare_seconds=time.monotonic()-started, compiled=False,
        frontend_construction_seconds=lower_started-started,
        frontend_lowering_seconds=lower_done-lower_started,
        canonical_json_write_seconds=project_started-lower_done,
        mpi_project_seconds=project_done-project_started,
        preparation_profiled=args.pilot,
        versions={'python':sys.version,'brian2':b.__version__,'numpy':np.__version__})
    if args.compile:
        tick = time.monotonic()
        compile_mpi_project(args.output/'mpi',
            mpicc='/atlas-home/0003/workspace/brian2-mpi-primary-20260909/mpi/bin/mpicc',
            rustc='/atlas-home/0003/workspace/brian2-lk-20260906/.cargo/bin/rustc',
            opt_level=3, panic_strategy='abort')
        report.update(compiled=True, compile_seconds=time.monotonic()-tick)
    hashing_started = time.monotonic()
    report['files'] = {str(f.relative_to(args.output)):
        {'bytes':f.stat().st_size, 'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest()}
        for f in args.output.rglob('*') if f.is_file()}
    report['artifact_hashing_seconds'] = time.monotonic()-hashing_started
    report['wall_seconds'] = time.monotonic()-started
    (args.output/'prepared.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'event':'prepared','neurons':neurons,'ranks':ranks,
                      'wall_seconds':report['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
