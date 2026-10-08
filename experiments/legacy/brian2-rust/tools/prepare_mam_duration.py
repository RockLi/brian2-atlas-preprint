"""Freeze a 10.5 s model and estimate output/runtime before MPI preparation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from mam_extend_duration import extend, audit_duration_delta, output_bytes


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f, 'sha256').hexdigest()


def prepare(args):
    sys.path.insert(0, str(args.python_source))
    from brian2_rust.protocol import attach_protocol, verify_protocol
    from brian2_rust.results import load_results
    started = time.perf_counter()
    baseline = json.loads((args.baseline / 'activity.json').read_text())
    expected = {'metastable': '75ff4455b1345a2d0713cf88c0e655840333fab93d4407c61bf3fb692c651fcc',
                'stabilized-ground': 'c9719baef5e9fbdfc2790345ddd7ba75f53bed149b181617739efa97b8182026'}
    assert sha(args.source / 'model.json') == baseline['model_sha256'] == expected[args.condition]
    for name, digest in baseline['result_sha256'].items():assert sha(args.results / name) == digest
    old = json.loads((args.source / 'model.json').read_text())
    verify_protocol(old)
    assert old['instance']['neuron_count'] == 4129924 and old['instance']['rng_seed'] == 1729
    data = load_results(old, args.results, include_times=False)
    spikes = data['metadata']['spike_count']
    last = sum(len(p['last_spikes']) for p in data['populations'])
    predicted = output_bytes(old, spikes, last)
    assert predicted['results_bytes'] == (args.results / 'results.bin').stat().st_size
    assert predicted['events_bytes'] == (args.results / 'events.bin').stat().st_size
    del data
    new = extend(old)
    attach_protocol(new);verify_protocol(new)
    delta = audit_duration_delta(old, new)
    assert new['protocol']['layers']['instance'] == old['protocol']['layers']['instance']
    args.output.mkdir(exist_ok=False)
    (args.output / 'model.json').write_text(json.dumps(new) + '\n')
    ranks = json.loads(args.run_audit.read_text())
    assert ranks['complete']
    # These measured rate scenarios are not bounds on future stochastic activity.
    measured = [dict(source=str(args.results / 'summary.json'), spikes=spikes, seconds=2.5)]
    for path in args.native_audit:
        report = json.loads(path.read_text())
        assert report['passed']
        expected_parameters = {'metastable': 'ee1a642c94b9d6e808627c54f78162e7f5d87f4140ba56496c5dfee3ed900d3e',
                               'stabilized-ground': '8f1c4846023e4a954ddb672b66f2a34c36ede8423d1011557800fe5149a36031'}
        assert report['parameters_sha256'] == expected_parameters[args.condition]
        assert len(report['physical_50ms_bin_counts']) == 50
        assert sum(report['physical_50ms_bin_counts']) + report['terminal_tick_events'] == report['spikes']
        measured.append(dict(source=str(path), spikes=report['spikes'], seconds=2.5, sha256=sha(path)))
    rate = max(p['spikes'] / p['seconds'] for p in measured)
    scenarios = {}
    for label, factor in [('observed_rate', 1), ('twice_observed_rate', 2)]:
        count = math.ceil(rate * 10.5 * factor)
        # Last-step spikes are independently bounded by the whole neuron count.
        scenarios[label] = dict(spikes=count, **output_bytes(new, count, 4129924))
    timings = ranks['synchronized_phases']
    simulation = timings['simulation_synchronized_max_seconds']
    initialization = timings['initialization_synchronized_max_seconds']
    report = dict(schema='b2-mam-duration-preparation-v1', condition=args.condition,
        simulation_started=False, mpi_prepared=False, requires_fresh_resource_admission=True,
        source_model_sha256=baseline['model_sha256'], model_sha256=sha(args.output / 'model.json'),
        instance_layer_sha256=new['protocol']['layers']['instance'], duration_seconds=10.5, **delta,
        existing_dump_formula_exact=True, existing_output=predicted, existing_last_step_spikes=last,
        measured_rate_sources=measured, output_scenarios=scenarios,
        runtime_projection=dict(baseline_simulation_seconds=simulation, baseline_initialization_seconds=initialization,
            projected_simulation_seconds=simulation * 4.2, projected_init_plus_simulation_seconds=initialization + simulation * 4.2,
            assumed_unchanged_simulation_rate=True, excludes_preparation_and_collection=True),
        proposed_runtime_envelope=dict(nodes=['25', '81', '83', '71'], ranks=32, ranks_per_node=8,
            cpu_quota_per_proxy=8, memory_mib_per_proxy=262144, minimum_host_reserve_gib=64,
            wall_guard_seconds=6000, per_file_limit_mib=24576, swap_mib=0,
            one_full_simulation_at_a_time=True, admitted=False),
        model_file_bytes=(args.output / 'model.json').stat().st_size,
        preparation_seconds=time.perf_counter() - started,
        source_sha256={str(p):sha(p) for p in [args.run_audit, args.source / 'model.json', Path(__file__), Path(__file__).with_name('mam_extend_duration.py')]},
        scope='Exact duration-only model preparation and measured workload scenarios. No simulation admitted; output, memory, runtime, exact 2.5 s prefix and scientific acceptance must still be validated.')
    (args.output / 'duration-preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'results', 'baseline', 'output', 'run-audit', 'python-source']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--native-audit', type=Path, action='append', default=[])
    p.add_argument('--condition', choices=['metastable', 'stabilized-ground'], required=True)
    prepare(p.parse_args())
