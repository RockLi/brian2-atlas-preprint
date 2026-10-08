"""Build a pinned Atlas consumer and repeat a bounded PD14 simulation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import tomllib


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-dir', type=Path)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    target = (args.target_dir or output / 'build').resolve()
    manifest = tomllib.loads((root / 'Cargo.toml').read_text())
    lock = tomllib.loads((root / 'Cargo.lock').read_text())
    dependency = manifest['dependencies']['b2-runner']
    engine = next(p for p in lock['package'] if p['name'] == 'b2-runner')
    assert engine['source'].endswith('#' + dependency['rev']), engine
    source = (root / manifest['bin'][0]['path']).resolve()
    start = time.time()
    commands = []

    def run(command, name):
        commands.append(command)
        with (output / name).open('w') as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=args.timeout)

    run(['cargo', 'build', '--release', '--locked', '--manifest-path', str(root / 'Cargo.toml'),
         '--target-dir', str(target)], 'build.log')
    run(['cargo', 'test', '--release', '--locked', '--manifest-path', str(root / 'Cargo.toml'),
         '--target-dir', str(target)], 'model-tests.log')
    binary = target / 'release' / ('atlas-preprint-pd14.exe' if os.name == 'nt' else 'atlas-preprint-pd14')
    options = ['--neuron-scale', '0.01', '--indegree-scale', '0.01', '--duration-ms', '100',
               '--repetitions', '2', '--warmups', '0', '--seed', '55', '--maximum-gib', '0.25']
    results = []
    for index in range(2):
        path = output / f'run-{index + 1}.json'
        run([str(binary), *options, '--output', str(path)], f'run-{index + 1}.log')
        result = json.loads(path.read_text())
        assert result['schema'] == 'b2-pd14-benchmark-v1'
        assert result['population_names'] == ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
        assert len(result['simulation_seconds']) == 2
        assert result['neuron_count'] == sum(result['population_neurons']) == 772
        assert 0 < result['synapse_count'] < 100000
        assert result['resident_topology_bytes'] > 0
        assert result['delivered_events'] > 0 and sum(result['spike_counts']) > 0
        assert all(math.isfinite(x) and x >= 0 for x in result['population_rates_hz'])
        results.append(result)
    fields = ['numeric_profile', 'population_names', 'population_neurons', 'neuron_count',
              'projection_count', 'synapse_count', 'estimated_topology_bytes',
              'resident_topology_bytes', 'spike_counts', 'delivered_events', 'population_rates_hz']
    for key in fields:
        assert results[0][key] == results[1][key], key
    report = {'schema': 'atlas-preprint-pd14-reproduction-v1', 'status': 'passed',
              'atlas_commit': dependency['rev'], 'atlas_git': dependency['git'],
              'engine_cargo_source': engine['source'], 'model_source_sha256': digest(source),
              'cargo_lock_sha256': digest(root / 'Cargo.lock'), 'binary_sha256': digest(binary),
              'platform': platform.platform(), 'started_unix': start, 'ended_unix': time.time(),
              'commands': commands, 'result_sha256': {f'run-{i}.json': digest(output / f'run-{i}.json') for i in [1, 2]},
              'deterministic_fields': fields, 'result': {key: results[0][key] for key in fields},
              'scope': 'New bounded reproduction at the pinned Atlas commit: 1% neuron and indegree scales, 100 ms, two fresh processes with two identical repetitions each. This does not reproduce full-scale scientific rates or historical benchmark timings.'}
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'report': str(output / 'validation.json')}))


if __name__ == '__main__':
    main()
