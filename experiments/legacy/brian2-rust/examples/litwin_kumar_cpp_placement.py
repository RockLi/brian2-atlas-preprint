"""Select Linux C++ CPU placement using byte-checked existing LK binaries.

Run only after competing long jobs finish, with the original CPU/NUMA binding.
This is configuration selection; the winner still needs fresh Rust/C++ timing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import psutil

from litwin_kumar_benchmark import environment, measured_process, sha256, write_json


def matching_rust_places(report, threads):
    samples = [r for r in report['samples']
               if r['backend'] == 'rust' and r['threads'] == threads]
    if not samples:
        raise ValueError(f'no Rust CPU-placement evidence for {threads} workers')
    orders = []
    for row in samples:
        summary = row.get('summary', {})
        cpus = summary.get('thread_cpus', [])
        if (not summary.get('thread_affinity') or len(cpus) != threads
                or len(set(cpus)) != threads
                or any(type(cpu) is not int or cpu < 0 for cpu in cpus)):
            raise ValueError('explicit unique Rust worker CPU placement required')
        orders.append(cpus)
    if any(order != orders[0] for order in orders):
        raise ValueError('Rust CPU placement changes across baseline repeats')
    return ','.join('{'+str(cpu)+'}' for cpu in orders[0])


def array_digest(project):
    files = sorted((project/'results').glob('*array_lk_*'))
    if not files:
        raise ValueError('no C++ model arrays to validate')
    return hashlib.sha256(''.join(sha256(p) for p in files).encode()).hexdigest()


def run(baseline, output, threads, repeats):
    if platform.system() != 'Linux':
        raise ValueError('this CPU-placement experiment requires Linux affinity')
    if repeats < 2 or not threads or len(set(threads)) != len(threads) or min(threads) < 1:
        raise ValueError('unique positive thread counts and at least two repeats required')
    report = json.loads((baseline/'report.json').read_text())
    expected_environment = json.loads((baseline/'environment.json').read_text())
    current = environment()
    expected_cpus = expected_environment.get('allowed_cpus')
    if not expected_cpus or sorted(psutil.Process().cpu_affinity()) != sorted(expected_cpus):
        raise ValueError('CPU allowance differs from the original performance experiment')
    if current['numa_policy'] != expected_environment.get('numa_policy'):
        raise ValueError('NUMA policy differs from the original performance experiment')
    if not report.get('cpp_fixed_worker_repeats_byte_exact'):
        raise ValueError('baseline must include verified repeatable C++ outputs')
    places, hashes, binaries = {}, {}, {}
    for n in threads:
        places[n] = matching_rust_places(report, n)
        project = baseline/f'cpp-{n}-build/project'
        hashes[n], binaries[n] = array_digest(project), sha256(project/'main')
        if not (project/'results/lk_warmup_seconds.txt').exists():
            raise ValueError('split-warmup executable required for both timing scopes')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'environment.json', current)
    write_json(output/'contract.json', {
        'baseline': str(baseline.resolve()), 'baseline_report_sha256': sha256(baseline/'report.json'),
        'cpp_binary_sha256': binaries, 'threads': threads, 'repeats': repeats,
        'bindings': ['spread', 'close', 'matched-rust'], 'matched_rust_places': places,
        'wait_policy': 'ACTIVE', 'purpose': 'configuration selection only; fresh confirmation required',
        'timing': 'post-warmup and full warmup+learning measured separately; no build time included'})
    schedule = [(n, binding, repeat) for n in threads
                for binding in ['spread', 'close', 'matched-rust'] for repeat in range(repeats)]
    np.random.default_rng(20260914).shuffle(schedule)
    rows = []
    for n, binding, repeat in schedule:
        project = baseline/f'cpp-{n}-build/project'
        env = {**os.environ, 'OMP_NUM_THREADS': str(n), 'OMP_WAIT_POLICY': 'ACTIVE',
               'OMP_PROC_BIND': 'close' if binding == 'matched-rust' else binding,
               'OMP_PLACES': places[n] if binding == 'matched-rust' else 'cores'}
        process = measured_process([str((project/'main').resolve())],
            output/f'cpp-{n}-{binding}-{repeat}', cwd=project, env=env)
        digest = array_digest(project)
        if digest != hashes[n]:
            raise AssertionError(f'C++ CPU placement changed full output: {n}/{binding}')
        post = float((project/'results/last_run_info.txt').read_text().split()[0])
        warm = float((project/'results/lk_warmup_seconds.txt').read_text())
        rows.append({'threads': n, 'binding': binding, 'repeat': repeat,
                     'post_warmup_seconds': post, 'whole_seconds': warm+post,
                     'result_sha256': digest, 'process': process})
        write_json(output/'samples.json', rows)
        print(n, binding, repeat, post, warm+post, flush=True)
    aggregates = []
    for n in threads:
        for binding in ['spread', 'close', 'matched-rust']:
            selected = [r for r in rows if r['threads'] == n and r['binding'] == binding]
            item = {'threads': n, 'binding': binding, 'n': len(selected)}
            for metric in ['post_warmup_seconds', 'whole_seconds']:
                values = [r[metric] for r in selected]
                item[metric] = {'median': float(np.median(values)), 'min': min(values), 'max': max(values)}
            aggregates.append(item)
    result = {'aggregates': aggregates, 'all_results_byte_exact': True,
              'selection_only': True, 'samples': rows}
    write_json(output/'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, nargs='+', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    run(**vars(parser.parse_args()))
