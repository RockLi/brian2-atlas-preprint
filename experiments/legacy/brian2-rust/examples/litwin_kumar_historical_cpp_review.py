"""Recompute archived Linux C++ comparators before accepting final performance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPORTS = ['remote23-baseline/report.json', 'remote23-release/whole-run/report.json',
           'remote23-release/final/performance/report.json',
           'remote23-release/final/whole_run/report.json']
POLICY = 'remote23-release/historical-cpp-sources/runtime-policy-report.json'


def summarize(samples, field):
    values = np.array([r[field] for r in samples], dtype=float)
    if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('finite positive raw timings required')
    if any(r['process']['returncode'] != 0 for r in samples):
        raise ValueError('failed process cannot establish a timing comparator')
    return dict(n=len(values), median_seconds=float(np.median(values)),
                min_seconds=float(values.min()), max_seconds=float(values.max()))


def review(root, linux_final_validation=None):
    sources, candidates = {}, []

    def read(relative):
        path = root/relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    for relative in REPORTS:
        report = read(relative)
        suite = report['suite']
        if (suite not in ('performance', 'whole_run') or report['profiled'] is not False
                or report['warmup_s'] != 10
                or report['timed_duration_s'] != (1 if suite == 'performance' else 11)
                or report['recording'] != 'identical full-history SpikeMonitor and StateMonitor'
                or report['cpp_fixed_worker_repeats_byte_exact'] is not True):
            raise ValueError('historical comparator scope or repeatability is invalid')
        aggregates = [r for r in report['aggregates'] if r['backend'] == 'cpp']
        samples = [r for r in report['samples'] if r['backend'] == 'cpp']
        workers = [r['threads'] for r in aggregates]
        keys = [(r['threads'], r['repeat']) for r in samples]
        if (len(workers) != len(set(workers)) or len(keys) != len(set(keys))
                or set(keys) != {(n, k) for n in workers for k in range(5)}):
            raise ValueError('complete historical five-repeat worker grid required')
        for recorded in aggregates:
            n = recorded['threads']
            run = report['builds'][f'cpp-{n}-build']['run']
            if (run['configuration']['ne'] != 4000 or run['configuration']['ni'] != 1000
                    or run['biological_seconds'] != 11 or run['recording_window_steps'] is not None):
                raise ValueError('full-scale full-history historical build required')
            computed = summarize([r for r in samples if r['threads'] == n], 'simulation_seconds')
            if any(recorded[k] != v for k, v in computed.items()):
                raise ValueError('historical aggregate differs from raw samples')
            candidates.append(dict(source=relative, suite=suite, threads=n,
                                   cohort='five-repeat benchmark', **computed))

    policy = read(POLICY)
    if policy['all_results_byte_exact'] is not True:
        raise ValueError('runtime policy output repeatability required')
    expected = {(n, p, k) for n in [4, 8, 16] for p in ['ACTIVE', 'PASSIVE'] for k in range(3)}
    keys = [(r['threads'], r['wait_policy'], r['repeat']) for r in policy['samples']]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('complete three-repeat runtime policy grid required')
    aggregates = {(r['threads'], r['wait_policy']): r for r in policy['aggregates']}
    if len(policy['aggregates']) != 6 or set(aggregates) != {(n, p) for n, p, _ in expected}:
        raise ValueError('complete runtime policy aggregates required')
    for (n, wait_policy), recorded in aggregates.items():
        group = [r for r in policy['samples'] if (r['threads'], r['wait_policy']) == (n, wait_policy)]
        if len({r['result_sha256'] for r in group}) != 1:
            raise ValueError('runtime policy output hashes differ')
        for suite, field in [('performance', 'post_warmup_seconds'), ('whole_run', 'whole_seconds')]:
            computed = summarize(group, field)
            if recorded[field] != computed['median_seconds'] or recorded['n'] != computed['n']:
                raise ValueError('runtime policy aggregate differs from raw samples')
            candidates.append(dict(source=POLICY, suite=suite, threads=n, wait_policy=wait_policy,
                cohort='three-repeat selection; whole time sums two native intervals', **computed))
    strongest = {suite: min((r for r in candidates if r['suite'] == suite),
                           key=lambda r: r['median_seconds']) for suite in ['performance', 'whole_run']}
    result = dict(source_reports_sha256=sources, candidates=candidates, strongest_historical=strongest,
        final_confirmation_reviewed=False,
        scope='Archived same-host C++ worker and runtime-policy cohorts; no cross-cohort pooling. '
              'Policy whole-run values sum split intervals and are retained as a conservative screening comparator. '
              'Final acceptance still requires independent confirmation of the current compiler/placement grid.')
    if linux_final_validation is not None:
        from litwin_kumar_linux_performance_review import review as review_linux
        _, final = review_linux(linux_final_validation)
        checks = {}
        for suite, gate in final['gates'].items():
            historical = [r for r in candidates if r['suite'] == suite]
            checks[suite] = dict(
                selected_cpp_median_no_slower_than_history=gate['cpp']['median_seconds'] <= strongest[suite]['median_seconds'],
                rust_range_beats_all_historical_ranges=gate['rust']['max_seconds'] < min(r['min_seconds'] for r in historical),
                independent_gate_passed=gate['passed'])
        result.update(final_confirmation_reviewed=True, final_source_reports_sha256=final['source_reports_sha256'],
                      checks=checks, all_passed=all(all(c.values()) for c in checks.values()))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--linux-final-validation', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop('output')
    result = review(**args)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(candidates=len(result['candidates']),
                         strongest_historical=result['strongest_historical'],
                         final_confirmation_reviewed=result['final_confirmation_reviewed'])))
