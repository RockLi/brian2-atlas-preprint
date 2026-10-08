"""Descriptive full-protocol sensitivity on matched network seeds and time bins."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from litwin_kumar_activity_review import review_data
from litwin_kumar_backend_comparison import full_jobs, matched_activity_metrics
from litwin_kumar_figures import csv_rows, science

CONDITIONS = {
    'uniform-delay': {'delay_distribution': 'uniform'},
    'euler-attenuation': {'trace_integration': 'euler-event'},
    'sources-10000': {'input_sources': 10000},
    'dt-005': {'dt_ms': .05},
    'joint': {'delay_distribution': 'uniform', 'trace_integration': 'euler-event',
              'input_sources': 10000, 'dt_ms': .05},
}


def matched_configuration(primary, variant, condition):
    historical = {'delay_distribution': 'fixed', 'trace_integration': 'event-driven'}
    baseline = {**historical, **primary}
    actual = {**historical, **variant}
    if condition not in CONDITIONS:
        raise ValueError('unknown sensitivity condition')
    if actual != {**baseline, **CONDITIONS[condition]}:
        raise ValueError(f'unplanned configuration difference: {condition}')
    if any(baseline.get(key) != value for key, value in
           {'delay_distribution': 'fixed', 'trace_integration': 'event-driven',
            'dt_ms': .1, 'input_sources': 1000}.items()):
        raise ValueError('primary run is not the declared sensitivity baseline')
    return baseline


def analyze(primary_suite, primary_analysis, sensitivity_suite, output):
    primary = full_jobs(primary_suite)
    jobs = json.loads((sensitivity_suite/'sensitivity_jobs.json').read_text())
    expected = {(seed, condition) for seed in primary for condition in CONDITIONS}
    actual = [(j['seed'], j['condition']) for j in jobs]
    if len(primary) < 3 or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('complete balanced sensitivity matrix with at least three seeds required')
    runs, paths = {}, {}
    for seed, path in primary.items():
        paths[seed, 'baseline'] = path
        runs[seed, 'baseline'] = json.loads((path/'result.json').read_text())
    for job in jobs:
        key = job['seed'], job['condition']
        paths[key] = sensitivity_suite/job['label']
        runs[key] = json.loads((paths[key]/'result.json').read_text())
        matched_configuration(runs[job['seed'], 'baseline']['configuration'],
                              runs[key]['configuration'], job['condition'])
    for key, run in runs.items():
        config = run['configuration']
        if (run['backend'] != 'rust' or config['scale'] != 1 or config['mode'] != 'learn'
                or config['seed'] != key[0] or config['duration_s'] != 2610
                or run['biological_seconds'] != config['duration_s']
                or not run['complete_training_protocol']):
            raise ValueError(f'incomplete full-scale protocol: {key}')
    output.mkdir(parents=True, exist_ok=False)
    rows, checks, episodes = [], [], {}
    for seed in sorted(primary):
        baseline_run = runs[seed, 'baseline']
        baseline_path = paths[seed, 'baseline']
        baseline_source = primary_analysis/baseline_path.name/'activity_source.npz'
        with np.load(baseline_source) as source:
            baseline_data = {name: source[name] for name in
                ['time_s', 'assembly_rates_hz', 'population_rate_hz', 'coverage']}
        for condition in ['baseline', *CONDITIONS]:
            key = seed, condition
            path, run = paths[key], runs[key]
            if run['topology_edges'] != baseline_run['topology_edges']:
                raise ValueError(f'projection edge counts differ: {key}')
            with np.load(baseline_path/'activity.npz') as a, np.load(path/'activity.npz') as b:
                digests = {}
                for name in ['ee_i', 'ee_j', 'ee_initial', 'membership_e', 'membership_i']:
                    x, y = a[name], b[name]
                    if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                        raise ValueError(f'initial structure differs: {key}, {name}')
                    digests[name] = hashlib.sha256(x.tobytes()).hexdigest()
                sizes = a['membership_e'].sum(axis=1)
            checks.append({'seed': seed, 'condition': condition,
                           'byte_exact_structure_sha256': digests,
                           'all_projection_edge_counts_equal': True})
            if condition == 'baseline':
                data = baseline_data
            else:
                destination = output/path.name
                destination.mkdir()
                science(path, destination)
                with np.load(destination/'activity_source.npz') as source:
                    data = {name: source[name] for name in baseline_data}
            # Full coverage is mandatory, including first/last 100 s. This
            # prevents a rolling-window or dt change from altering the interval.
            _, drift = review_data(data, sizes, run['configuration'])
            review_data(baseline_data, sizes, baseline_run['configuration'])
            metrics, events = matched_activity_metrics(
                {'rust': baseline_data, 'cpp': data}, sizes, baseline_run['configuration'])
            values = metrics['cpp']  # Reuse the paired-interval calculation only.
            weights = run['weights']
            rows.append({'seed': seed, 'condition': condition,
                'within_between_ratio': weights['within_mean_pf']/weights['between_mean_pf'],
                'upper_bound_fraction': weights['upper_bound_fraction'],
                'inhibitory_upper_fraction': weights['inhibitory_upper_fraction'],
                **values, **drift})
            episodes[f'{seed}/{condition}'] = events['cpp']
            csv_rows(output/'sensitivity_seed_metrics.csv', rows)
    report = {'seed_metrics': rows, 'initial_structure_checks': checks,
        'conditions': CONDITIONS, 'analysis_completed': True,
        'activity_sampling': 'same 50 ms grid; complete 9 s baseline and 1000 s spontaneous interval in every condition',
        'structure_scope': 'EE edges, initial EE weights, both memberships byte exact; all projection edge counts equal',
        'interpretation': 'Descriptive seed-level sensitivity. Same network seed does not match physical stochastic input across dt/source discretizations. No deterministic long-trajectory convergence or powered statistical equivalence claim.',
        'timing': 'Concurrent scientific timings are excluded from performance acceptance.'}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output/'reactivation_episodes.json').write_text(json.dumps(episodes, indent=2, allow_nan=False)+'\n')
    plot(rows, output)
    return report


def plot(rows, output):
    import matplotlib.pyplot as plt
    order = ['baseline', *CONDITIONS]
    labels = ['Baseline', 'Uniform\ndelay', 'Euler\nattenuation', '10,000\nsources', 'dt 0.05 ms', 'Joint']
    panels = [('within_between_ratio', 'Within / between E→E weight', 1),
              ('population_conditioned_excess_change', 'Conditioned selectivity change', 1),
              ('spontaneous_population_hz', 'Spontaneous E rate (Hz)', 1),
              ('late_over_early_population_rate', 'Last / first 100 s E rate', 1)]
    with plt.rc_context({'font.size': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 5), layout='constrained')
        seeds = sorted({r['seed'] for r in rows})
        for letter, ax, (metric, label, factor) in zip('abcd', axes.flat, panels):
            for index, seed in enumerate(seeds):
                values = [next(r[metric] for r in rows if r['seed'] == seed and r['condition'] == c) for c in order]
                offset = (index-(len(seeds)-1)/2)*.09
                ax.plot(np.arange(len(order))+offset, np.array(values)*factor, 'o', markersize=4, label=str(seed))
            ax.set(xticks=np.arange(len(order)), xticklabels=labels, ylabel=label)
            ax.text(.02, .97, letter, va='top', transform=ax.transAxes, fontweight='bold')
        axes[0, 0].legend(title='Network seed', frameon=False, fontsize=6)
        fig.suptitle('Full 2,610 s sensitivity · descriptive paired network seeds', fontsize=10)
        fig.savefig(output/'sensitivity_comparison.pdf')
        fig.savefig(output/'sensitivity_comparison.png', dpi=200)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['primary-suite', 'primary-analysis', 'sensitivity-suite', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    analyze(**vars(parser.parse_args()))
