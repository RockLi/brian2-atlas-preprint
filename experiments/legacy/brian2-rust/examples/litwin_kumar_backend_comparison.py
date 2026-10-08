"""Compare completed long Rust/C++ protocols on identical observed time bins."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from litwin_kumar_analysis import (population_conditioned_activity,
                                  reactivation_episodes, selective_activity)
from litwin_kumar_figures import csv_rows


def matched_activity_metrics(data, sizes, config):
    """Use the backend intersection, not the shorter ablation-cohort coverage."""
    time = data['rust']['time_s']
    if not np.array_equal(time, data['cpp']['time_s']):
        raise ValueError('backend activity time grids differ')
    if len(time) < 2:
        raise ValueError('at least two activity bins required')
    common = data['rust']['coverage'] & data['cpp']['coverage']
    baseline = common & (time >= min(1., config['warmup_s']/2)) & (time < config['warmup_s'])
    spontaneous = common & (time >= config['warmup_s']+config['training_s'])
    bin_s = float(time[1]-time[0])
    if baseline.sum() < 20 or not spontaneous.any():
        raise ValueError('insufficient matched baseline or spontaneous coverage')
    rows, episodes = {}, {}
    for backend, item in data.items():
        adjusted, fractions = population_conditioned_activity(
            item['assembly_rates_hz'], item['population_rate_hz'], sizes,
            config['ne'], bin_s, baseline)
        conditioned = selective_activity(adjusted, np.zeros_like(time), baseline, spontaneous)
        events, event_metrics = reactivation_episodes(
            adjusted, np.zeros_like(time), time, baseline, spontaneous, common)
        rows[backend] = {
            'common_baseline_seconds': float(baseline.sum()*bin_s),
            'common_spontaneous_seconds': float(spontaneous.sum()*bin_s),
            'population_conditioned_excess_change': conditioned['selective_excess_change_hz'],
            'population_conditioned_occupancy': conditioned['selective_occupancy'],
            **selective_activity(item['assembly_rates_hz'], item['population_rate_hz'], baseline, spontaneous),
            **event_metrics}
        episodes[backend] = {'baseline_spike_fractions': fractions.tolist(), 'episodes': events}
    return rows, episodes


def full_jobs(suite):
    jobs = [j for j in json.loads((suite/'science_jobs.json').read_text()) if j['condition'] == 'full']
    if not jobs or len({j['seed'] for j in jobs}) != len(jobs):
        raise ValueError('expected unique full-model seeds')
    return {j['seed']: suite/j['label'] for j in jobs}


def compare(rust_suite, cpp_suite, rust_analysis, cpp_analysis, output):
    suites = {'rust': full_jobs(rust_suite), 'cpp': full_jobs(cpp_suite)}
    if set(suites['rust']) != set(suites['cpp']):
        raise ValueError('unmatched backend seeds')
    analyses = {'rust': rust_analysis, 'cpp': cpp_analysis}
    rows, checks, episode_data = [], [], {}
    for seed in sorted(suites['rust']):
        runs = {backend: json.loads((paths[seed]/'result.json').read_text())
                for backend, paths in suites.items()}
        config = runs['rust']['configuration']
        if config != runs['cpp']['configuration']:
            raise ValueError(f'backend configuration mismatch: seed {seed}')
        for backend, run in runs.items():
            if (run['backend'] != backend or config['scale'] != 1 or config['mode'] != 'learn'
                    or run['biological_seconds'] != config['duration_s']):
                raise ValueError(f'incomplete full-scale protocol: {backend}, seed {seed}')
        if runs['rust']['topology_edges'] != runs['cpp']['topology_edges']:
            raise ValueError(f'edge count mismatch: seed {seed}')
        # NPZ loads only requested members: never read the multi-GB voltage arrays.
        with np.load(suites['rust'][seed]/'activity.npz') as rust, np.load(suites['cpp'][seed]/'activity.npz') as cpp:
            digests = {}
            for name in ['ee_i', 'ee_j', 'ee_initial', 'membership_e', 'membership_i']:
                a, b = rust[name], cpp[name]
                if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                    raise ValueError(f'initial structure mismatch: seed {seed}, {name}')
                digests[name] = hashlib.sha256(a.tobytes()).hexdigest()
            sizes = rust['membership_e'].sum(axis=1)
        checks.append({'seed': seed, 'byte_exact_fields_sha256': digests,
                       'all_projection_edge_counts_equal': True})
        data = {}
        for backend in suites:
            file = analyses[backend]/suites[backend][seed].name/'activity_source.npz'
            with np.load(file) as source:
                data[backend] = {name: source[name] for name in
                    ['time_s', 'assembly_rates_hz', 'population_rate_hz', 'coverage']}
        metrics, episodes = matched_activity_metrics(data, sizes, config)
        episode_data[str(seed)] = episodes
        for backend, values in metrics.items():
            weights = runs[backend]['weights']
            rows.append({'seed': seed, 'backend': backend,
                'within_between_ratio': weights['within_mean_pf']/weights['between_mean_pf'],
                'upper_bound_fraction': weights['upper_bound_fraction'],
                'inhibitory_upper_fraction': weights['inhibitory_upper_fraction'], **values})
    output.mkdir(parents=True, exist_ok=False)
    csv_rows(output/'backend_seed_metrics.csv', rows)
    report = {'seed_metrics': rows, 'initial_structure_checks': checks,
        'activity_sampling': 'intersection of Rust/C++ full-model retained bins within each seed; independent of ablation recording coverage',
        'interpretation': 'Descriptive paired network seeds. Backend stochastic inputs differ; this is not a trajectory identity or powered statistical equivalence test.',
        'structure_scope': 'E-to-E edge arrays, initial E-to-E weights, both membership arrays are byte exact; other projection edge counts and complete configurations match.',
        'timing': 'Concurrent long scientific runs are excluded from performance evidence.'}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output/'reactivation_episodes.json').write_text(json.dumps(episode_data, indent=2, allow_nan=False)+'\n')
    plot(rows, output)
    return report


def plot(rows, output):
    import matplotlib.pyplot as plt
    panels = [('within_between_ratio', 'Within / between E→E weight', 1.),
              ('population_conditioned_excess_change', 'Population-conditioned\nselectivity change', 1.),
              ('spontaneous_population_hz', 'Spontaneous E rate (Hz)', 1.),
              ('sustained_selective_occupancy', 'Sustained selective occupancy (%)', 100.)]
    with plt.rc_context({'font.size': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.5), layout='constrained')
        for letter, ax, (metric, label, factor) in zip('abcd', axes.flat, panels):
            for seed in sorted({r['seed'] for r in rows}):
                points = [next(r for r in rows if r['seed'] == seed and r['backend'] == backend)[metric]
                          for backend in ['rust', 'cpp']]
                ax.plot([0, 1], np.array(points)*factor, 'o-', linewidth=.8, markersize=4, label=str(seed))
            ax.set(xticks=[0, 1], xticklabels=['Rust', 'Brian2 C++'], ylabel=label, xlim=(-.25, 1.25))
            ax.text(.02, .97, letter, va='top', transform=ax.transAxes,
                    fontweight='bold', fontsize=11)
        axes[0, 0].legend(title='Network seed', frameon=False, fontsize=6, title_fontsize=7)
        fig.suptitle('Full-protocol backend comparison · matched observed intervals', fontsize=10)
        fig.savefig(output/'backend_comparison.pdf')
        fig.savefig(output/'backend_comparison.png', dpi=200)
        plt.close(fig)
    (output/'caption.md').write_text(
        'Each connected pair uses the same network seed and observed time bins. '
        'Endpoints follow the complete 2,610 s protocol. Population-conditioned scores use '
        'each backend’s pretraining spike fractions; sustained episodes exceed the baseline '
        'mean + 3 SD for at least 100 ms. This is a descriptive diagnostic, not a calibrated '
        'correlated-spike null or evidence of statistical equivalence. Raw values and coverage '
        'are in backend_seed_metrics.csv. No timing from these concurrent runs is used.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['rust-suite', 'cpp-suite', 'rust-analysis', 'cpp-analysis', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    compare(**vars(parser.parse_args()))
