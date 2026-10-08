"""Seed-level, matched-control analysis of completed LK scientific protocols.

Assemblies, neurons and edges are not independent experimental replicates.
All activity comparisons within a seed use the same retained time bins.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from litwin_kumar_figures import csv_rows, science


def sign_flip_test(differences):
    """Two-sided exact paired test under sign symmetry; no Monte Carlo p-values."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not 1 <= len(values) <= 20 or not np.isfinite(values).all():
        raise ValueError('exact sign-flip test requires 1 to 20 finite paired values')
    scale = np.max(np.abs(values))
    if scale == 0:
        return 1.
    values = values/scale
    observed = abs(values.mean())
    exceedances = sum(abs(np.dot(signs, values)/len(values)) >= observed-1e-14
                      for signs in itertools.product([-1, 1], repeat=len(values)))
    return exceedances/2**len(values)


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(
        pvalues[order]*(len(pvalues)-np.arange(len(pvalues)))))
    return adjusted


def selective_activity(rates, population, baseline, spontaneous):
    """Remove common population activity before measuring assembly selectivity."""
    if baseline.sum() < 20 or not spontaneous.any():
        raise ValueError('insufficient common baseline or spontaneous activity')
    residual = rates-population[None, :]
    before, after = residual[:, baseline], residual[:, spontaneous]
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('missing activity in supposedly covered bins')
    threshold = before.mean(axis=1)+3*before.std(axis=1, ddof=1)
    return {
        'selective_excess_change_hz': float(after.max(axis=0).mean()-before.max(axis=0).mean()),
        'selective_occupancy': float((after > threshold[:, None]).mean()),
        'spontaneous_population_hz': float(population[spontaneous].mean()),
    }


def population_conditioned_activity(rates, population, sizes, population_size, bin_s, baseline):
    """Standardize against a baseline-fitted conditional-binomial count reference.

    Conditioning on the current population count removes a common rate change
    and its count-variance increase. This is a diagnostic reference, not a
    calibrated null for correlated network spikes.
    """
    rates, population, sizes = map(np.asarray, (rates, population, sizes))
    baseline = np.asarray(baseline, dtype=bool)
    if (rates.ndim != 2 or population.shape != (rates.shape[1],)
            or baseline.shape != population.shape or sizes.shape != (rates.shape[0],)
            or bin_s <= 0 or not np.isfinite(bin_s)
            or np.any(sizes <= 0) or np.any(sizes >= population_size)
            or baseline.sum() < 20):
        raise ValueError('invalid population-conditioned activity layout')
    total = population*population_size*bin_s
    counts = rates*sizes[:, None]*bin_s
    if (not np.isfinite(total[baseline]).all() or total[baseline].sum() <= 0
            or not np.isfinite(counts[:, baseline]).all()):
        raise ValueError('finite nonempty baseline spike counts required')
    fractions = counts[:, baseline].sum(axis=1)/total[baseline].sum()
    if np.any((fractions <= 0) | (fractions >= 1)):
        raise ValueError('each baseline assembly must carry a nonzero proper fraction of spikes')
    expected = fractions[:, None]*total
    variance = fractions[:, None]*(1-fractions[:, None])*total
    score = np.divide(counts-expected, np.sqrt(np.maximum(variance, 0)),
                      out=np.full_like(expected, np.nan), where=variance > 0)
    score[:, total == 0] = np.where(counts[:, total == 0] == 0, 0., np.nan)
    return score, fractions


def reactivation_episodes(rates, population, time, baseline, spontaneous, coverage,
                         minimum_duration_s=.1):
    """Describe sustained dominant-assembly excess, never bridge unobserved bins."""
    rates, population, time = map(np.asarray, (rates, population, time))
    baseline, spontaneous, coverage = [np.asarray(mask, dtype=bool)
                                      for mask in (baseline, spontaneous, coverage)]
    if (rates.ndim != 2 or time.ndim != 1 or rates.shape[0] < 1
            or rates.shape[1] != len(time) or len(time) < 2
            or any(array.shape != time.shape for array in
                   [population, baseline, spontaneous, coverage])):
        raise ValueError('reactivation arrays must share one time grid')
    dt = float(time[1]-time[0])
    if dt <= 0 or not np.allclose(np.diff(time), dt, rtol=0, atol=1e-10):
        raise ValueError('reactivation requires an equally spaced time grid')
    if (not np.isfinite(minimum_duration_s) or minimum_duration_s <= 0
            or baseline.sum() < 20 or not coverage[baseline].all()):
        raise ValueError('positive minimum duration and at least 20 covered baseline bins required')
    residual = rates-population[None, :]
    if not np.isfinite(residual[:, baseline]).all():
        raise ValueError('baseline rates must be finite')
    threshold = residual[:, baseline].mean(axis=1)+3*residual[:, baseline].std(axis=1, ddof=1)
    finite = np.isfinite(residual).all(axis=0)
    valid = coverage & spontaneous & finite
    dominant = np.argmax(np.where(np.isfinite(residual), residual, -np.inf), axis=0)
    active = valid & (residual[dominant, np.arange(len(time))] > threshold[dominant])
    labels = np.where(active, dominant, -1)
    boundaries = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
    episodes = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if labels[start] < 0 or (stop-start)*dt < minimum_duration_s-1e-10:
            continue
        episodes.append({'assembly': int(labels[start])+1,
            'start_s': float(time[start]-dt/2), 'stop_s': float(time[stop-1]+dt/2),
            'observed_duration_s': float((stop-start)*dt),
            'left_censored': bool(start == 0 or not valid[start-1]),
            'right_censored': bool(stop == len(time) or not valid[stop]),
            'start_bin': int(start), 'stop_bin': int(stop)})
    duration = sum(row['observed_duration_s'] for row in episodes)
    return episodes, {
        'reactivation_episode_count': len(episodes),
        'reactivated_assembly_count': len({row['assembly'] for row in episodes}),
        'observed_episode_duration_median_s': (float(np.median(
            [row['observed_duration_s'] for row in episodes])) if episodes else None),
        'sustained_selective_occupancy': duration/(valid.sum()*dt) if valid.any() else None,
        'adjacent_dominant_switches': sum(a['stop_bin'] == b['start_bin'] and
            a['assembly'] != b['assembly'] for a, b in zip(episodes[:-1], episodes[1:])),
        'censored_episode_count': sum(row['left_censored'] or row['right_censored']
                                     for row in episodes),
    }


def plot_seed_comparisons(rows, output):
    import matplotlib.pyplot as plt
    order = [condition for condition in ['full', 'no_stimulation', 'no_istdp', 'no_normalization']
             if any(row['condition'] == condition for row in rows)]
    labels = {'full': 'Full model', 'no_stimulation': 'No patterned\ninput',
              'no_istdp': 'No iSTDP', 'no_normalization': 'No weight\nnormalization'}
    seeds = sorted({row['seed'] for row in rows})
    lookup = {(row['seed'], row['condition']): row for row in rows}
    rate_log = all(row['spontaneous_population_hz'] > 0 for row in rows)
    panels = [('within_between_ratio', 'Within / between E→E weight', 1.),
              ('population_conditioned_excess_change', 'Population-conditioned\nselectivity change', 1.),
              ('spontaneous_population_hz', 'Spontaneous E rate (Hz)', 1.),
              ('upper_bound_fraction', 'E→E weights at upper bound (%)', 100.)]
    with plt.rc_context({'font.size': 8, 'pdf.fonttype': 42, 'savefig.dpi': 300,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), layout='constrained')
        for letter, ax, (metric, ylabel, factor) in zip('abcd', axes.flat, panels):
            for index, seed in enumerate(seeds):
                values = [lookup[seed, condition][metric]*factor for condition in order]
                ax.plot(np.arange(len(order)), values, 'o-', linewidth=.8, markersize=3,
                        color=plt.get_cmap('tab10')(index % 10), label=str(seed), alpha=.8)
            ax.text(.02, .98, letter, transform=ax.transAxes, va='top',
                    fontweight='bold', fontsize=11)
            ax.set(xticks=np.arange(len(order)), xticklabels=[labels[c] for c in order], ylabel=ylabel)
            if metric == 'within_between_ratio':
                maximum = max(row[metric] for row in rows)
                minimum = min(row[metric] for row in rows)
                ax.set_ylim(min(.8, minimum*.95), max(1.2, maximum*1.05))
                ax.axhline(1., color='#888888', linewidth=.6, linestyle='--')
            elif metric == 'population_conditioned_excess_change':
                ax.axhline(0., color='#888888', linewidth=.6, linestyle='--')
            elif metric == 'spontaneous_population_hz' and rate_log:
                ax.set_yscale('log')
                ax.set_ylabel(ylabel+' · log scale')
            elif metric == 'upper_bound_fraction':
                ax.set_ylim(0, max(1., max(row[metric] for row in rows)*factor*1.1))
        axes[0, 0].legend(title='Independent seed', frameon=False, fontsize=6, title_fontsize=7)
        fig.suptitle(f'LK2014 triplet variant — {len(seeds)} paired network seeds', fontsize=10)
        fig.savefig(output/'seed_comparisons.pdf')
        fig.savefig(output/'seed_comparisons.png')
        plt.close(fig)
    (output/'caption.md').write_text(
        'Each line connects matched conditions for one independently constructed network/input seed. '
        'All seeds are shown; neurons, memberships and synapses are not independent replicates.\n\n'
        'a, Final mean E→E weight within a shared stimulus membership divided by the mean over '
        'edges with no shared membership. b, Change from pretraining to spontaneous activity '
        'in the mean maximum population-conditioned assembly score. For assembly count X and '
        'population count K in a 50 ms bin, the score is (X-pK)/sqrt(p(1-p)K), with p fitted '
        'from the pretraining count fraction. This dimensionless diagnostic accounts for '
        'common population-rate scaling; it is not a calibrated null for correlated spikes. '
        'c, Spontaneous E population rate. Activity panels use the identical intersection of '
        'retained 50 ms bins across conditions within each seed. d, Fraction of E→E weights '
        'at the 21.4 pF upper bound.\n\n'
        'The source values, coverage, paired differences and exploratory multiplicity-adjusted '
        'sign-flip tests are in seed_metrics.csv and report.json. These metrics alone do not '
        'establish attractor dynamics or manuscript readiness.\n\n'
        + ('Panel c uses a logarithmic rate axis; all source rates are positive.\n' if rate_log else '')
        + 'Retained activity coverage per seed: '
        + '; '.join(f'{seed}: {lookup[seed, order[0]]["common_baseline_seconds"]:g} s pretraining, '
                    f'{lookup[seed, order[0]]["common_spontaneous_seconds"]:g} s spontaneous'
                    for seed in seeds)
        + '. These durations describe retained bins, which can be separated by recording gaps; '
        'they are not necessarily continuous intervals or the entire simulated phase.\n')


def analyze(suite, output):
    jobs = json.loads((suite/'science_jobs.json').read_text())
    if len({(j['seed'], j['condition']) for j in jobs}) != len(jobs):
        raise ValueError('duplicate seed/condition in scientific manifest')
    # Fail before creating artifacts if any required simulation remains incomplete.
    for job in jobs:
        result = suite/job['label']/'result.json'
        if not result.exists():
            raise ValueError(f'incomplete scientific job: {job["label"]}')
        run = json.loads(result.read_text())
        config = run['configuration']
        if (config['scale'] != 1 or config['mode'] != 'learn'
                or run['biological_seconds'] < config['duration_s']):
            raise ValueError(f'full de novo protocol required: {job["label"]}')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in sorted({j['seed'] for j in jobs}):
        selected = [j for j in jobs if j['seed'] == seed]
        if 'full' not in {j['condition'] for j in selected}:
            raise ValueError('each seed requires the full-model condition')
        data, reports, time = {}, {}, None
        for job in selected:
            destination = output/job['label']
            destination.mkdir()
            run, membership, *_, diagnostics = science(suite/job['label'], destination)
            with np.load(destination/'activity_source.npz') as source:
                data[job['condition']] = {key: source[key].copy() for key in
                    ['time_s', 'assembly_rates_hz', 'population_rate_hz', 'coverage']}
            data[job['condition']]['assembly_sizes'] = membership.sum(axis=1)
            current = data[job['condition']]['time_s']
            if time is not None and not np.array_equal(current, time):
                raise ValueError('activity time grids differ across controls')
            time = current
            reports[job['condition']] = run
        common = np.logical_and.reduce([d['coverage'] for d in data.values()])
        config = reports['full']['configuration']
        for condition, run in reports.items():
            differences = {key for key in config if run['configuration'][key] != config[key]}
            expected = {'full': set(), 'no_stimulation': {'stimulation'},
                        'no_istdp': {'inhibitory_plasticity'},
                        'no_normalization': {'normalization'}}[condition]
            if differences != expected:
                raise ValueError(f'unmatched configuration for {condition}: {differences}')
        baseline = common & (time >= min(1., config['warmup_s']/2)) & (time < config['warmup_s'])
        spontaneous = common & (time >= config['warmup_s']+config['training_s'])
        bin_s = float(time[1]-time[0])
        for condition, item in data.items():
            weights = reports[condition]['weights']
            adjusted, fractions = population_conditioned_activity(
                item['assembly_rates_hz'], item['population_rate_hz'],
                item['assembly_sizes'], config['ne'], bin_s, baseline)
            adjusted_metrics = selective_activity(adjusted, np.zeros_like(time), baseline, spontaneous)
            episodes, episode_metrics = reactivation_episodes(
                adjusted, np.zeros_like(time), time,
                baseline, spontaneous, common)
            (output/f'reactivation-seed-{seed}-{condition}.json').write_text(
                json.dumps({'baseline_spike_fractions': fractions.tolist(),
                            'bin_s': bin_s, 'episodes': episodes}, indent=2, allow_nan=False)+'\n')
            rows.append({'seed': seed, 'condition': condition,
                'within_between_ratio': weights['within_mean_pf']/weights['between_mean_pf'],
                'upper_bound_fraction': weights['upper_bound_fraction'],
                'inhibitory_upper_fraction': weights['inhibitory_upper_fraction'],
                'common_baseline_seconds': float(baseline.sum()*bin_s),
                'common_spontaneous_seconds': float(spontaneous.sum()*bin_s),
                **episode_metrics,
                'population_conditioned_excess_change': adjusted_metrics['selective_excess_change_hz'],
                'population_conditioned_occupancy': adjusted_metrics['selective_occupancy'],
                **selective_activity(item['assembly_rates_hz'], item['population_rate_hz'],
                                     baseline, spontaneous)})
    csv_rows(output/'seed_metrics.csv', rows)
    contrasts = []
    full = {r['seed']: r for r in rows if r['condition'] == 'full'}
    for control in sorted({r['condition'] for r in rows}-{'full'}):
        controls = {r['seed']: r for r in rows if r['condition'] == control}
        if set(controls) != set(full):
            raise ValueError(f'unbalanced matched seeds for {control}')
        seeds = sorted(full)
        for metric in ['within_between_ratio', 'selective_excess_change_hz',
                       'population_conditioned_excess_change', 'spontaneous_population_hz']:
            differences = [full[seed][metric]-controls[seed][metric] for seed in seeds]
            contrasts.append({'control': control, 'metric': metric, 'seeds': seeds,
                'paired_full_minus_control': differences, 'n_seeds': len(seeds),
                'mean_difference': float(np.mean(differences)),
                'two_sided_sign_flip_p': sign_flip_test(differences)})
    for row, adjusted in zip(contrasts, holm_adjust([c['two_sided_sign_flip_p'] for c in contrasts])):
        row['holm_adjusted_p'] = float(adjusted)
    report = {'replicate': 'independent network/input seed', 'seed_metrics': rows,
        'paired_contrasts': contrasts,
        'activity_sampling': 'identical intersection of retained 50 ms time bins across all conditions within a seed',
        'selectivity': 'maximum assembly rate minus population rate, averaged across bins; spontaneous minus baseline',
        'occupancy': 'fraction of assembly/bins above the baseline residual mean + 3 SD; descriptive',
        'population_conditioning': 'For population count K and assembly count X per bin, fit p=sum(X_baseline)/sum(K_baseline); score=(X-p*K)/sqrt(p*(1-p)*K). This removes a common rate scaling and associated conditional-binomial count variance. The reference is diagnostic, not a calibrated null for correlated network spikes; raw residual-rate results are also reported.',
        'reactivation_episodes': 'dominant population-conditioned assembly score above its baseline mean + 3 SD for at least 100 ms; coverage gaps split episodes, observation-boundary censoring is flagged; durations are observed durations and are not corrected for censoring',
        'inference': 'exploratory two-sided paired sign-flip test assumes symmetric seed-level differences; Holm correction across all reported contrasts',
        'small_sample_limit': 'three paired seeds have a minimum attainable two-sided p-value of 0.25; no significant result can be claimed at 0.05',
        'interpretation': 'selectivity and weight enrichment require scientific review; these metrics alone do not establish attractor dynamics or manuscript readiness'}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plot_seed_comparisons(rows, output)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.suite, args.output)
