"""Review complete spontaneous activity with fixed windows and explicit rate drift.

Consumes existing scientific analysis arrays, without loading raw voltage or
spike histories. These descriptive diagnostics do not establish stationarity or
attractor dynamics.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from litwin_kumar_analysis import population_conditioned_activity
from litwin_kumar_figures import csv_rows


def review_data(data, sizes, config):
    time = np.asarray(data['time_s'])
    coverage = np.asarray(data['coverage'], dtype=bool)
    population = np.asarray(data['population_rate_hz'])
    rates = np.asarray(data['assembly_rates_hz'])
    if (time.ndim != 1 or len(time) < 2 or coverage.shape != time.shape
            or population.shape != time.shape or rates.shape != (len(sizes), len(time))):
        raise ValueError('inconsistent activity arrays')
    dt = float(time[1]-time[0])
    if dt <= 0 or not np.allclose(np.diff(time), dt, rtol=0, atol=1e-9):
        raise ValueError('activity requires a uniform time grid')
    end = config['warmup_s']+config['training_s']
    duration = config['duration_s']-end
    baseline_start = min(1., config['warmup_s']/2)
    baseline = (time >= baseline_start) & (time < config['warmup_s'])
    spontaneous = (time >= end) & (time < config['duration_s'])
    required = baseline | spontaneous
    if (duration < 200 or baseline.sum() < 20
            or not np.isclose(baseline.sum()*dt, config['warmup_s']-baseline_start,
                              atol=1e-8, rtol=0)
            or not np.isclose(spontaneous.sum()*dt, duration, atol=1e-8, rtol=0)
            or not coverage[required].all() or not np.isfinite(population[required]).all()
            or not np.isfinite(rates[:, required]).all()):
        raise ValueError('complete baseline and at least 200 s of spontaneous coverage required')
    score, fractions = population_conditioned_activity(
        rates, population, sizes, config['ne'], dt, baseline)
    relative = time[spontaneous]-end
    values = population[spontaneous]
    score = score[:, spontaneous]
    early, late = relative < 100, relative >= duration-100
    early_rate, late_rate = float(values[early].mean()), float(values[late].mean())
    bins_per_second = round(1/dt)
    if not np.isclose(bins_per_second*dt, 1., atol=1e-9, rtol=0):
        raise ValueError('activity bins must divide one second')
    if len(values) % bins_per_second:
        raise ValueError('spontaneous interval must contain complete one-second bins')
    return {
        'time_after_training_s': relative, 'conditioned_score': score,
        'population_rate_hz': values, 'baseline_fractions': fractions,
        'population_time_1s': relative.reshape(-1, bins_per_second).mean(axis=1),
        'population_rate_1s_hz': values.reshape(-1, bins_per_second).mean(axis=1),
    }, {
        'spontaneous_seconds': duration, 'bin_s': dt,
        'baseline_population_hz': float(population[baseline].mean()),
        'first_100s_population_hz': early_rate, 'last_100s_population_hz': late_rate,
        'late_minus_early_population_hz': late_rate-early_rate,
        'late_over_early_population_rate': late_rate/early_rate if early_rate else None,
        'score_below_display_fraction': float(np.mean(score < -3)),
        'score_above_display_fraction': float(np.mean(score > 15)),
    }


def review(suite, analysis, output):
    jobs = [j for j in json.loads((suite/'science_jobs.json').read_text())
            if j['condition'] == 'full']
    if not jobs or len({j['seed'] for j in jobs}) != len(jobs):
        raise ValueError('unique full-model seeds required')
    prepared, rows = [], []
    for job in sorted(jobs, key=lambda j: j['seed']):
        run = json.loads((suite/job['label']/'result.json').read_text())
        config = run['configuration']
        if (config['scale'] != 1 or config['mode'] != 'learn'
                or run['biological_seconds'] != config['duration_s']):
            raise ValueError('complete full-scale de novo protocol required')
        with np.load(analysis/job['label']/'activity_source.npz') as source:
            data = {key: source[key] for key in
                    ['time_s', 'coverage', 'population_rate_hz', 'assembly_rates_hz']}
        with np.load(analysis/job['label']/'connectivity_source.npz') as source:
            sizes = source['membership'].sum(axis=1)
        arrays, metrics = review_data(data, sizes, config)
        rows.append({'seed': job['seed'], 'backend': run['backend'], **metrics})
        prepared.append(arrays)
    output.mkdir(parents=True, exist_ok=False)
    for row, arrays in zip(rows, prepared):
        np.savez_compressed(output/f'activity-seed-{row["seed"]}.npz', **arrays)
    csv_rows(output/'rate_drift.csv', rows)
    report = {'seed_metrics': rows,
        'selection': 'all full-model seeds; fixed first/last 100 s and final 10 s, no peak selection',
        'interpretation': 'Descriptive review; nonzero early-to-late rate differences must be reported. Neither this diagnostic nor thresholded episodes prove stationarity or attractor dynamics.',
        'display': 'shared score scale [-3, 15], with clipping fractions reported and colorbar extensions; raw scores retained in source NPZ',
        'source_analysis': str(analysis.resolve())}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plot(prepared, rows, output)
    return report


def plot(prepared, rows, output):
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.25*len(rows)),
                                 squeeze=False, layout='constrained')
        zoom, zoom_axes = plt.subplots(len(rows), 1, figsize=(7, 1.8*len(rows)),
                                       squeeze=False, layout='constrained')
        for axes_row, zoom_row, arrays, row in zip(axes, zoom_axes, prepared, rows):
            time, score = arrays['time_after_training_s'], arrays['conditioned_score']
            duration = row['spontaneous_seconds']
            windows = [(0, 100), (duration-100, duration), (duration-10, duration)]
            for ax, (start, stop) in zip([*axes_row[:2], zoom_row[0]], windows):
                selected = (time >= start) & (time < stop)
                artist = ax.imshow(score[:, selected], aspect='auto', origin='lower',
                    extent=[start, stop, .5, score.shape[0]+.5], vmin=-3, vmax=15,
                    cmap='viridis', interpolation='nearest', rasterized=True)
                ax.set(xlabel='Time after training (s)', ylabel=f'Seed {row["seed"]}\nAssembly',
                       yticks=[1, score.shape[0]//2, score.shape[0]])
            axes_row[2].plot(arrays['population_time_1s'], arrays['population_rate_1s_hz'],
                             lw=.7, color='#0072B2')
            axes_row[2].axhline(row['baseline_population_hz'], color='#777777', ls='--', lw=.7)
            axes_row[2].set(xlabel='Time after training (s)', ylabel='E population rate (Hz)',
                            ylim=(0, None))
        for ax, title in zip(axes[0], ['First 100 s', 'Last 100 s', 'Population activity · 1 s bins']):
            ax.set_title(title)
        fig.colorbar(axes[-1, 1].images[0], ax=list(axes[:, :2].flat), shrink=.7, extend='both',
                     label='Population-conditioned assembly score')
        zoom.colorbar(zoom_axes[-1, 0].images[0], ax=list(zoom_axes.flat), shrink=.7, extend='both',
                      label='Population-conditioned assembly score')
        backend = ', '.join(sorted({r['backend'] for r in rows}))
        fig.suptitle(f'{backend} · complete spontaneous intervals · descriptive review', fontsize=11)
        zoom.suptitle(f'{backend} · fixed final 10 s · descriptive review', fontsize=11)
        for name, figure in [('activity_overview', fig), ('activity_zoom', zoom)]:
            figure.savefig(output/(name+'.pdf'))
            figure.savefig(output/(name+'.png'), dpi=180)
            plt.close(figure)
    (output/'caption.md').write_text(
        'Every full-model seed is shown after the complete training protocol. '
        'All seeds use the first/last 100 s and final 10 s of spontaneous activity; '
        'windows are selected by time, not response amplitude. '
        'The population panels use complete 1 s bins; dashed lines show pretraining means. '
        'Population-conditioned assembly scores account for each assembly’s baseline spike fraction. '
        'The shared colour scale clips at -3 and 15; raw scores and clipping fractions are supplied. '
        'The rate_drift.csv file reports early and late means explicitly. These diagnostics are '
        'not calibrated significance tests or proof of steady-state attractor dynamics.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['suite', 'analysis', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    review(**vars(parser.parse_args()))
