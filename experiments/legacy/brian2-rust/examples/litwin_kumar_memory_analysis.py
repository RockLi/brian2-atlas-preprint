"""Verify rolling/full state equivalence and report both native and total RSS."""
import argparse
import json
from pathlib import Path

import numpy as np

from litwin_kumar_figures import csv_rows

MONITOR_FIELDS = {'exc_spike_i', 'exc_spike_t', 'inh_spike_i', 'inh_spike_t',
                  'voltage_t', 'voltage_v'}


def verify_window(full_path, rolling_path, *, dt_s, horizon_s, window_steps):
    """Check dynamics, cumulative counters, and exact retained monitor suffixes."""
    if dt_s <= 0 or window_steps < 1 or horizon_s <= 0:
        raise ValueError('positive clock, duration, and window required')
    end_tick = round(horizon_s/dt_s)
    if not np.isclose(end_tick*dt_s, horizon_s, rtol=0, atol=1e-10):
        raise ValueError('horizon does not align with clock')
    first_tick = max(0, end_tick-window_steps)
    def exact(name, actual, expected):
        if (actual.shape != expected.shape or actual.dtype != expected.dtype
                or actual.tobytes() != expected.tobytes()):
            raise AssertionError(f'rolling/full mismatch: {name}')
    with np.load(full_path) as full, np.load(rolling_path) as rolling:
        if set(full.files) != set(rolling.files) or not MONITOR_FIELDS <= set(full.files):
            raise ValueError('state files do not share the complete monitor schema')
        fields = sorted(set(full.files)-MONITOR_FIELDS)
        if not {'exc_spike_total', 'inh_spike_total'} <= set(fields):
            raise ValueError('cumulative spike counters must be verified separately from monitors')
        for name in fields:
            exact(name, rolling[name], full[name])
        sizes = {}
        for label in ['exc', 'inh']:
            times = full[label+'_spike_t']
            keep = np.rint(times/dt_s).astype(np.int64) >= first_tick
            exact(label+'_spike_t', rolling[label+'_spike_t'], times[keep])
            exact(label+'_spike_i', rolling[label+'_spike_i'], full[label+'_spike_i'][keep])
            sizes[label+'_retained_spikes'] = int(keep.sum())
        times = full['voltage_t']
        keep = np.rint(times/dt_s).astype(np.int64) >= first_tick
        exact('voltage_t', rolling['voltage_t'], times[keep])
        exact('voltage_v', rolling['voltage_v'], full['voltage_v'][:, keep])
        sizes['retained_voltage_frames'] = int(keep.sum())
    return {'dynamics_fields_byte_exact': fields, 'monitor_suffixes_byte_exact': sorted(MONITOR_FIELDS),
            'retained_from_tick': first_tick, 'window_steps': window_steps, **sizes}


def analyze(suite, output, allow_incomplete=False, version_label=None):
    source = suite/'report.json'
    if source.exists():
        experiments = json.loads(source.read_text())['experiments']
    elif allow_incomplete:
        source = suite/'memory.json'
        experiments = json.loads(source.read_text())
    else:
        raise ValueError('final memory report required')
    keys = [(r['backend'], r['window_steps'] is not None, r['horizon_s']) for r in experiments]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate memory condition/horizon')
    required = {(backend, rolling, horizon) for backend, rolling in
                [('rust', False), ('rust', True), ('cpp', False)] for horizon in [1., 10., 100., 1000.]}
    missing = sorted(required-set(keys))
    if missing and not allow_incomplete:
        raise ValueError(f'incomplete memory horizon grid: {missing}')
    rows, checks = [], []
    for run in experiments:
        process = run['process']
        window = run['window_steps']
        rows.append({'backend': run['backend'], 'recording': 'full' if window is None else 'rolling',
            'horizon_s': run['horizon_s'], 'window_steps': window,
            'window_seconds': None if window is None else window*run['run']['configuration']['dt_ms']/1000,
            'peak_native_rss_bytes': process['peak_native_rss_bytes'],
            'peak_process_tree_rss_bytes': process['peak_process_tree_rss_bytes'],
            'rss_complete': process['rss_complete']})
        if run['backend'] != 'rust' or window is None:
            continue
        horizon = run['horizon_s']
        full = [r for r in experiments if r['backend']=='rust' and r['window_steps'] is None and r['horizon_s']==horizon]
        if not full:
            if allow_incomplete: continue
            raise ValueError('rolling run lacks its full-history counterpart')
        if full[0]['run']['configuration'] != run['run']['configuration']:
            raise ValueError('unmatched rolling/full model configuration')
        base = suite/f'rust-full-{horizon:g}s'/'state.npz'
        bounded = suite/f'rust-{window}-{horizon:g}s'/'state.npz'
        checks.append({'horizon_s': horizon, **verify_window(base, bounded,
            dt_s=run['run']['configuration']['dt_ms']/1000, horizon_s=horizon, window_steps=window)})
    complete = not missing and all(row['rss_complete'] and row['peak_native_rss_bytes'] is not None for row in rows)
    report = {'complete_required_grid': complete, 'missing_conditions': missing,
        'version_label': version_label,
        'source': str(source), 'measurements': rows, 'rolling_full_equivalence': checks,
        'replication': 'one independent execution per condition/horizon; sampled peaks are descriptive, not confidence bounds',
        'measurement': 'peak native-process RSS and peak summed process-tree RSS are distinct; tree RSS can double-count shared pages',
        'interpretation': 'Different retained history is intentional. Exact dynamics/counters and suffix checks verify that reducing recording does not alter the simulation; neither an OOM nor a universal constant-RSS theorem is inferred.'}
    output.mkdir(parents=True, exist_ok=False)
    csv_rows(output/'memory_source.csv', rows)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plot(rows, output, complete, version_label=version_label)
    return report


def plot(rows, output, complete, version_label=None):
    import matplotlib.pyplot as plt
    styles = [('rust', 'full', '#0072B2', '-', 'Rust · full history'),
              ('rust', 'rolling', '#009E73', '--', 'Rust · rolling window'),
              ('cpp', 'full', '#D55E00', '-', 'Brian2 C++ · full history')]
    with plt.rc_context({'font.size': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.9), layout='constrained')
        for ax, key, title in zip(axes, ['peak_native_rss_bytes', 'peak_process_tree_rss_bytes'],
                                 ['Native simulation process', 'Complete process tree']):
            for backend, recording, color, linestyle, label in styles:
                values = sorted([r for r in rows if r['backend']==backend and r['recording']==recording
                                 and r['rss_complete'] and r[key] is not None], key=lambda r:r['horizon_s'])
                ax.plot([r['horizon_s'] for r in values], [r[key]/2**30 for r in values],
                        marker='o', markersize=4, color=color, linestyle=linestyle, label=label)
            ax.set(xscale='log', xlabel='Biological duration (s)', ylabel='Peak sampled RSS (GiB)', title=title)
            ax.set_xticks([1, 10, 100, 1000], labels=['1', '10', '100', '1,000'])
            ax.set_ylim(bottom=0)
        axes[0].legend(frameon=False, fontsize=6.5)
        fig.suptitle('LK2014 recording memory' +
                     (f' · {version_label}' if version_label else '') +
                     ('' if complete else ' · interim horizons'), fontsize=10)
        fig.savefig(output/'memory_comparison.pdf')
        fig.savefig(output/'memory_comparison.png', dpi=200)
        plt.close(fig)
    (output/'caption.md').write_text(
        (f'Implementation version: {version_label}. ' if version_label else '') +
        'Separate runs at each available horizon; one execution per point. Left: peak sampled '
        'native simulation RSS. Right: peak summed process-tree RSS, including model construction, '
        'compilation and export; shared pages may be counted twice. Window length and all sampled '
        'peaks are in memory_source.csv. Rust full/rolling dynamics, cumulative spike counters and '
        'retained monitor suffixes are checked byte-for-byte in report.json. Missing horizons and '
        'incomplete RSS samples remain explicit; no C++ OOM is inferred.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--version-label', help='Explicit implementation label for report, figure and caption')
    analyze(**vars(parser.parse_args()))
