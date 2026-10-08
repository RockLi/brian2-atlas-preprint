"""Plot paired backend scaling, preserving its separation from confirmation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from litwin_kumar_figures import COLORS, csv_rows


def validated_report(path, suite, workers, repeats):
    report = json.loads((path/'report.json').read_text())
    if report.get('suite') != suite or report.get('profiled') is not False:
        raise ValueError('explicit unprofiled timing scope required')
    if report.get('all_outputs_byte_exact') is not True:
        raise ValueError('complete output repeatability evidence required')
    if (report.get('warmup_s') != 10
            or report.get('timed_duration_s') != (1 if suite == 'performance' else 11)
            or report.get('recording') != 'identical full-history SpikeMonitor and StateMonitor'):
        raise ValueError('10 s warmup, declared timing interval and full-history recording required')
    samples = report['samples']
    keys = [(r['backend'], r['threads'], r['repeat']) for r in samples]
    required = {(backend, n, k) for backend in ['rust', 'cpp']
                for n in workers for k in range(repeats)}
    if len(keys) != len(set(keys)) or set(keys) != required:
        raise ValueError('complete unique paired backend/thread/repeat grid required')
    for row in samples:
        if row['backend'] == 'rust':
            summary = row.get('summary', {})
            if (summary.get('neuron_count') != 5000
                    or summary.get('threads') != row['threads']
                    or summary.get('phase_profile', {}).get('enabled') is not False):
                raise ValueError('full-scale unprofiled runtime and worker evidence required')
    aggregates = []
    for backend in ['rust', 'cpp']:
        for n in workers:
            values = np.array([r['simulation_seconds'] for r in samples
                               if r['backend'] == backend and r['threads'] == n])
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError('finite positive timings required')
            aggregates.append({'backend': backend, 'threads': n, 'n': repeats,
                'median_seconds': float(np.median(values)),
                'min_seconds': float(values.min()), 'max_seconds': float(values.max())})
    best_cpp = min((r for r in aggregates if r['backend'] == 'cpp'),
                   key=lambda r: r['median_seconds'])
    if report.get('new_cpp_best') != best_cpp:
        raise ValueError('C++ selection disagrees with raw timing observations')
    return report, aggregates


def render(paired_scaling, output, machine_label):
    import matplotlib.pyplot as plt
    contract = json.loads((paired_scaling/'contract.json').read_text())
    workers, repeats = contract['workers'], contract['repeats']
    if (not workers or len(workers) != len(set(workers))
            or any(not isinstance(n, int) or n < 1 for n in workers)
            or not isinstance(repeats, int) or repeats < 3):
        raise ValueError('unique positive workers and at least three repetitions required')
    workers = sorted(workers)
    sources = {suite: paired_scaling/suite for suite in ['performance', 'whole_run']}
    data = {suite: validated_report(path, suite, workers, repeats)
            for suite, path in sources.items()}
    output.mkdir(parents=True, exist_ok=False)
    csv_rows(output/'timing_source.csv', [
        {'suite': suite, **{key: row[key] for key in
         ['backend', 'threads', 'repeat', 'simulation_seconds']}}
        for suite, (report, _) in data.items() for row in report['samples']])
    with plt.rc_context({'font.size': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), layout='constrained')
        for ax, (suite, (report, aggregates)) in zip(axes, data.items()):
            for backend in ['rust', 'cpp']:
                rows = [r for r in aggregates if r['backend'] == backend]
                medians = np.array([r['median_seconds'] for r in rows])
                ax.errorbar(workers, medians,
                    yerr=[medians-[r['min_seconds'] for r in rows],
                          [r['max_seconds'] for r in rows]-medians],
                    fmt='o-', color=COLORS[backend], markersize=3, capsize=2,
                    linewidth=1, label='Rust' if backend == 'rust' else 'Brian2 C++')
                for row in rows:
                    values = [r['simulation_seconds'] for r in report['samples']
                              if r['backend'] == backend and r['threads'] == row['threads']]
                    ax.scatter(np.full(len(values), row['threads']), values,
                               color=COLORS[backend], s=8, alpha=.45)
            ax.set_xscale('log', base=2)
            ax.set(xticks=workers, xticklabels=[str(n) for n in workers],
                   xlabel='Worker threads', ylabel='Simulation + recording (s)', ylim=(0, None),
                   title='Post-warmup · 10–11 s' if suite == 'performance' else 'Complete run · 0–11 s')
            ax.legend(frameon=False, fontsize=7)
        fig.suptitle(f'{machine_label} · N = 5,000 · paired scaling, n = {repeats}', fontsize=10)
        fig.savefig(output/'paired_scaling.pdf')
        fig.savefig(output/'paired_scaling.png', dpi=220)
        plt.close(fig)
    summary = {'machine_label': machine_label, 'workers': workers, 'repeats': repeats,
        'aggregates': {suite: rows for suite, (_, rows) in data.items()},
        'best_cpp': {suite: report['new_cpp_best'] for suite, (report, _) in data.items()},
        'source_reports_sha256': {suite: hashlib.sha256((path/'report.json').read_bytes()).hexdigest()
                                  for suite, path in sources.items()},
        'contract_sha256': hashlib.sha256((paired_scaling/'contract.json').read_bytes()).hexdigest(),
        'interpretation': 'Paired scaling is separate from independent fixed-setting confirmation. '
                          'A changed best C++ thread count requires comparison with earlier valid measurements.'}
    (output/'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    profiles = {suite: report['cpp_profile'] for suite, (report, _) in data.items()}
    (output/'caption.md').write_text(
        f'Both backends use {repeats} fresh process runs per worker setting, randomized within each timing scope. '
        'Lines connect medians; whiskers span minimum–maximum; small points show individual observations. '
        'Both use full-history monitors. Compilation and export are excluded. '
        f"C++ profiles are {profiles['performance']} (post-warmup) and {profiles['whole_run']} (complete run). "
        'Matched initialization and input distributions do not imply identical backend random streams or event counts. '
        'All Rust output bytes match the frozen reference; C++ output bytes match its own fixed-worker reference. '
        'This scaling batch does not replace the separate five-repeat fixed-setting confirmation. '
        'Raw observations and source hashes accompany this figure.\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['paired-scaling', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--machine-label', required=True)
    render(**vars(parser.parse_args()))
