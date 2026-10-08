"""Review fixed Rust8 confirmation and measured scaling with raw timing points."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from litwin_kumar_figures import COLORS, csv_rows


def validated_report(path, suite):
    report = json.loads((path/'report.json').read_text())
    if report.get('suite') != suite or report.get('profiled') is not False:
        raise ValueError('explicit unprofiled timing scope required')
    if not report.get('all_outputs_byte_exact'):
        raise ValueError('full output repeatability evidence required')
    samples = report['samples']
    if any(row.get('summary', {}).get('neuron_count') != 5000
           for row in samples if row['backend'] == 'rust'):
        raise ValueError('full 5000-neuron runtime evidence required')
    keys = [(r['backend'], r['threads'], r['repeat']) for r in samples]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate process timing sample')
    groups = {(r['backend'], r['threads']) for r in samples}
    aggregates = []
    for backend, threads in sorted(groups):
        values = np.array([r['simulation_seconds'] for r in samples
                           if r['backend'] == backend and r['threads'] == threads])
        if len(values) != 5 or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('five finite positive timing repeats required per configuration')
        aggregates.append({'backend': backend, 'threads': threads, 'n': len(values),
            'median_seconds': float(np.median(values)), 'min_seconds': float(values.min()),
            'max_seconds': float(values.max())})
    if len([r for r in aggregates if r['backend'] == 'cpp']) != 1:
        raise ValueError('one selected C++ comparator required')
    cpp = next(r for r in aggregates if r['backend'] == 'cpp')
    rust = next(r for r in aggregates if r['backend'] == 'rust' and r['threads'] == 8)
    gate = report['fixed_rust8_gate']
    if (gate['rust'] != rust or gate['cpp'] != cpp
            or gate['passed'] != (rust['max_seconds'] < cpp['min_seconds'])
            or not np.isclose(gate['speedup'], cpp['median_seconds']/rust['median_seconds'], rtol=1e-14, atol=0)):
        raise ValueError('reported fixed gate does not agree with raw samples')
    return report, aggregates


def render(performance, whole_run, output, machine_label):
    import matplotlib.pyplot as plt
    sources = {'performance': performance, 'whole_run': whole_run}
    data = {suite: validated_report(path, suite) for suite, path in sources.items()}
    output.mkdir(parents=True, exist_ok=False)
    csv_rows(output/'timing_source.csv', [{'suite': suite, **{key: row[key] for key in
        ['backend', 'threads', 'repeat', 'simulation_seconds']}}
        for suite, (report, _) in data.items() for row in report['samples']])
    colors = {backend: COLORS[backend] for backend in ['rust', 'cpp']}
    with plt.rc_context({'font.size': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), layout='constrained')
        for col, suite in enumerate(['performance', 'whole_run']):
            report, aggregates = data[suite]
            gate = report['fixed_rust8_gate']
            ax = axes[0, col]
            for index, backend in enumerate(['rust', 'cpp']):
                row = gate[backend]
                values = [r['simulation_seconds'] for r in report['samples']
                          if r['backend'] == backend and r['threads'] == row['threads']]
                ax.scatter(index+np.linspace(-.09, .09, len(values)), values,
                           s=16, color=colors[backend], zorder=3)
                median = row['median_seconds']
                ax.errorbar(index, median, yerr=[[median-row['min_seconds']], [row['max_seconds']-median]],
                            fmt='_', markersize=15, capsize=5, color='black', linewidth=.8, zorder=4)
            ax.set(xticks=[0, 1], xticklabels=['Rust · 8 workers', f"C++ · {gate['cpp']['threads']} workers"],
                   ylabel='Simulation + recording (s)', xlim=(-.5, 1.5),
                   ylim=(0, max(gate[b]['max_seconds'] for b in colors)*1.2),
                   title='Post-warmup · 10–11 s' if suite == 'performance' else 'Complete run · 0–11 s')
            ax.text(.98, .97, f"{gate['speedup']:.2f}× speedup", ha='right', va='top', transform=ax.transAxes)
            ax = axes[1, col]
            rows = sorted([r for r in aggregates if r['backend'] == 'rust'], key=lambda r:r['threads'])
            x = [r['threads'] for r in rows]
            y = np.array([r['median_seconds'] for r in rows])
            ax.errorbar(x, y, yerr=[y-[r['min_seconds'] for r in rows], [r['max_seconds'] for r in rows]-y],
                        fmt='o-', markersize=3, capsize=2, color=colors['rust'], label='Rust', linewidth=1)
            cpp = gate['cpp']
            ax.axhspan(cpp['min_seconds'], cpp['max_seconds'], alpha=.12, color=colors['cpp'])
            ax.axhline(cpp['median_seconds'], color=colors['cpp'], linestyle='--', linewidth=.9,
                       label=f"C++ · {cpp['threads']} workers")
            ax.set_xscale('log', base=2)
            ax.set(xticks=x, xticklabels=[str(n) for n in x], xlabel='Rust worker threads',
                   ylabel='Simulation + recording (s)', ylim=(0, max(y)*1.15))
            ax.legend(frameon=False, fontsize=7)
        for letter, ax in zip('abcd', axes.flat):
            ax.text(.02, .98, letter, va='top', transform=ax.transAxes, fontweight='bold', fontsize=10)
        fig.suptitle(f'{machine_label} · N = 5,000 · five independent process runs per setting', fontsize=10)
        fig.savefig(output/'performance_confirmation.pdf')
        fig.savefig(output/'performance_confirmation.png', dpi=220)
        plt.close(fig)
    summary = {'machine_label': machine_label, 'gates': {suite: r['fixed_rust8_gate'] for suite, (r, _) in data.items()},
        'source_reports_sha256': {suite: hashlib.sha256((path/'report.json').read_bytes()).hexdigest() for suite, path in sources.items()},
        'scope': 'fixed eight-worker Rust confirmation; lower panels show measured Rust scaling with selected C++ control only'}
    (output/'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    profiles = {suite: next(item['cpp_compilation']['profile']
        for item in report['event_volume_evidence'].values() if item['cpp_compilation'])
        for suite, (report, _) in data.items()}
    ratios = {suite: report['build_run_event_count_ratio'] for suite, (report, _) in data.items()}
    (output/'caption.md').write_text(
        'Five independent process runs per configuration; dots are raw observations, black marks are medians, '
        'and whiskers span the minimum–maximum. Rust8 was fixed before this confirmation. '
        'The upper panels separately measure the final biological second after 10 s warmup and the complete 11 s simulation. '
        'Compilation and export are excluded. Both backends use full-history monitors. '
        'The lower panels show Rust scaling; the dashed line and shaded band are the selected C++ control '
        'at the worker count shown, not a C++ scaling curve. '
        f"C++ profiles: {profiles['performance']} for post-warmup; {profiles['whole_run']} for the complete run. "
        'Backend random streams differ despite matched network initialization and input distributions. '
        f"The original workload event-count ratios were {ratios['performance']:.3f} (post-warmup) "
        f"and {ratios['whole_run']:.3f} (complete run). "
        'Timing source data are in timing_source.csv; source hashes and gates are in report.json. '
        f'This figure covers {machine_label}; other hardware and the complete scientific evidence remain separate.\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['performance', 'whole-run', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--machine-label', required=True)
    render(**vars(parser.parse_args()))
