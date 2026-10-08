"""Plot Linux post-warmup worker scaling from the completed selection grid."""
import argparse
import json
from pathlib import Path

import numpy as np

from litwin_kumar_figures import COLORS, csv_rows
from litwin_kumar_linux_performance_review import aggregate_samples, review


def render(final_validation, output):
    import matplotlib.pyplot as plt
    _, checked = review(final_validation)
    pilot = json.loads((final_validation/'performance-rust-selection/report.json').read_text())
    rust = [r for r in aggregate_samples(pilot, 'performance', 2) if r['backend'] == 'rust']
    candidates = json.loads((final_validation/'cpp-placement-candidates.json').read_text())['candidates']
    cpp = []
    for n in sorted({r['threads'] for r in candidates}):
        best = min((r for r in candidates if r['threads'] == n), key=lambda r:r['performance']['median_seconds'])
        cpp.append(dict(backend='cpp', threads=n, n=2, profile=best['profile'],
                        binding=best['binding'], **best['performance']))
    rows = [dict(profile='', binding='', **r) for r in rust] + cpp
    output.mkdir(parents=True, exist_ok=False)
    csv_rows(output/'scaling_source.csv', rows)
    with plt.rc_context({'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), layout='constrained')
        for backend, groups in [('rust', rust), ('cpp', cpp)]:
            groups = sorted(groups, key=lambda r:r['threads'])
            n = np.array([r['threads'] for r in groups])
            median, low, high = (np.array([r[k] for r in groups]) for k in
                                ['median_seconds', 'min_seconds', 'max_seconds'])
            baseline = next(r['median_seconds'] for r in groups if r['threads'] == 1)
            label = 'Rust' if backend == 'rust' else 'C++ best measured per worker count'
            axes[0].errorbar(n, median, yerr=[median-low, high-median], fmt='o-', capsize=3,
                             markersize=4, color=COLORS[backend], label=label)
            speedup = baseline/median
            axes[1].errorbar(n, speedup, yerr=[speedup-baseline/high, baseline/low-speedup],
                            fmt='o-', capsize=3, markersize=4, color=COLORS[backend], label=label)
        for ax in axes:
            ax.set_xscale('log', base=2)
            ax.set(xlabel='Worker count', xticks=[1, 2, 4, 8, 16, 32, 48],
                   xticklabels=['1', '2', '4', '8', '16', '32', '48'], ylim=(0, None))
            ax.tick_params(axis='x', labelrotation=45)
        axes[0].set(ylabel='Simulation + recording (s)', title='Absolute post-warmup time')
        axes[1].set(ylabel='Speedup vs own 1-worker median', title='Worker scaling')
        axes[0].legend(fontsize=6, frameon=False, loc='upper right')
        fig.suptitle('Linux 23 · N = 5,000 · selection only · n = 2', fontsize=11)
        fig.savefig(output/'linux_scaling.pdf')
        fig.savefig(output/'linux_scaling.png', dpi=220)
        plt.close(fig)
    one = {b: next(r for r in rows if r['backend'] == b and r['threads'] == 1) for b in ['rust', 'cpp']}
    report = dict(source_reports_sha256=checked['source_reports_sha256'], selection_only=True,
                  timings=rows, rust_over_cpp_one_worker_time=one['rust']['median_seconds']/one['cpp']['median_seconds'],
                  final_confirmation_separate=True)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output/'caption.md').write_text(
        'Full-scale biological10–11s, identical full-history monitors, compilation/export excluded. '
        'Two observations per point. Rust workers1–48; C++ workers1–16 use the lowest median among '
        'three compiler profiles and three placements at each worker count. This is an optimistically '
        'selected C++ envelope, not one fixed compiler/placement curve. Bars show observed min/max; '
        'normalized bars fix each backend one-worker median, not confidence intervals. '
        'No C++32/48 point from the current grid is invented; the historical strict scan is separate. '
        'Independent final n5 confirmation selected Rust16 versus C++4 for this scope. '
        'Different worker counts mean the elapsed-time win does not establish better core efficiency. '
        'Whole-run pilot scopes differ (single versus split native intervals), so they are omitted here.\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final-validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    render(**vars(parser.parse_args()))
