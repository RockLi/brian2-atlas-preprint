"""Plot only a completed and independently audited warmup comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(folder):
    source = folder / 'results.json'
    result = json.loads(source.read_text())
    assert result['completed_cases'] == 10 and result['recounted_trials'] == 360
    assert result['all_gates_zero'] and result['all_gains_one'] and result['all_matched_inputs_exact']
    cases, seeds = result['cases'], [11, 23, 47, 83, 131]
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.dpi': 160})
    colors = ['#2763d9', '#cc5522']
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, item in enumerate(result['primary_effects']):
        ax.plot([0, 1], [item['short_drift_hz'], item['long_drift_hz']], marker='o',
                label=f"Seed {item['seed']}", alpha=.8)
    means = [np.mean([r[key] for r in result['primary_effects']]) for key in ['short_drift_hz', 'long_drift_hz']]
    ax.scatter([0, 1], means, marker='_', s=1000, linewidths=3, c='black', zorder=5, label='Mean')
    ax.set(xticks=[0, 1], xticklabels=['100 ms warmup', '10.1 s warmup'], xlim=(-.3, 1.3),
           ylabel='Mean absolute pre-to-final cue response change (Hz)',
           title='Frozen FlyWire: preset response-drift comparison')
    ax.grid(axis='y', alpha=.2)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(folder / f'warmup_drift.{ext}', bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey='row')
    for col, arm in enumerate([100, 10100]):
        group = [c for c in cases if c['warmup_ms'] == arm]
        assert sorted(c['seed'] for c in group) == seeds
        for row, suffix in enumerate(['_hz', '_evoked_hz']):
            ax = axes[row, col]
            for cue, color in zip(['A', 'B'], colors):
                lines = np.array([[c['probes'][b][cue + suffix] for b in ['pre', 'post', 'final']] for c in group])
                for values in lines:
                    ax.plot(range(3), values, color=color, alpha=.22, lw=1)
                ax.plot(range(3), lines.mean(axis=0), color=color, lw=2.5, marker='o', label=f'Cue {cue}')
            ax.set(xticks=range(3), xticklabels=['Early', 'Middle', 'Late'], xlim=(-.15, 2.15))
            ax.axhline(0, color='#667085', lw=.7)
            ax.grid(axis='y', alpha=.2)
            if row == 0:
                ax.set_title('100 ms warmup' if arm == 100 else '10.1 s warmup')
            if col == 0:
                ax.set_ylabel('Raw MBON01 rate (Hz)' if row == 0 else 'Response minus baseline (Hz)')
    axes[0, 1].legend(frameon=False)
    fig.suptitle('No learning or teaching: raw and baseline-subtracted cue responses', y=.99)
    fig.text(.5, .02, 'Thin lines: all 5 seeds. Bold: mean. Probes: 2 trials per cue; baseline: 50 ms.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .06, 1, .95])
    for ext in ['png', 'pdf']:
        fig.savefig(folder / f'warmup_responses.{ext}', bbox_inches='tight')
    plt.close(fig)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (folder / 'figure_provenance.json').write_text(json.dumps(dict(results_sha256=sha(source),
            plot_source_sha256=sha(Path(__file__)), seeds=seeds, matplotlib_version=matplotlib.__version__,
            note='All seeds shown. Connecting lines aid comparison; no confidence intervals or fitted trajectories.'), indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    main(p.parse_args().report.resolve())
