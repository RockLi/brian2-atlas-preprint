"""Plot audited 25-case CPU results, retaining individual seeds and raw rates."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

SEEDS = [11, 23, 47, 83, 131]
CONDITIONS = ['paired', 'frozen', 'teaching_off', 'shuffled_reward', 'reversal']
LABELS = ['Paired', 'Frozen', 'Teaching off', 'Shuffled reward', 'Reversal']


def plot(report):
    source = report / 'final_report.json'
    data = json.loads(source.read_text())
    audit = json.loads((report / 'completion_audit.json').read_text())
    expected = {(seed, condition) for seed in SEEDS for condition in CONDITIONS}
    assert data['completed_cases'] == 25 and data['all_correctness_gates_passed']
    assert data['seeds'] == SEEDS and data['conditions'] == CONDITIONS
    assert len(audit) == 25 and {(a['seed'], a['condition']) for a in audit} == expected
    rows = {(r['seed'], r['condition']): r for r in data['rows']}
    assert len(data['rows']) == 25 and set(rows) == expected
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    markers = ['o', 's', '^', 'D', 'v']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, metric, title in zip(axes, ('post_effect_hz', 'final_effect_hz'),
                                 ('After first training', 'After second training')):
        ax.axhline(0, color='#94a3b8', lw=1, zorder=0)
        for x, condition in enumerate(CONDITIONS):
            values = [rows[seed, condition][metric] for seed in SEEDS]
            for offset, value, marker in zip(np.linspace(-.17, .17, 5), values, markers):
                ax.scatter(x + offset, value, marker=marker, s=42, color='#2563eb',
                           edgecolors='white', linewidths=.5, zorder=3)
            ax.hlines(np.mean(values), x - .25, x + .25, color='#111827', lw=2.5, zorder=4)
        ax.set(title=title, xticks=range(5), xticklabels=LABELS)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Change in B − A vs same-seed frozen (Hz)')
    handles = [Line2D([], [], marker=m, color='#2563eb', ls='', label=f'Seed {s}')
               for s, m in zip(SEEDS, markers)]
    handles.append(Line2D([], [], color='#111827', lw=2.5, label='Mean'))
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False)
    fig.suptitle('FlyWire CPU v1 · Preset MBON01 contrast', fontsize=15)
    fig.subplots_adjust(top=.84, bottom=.25, wspace=.12)
    for suffix in ('png', 'pdf'):
        fig.savefig(report / f'preset_effects.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.8), sharey=True)
    blocks = ('pre', 'post', 'final')
    colors = ('#2563eb', '#c05621')
    for ax, condition, label in zip(axes.flat, CONDITIONS, LABELS):
        for cue, color in zip(('A_hz', 'B_hz'), colors):
            values = np.array([[rows[seed, condition]['contrasts'][block][cue]
                                for block in blocks] for seed in SEEDS])
            for values_for_seed in values:
                ax.plot(range(3), values_for_seed, color=color, alpha=.18, lw=.8)
            ax.plot(range(3), values.mean(axis=0), color=color, marker='o', lw=2.4,
                    label=f'Cue {cue[0]}')
        ax.set(title=label, xticks=range(3), xticklabels=['Before', 'After\nfirst', 'After\nsecond'])
        ax.set_xlim(-.18, 2.18)
        ax.grid(axis='y', alpha=.15)
    maximum = max(row['contrasts'][block][cue] for row in rows.values()
                  for block in blocks for cue in ('A_hz', 'B_hz'))
    axes[0, 0].set_ylim(-.6, max(1, maximum * 1.08))
    for ax in axes[:, 0]:
        ax.set_ylabel('MBON01 probe response (Hz)')
    legend = axes[1, 2]
    legend.axis('off')
    legend.legend(handles=[Line2D([], [], color=c, marker='o', lw=2.4, label=f'Cue {cue}')
                           for cue, c in zip('AB', colors)], loc='upper left', frameon=False)
    legend.text(0, .62, 'Thin lines: individual seeds\nBold lines: mean of 5 seeds\n\n'
                'Each probe: 2 trials per cue\nReadout: 2 bilateral MBON01 neurons\n\n'
                'Learning and teaching gates\nare off during all probes.',
                transform=legend.transAxes, va='top', linespacing=1.5)
    fig.suptitle('FlyWire CPU v1 · Raw cue responses', fontsize=15)
    fig.subplots_adjust(top=.9, bottom=.1, hspace=.36, wspace=.25)
    for suffix in ('png', 'pdf'):
        fig.savefig(report / f'raw_cue_responses.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    (report / 'figure_provenance.json').write_text(json.dumps({
        'final_report_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'plot_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'seeds': SEEDS, 'conditions': CONDITIONS,
        'note': 'All five seeds shown; means are descriptive, not confidence intervals.'
    }, indent=2) + '\n')
    print(f'Wrote audited-result PNG/PDF figures to {report}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    plot(parser.parse_args().report)
