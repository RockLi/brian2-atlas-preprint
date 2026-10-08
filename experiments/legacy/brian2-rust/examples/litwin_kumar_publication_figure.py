"""Compose the LK scientific and systems figure from checked source artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def render(root, output, draft=False, linux_final_validation=None, control_reproduction=None):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, findfont
    from litwin_kumar_linux_performance_review import review as review_linux

    findfont(FontProperties(family='Arial'), fallback_to_default=False)

    index = json.loads((root/'ARTIFACT_INDEX.json').read_text())['files']
    sources = {}

    def checked(relative):
        path = root/relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if relative not in index or digest != index[relative]['sha256']:
            raise ValueError(f'figure source differs from evidence index: {relative}')
        sources[relative] = digest
        return path

    def table(relative):
        with checked(relative).open() as stream:
            return list(csv.DictReader(stream))

    matrix_rows = table('remote23-release/assembly-formation-review-final/assembly_matrix_source.csv')
    selected = [r for r in matrix_rows if r['backend'] == 'rust' and int(r['seed']) == 20260906]
    if len(selected) != 400 or len({(r['post_assembly'], r['pre_assembly']) for r in selected}) != 400:
        raise ValueError('complete representative assembly matrix required')
    matrix = np.full((20, 20), np.nan)
    for row in selected:
        if abs(float(row['initial_mean_pf'])-2.76) > 1e-10:
            raise ValueError('uniform unstructured initial weights required')
        matrix[int(row['post_assembly'])-1, int(row['pre_assembly'])-1] = float(row['final_mean_pf'])
    if not np.isfinite(matrix).all():
        raise ValueError('finite complete connectivity matrix required')
    raster_path = checked('remote23-release/primary-raster-evidence/results/primary-raster-sources/rust-seed-20260906.npz')
    with np.load(raster_path) as source:
        membership, ids, times = (source[k] for k in ['membership_e', 'exc_i', 'exc_t'])
    if membership.shape != (20, 4000) or len(ids) != len(times) or np.any((times < 2600) | (times >= 2610)):
        raise ValueError('full-scale final-ten-second raster required')
    first = np.where(membership.any(axis=0), membership.argmax(axis=0), 20)
    order = np.lexsort((np.arange(4000), first))
    rank = np.empty(4000, dtype=int)
    rank[order] = np.arange(4000)
    control_path = checked('remote23-release/control-comparison-review-final/report.json')
    control_report = json.loads(control_path.read_text())
    controls = control_report['seed_metrics']
    conditions = ['full', 'no_stimulation', 'no_istdp', 'no_normalization']
    seeds = [20260906, 20260907, 20260908]
    lookup = {(r['seed'], r['condition']): r for r in controls}
    if len(controls) != 12 or set(lookup) != {(s, c) for s in seeds for c in conditions}:
        raise ValueError('complete three-seed four-condition comparison required')
    if any(abs(r['common_baseline_seconds']-9) > 1e-9 or
           abs(r['common_spontaneous_seconds']-110) > 1e-9 for r in controls):
        raise ValueError('figure caption requires the checked9s/110s common coverage')
    memory = table('remote23-release/memory-current-version-review/memory_source.csv')
    memory_keys = [(r['backend'], r['recording'], float(r['horizon_s'])) for r in memory]
    if len(memory_keys) != 12 or set(memory_keys) != {(b, m, t) for b, m in
            [('rust', 'full'), ('rust', 'rolling'), ('cpp', 'full')] for t in [1, 10, 100, 1000]}:
        raise ValueError('complete memory grid required')
    if any(r['rss_complete'] != 'True' for r in memory):
        raise ValueError('complete RSS observations required')
    scaling = table('remote27-release/paired-scaling-review/timing_source.csv')
    scaling = [r for r in scaling if r['suite'] == 'performance']
    scaling_keys = [(r['backend'], int(r['threads']), int(r['repeat'])) for r in scaling]
    if len(scaling_keys) != 36 or set(scaling_keys) != {(b, n, k) for b in ['rust', 'cpp']
            for n in [1, 2, 4, 8, 16, 20] for k in range(3)}:
        raise ValueError('complete Mac paired scaling grid required')
    mac = json.loads(checked('remote27-release/performance-confirmation-review-consistent/report.json').read_text())
    gates = {'27': mac['gates']}
    linux = None
    historical = None
    if linux_final_validation is not None:
        _, linux = review_linux(linux_final_validation)
        gates['23'] = linux['gates']
    if not draft:
        if linux is None or control_reproduction is None:
            raise ValueError('final figure requires Linux confirmation and complete control source reproduction')
        from litwin_kumar_historical_cpp_review import review as review_historical
        historical = review_historical(root, linux_final_validation)
        if historical.get('all_passed') is not True:
            raise ValueError('final Linux confirmation must also beat historical C++ comparators')
        reproduction = json.loads(control_reproduction.read_text())
        if (reproduction.get('all_passed') is not True or len(reproduction.get('checks', [])) != 12
                or {(r['seed'], r['condition']) for r in reproduction['checks']} != set(lookup)
                or any(r['metrics'] != 16 for r in reproduction['checks'])):
            raise ValueError('complete source-data reproduction required')
        evidence = control_reproduction.parent
        control_manifest = evidence/'manifest.json'
        if hashlib.sha256(control_manifest.read_bytes()).hexdigest() != reproduction['source_manifest_sha256']:
            raise ValueError('control reproduction is not bound to the adjacent source manifest')
        source_report = evidence/'analysis-report.json'
        files = json.loads(control_manifest.read_text())['files']
        if (hashlib.sha256(source_report.read_bytes()).hexdigest() != files['analysis-report.json']
                or json.loads(source_report.read_text()) != control_report):
            raise ValueError('control reproduction describes a different figure dataset')
        sources[str(control_reproduction)] = hashlib.sha256(control_reproduction.read_bytes()).hexdigest()
    for host, cases in gates.items():
        for scope, gate in cases.items():
            rust, cpp = gate['rust'], gate['cpp']
            if (rust['n'] != 5 or cpp['n'] != 5 or
                    not np.isclose(gate['speedup'], cpp['median_seconds']/rust['median_seconds'], rtol=1e-14)):
                raise ValueError('five-repeat independent confirmation required')
            if not draft and not (gate['passed'] and rust['max_seconds'] < cpp['min_seconds']):
                raise ValueError(f'performance acceptance incomplete: {host}/{scope}')
    output.mkdir(parents=True, exist_ok=False)
    colors = {'rust': '#0072B2', 'rolling': '#009E73', 'cpp': '#D55E00'}
    with plt.rc_context({'font.family': 'Arial', 'font.size': 5.5, 'pdf.fonttype': 42, 'axes.titlesize': 6,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig = plt.figure(figsize=(180/25.4, 170/25.4), layout='constrained')
        grid = fig.add_gridspec(4, 2)
        ax = fig.add_subplot(grid[0, 0])
        im = ax.imshow(matrix, origin='lower', vmin=1.78, vmax=21.4, cmap='viridis',
                       extent=(.5, 20.5, .5, 20.5), aspect='auto')
        ax.set(xlabel='Presynaptic membership', ylabel='Postsynaptic\nmembership',
               xticks=[1, 10, 20], yticks=[1, 10, 20])
        ax.set_title('a  Learned EE connectivity · 2,610 s', loc='left')
        fig.colorbar(im, ax=ax, label='Mean EE weight (pF)', fraction=.05, pad=.02)
        raster_grid = grid[0, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=.02)
        ax = fig.add_subplot(raster_grid[0, 0])
        ax.plot(times, rank[ids], '.', color=colors['rust'], markersize=.25)
        ax.set(xlim=(2600, 2610), ylim=(0, 4000), yticks=[0, 2000, 4000], ylabel='E neuron\nrank')
        ax.tick_params(labelbottom=False)
        ax.set_title('b  Spontaneous activity · representative seed', loc='left')
        rate = fig.add_subplot(raster_grid[1, 0], sharex=ax)
        bins = np.linspace(2600, 2610, 201)
        rate.plot((bins[1:]+bins[:-1])/2, np.histogram(times, bins)[0]/200,
                  color=colors['rust'], linewidth=.6)
        rate.set(xlabel='Biological time (s)', ylabel='E rate\n(Hz)')
        labels = ['Full', 'No patterned\ninput', 'No iSTDP', 'No weight\nnormalization']
        for col, metric, title, ylabel in [
                (0, 'within_between_ratio', 'c  Structural enrichment', 'Within / between\nEE weight'),
                (1, 'spontaneous_population_hz', 'd  Activity under matched controls', 'Spontaneous E rate (Hz)\nlog scale')]:
            ax = fig.add_subplot(grid[1, col])
            for i, seed in enumerate(seeds):
                ax.plot(range(4), [lookup[seed, c][metric] for c in conditions],
                        marker=['o', 's', '^'][i], linestyle=['-', '--', ':'][i],
                        color=[colors['rust'], colors['cpp'], colors['rolling']][i],
                        markersize=3, linewidth=.8, label=str(seed)[-2:])
            ax.set(xticks=range(4), xticklabels=labels, ylabel=ylabel, xlim=(-.2, 3.2))
            ax.set_title(title, loc='left')
            if col == 0:
                ax.axhline(1, color='.5', linestyle='--', linewidth=.6)
                ax.legend(title='Seed suffix', loc='upper center', fontsize=6, title_fontsize=6, frameon=False)
            else:
                ax.set_yscale('log')
        for col, field, title in [(0, 'peak_native_rss_bytes', 'e  Native execution RSS'),
                                  (1, 'peak_process_tree_rss_bytes', 'f  Process-tree RSS')]:
            ax = fig.add_subplot(grid[2, col])
            for backend, recording, color in [('rust', 'full', colors['rust']),
                    ('rust', 'rolling', colors['rolling']), ('cpp', 'full', colors['cpp'])]:
                values = sorted((r for r in memory if (r['backend'], r['recording']) == (backend, recording)),
                                key=lambda r: float(r['horizon_s']))
                ax.plot([float(r['horizon_s']) for r in values], [float(r[field])/1e9 for r in values],
                        marker='^' if recording == 'rolling' else ('s' if backend == 'cpp' else 'o'),
                        linestyle='--' if recording == 'rolling' else '-', markersize=3, linewidth=.8, color=color,
                        label=('Rust' if backend == 'rust' else 'C++')+' · '+recording)
            ax.set(xscale='log', xlabel='Simulated duration (s)', ylabel='Peak RSS (GB)', ylim=(0, None))
            ax.set_title(title+' · one worker', loc='left')
            ax.legend(fontsize=6, frameon=False, loc='upper left' if col == 0 else 'center left')
        ax = fig.add_subplot(grid[3, 0])
        performance_source = []
        for i, (host, scope) in enumerate([('27', 'performance'), ('27', 'whole_run'),
                                          ('23', 'performance'), ('23', 'whole_run')]):
            if host not in gates:
                continue
            gate = gates[host][scope]
            lower = gate['cpp']['min_seconds']/gate['rust']['max_seconds']
            upper = gate['cpp']['max_seconds']/gate['rust']['min_seconds']
            median = gate['speedup']
            ax.errorbar(i, median, yerr=[[median-lower], [upper-median]], fmt='o', capsize=3,
                        markersize=4, color=colors['rust'])
            ax.annotate(f'{median:.2f}×', (i, median), xytext=(0, -12),
                        textcoords='offset points', ha='center', fontsize=6)
            performance_source.append(dict(host=host, scope=scope, speedup=median,
                                           ratio_envelope_low=lower, ratio_envelope_high=upper))
        if '23' not in gates:
            ax.axvspan(1.55, 3.45, color='.95', zorder=-2)
            ax.text(2.5, 1.3, 'Linux confirmation\npending', ha='center', va='center', color='.4')
        ax.axhline(1., color='.5', linestyle='--', linewidth=.6)
        ax.set(xticks=range(4), xticklabels=['27\npost', '27\nwhole', '23\npost', '23\nwhole'],
               xlim=(-.45, 3.45), ylim=(.7, max(1.75, max(r['ratio_envelope_high'] for r in performance_source)*1.1)),
               ylabel='C++ / Rust time\n(ratio of medians)')
        ax.set_title('g  Independent performance confirmation · n = 5', loc='left')
        ax = fig.add_subplot(grid[3, 1])
        for backend in ['rust', 'cpp']:
            workers = [1, 2, 4, 8, 16, 20]
            groups = [np.array([float(r['simulation_seconds']) for r in scaling
                       if r['backend'] == backend and int(r['threads']) == n]) for n in workers]
            baseline = np.median(groups[0])
            median = np.array([baseline/np.median(g) for g in groups])
            low = np.array([baseline/g.max() for g in groups])
            high = np.array([baseline/g.min() for g in groups])
            ax.errorbar(workers, median, yerr=[median-low, high-median], fmt='o-', markersize=3,
                        capsize=2, linewidth=.8, color=colors[backend], label='Rust' if backend == 'rust' else 'C++')
        ax.set_xscale('log', base=2)
        ax.set(xlabel='Worker count', ylabel='Speedup vs own\n1-worker median',
               xticks=[1, 2, 4, 8, 16, 20], xticklabels=['1', '2', '4', '8', '16', '20'], xlim=(.9, 26), ylim=(0, None))
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_title('h  Mac 27 scaling · post-warmup · n = 3', loc='left')
        ax.legend(frameon=False, fontsize=6)
        title = 'Assembly formation and execution · N = 5,000'
        if draft:
            title += '\nDRAFT · incomplete acceptance evidence'
        fig.suptitle(title, fontsize=7)
        fig.savefig(output/'figure3_litwin_kumar.pdf')
        fig.savefig(output/'figure3_litwin_kumar.png', dpi=300)
        plt.close(fig)
    summary = dict(draft=draft, figure_data_checks_passed=not draft,
                   overall_release_audit='COMPLETION_AUDIT.md', representative_seed=20260906,
                   figure_dimensions_mm=[180, 170], font_family='Arial', font_size_range_pt=[5.5, 7],
                   figure_sources_sha256=sources, performance_ratios=performance_source,
                   linux_review=linux, historical_cpp_review=historical,
                   control_source_reproduction_required_for_final=True)
    (output/'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (output/'caption.md').write_text(
        'a, Mean final EE weights over existing edges between overlapping stimulus memberships. '
        'The representative Rust seed 20260906 began with uniform 2.76 pF weights. Memberships are not disjoint clusters. '
        'b, All E spikes from the same seed in [2600,2610) s; neurons are ordered by first membership, with '
        'unassigned neurons last. Population rates use 50 ms bins. This representative seed/window is a review '
        'convention, not a preregistered selection. Companion figures show both backends and all seeds. '
        'c,d, Three independent matched seeds, 9 s common baseline and 110 s noncontiguous retained spontaneous '
        'activity per seed, following complete 2610 s simulations. Lines pair seeds across conditions. '
        'No-iSTDP activity is hyperactive; higher conditioned scores alone do not indicate better assemblies. '
        'These descriptive controls do not establish attractor dynamics or significance (all 12 Holm p=1).\n\n'
        'e,f, Single observations at each horizon, full-history Rust/C++ versus Rust retaining 1 s. '
        'Memory uses the one-worker JSON-optimized cohort before the final edge-CodeRunner parallelization. '
        'Native and summed process-tree RSS are distinct; tree totals can double-count shared pages. '
        'Rust has lower native RSS here, while its total process-tree RSS exceeds C++. No OOM occurred.\n\n'
        'g, Separate independent n=5 confirmations: post denotes 10–11 s and whole denotes 0–11 s of biological '
        'simulation plus recording, excluding export/compilation. Dots are ratios of medians; whiskers span '
        'C++minimum/Rustmaximum to C++maximum/Rustminimum, not confidence intervals. Worker/compiler details '
        'are in the host reports; historical C++ comparator selection is reviewed separately. Linux uses Rust16/32 '
        'versus C++4 workers, so elapsed-time wins do not establish better core efficiency. The separate Linux '
        'scaling figure includes absolute timings and its slower one-worker Rust result. Missing Linux values are '
        'explicit in drafts. h, Separate Mac n=3 paired scaling cohort, normalized to each backend own '
        'one-worker median. Whiskers reflect observed timing extrema with a fixed denominator; these are '
        'not independent confirmation repeats or confidence intervals. All source paths and hashes are in report.json.\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draft', action='store_true')
    parser.add_argument('--linux-final-validation', type=Path)
    parser.add_argument('--control-reproduction', type=Path)
    render(**vars(parser.parse_args()))
