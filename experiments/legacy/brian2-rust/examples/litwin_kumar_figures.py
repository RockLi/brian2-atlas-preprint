"""Render measured LK evidence and source data, explicitly labelling incomplete protocols."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'rust': '#0072B2', 'cpp': '#D55E00', 'within': '#009E73', 'between': '#777777'}


def assembly_weight_matrix(source, target, weights, membership):
    """Mean over existing edges; overlapping memberships contribute to each relevant pair."""
    groups, neurons = membership.shape
    means = np.full((groups, groups), np.nan)
    counts = np.zeros((groups, groups), dtype=np.int64)
    for pre in range(groups):
        mask = membership[pre, source]
        sums_by_target = np.bincount(target[mask], weights=weights[mask], minlength=neurons)
        counts_by_target = np.bincount(target[mask], minlength=neurons)
        sums = membership @ sums_by_target
        counts[:, pre] = membership @ counts_by_target
        np.divide(sums, counts[:, pre], out=means[:, pre], where=counts[:, pre] > 0)
    return means, counts


def csv_rows(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def missing(ax, text):
    ax.text(.5, .5, text, transform=ax.transAxes, ha='center', va='center',
            fontsize=8, color='#666666', wrap=True)
    ax.set_axis_off()


def science(path, destination):
    report = json.loads((path/'result.json').read_text())
    config = report['configuration']
    with np.load(path/'activity.npz') as activity:
        membership = activity['membership_e']
        source, target = activity['ee_i'], activity['ee_j']
        initial, counts = assembly_weight_matrix(source, target, activity['ee_initial'], membership)
        final, _ = assembly_weight_matrix(source, target, activity['ee_final'], membership)
        spike_i, spike_t = activity['exc_spike_i'].copy(), activity['exc_spike_t'].copy()
        within = np.zeros(len(source), dtype=bool)
        for members in membership:
            within |= members[source] & members[target]
        initial_metrics = {
            'within_mean_pf': float(activity['ee_initial'][within].mean()) if within.any() else None,
            'between_mean_pf': float(activity['ee_initial'][~within].mean()) if (~within).any() else None,
            'mean_absolute_row_sum_error_pf': float(np.mean(np.abs(
                np.bincount(target, weights=activity['ee_initial'], minlength=config['ne'])-
                np.bincount(target, minlength=config['ne'])*2.76))),
            'upper_bound_fraction': float(np.mean(activity['ee_initial'] >= 21.4-1e-12))}
    np.savez_compressed(destination/'connectivity_source.npz', initial=initial, final=final,
                        edge_counts=counts, membership=membership)
    csv_rows(destination/'connectivity.csv', [
        {'post_assembly': post+1, 'pre_assembly': pre+1,
         'existing_edges': int(counts[post, pre]),
         'initial_mean_pf': initial[post, pre], 'final_mean_pf': final[post, pre]}
        for post in range(len(membership)) for pre in range(len(membership))])
    trajectories = report.get('trajectory', []) or [
        {'time_s': report['biological_seconds'], 'weights': report['weights']}]
    trajectories = [{'time_s': 0, 'weights': initial_metrics}, *trajectories]
    csv_rows(destination/'weight_trajectory.csv', [
        {'time_s': item['time_s'], 'within_pf': item['weights']['within_mean_pf'],
         'between_pf': item['weights']['between_mean_pf'],
         'mean_row_error_pf': item['weights']['mean_absolute_row_sum_error_pf'],
         'upper_fraction': item['weights']['upper_bound_fraction']}
        for item in trajectories])
    # Full segments are read once, discarding duplicated retained history at boundaries.
    bin_s = .05
    nbin = int(np.ceil(report['biological_seconds']/bin_s))
    rates = np.zeros((len(membership), nbin))
    population = np.zeros(nbin)
    observed_seconds = np.zeros(nbin)
    segments = report.get('segments', [])
    if not segments:
        # C++ and NumPy execute one complete run and retain their monitor in
        # activity.npz. Missing segment files in a segmented run stay missing.
        segments = [{'start_s': report.get('initial_time_seconds', 0.),
                     'stop_s': report['biological_seconds'], 'inline_activity': True}]
    for index, segment in enumerate(segments, start=1):
        file = path/'segments'/f'{index:04d}'/'spikes.npz'
        if not segment.get('inline_activity') and not file.exists():
            continue
        start, stop = segment['start_s'], segment['stop_s']
        window = report['recording_window_steps']
        if window is not None:
            start = max(start, stop-window*config['dt_ms']/1000)
        first = max(0, int(np.floor(start/bin_s+1e-10)))
        last = min(nbin, int(np.ceil(stop/bin_s-1e-10)))
        if last <= first:
            continue
        if segment.get('inline_activity'):
            ids, times = spike_i, spike_t
        else:
            with np.load(file) as spikes:
                ids, times = spikes['exc_i'], spikes['exc_t']
        epsilon = min(1e-10, config['dt_ms']/1000/4)
        valid = (times >= start-epsilon) & (times < stop-epsilon)
        ids, bins = ids[valid], np.floor(times[valid]/bin_s+1e-10).astype(int)
        population[first:last] += np.bincount(bins, minlength=nbin)[first:last]/(config['ne']*bin_s)
        for assembly, members in enumerate(membership):
            if members.any():
                rates[assembly, first:last] += np.bincount(bins[members[ids]], minlength=nbin)[first:last]/(members.sum()*bin_s)
        left = np.arange(first, last)*bin_s
        observed_seconds[first:last] += np.maximum(0., np.minimum(left+bin_s, stop)-np.maximum(left, start))
    # Contiguous partial pieces can complete a bin; a missing interval cannot.
    covered = np.isclose(observed_seconds, bin_s, rtol=0, atol=1e-9)
    rates[:, ~covered] = np.nan
    rates[~membership.any(axis=1)] = np.nan
    population[~covered] = np.nan
    bin_centers = (np.arange(nbin)+.5)*bin_s
    np.savez_compressed(destination/'activity_source.npz', time_s=bin_centers,
                        assembly_rates_hz=rates, population_rate_hz=population,
                        coverage=covered, retained_spike_i=spike_i, retained_spike_t=spike_t)
    train_end = config['warmup_s']+config['training_s']
    baseline = covered & (bin_centers >= min(1., config['warmup_s']/2)) & (bin_centers < config['warmup_s'])
    spontaneous = covered & (bin_centers >= train_end)
    diagnostics = {'biological_seconds': report['biological_seconds'],
        'complete_training_protocol': report['complete_training_protocol'],
        'complete_default_protocol': report['biological_seconds'] >= config['duration_s'],
        'covered_activity_seconds': float(covered.sum()*bin_s),
        'baseline_seconds': float(baseline.sum()*bin_s),
        'spontaneous_seconds': float(spontaneous.sum()*bin_s),
        'within_between_ratio': report['weights']['within_mean_pf']/report['weights']['between_mean_pf'],
        'activation_definition': '50 ms bins above each assembly pretraining mean + 3 SD; descriptive, not a significance test'}
    responses = []
    period = config['stimulus_s']+config['gap_s']
    for presentation in range(config['assemblies']*config['repetitions']):
        onset = config['warmup_s']+presentation*period
        assembly = presentation % config['assemblies']
        on = (bin_centers >= onset) & (bin_centers < onset+config['stimulus_s'])
        off = (bin_centers >= onset+config['stimulus_s']) & (bin_centers < onset+period)
        if not on.any() or not np.all(covered[on]) or on.sum()*bin_s < config['stimulus_s']-1e-8:
            continue
        responses.append({'presentation': presentation+1, 'assembly': assembly+1,
            'onset_s': onset, 'member_evoked_hz': float(np.mean(rates[assembly, on])),
            'population_evoked_hz': float(np.mean(population[on])),
            'member_gap_hz': float(np.mean(rates[assembly, off]))
                if off.any() and np.all(covered[off]) and off.sum()*bin_s >= config['gap_s']-1e-8 else None})
    csv_rows(destination/'stimulus_responses.csv', responses)
    diagnostics['complete_stimulus_presentations_observed'] = len(responses)
    diagnostics['mean_evoked_member_hz'] = float(np.mean([r['member_evoked_hz'] for r in responses])) if responses else None
    if baseline.sum() >= 20 and spontaneous.any():
        thresholds = np.nanmean(rates[:, baseline], axis=1)+3*np.nanstd(rates[:, baseline], axis=1, ddof=1)
        activation = rates[:, spontaneous] > thresholds[:, None]
        starts = activation & ~np.pad(activation[:, :-1], ((0, 0), (1, 0)))
        diagnostics['activation_threshold_hz'] = thresholds.tolist()
        diagnostics['activation_event_counts'] = starts.sum(axis=1).tolist()
        diagnostics['activation_occupancy'] = activation.mean(axis=1).tolist()
        diagnostics['assemblies_with_activation'] = int(np.any(activation, axis=1).sum())
    (destination/'science_diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    return report, membership, initial, final, spike_i, spike_t, trajectories, diagnostics


def render(args):
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelsize': 8,
        'axes.titlesize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'savefig.dpi': 300})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), layout='constrained')
    for letter, ax in zip('abcdef', axes.flat):
        ax.text(-.16, 1.08, letter, transform=ax.transAxes, fontsize=11, fontweight='bold')
    caption = []
    complete = False
    if args.science_run:
        report, members, initial, final, ids, times, trajectory, diag = science(args.science_run, args.output)
        complete = diag['complete_default_protocol'] and report['configuration']['scale'] == 1.
        image = axes[0, 0].imshow(final, origin='lower', cmap='viridis', aspect='equal',
                                   vmin=1.78, vmax=21.4,
                                   extent=(.5, len(members)+.5, .5, len(members)+.5))
        fig.colorbar(image, ax=axes[0, 0], shrink=.8, orientation='horizontal',
                     label='Mean weight (pF)', ticks=[1.78, 10, 21.4])
        axes[0, 0].set(xlabel='Presynaptic assembly', ylabel='Postsynaptic assembly',
                      title='Assembly connectivity')
        for field, label, color in [('within_mean_pf', 'Within', COLORS['within']),
                                     ('between_mean_pf', 'Between', COLORS['between'])]:
            axes[0, 1].plot([x['time_s'] for x in trajectory],
                [x['weights'][field] for x in trajectory],
                label=label, color=color, linewidth=1.3,
                linestyle='-' if report.get('trajectory') else 'None',
                marker=None if report.get('trajectory') else 'o')
        axes[0, 1].set(xlabel='Biological time (s)', ylabel='Mean weight (pF)',
                      title='Weight evolution' if report.get('trajectory') else 'Weight endpoints')
        axes[0, 1].legend(frameon=False, fontsize=7)
        end = report['biological_seconds']
        retained_s = (report['recording_window_steps']*report['configuration']['dt_ms']/1000
                      if report['recording_window_steps'] else end)
        start = max(0, end-min(5, retained_s))
        selected = (times >= start) & (times < end)
        # Primary-membership order is a display convention only; analysis keeps overlaps.
        primary = np.where(members.any(axis=0), members.argmax(axis=0), len(members))
        order = np.argsort(primary, kind='stable')
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        axes[0, 2].scatter(times[selected], rank[ids[selected]], s=.12, color='#222222', rasterized=True)
        axes[0, 2].set(xlabel='Biological time (s)', ylabel='E neuron (ordered)',
                      title='Retained activity', xlim=(start, end), ylim=(0, len(order)))
        caption += ['a, Mean weight over existing edges between overlapping stimulus memberships; absent edges excluded.',
                    'b, Within/shared-membership and between/disjoint-membership E→E means. Individual seeds are the independent replicate.',
                    'c, Final five seconds available in the retained monitor; neurons ordered by first membership, then non-members.']
        if not report.get('trajectory'):
            caption += ['Only baseline and final weights were recorded for this run; no intermediate weight trajectory is inferred.']
    else:
        for ax in axes[0]:
            missing(ax, 'Scientific run not supplied')
    if args.performance:
        benchmark = json.loads((args.performance/'report.json').read_text())
        csv_rows(args.output/'timing_samples.csv', [
            {k: row[k] for k in ['backend', 'threads', 'repeat', 'simulation_seconds']}
            for row in benchmark['samples']])
        for backend in ['rust', 'cpp']:
            rows = sorted([x for x in benchmark['aggregates'] if x['backend'] == backend], key=lambda x: x['threads'])
            x, y = [r['threads'] for r in rows], np.array([r['median_seconds'] for r in rows])
            axes[1, 0].errorbar(x, y, yerr=[y-[r['min_seconds'] for r in rows],
                                  np.array([r['max_seconds'] for r in rows])-y],
                              marker='o', capsize=2, color=COLORS[backend], label=backend)
            baseline = next((r['median_seconds'] for r in rows if r['threads'] == 1), None)
            if baseline is not None:
                axes[1, 1].plot(x, baseline/y, 'o-', color=COLORS[backend], label=backend)
            else:
                axes[1, 1].text(.04, .06, f'{backend}: one-worker timing not sampled',
                               transform=axes[1, 1].transAxes, fontsize=6)
        scope = ('Complete simulation' if benchmark.get('suite') == 'whole_run'
                 else 'Post-warmup interval')
        axes[1, 0].set(title=scope, xlabel='Worker threads', ylabel='Simulation time (s)')
        axes[1, 1].set(title='Within-backend scaling', xlabel='Worker threads', ylabel='Speedup over one worker')
        for ax in axes[1, :2]:
            ax.legend(frameon=False)
        caption += [f'd, {scope}. Median native simulation plus recording time; bars show minimum–maximum across independent process runs. '
                    f'Profiled: {benchmark.get("profiled", False)}. '+benchmark.get('timing_contract', ''),
                    'e, Each backend normalized to its own one-worker median when sampled; no unmeasured baseline is inferred. Full-history monitors match; external RNG algorithms differ.']
    else:
        for ax in axes[1, :2]:
            missing(ax, 'Performance not yet measured')
    if args.memory:
        memory = json.loads((args.memory/'report.json').read_text())['experiments']
        rows = []
        for backend, bounded in [('rust', False), ('rust', True), ('cpp', False)]:
            selected = sorted([r for r in memory if r['backend'] == backend and
                               (r['window_steps'] is not None) == bounded], key=lambda x: x['horizon_s'])
            selected = [r for r in selected if r['process']['rss_complete'] and
                        r['process'].get('peak_native_rss_bytes') is not None]
            label = f'{backend} '+('rolling' if bounded else 'full')
            axes[1, 2].plot([r['horizon_s'] for r in selected],
                           [r['process']['peak_native_rss_bytes']/2**30 for r in selected],
                           'o--' if bounded else 'o-', color=COLORS[backend], label=label)
            rows += [{'condition': label, 'horizon_s': r['horizon_s'],
                      'peak_native_rss_bytes': r['process']['peak_native_rss_bytes'],
                      'peak_process_tree_rss_bytes': r['process']['peak_process_tree_rss_bytes']} for r in selected]
        csv_rows(args.output/'memory_samples.csv', rows)
        axes[1, 2].set(xlabel='Run horizon (biological s)', ylabel='Peak native RSS (GiB)', title='Measured memory')
        axes[1, 2].legend(frameon=False, fontsize=6.5)
        caption += ['f, Peak sampled native-process RSS; horizons are separate runs. End-to-end process-tree RSS is exported separately, including construction, compilation and export; shared pages can be counted twice in the tree total.']
    else:
        missing(axes[1, 2], 'Memory not yet measured')
    full_evidence = complete and args.performance is not None and args.memory is not None
    fig.suptitle('LK2014 triplet variant — '+('measured evidence' if full_evidence else 'development evidence; incomplete validation'), fontsize=10)
    fig.savefig(args.output/'figure3_litwin_kumar.pdf')
    fig.savefig(args.output/'figure3_litwin_kumar.png')
    plt.close(fig)
    (args.output/'caption.md').write_text('\n\n'.join(caption)+
        '\n\nThis figure does not establish manuscript readiness. Multiple seeds, matched controls and numerical sensitivity checks must be reviewed separately.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--science-run', type=Path)
    parser.add_argument('--performance', type=Path)
    parser.add_argument('--memory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    render(parser.parse_args())
