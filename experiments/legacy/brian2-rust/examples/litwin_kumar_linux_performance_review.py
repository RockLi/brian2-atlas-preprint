"""Review Linux compiler/placement selection and independent confirmation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from litwin_kumar_figures import COLORS, csv_rows

SUITES = ('performance', 'whole_run')
METRICS = ('backend', 'threads', 'n', 'median_seconds', 'min_seconds', 'max_seconds')


def aggregate_samples(report, suite, repeats):
    if (report.get('suite') != suite or report.get('profiled') is not False
            or report.get('warmup_s') != 10
            or report.get('timed_duration_s') != (1 if suite == 'performance' else 11)
            or report.get('recording') != 'identical full-history SpikeMonitor and StateMonitor'):
        raise ValueError('explicit unprofiled timing and recording scope required')
    if not (report.get('rust_worker_results_byte_exact') is True
            and report.get('cpp_fixed_worker_repeats_byte_exact') is True):
        raise ValueError('backend repeatability evidence required')
    counts = report['thread_counts']
    if set(counts) != {'rust', 'cpp'} or any(
            not ns or len(ns) != len(set(ns)) or any(type(n) is not int or n < 1 for n in ns)
            for ns in counts.values()):
        raise ValueError('unique positive worker counts required')
    samples = report['samples']
    keys = [(r['backend'], r['threads'], r['repeat']) for r in samples]
    required = {(b, n, k) for b, ns in counts.items() for n in ns for k in range(repeats)}
    if len(keys) != len(set(keys)) or set(keys) != required:
        raise ValueError('complete unique backend/thread/repeat grid required')
    for row in samples:
        if row['backend'] == 'rust':
            summary = row.get('summary', {})
            if (summary.get('neuron_count') != 5000 or summary.get('threads') != row['threads']
                    or summary.get('phase_profile', {}).get('enabled') is not False):
                raise ValueError('full-scale unprofiled worker evidence required')
    rows = []
    for backend, ns in counts.items():
        for n in ns:
            values = np.array([r['simulation_seconds'] for r in samples
                               if (r['backend'], r['threads']) == (backend, n)], dtype=float)
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError('finite positive timings required')
            rows.append(dict(backend=backend, threads=n, n=repeats,
                             median_seconds=float(np.median(values)),
                             min_seconds=float(values.min()), max_seconds=float(values.max())))
    return rows


def check_aggregate(recorded, computed):
    if any(recorded.get(k) != computed[k] for k in METRICS):
        raise ValueError('recorded aggregate disagrees with raw samples')


def review(root):
    sources = {}

    def read(name):
        path = root/name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    contract = read('contract.json')
    profiles = contract['cpp_profiles']
    workers = contract['cpp_workers']
    bindings = contract['cpp_placements']
    repeats = contract['cpp_selection_repeats']
    if (set(profiles) != {'strict', 'native-strict', 'brian2-default'}
            or set(bindings) != {'spread', 'close', 'matched-rust'} or repeats != 2
            or len(workers) != len(set(workers)) or set(workers) != {1, 2, 4, 8, 16}):
        raise ValueError('complete declared Linux compiler/worker/placement search required')
    samples = read('cpp-placement-samples.json')
    keys = [(r['profile'], r['threads'], r['binding'], r['repeat']) for r in samples]
    expected = {(p, n, b, k) for p in profiles for n in workers for b in bindings for k in range(repeats)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('complete unique compiler/placement selection grid required')
    candidates = []
    for profile in profiles:
        for n in workers:
            for binding in bindings:
                group = [r for r in samples if (r['profile'], r['threads'], r['binding']) == (profile, n, binding)]
                item = dict(profile=profile, threads=n, binding=binding, n=repeats)
                for suite in SUITES:
                    values = np.array([r[suite] for r in group], dtype=float)
                    if not np.isfinite(values).all() or np.any(values <= 0):
                        raise ValueError('finite positive selection timings required')
                    item[suite] = dict(median_seconds=float(np.median(values)),
                                       min_seconds=float(values.min()), max_seconds=float(values.max()))
                candidates.append(item)
    saved = read('cpp-placement-candidates.json')
    if saved.get('all_outputs_byte_exact') is not True or saved['candidates'] != candidates:
        raise ValueError('placement aggregates or repeatability disagree with source data')
    recorded_gates = read('linux-final-gates.json')
    data, gates = {}, {}
    for suite in SUITES:
        pilot = read(suite+'-rust-selection/report.json')
        pilot_rows = aggregate_samples(pilot, suite, 2)
        if set(pilot['thread_counts']['rust']) != set(contract['rust_selection_workers']):
            raise ValueError('Rust selection worker grid differs from contract')
        best_rust = min((r for r in pilot_rows if r['backend'] == 'rust'), key=lambda r:r['median_seconds'])
        best_cpp = min(candidates, key=lambda r:r[suite]['median_seconds'])
        fixed = read(suite+'-fixed-selection.json')
        check_aggregate(fixed['rust'], best_rust)
        if fixed['cpp'] != best_cpp:
            raise ValueError('fixed C++ selection is not the fastest measured candidate')
        report = read(suite+'-confirmation/report.json')
        rows = aggregate_samples(report, suite, 5)
        if (report['thread_counts']['cpp'] != [best_cpp['threads']]
                or set(report['thread_counts']['rust']) != {1, best_rust['threads']}
                or report['cpp_compile_profile'] != best_cpp['profile']):
            raise ValueError('confirmation configuration differs from fixed selection')
        event_counts = []
        for build in report['builds'].values():
            run = build['run']
            if run['configuration']['scale'] != 1 or run['recording_window_steps'] is not None:
                raise ValueError('full-scale full-history build required')
            event_counts.append(run['timed_interval']['synaptic_events'] if suite == 'performance'
                                else run['synaptic_events_from_full_history'])
        if not event_counts or min(event_counts) <= 0:
            raise ValueError('positive timed event volumes required')
        ratio = max(event_counts)/min(event_counts)
        if ratio > 1.25 or ratio != report['build_run_event_count_ratio']:
            raise ValueError('event workload evidence disagrees with guard')
        gate = {backend: next(r for r in rows if r['backend'] == backend and r['threads'] == n)
                for backend, n in [('rust', best_rust['threads']), ('cpp', best_cpp['threads'])]}
        gate.update(speedup=gate['cpp']['median_seconds']/gate['rust']['median_seconds'],
                    passed=gate['rust']['max_seconds'] < gate['cpp']['min_seconds'],
                    cpp_selection=best_cpp, event_count_ratio=ratio)
        recorded = recorded_gates[suite]
        for backend in ['rust', 'cpp']:
            check_aggregate(recorded[backend], gate[backend])
        if any(recorded.get(k) != gate[k] for k in ['speedup', 'passed', 'cpp_selection']):
            raise ValueError('recorded acceptance gate disagrees with raw confirmation')
        data[suite], gates[suite] = report, gate
    completed = read('completed.json')
    if (completed.get('completed') is not True
            or completed.get('both_scopes_passed') != all(g['passed'] for g in gates.values())):
        raise ValueError('completion flag disagrees with confirmation')
    return data, dict(gates=gates, source_reports_sha256=sources,
        both_scopes_passed=all(g['passed'] for g in gates.values()),
        scope='Linux preselected configuration, independent five-repeat compiled simulation and recording; build/export excluded',
        limitations='Compiler/placement search is the declared measured grid, not an exhaustive optimizer. Backend random streams differ; event-volume guard is not statistical equivalence.')


def render(final_validation, output):
    import matplotlib.pyplot as plt
    data, summary = review(final_validation)
    output.mkdir(parents=True, exist_ok=False)
    csv_rows(output/'confirmation_source.csv', [dict(suite=suite, **{k:row[k] for k in
        ['backend', 'threads', 'repeat', 'simulation_seconds']})
        for suite, report in data.items() for row in report['samples']])
    with plt.rc_context({'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), layout='constrained')
        for ax, suite in zip(axes, SUITES):
            gate = summary['gates'][suite]
            for index, backend in enumerate(['rust', 'cpp']):
                group = gate[backend]
                values = [r['simulation_seconds'] for r in data[suite]['samples']
                          if (r['backend'], r['threads']) == (backend, group['threads'])]
                ax.scatter(index+np.linspace(-.08, .08, 5), values, s=20, color=COLORS[backend])
                median = group['median_seconds']
                ax.errorbar(index, median, yerr=[[median-group['min_seconds']], [group['max_seconds']-median]],
                            fmt='_', markersize=15, capsize=5, color='black')
            ax.set(xticks=[0, 1], xticklabels=[f"{b} · {gate[b]['threads']} workers" for b in ['rust', 'cpp']],
                   ylabel='Simulation + recording (s)', xlim=(-.5, 1.5),
                   ylim=(0, max(gate[b]['max_seconds'] for b in ['rust', 'cpp'])*1.25),
                   title='Post-warmup · 10–11 s' if suite == 'performance' else 'Complete run · 0–11 s')
            ax.text(.98, .96, f"{gate['speedup']:.2f}× · {'PASS' if gate['passed'] else 'FAIL'}",
                    ha='right', va='top', transform=ax.transAxes)
        fig.suptitle('Linux 23 · N = 5,000 · independent confirmation', fontsize=11)
        fig.savefig(output/'linux_confirmation.pdf')
        fig.savefig(output/'linux_confirmation.png', dpi=220)
        plt.close(fig)
    (output/'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    descriptions = '; '.join(f"{suite}: Rust {g['rust']['threads']} workers, C++ {g['cpp']['threads']} workers, "
                            f"{g['cpp_selection']['profile']}, {g['cpp_selection']['binding']} placement"
                            for suite, g in summary['gates'].items())
    (output/'caption.md').write_text(
        'Five independent fresh-process observations per selected configuration; dots are observations, '
        'black marks show medians and minimum–maximum ranges. Configurations were selected in separate '
        'two-repeat pilots. '+descriptions+'. Compilation and export are excluded; both backends retain '
        'full monitor history. The complete run includes biological warmup in one native run. '
        'PASS requires every Rust observation to be faster than every selected C++ observation. '
        +summary['limitations']+' Source report hashes and gates are in report.json.\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final-validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    render(**vars(parser.parse_args()))
