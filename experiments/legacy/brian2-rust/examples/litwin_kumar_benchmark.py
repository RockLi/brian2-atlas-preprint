"""Auditable LK comparison: separate correctness, warm compute and memory experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT/'examples/litwin_kumar_device.py'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def measured_process(command, directory, *, cwd=None, env=None):
    """Sample the sum of live process-tree RSS; never substitute missing samples."""
    directory.mkdir(parents=True, exist_ok=False)
    samples, errors = [], set()
    started = time.monotonic()
    with (directory/'process.log').open('w') as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
        root = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                processes = [root, *root.children(recursive=True)]
                rss, present, native_rss, native_processes = 0, 0, 0, 0
                for child in processes:
                    try:
                        child_rss = child.memory_info().rss
                        rss += child_rss
                        present += 1
                        if child.name() in {'b2-native', 'main', 'b2-native.exe', 'main.exe'}:
                            native_rss += child_rss
                            native_processes += 1
                    except psutil.NoSuchProcess:
                        pass
                samples.append({'wall_seconds': time.monotonic()-started,
                                'process_tree_rss_bytes': rss,
                                'native_simulation_rss_bytes': native_rss if native_processes else None,
                                'live_processes': present})
            except (psutil.AccessDenied, psutil.NoSuchProcess, PermissionError) as error:
                errors.add(type(error).__name__)
            time.sleep(.1)
    report = {'command': [str(x) for x in command], 'cwd': str(cwd) if cwd else None,
              'returncode': process.returncode, 'wall_seconds': time.monotonic()-started,
              'rss_sample_interval_s': .1, 'rss_errors': sorted(errors),
              'peak_process_tree_rss_bytes': max((x['process_tree_rss_bytes'] for x in samples), default=None),
              'peak_native_rss_bytes': max((x['native_simulation_rss_bytes'] for x in samples
                                          if x['native_simulation_rss_bytes'] is not None), default=None),
              'rss_complete': bool(samples) and not errors, 'samples': samples}
    write_json(directory/'process.json', report)
    if process.returncode:
        raise RuntimeError(f'command failed; see {directory / "process.log"}')
    return report


def environment():
    versions = {}
    for command in [['rustc', '--version', '--verbose'],
                    [os.environ.get('CXX', 'c++'), '--version'], ['git', 'rev-parse', 'HEAD']]:
        result = subprocess.run(command, capture_output=True, text=True)
        versions[' '.join(command)] = result.stdout.strip() or result.stderr.strip()
    hardware = None
    affinity = (psutil.Process().cpu_affinity()
                if hasattr(psutil.Process, 'cpu_affinity') else None)
    numa = None
    if sys.platform == 'linux' and shutil.which('numactl'):
        result = subprocess.run(['numactl', '--show'], capture_output=True, text=True)
        numa = {'returncode': result.returncode, 'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip()}
    if sys.platform == 'darwin':
        command = ['sysctl', 'machdep.cpu.brand_string', 'hw.perflevel0.physicalcpu',
                   'hw.perflevel1.physicalcpu']
        result = subprocess.run(command, capture_output=True, text=True)
        hardware = {'command': command, 'returncode': result.returncode,
                    'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}
    return {'platform': platform.platform(), 'machine': platform.machine(),
            'python': sys.version, 'logical_cpus': psutil.cpu_count(),
            'physical_cpus': psutil.cpu_count(logical=False),
            'memory_bytes': psutil.virtual_memory().total, 'versions': versions,
            'hardware_details': hardware,
            'allowed_cpus': affinity, 'numa_policy': numa,
            'source_sha256': {str(path.relative_to(ROOT)): sha256(path)
                              for path in [RUNNER, Path(__file__), ROOT/'examples/litwin_kumar_model.py',
                                           *sorted((ROOT/'python/brian2_rust').glob('*.py')),
                                           *sorted((ROOT/'src').glob('*.rs'))]},
            'runner_sha256': sha256(ROOT/'target/release/b2-runner'),
            'CXX': os.environ.get('CXX'),
            'thread_environment': {key: os.environ.get(key) for key in
                ['OMP_PROC_BIND', 'OMP_PLACES', 'OMP_WAIT_POLICY', 'OMP_NUM_THREADS',
                 'B2_THREAD_AFFINITY', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
            'OMP_PROC_BIND': os.environ.get('OMP_PROC_BIND')}


def build_run(output, label, backend, scale, duration_s, threads=1, extra=()):
    destination = output/label
    command = [sys.executable, str(RUNNER), '--backend', backend,
               '--network-scale', str(scale), '--duration-ms', str(duration_s*1000),
               '--threads', str(threads), '--output', str(destination), *extra]
    process = measured_process(command, output/(label+'-measurement'))
    result = json.loads((destination/'result.json').read_text())
    return destination, result, process


def compare_states(left, right, *, exact=False):
    comparisons = {}
    with np.load(left) as a, np.load(right) as b:
        if set(a.files) != set(b.files):
            raise AssertionError('state fields differ')
        for name in sorted(a.files):
            x, y = a[name], b[name]
            same_bytes = x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
            if exact:
                if not same_bytes:
                    raise AssertionError(f'byte mismatch: {name}')
            else:
                np.testing.assert_allclose(x, y, rtol=1e-11, atol=1e-13, err_msg=name)
            comparisons[name] = {'byte_exact': same_bytes,
                'max_absolute_error': float(np.max(np.abs(x.astype(float)-y.astype(float)), initial=0))}
    return comparisons


def correctness(args):
    cases = [('numpy', 'numpy', []), ('reference', 'rust', ['--engine', 'reference']),
             ('rust', 'rust', []), ('cpp', 'cpp', [])]
    directories, measurements = {}, {}
    for label, backend, extra in cases:
        directory, report, process = build_run(args.output, label, backend, .01, .1,
            extra=['--input-mode', 'replay', '--warmup-s', '.02', *extra,
                   *(['--cpp-profile', args.cpp_profile] if backend == 'cpp' else [])])
        directories[label] = directory
        measurements[label] = {'run': report, 'process': process}
    comparison = {label: compare_states(directories['numpy']/'state.npz', path/'state.npz')
                  for label, path in directories.items() if label != 'numpy'}
    return {'case': 'common frozen external input, 40 E / 10 I, 100 ms',
            'comparisons': comparison, 'measurements': measurements}


def performance(args, profile=False, whole_run=False):
    # Compile one C++ program per OpenMP setting; Rust's worker pool is runtime selected.
    thread_counts = {'rust': args.threads,
                     'cpp': getattr(args, 'cpp_threads', None) or args.threads}
    def interval(report):
        if whole_run:
            return {'synaptic_events': report['synaptic_events_from_full_history'],
                    'rates_hz': report['rates_hz']}
        return report['timed_interval']

    builds, measurements = {}, {}
    for backend in ['rust', 'cpp']:
        for threads in ([1] if backend == 'rust' else thread_counts['cpp']):
            label = f'{backend}-{threads}-build'
            path, report, process = build_run(args.output, label, backend,
                args.network_scale, args.warmup_s+args.duration_s, threads,
                extra=['--warmup-s', str(args.warmup_s), '--segment-seconds',
                       str(args.warmup_s+args.duration_s),
                       *(['--disable-phase-segmentation'] if whole_run else ['--split-warmup']),
                       *(['--cpp-profile', args.cpp_profile] if backend == 'cpp' else []),
                       *(['--profile'] if profile else [])])
            builds[(backend, threads)] = path
            measurements[label] = {'run': report, 'process': process}
            write_json(args.output/'builds.json', measurements)
    events = [interval(entry['run'])['synaptic_events'] for entry in measurements.values()]
    workload_ratio = max(events)/min(events) if min(events) else None
    if workload_ratio is None or workload_ratio > 1.25:
        write_json(args.output/'workload_rejected.json', {
            'event_count_ratio': workload_ratio, 'maximum_allowed_ratio': 1.25,
            'meaning': 'engineering guard against grossly unmatched work, not a statistical equivalence test'})
        raise RuntimeError('timed event volumes are not comparable; inspect builds.json')
    schedule = [(backend, threads, repeat) for repeat in range(args.repeats)
                for backend in ['rust', 'cpp'] for threads in thread_counts[backend]]
    np.random.default_rng(20260906).shuffle(schedule)
    samples, rust_hash, cpp_hashes = [], None, {}
    for backend, threads, repeat in schedule:
        built = builds[(backend, 1 if backend == 'rust' else threads)]
        label = f'{backend}-{threads}-repeat-{repeat}'
        env = os.environ.copy()
        env['B2_NUM_THREADS'] = str(threads)
        env['OMP_NUM_THREADS'] = str(threads)
        env['B2_AOT_PROFILE_PHASES'] = str(int(profile))
        if backend == 'rust':
            native = Path(measurements['rust-1-build']['run']['last_run_directory'])/'native'
            result_dir = args.output/(label+'-result')
            command = [str(native/'b2-native'), str(native/'instance.bin'), str(result_dir)]
            cwd = None
        else:
            result_dir = built/'project/results'
            command, cwd = [str(built/'project/main')], built/'project'
        process = measured_process(command, args.output/label, cwd=cwd, env=env)
        if backend == 'rust':
            summary = json.loads((result_dir/'summary.json').read_text())
            seconds = summary['timings']['simulation_and_recording_seconds']
            digest = sha256(result_dir/'results.bin')
            if rust_hash is None:
                rust_hash = digest
            if digest != rust_hash:
                raise AssertionError('Rust internal result differs across worker counts/repeats')
            shutil.rmtree(result_dir)
        else:
            seconds = float((result_dir/'last_run_info.txt').read_text().split()[0])
            summary = {'simulation_seconds': seconds}
            digest = hashlib.sha256(''.join(sha256(path) for path in
                sorted(result_dir.glob('*array_lk_*'))).encode()).hexdigest()
            if threads in cpp_hashes and cpp_hashes[threads] != digest:
                raise AssertionError('C++ numerical results differ between repeats at fixed worker count')
            cpp_hashes[threads] = digest
            if profile:
                summary['code_object_profile_seconds'] = {
                    line.split()[0]: float(line.split()[1])
                    for line in (result_dir/'profiling_info.txt').read_text().splitlines() if line.strip()}
        samples.append({'backend': backend, 'threads': threads, 'repeat': repeat,
                        'simulation_seconds': seconds, 'summary': summary, 'process': process})
        write_json(args.output/'samples.json', samples)
    aggregates = []
    for backend in ['rust', 'cpp']:
        for threads in thread_counts[backend]:
            values = [x['simulation_seconds'] for x in samples
                      if x['backend'] == backend and x['threads'] == threads]
            build = measurements[f'{backend}-{1 if backend == "rust" else threads}-build']['run']
            events = interval(build)['synaptic_events']
            aggregates.append({'backend': backend, 'threads': threads, 'n': len(values),
                               'median_seconds': float(np.median(values)),
                               'min_seconds': min(values), 'max_seconds': max(values),
                               'build_run_rates_hz': interval(build)['rates_hz'],
                               'build_run_synaptic_events': events,
                               'events_per_second_at_median': events/float(np.median(values))})
    return {'profiled': profile, 'builds': measurements, 'samples': samples, 'aggregates': aggregates,
            'cpp_compile_profile': args.cpp_profile,
            'thread_counts': thread_counts,
            'rust_worker_results_byte_exact': True,
            'cpp_fixed_worker_repeats_byte_exact': True,
            'warmup_s': args.warmup_s,
            'timed_duration_s': args.warmup_s+args.duration_s if whole_run else args.duration_s,
            'build_run_event_count_ratio': workload_ratio,
            'recording': 'identical full-history SpikeMonitor and StateMonitor',
            'input_contract': 'same distribution, topology and initial state; backend RNG streams differ',
            'timing_contract': ('fresh process, compiled binary; complete simulation including biological warmup; build-run excluded; randomized order'
                               if whole_run else
                               'fresh process, compiled binary; post-warmup simulation interval only; build-run excluded; randomized order'),
            'warmup_execution': ('both binaries begin at biological time zero and execute the entire warmup and learning interval in one native run'
                                 if whole_run else
                                 'Rust timed binary starts from the saved in-memory warmup endpoint; C++ repeats warmup before its separately timed final run; process wall time includes these differences')}


def whole_run(args):
    """Confirm speed with the same complete warmup execution in both binaries."""
    return performance(args, whole_run=True)


def profiling(args):
    return performance(args, profile=True)


def memory(args):
    reports = []
    for duration in args.horizons_s:
        for backend, window in [('rust', None), ('rust', args.window_steps), ('cpp', None)]:
            label = f'{backend}-{window or "full"}-{duration:g}s'
            extra = ['--warmup-s', '.02', '--segment-seconds', str(duration),
                     '--disable-phase-segmentation']
            if window is not None:
                extra += ['--bounded-window-steps', str(window)]
            if backend == 'rust':
                extra += ['--compact-artifacts']
            else:
                extra += ['--cpp-profile', args.cpp_profile]
            _, run, process = build_run(args.output, label, backend, args.network_scale,
                                       duration, args.threads[0], extra=extra)
            reports.append({'backend': backend, 'window_steps': window, 'horizon_s': duration,
                            'run': run, 'process': process})
            write_json(args.output/'memory.json', reports)
    return {'experiments': reports,
            'measurement': 'native simulation RSS and end-to-end process-tree RSS are recorded separately; raw wall-time samples',
            'caveat': 'RSS is summed across processes and can double-count shared pages; no inferred OOM'}



def convergence(args):
    """Use identical fine-grid input events, aggregated for coarser Euler steps."""
    comparisons = []
    for seed in args.seeds:
        paths = {}
        for dt_ms in [.1, .05, .025]:
            label = f'seed-{seed}-dt-{dt_ms:g}'
            path, report, process = build_run(args.output, label, 'rust', .01, .1,
                extra=['--seed', str(seed), '--input-mode', 'replay',
                       '--replay-base-dt-ms', '.025', '--dt-ms', str(dt_ms),
                       '--warmup-s', '.02', '--compact-artifacts'])
            paths[dt_ms] = path
        with np.load(paths[.025]/'state.npz') as reference:
            for dt_ms in [.1, .05]:
                with np.load(paths[dt_ms]/'state.npz') as candidate:
                    row = {'seed': seed, 'dt_ms': dt_ms, 'reference_dt_ms': .025}
                    for field in ['exc_v', 'inh_v', 'ee_w', 'ie_w']:
                        difference = candidate[field]-reference[field]
                        row[field+'_rmse'] = float(np.sqrt(np.mean(difference**2)))
                    for population in ['exc', 'inh']:
                        row[population+'_spike_count_difference'] = int(
                            len(candidate[population+'_spike_i'])-len(reference[population+'_spike_i']))
                    comparisons.append(row)
        write_json(args.output/'convergence.json', comparisons)
    return {'comparisons': comparisons,
            'scope': 'small complete-model integration check; not full-scale assembly sensitivity',
            'input_contract': 'counts drawn at 0.025 ms and summed into 0.05/0.1 ms bins',
            'units': 'membrane RMSE in volts, weight RMSE in pF',
            'interpretation': 'reset discontinuities can make final-voltage error nonmonotonic; inspect spikes and all seeds'}



def science(args):
    """Matched-seed full protocols; each condition is an independently constructed network."""
    conditions = {'full': [], 'no_stimulation': ['--disable-stimulation'],
                  'no_istdp': ['--disable-istdp'],
                  'no_normalization': ['--disable-normalization']}
    jobs = []
    for seed in args.seeds:
        for condition in args.conditions:
            label = f'{condition}-seed-{seed}'
            # Retain full 100 s segments for the primary model; cap potentially
            # pathological ablation activity at 10 s, with exact cumulative rates.
            window = 1000000 if condition in ['full', 'no_stimulation'] else 100000
            jobs.append({'label': label, 'seed': seed, 'condition': condition,
                'command': [sys.executable, str(RUNNER), '--backend', 'rust',
                    '--network-scale', '1', '--seed', str(seed),
                    '--threads', str(args.threads[-1]), '--segment-seconds', '100',
                    '--bounded-window-steps', str(window), '--checkpoint',
                    *(['--keep-phase-checkpoints'] if condition == 'full' else []),
                    '--compact-artifacts', '--output', str(args.output/label),
                    *conditions[condition]]})
    write_json(args.output/'science_jobs.json', jobs)
    results = []
    for job in jobs:
        process = measured_process(job['command'], args.output/(job['label']+'-measurement'))
        report = json.loads((args.output/job['label']/'result.json').read_text())
        if not report['complete_training_protocol']:
            raise AssertionError('scientific job ended before training completed')
        results.append({'label': job['label'], 'seed': job['seed'],
            'condition': job['condition'], 'run': report, 'process': process})
        write_json(args.output/'science_results.json', results)
    return {'experiments': results,
            'replicate': 'independent network/input seed; assemblies and edges are not independent replicates',
            'protocol': '10 s warmup + 1600 s training + 1000 s spontaneous, learning active',
            'recording': 'primary/no-stimulation retain 100 s; potentially pathological ablations retain 10 s; cumulative counts are complete'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['correctness', 'performance', 'whole_run', 'profiling', 'memory', 'convergence', 'science'], default='correctness')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--network-scale', type=float, default=1.)
    p.add_argument('--duration-s', type=float, default=1.)
    p.add_argument('--warmup-s', type=float, default=10.)
    p.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--cpp-threads', type=int, nargs='+',
                   help='optional C++ thread grid for confirmation after a separate full scaling scan')
    p.add_argument('--cpp-profile', choices=['strict', 'native-strict', 'brian2-default'], default='strict')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--horizons-s', type=float, nargs='+', default=[1., 10., 100., 1000.])
    p.add_argument('--seeds', type=int, nargs='+', default=[20260906, 20260907, 20260908])
    p.add_argument('--conditions', nargs='+', choices=['full', 'no_stimulation', 'no_istdp', 'no_normalization'], default=['full', 'no_stimulation', 'no_istdp', 'no_normalization'])
    p.add_argument('--window-steps', type=int, default=10000)
    args = p.parse_args()
    if args.repeats < 2:
        p.error('at least two independent warm process executions are required')
    if args.cpp_profile != 'strict' and args.suite in ['science', 'convergence']:
        p.error('cpp-profile does not apply to a Rust-only suite')
    if 1 not in args.threads or len(set(args.threads)) != len(args.threads) or min(args.threads) < 1:
        p.error('thread counts must be unique, positive, and include one worker')
    if args.cpp_threads is not None and (len(set(args.cpp_threads)) != len(args.cpp_threads)
                                        or min(args.cpp_threads) < 1):
        p.error('C++ thread counts must be unique and positive')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'environment.json', environment())
    report = globals()[args.suite](args)
    write_json(args.output/'report.json', {'suite': args.suite, **report})


if __name__ == '__main__':
    main()
