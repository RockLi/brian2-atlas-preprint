"""Run the same LK2014 triplet-variant model on Rust, C++ or NumPy."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
import brian2_rust  # noqa: F401,E402
from litwin_kumar_model import LKConfig, make_network  # noqa: E402


def weight_metrics(model):
    weights = np.asarray(model.synapses['ee'].w[:])
    source, target = model.edges['ee']
    within = np.zeros(len(weights), dtype=bool)
    per_assembly = []
    for members in model.membership['exc']:
        selected = members[source] & members[target]
        within |= selected
        per_assembly.append(float(weights[selected].mean()) if selected.any() else None)
    sums = np.bincount(target, weights=weights, minlength=model.config.ne)
    target_sum = np.asarray(model.exc.weight_target[:])
    inhibitory = np.asarray(model.synapses['ie'].w[:])
    return {
        'within_mean_pf': float(weights[within].mean()) if within.any() else None,
        'between_mean_pf': float(weights[~within].mean()) if (~within).any() else None,
        'assembly_means_pf': per_assembly,
        'max_row_sum_error_pf': float(np.max(np.abs(sums-target_sum))),
        'mean_absolute_row_sum_error_pf': float(np.mean(np.abs(sums-target_sum))),
        'lower_bound_fraction': float(np.mean(weights <= 1.78+1e-12)),
        'upper_bound_fraction': float(np.mean(weights >= 21.4-1e-12)),
        'finite': bool(np.all(np.isfinite(weights))),
        'inhibitory_mean_pf': float(inhibitory.mean()),
        'inhibitory_lower_fraction': float(np.mean(inhibitory <= 48.7+1e-12)),
        'inhibitory_upper_fraction': float(np.mean(inhibitory >= 243-1e-12)),
    }


def delivered_events_from_history(model, duration_s, start_s=0.):
    """Independent event-work count for full-history, zero-origin executions."""
    end_tick = round(duration_s*1000/model.config.dt_ms)
    start_tick = round(start_s*1000/model.config.dt_ms)
    delay_ticks = round(model.config.delay_ms/model.config.dt_ms)
    counts, eligible = {}, {}
    for label, size in [('exc', model.config.ne), ('inh', model.config.ni)]:
        monitor = model.spikes[label]
        ids = np.asarray(monitor.i[:])
        ticks = np.rint(np.asarray(monitor.t[:]/b.second)*1000/model.config.dt_ms).astype(np.int64)
        counts[label] = np.bincount(ids[ticks >= start_tick], minlength=size)
        eligible[label] = np.bincount(ids[(ticks+delay_ticks < end_tick) &
                                         (ticks+delay_ticks >= start_tick)], minlength=size)
    total = 0
    for label, (source, target) in model.edges.items():
        pre = 'exc' if label[0] == 'e' else 'inh'
        post = 'exc' if label[1] == 'e' else 'inh'
        total += int(np.dot(np.bincount(source, minlength=len(eligible[pre])), eligible[pre]))
        if label in ['ee', 'ie']:
            total += int(np.dot(np.bincount(target, minlength=len(counts[post])), counts[post]))
    return total


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['rust', 'cpp', 'numpy'], default='rust')
    p.add_argument('--cpp-profile', choices=['strict', 'native-strict', 'brian2-default'],
                   default='strict', help='C++ arithmetic/CPU flags; defaults are measured separately from strict numerical reference')
    p.add_argument('--engine', choices=['aot', 'reference'], default='aot')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--split-warmup', action='store_true',
                   help='time the post-warmup interval separately in C++ and Rust')
    p.add_argument('--network-scale', type=float, default=1.)
    p.add_argument('--duration-ms', type=float)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--mode', choices=['learn', 'clustered'], default='learn')
    p.add_argument('--seed', type=int, default=20260906)
    p.add_argument('--warmup-s', type=float, default=10.)
    p.add_argument('--train-repetitions', type=int, default=20)
    p.add_argument('--stimulus-s', type=float, default=1.)
    p.add_argument('--gap-s', type=float, default=3.)
    p.add_argument('--spontaneous-s', type=float, default=1000.)
    p.add_argument('--bounded-window-steps', type=int)
    p.add_argument('--input-mode', choices=['poisson', 'replay'], default='poisson')
    p.add_argument('--replay-base-dt-ms', type=float)
    p.add_argument('--disable-plasticity', action='store_true')
    p.add_argument('--disable-istdp', action='store_true')
    p.add_argument('--disable-normalization', action='store_true')
    p.add_argument('--disable-stimulation', action='store_true')
    p.add_argument('--learning-multiplier', type=float, default=1.)
    p.add_argument('--inhibitory-stimulus-factor', type=float, default=0.)
    p.add_argument('--delay-ms', type=float, default=1.5)
    p.add_argument('--delay-distribution', choices=['fixed', 'uniform'], default='fixed')
    p.add_argument('--trace-integration', choices=['event-driven', 'euler', 'euler-event'], default='event-driven')
    p.add_argument('--input-sources', type=int, default=1000)
    p.add_argument('--dt-ms', type=float, default=.1)
    p.add_argument('--segment-seconds', type=float, default=100.)
    p.add_argument('--disable-phase-segmentation', action='store_true',
                   help='use only requested segment boundaries for compute comparisons')
    p.add_argument('--checkpoint', action='store_true')
    p.add_argument('--keep-phase-checkpoints', action='store_true',
                   help='retain immutable warmup and training endpoint checkpoints')
    p.add_argument('--restore', type=Path)
    p.add_argument('--compact-artifacts', action='store_true',
                   help='remove completed Rust build payloads after recording segment data')
    p.add_argument('--output', type=Path, required=True)
    return p


def run(args):
    if args.threads < 1:
        raise ValueError('threads must be positive')
    if args.keep_phase_checkpoints and (not args.checkpoint or args.disable_phase_segmentation):
        raise ValueError('phase checkpoint retention requires checkpoint and phase segmentation')
    if args.bounded_window_steps is not None and args.backend != 'rust':
        raise ValueError('rolling monitor is a Rust capability; compare recording semantics explicitly')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = LKConfig(scale=args.network_scale, seed=args.seed, dt_ms=args.dt_ms,
        mode=args.mode, warmup_s=args.warmup_s, stimulus_s=args.stimulus_s,
        gap_s=args.gap_s, repetitions=args.train_repetitions, spontaneous_s=args.spontaneous_s,
        plasticity=not args.disable_plasticity,
        inhibitory_plasticity=not args.disable_istdp,
        normalization=not args.disable_normalization,
        stimulation=not args.disable_stimulation,
        learning_multiplier=args.learning_multiplier,
        inhibitory_stimulus_factor=args.inhibitory_stimulus_factor,
        delay_ms=args.delay_ms, delay_distribution=args.delay_distribution,
        trace_integration=args.trace_integration, input_sources=args.input_sources)
    duration_s = config.duration_s if args.duration_ms is None else args.duration_ms/1000
    if duration_s <= 0:
        raise ValueError('duration must be positive')
    if args.split_warmup and (not 0 < config.warmup_s < duration_s or
                             args.disable_phase_segmentation):
        raise ValueError('split warmup requires an internal warmup boundary and phase segmentation')
    if args.split_warmup and (args.bounded_window_steps is not None or args.restore):
        raise ValueError('timed interval work accounting requires full history from zero')
    for dt_s in [config.dt_ms/1000, config.normalization_ms/1000]:
        if not np.isclose(duration_s/dt_s, round(duration_s/dt_s)):
            raise ValueError('duration must be divisible by neuron and normalization clock periods')
    if (args.checkpoint or args.restore) and args.backend != 'rust':
        raise ValueError('this runner exposes checkpoint continuation only on Rust')
    if args.replay_base_dt_ms is not None and args.input_mode != 'replay':
        raise ValueError('replay-base-dt-ms requires input-mode replay')
    if args.segment_seconds <= 0:
        raise ValueError('segment-seconds must be positive')
    cpp_options = None
    if args.backend == 'rust':
        if args.profile:
            os.environ['B2_AOT_PROFILE_PHASES'] = '1'
        for boundary in [args.segment_seconds, config.warmup_s,
                         config.warmup_s+config.training_s]:
            ticks = boundary/(config.normalization_ms/1000)
            if not np.isclose(ticks, round(ticks), rtol=0, atol=1e-8):
                raise ValueError('segment and phase boundaries must align with normalization clock')
    (output/'configuration.json').write_text(json.dumps(config.to_dict(), indent=2)+'\n')
    input_configuration = {'mode': args.input_mode, 'base_dt_ms': args.replay_base_dt_ms}
    (output/'input_configuration.json').write_text(json.dumps(input_configuration, indent=2)+'\n')
    started = time.perf_counter()
    if args.backend == 'rust':
        b.set_device('rust_standalone', runner=ROOT/'target/release/b2-runner',
            directory=output/'project', engine=args.engine, threads=args.threads,
            recording_window_steps=args.bounded_window_steps)
    elif args.backend == 'cpp':
        b.prefs.devices.cpp_standalone.openmp_threads = 0 if args.threads == 1 else args.threads
        if args.cpp_profile == 'brian2-default':
            b.prefs.codegen.cpp.extra_compile_args = None
        else:
            flags = ['-O3', '-std=c++17', '-fno-fast-math', '-ffp-contract=off']
            if args.cpp_profile == 'native-strict':
                flags += ['-mcpu=native' if platform.machine() in ['arm64', 'aarch64']
                          else '-march=native']
            b.prefs.codegen.cpp.extra_compile_args = flags
        from brian2.codegen.cpp_prefs import get_compiler_and_args
        compiler, flags = get_compiler_and_args()
        cpp_options = {'profile': args.cpp_profile, 'compiler': compiler, 'effective_flags': flags,
                       'numerical_reference': args.cpp_profile != 'brian2-default'}
        b.prefs.devices.cpp_standalone.extra_make_args_unix = ['-j2']
        b.set_device('cpp_standalone', build_on_run=False)
    else:
        b.set_device('runtime')
        b.prefs.codegen.target = 'numpy'
    model = make_network(config, replay_duration_s=duration_s if args.input_mode == 'replay' else None,
                         replay_base_dt_ms=args.replay_base_dt_ms)
    constructed = time.perf_counter()
    trajectory, segments = [], []
    if args.restore:
        saved_config = json.loads((args.restore.parent/'configuration.json').read_text())
        saved_config = {'delay_distribution': 'fixed', 'trace_integration': 'event-driven',
                        **saved_config}
        if saved_config != config.to_dict():
            raise ValueError('checkpoint configuration differs from the requested model')
        if (args.restore.parent/'input_configuration.json').exists():
            saved_input = json.loads((args.restore.parent/'input_configuration.json').read_text())
        else:
            old_report = json.loads((args.restore.parent/'result.json').read_text())
            saved_input = {'mode': old_report['input_mode'], 'base_dt_ms': old_report.get('replay_base_dt_ms')}
        if saved_input != input_configuration:
            raise ValueError('checkpoint input protocol differs from the requested model')
        model.network.restore('latest', filename=args.restore, restore_random_state=True)
    initial_time_s = float(model.network.t/b.second)
    if args.restore and (args.restore.parent/'progress.json').exists():
        history = json.loads((args.restore.parent/'progress.json').read_text())
        prefix = [i for i, segment in enumerate(history['segments'])
                  if segment['stop_s'] <= initial_time_s+1e-10]
        if not prefix or not np.isclose(history['segments'][prefix[-1]]['stop_s'], initial_time_s,
                                       rtol=0, atol=1e-10):
            raise ValueError('checkpoint time does not match its recorded segment history')
        segments = [history['segments'][i] for i in prefix]
        trajectory = [history['trajectory'][i] for i in prefix]
        for index in prefix:
            shutil.copytree(args.restore.parent/'segments'/f'{index+1:04d}',
                            output/'segments'/f'{index+1:04d}')
    restored_segments = len(segments)
    if args.backend == 'rust':
        boundaries = {duration_s}
        if not args.disable_phase_segmentation:
            boundaries.update([config.warmup_s, config.warmup_s+config.training_s])
        boundaries.update(np.arange(args.segment_seconds, duration_s, args.segment_seconds))
        for stop in sorted(float(t) for t in boundaries if initial_time_s < t <= duration_s):
            current = float(model.network.t/b.second)
            segment_started = time.perf_counter()
            model.network.run((stop-current)*b.second, namespace={}, profile=args.profile)
            summary = json.loads((b.get_device().last_run_directory/'rust/summary.json').read_text())
            segments.append({'start_s': current, 'stop_s': stop, **summary['timings'],
                'host_run_wall_seconds': time.perf_counter()-segment_started,
                'build_timings': dict(b.get_device().last_build_timings)})
            trajectory.append({'time_s': stop, 'weights': weight_metrics(model),
                               'spike_totals': {name: int(np.sum(group.spike_total[:]))
                                for name, group in [('exc', model.exc), ('inh', model.inh)]}})
            segment_dir = output/'segments'/f'{len(segments):04d}'
            segment_dir.mkdir(parents=True)
            (segment_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            np.savez_compressed(segment_dir/'spikes.npz',
                **{f'{label}_{field}': np.asarray(getattr(monitor, field)[:]/b.second
                    if field == 't' else getattr(monitor, field)[:])
                   for label, monitor in model.spikes.items() for field in ['i', 't']})
            (output/'progress.json').write_text(json.dumps({
                'segments': segments, 'trajectory': trajectory}, indent=2)+'\n')
            if args.checkpoint:
                model.network.store('latest', filename=output/'checkpoint.pkl')
                if args.keep_phase_checkpoints:
                    for label, boundary in [('warmup', config.warmup_s),
                                            ('training', config.warmup_s+config.training_s)]:
                        if np.isclose(stop, boundary, rtol=0, atol=1e-10):
                            # store publishes by atomic replace. A hard link retains
                            # this immutable inode through subsequent checkpoints.
                            os.link(output/'checkpoint.pkl', output/f'checkpoint-{label}.pkl')
            if args.compact_artifacts:
                run_directory = b.get_device().last_run_directory
                (run_directory/'model.json').unlink()
                for subdirectory in ['native', 'rust']:
                    shutil.rmtree(run_directory/subdirectory)
            print(f'completed {stop:g}/{duration_s:g} biological seconds', flush=True)
        if len(segments) == restored_segments:
            raise ValueError('requested end time must be after checkpoint time')
    else:
        if args.split_warmup:
            model.network.run(config.warmup_s*b.second, namespace={}, profile=False)
            if args.backend == 'cpp':
                b.get_device().insert_code('main', '''
                {
                    std::ofstream _lk_warmup_time(brian::results_dir + "lk_warmup_seconds.txt");
                    _lk_warmup_time.precision(17);
                    _lk_warmup_time << Network::_last_run_time << std::endl;
                    if (!_lk_warmup_time.good()) return 2;
                }
                ''')
            model.network.run((duration_s-config.warmup_s)*b.second, namespace={}, profile=args.profile)
        else:
            model.network.run(duration_s*b.second, namespace={}, profile=args.profile)
    if args.backend == 'cpp':
        b.get_device().build(directory=str(output/'project'), run=True, with_output=False)
    finished = time.perf_counter()
    metadata = {}
    if args.backend == 'rust':
        metadata = summary
        simulation_s = sum(segment['simulation_and_recording_seconds'] for segment in segments)
    elif args.backend == 'cpp':
        cpp_final_interval_s = float((output/'project/results/last_run_info.txt').read_text().split()[0])
        cpp_warmup_s = (float((output/'project/results/lk_warmup_seconds.txt').read_text())
                        if args.split_warmup else 0.)
        simulation_s = cpp_warmup_s+cpp_final_interval_s
    else:
        simulation_s = finished-constructed
    report = {
        'model': 'LK2014 Fig.5 triplet variant', 'paper_reproduction': False,
        'backend': args.backend, 'engine': args.engine if args.backend == 'rust' else None,
        'cpp_compilation': cpp_options,
        'profiled': args.profile,
        'code_object_profile_seconds': ({name: float(seconds/b.second)
            for name, seconds in model.network.profiling_info} if args.profile else None),
        'threads': args.threads, 'input_mode': args.input_mode,
        'replay_base_dt_ms': args.replay_base_dt_ms,
        'configuration': config.to_dict(), 'biological_seconds': duration_s,
        'initial_time_seconds': initial_time_s, 'segments': segments, 'trajectory': trajectory,
        'restored_segment_count': restored_segments,
        'retained_phase_checkpoints': {label: {'time_s': boundary,
            'path': str(output/f'checkpoint-{label}.pkl')}
            for label, boundary in [('warmup', config.warmup_s),
                                   ('training', config.warmup_s+config.training_s)]
            if (output/f'checkpoint-{label}.pkl').exists()},
        'continuation_simulation_seconds': (sum(segment['simulation_and_recording_seconds']
                                               for segment in segments[restored_segments:])
                                            if args.backend == 'rust' else simulation_s),
        'complete_training_protocol': duration_s >= config.warmup_s+config.training_s,
        'topology_edges': {name: len(edges[0]) for name, edges in model.edges.items()},
        'recording_window_steps': args.bounded_window_steps,
        'construction_seconds': constructed-started,
        'last_run_directory': str(b.get_device().last_run_directory) if args.backend == 'rust' else None,
        'build_and_run_seconds': finished-constructed,
        'simulation_seconds': simulation_s,
        'simulation_seconds_scope': 'sum of all scheduled simulation intervals, including warmup and restored history when available',
        'build_timings': ({key: sum(segment.get('build_timings', {}).get(key, 0.)
                                  for segment in segments[restored_segments:])
                           for key in ['validation_seconds', 'generate_seconds', 'compile_seconds']}
                          if args.backend == 'rust' else
                          b.get_device().timers if args.backend == 'cpp' else None),
        'spike_totals': {label: int(np.sum(group.spike_total[:])) for label, group in [('exc',model.exc),('inh',model.inh)]},
        'retained_spikes': {name: len(monitor.i) for name, monitor in model.spikes.items()},
        'synaptic_events_from_full_history': (delivered_events_from_history(model, duration_s)
            if args.bounded_window_steps is None and initial_time_s == 0 else None),
        'weights': weight_metrics(model), 'native_metadata': metadata,
        'environment': {'python': sys.version, 'brian2': b.__version__,
                        'numpy': np.__version__, 'platform': platform.platform()},
    }
    for label, size in [('exc', config.ne), ('inh', config.ni)]:
        report.setdefault('rates_hz', {})[label] = report['spike_totals'][label]/size/duration_s
    if args.split_warmup:
        report['timed_interval'] = {
            'start_s': config.warmup_s, 'stop_s': duration_s,
            'duration_s': duration_s-config.warmup_s,
            'simulation_seconds': (sum(segment['simulation_and_recording_seconds']
                                       for segment in segments if segment['start_s'] >= config.warmup_s-1e-10)
                                   if args.backend == 'rust' else
                                   cpp_final_interval_s if args.backend == 'cpp' else None),
            'synaptic_events': delivered_events_from_history(model, duration_s, config.warmup_s),
            'rates_hz': {label: int(np.sum(np.rint(np.asarray(monitor.t[:]/b.second)*1000/config.dt_ms) >=
                                         round(config.warmup_s*1000/config.dt_ms))) /
                         (len(model.exc) if label == 'exc' else len(model.inh)) /
                         (duration_s-config.warmup_s) for label, monitor in model.spikes.items()}}
    snapshot = model.snapshot()
    if not all(np.all(np.isfinite(value)) for value in snapshot.values()):
        raise RuntimeError('non-finite model state; reject the run')
    np.savez(output/'state.npz', **snapshot)
    np.savez(output/'activity.npz',
             ee_i=model.edges['ee'][0], ee_j=model.edges['ee'][1],
             ee_initial=model.initial_weights['ee'], ee_final=snapshot['ee_w'],
             membership_e=model.membership['exc'], membership_i=model.membership['inh'],
             **{name:value for name,value in snapshot.items() if 'spike_' in name or name.startswith('voltage_')})
    report['output_seconds'] = time.perf_counter()-finished
    (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key:report[key] for key in ['backend','biological_seconds','simulation_seconds','rates_hz','weights']}, indent=2))
    return report


if __name__ == '__main__':
    run(parser().parse_args())
