"""Fresh-process recovery acceptance for the complete LK model, not phase clocks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
import brian2_rust  # noqa: E402,F401
from litwin_kumar_model import LKConfig, make_network  # noqa: E402


def child(args):
    output = args.output.resolve()
    config = LKConfig(scale=args.network_scale, seed=args.seed, warmup_s=.02)
    if args.backend == 'rust':
        b.set_device('rust_standalone', runner=ROOT/'target/release/b2-runner',
                     directory=output/args.phase, engine='aot', threads=args.threads,
                     recording_window_steps=None if args.full_history else args.window_steps)
    else:
        b.set_device('runtime')
        b.prefs.codegen.target = 'numpy'
    model = make_network(config)
    checkpoint = output/'midpoint.pkl'
    started = time.perf_counter()
    if args.phase == 'uninterrupted':
        model.network.run(2*args.segment_s*b.second)
    elif args.phase == 'save':
        model.network.run(args.segment_s*b.second)
        model.network.store('midpoint', filename=checkpoint)
    elif args.phase == 'restore':
        model.network.restore('midpoint', filename=checkpoint, restore_random_state=True)
        model.network.run(args.segment_s*b.second)
    np.savez(output/f'{args.phase}.npz', **model.snapshot())
    (output/f'{args.phase}.json').write_text(json.dumps({
        'pid': os.getpid(), 'time_s': float(model.network.t/b.second),
        'wall_seconds': time.perf_counter()-started,
        'configuration': config.to_dict(),
        'backend': args.backend,
    }, indent=2)+'\n')
    if args.compact_artifacts and (output/args.phase).exists():
        shutil.rmtree(output/args.phase)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=['rust', 'numpy'], default='rust')
    p.add_argument('--phase', choices=['uninterrupted', 'save', 'restore'])
    p.add_argument('--network-scale', type=float, default=.02)
    p.add_argument('--segment-s', type=float, default=.2)
    p.add_argument('--window-steps', type=int, default=1000)
    p.add_argument('--full-history', action='store_true',
                   help='retain complete monitors on both backends; required for NumPy')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=20260906)
    p.add_argument('--compact-artifacts', action='store_true')
    args = p.parse_args()
    if args.backend == 'numpy' and (not args.full_history or args.threads != 1):
        p.error('NumPy recovery requires --full-history --threads 1')
    if args.phase:
        child(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    for phase in ['uninterrupted', 'save', 'restore']:
        command = [sys.executable, __file__, '--phase', phase,
                   '--backend', args.backend,
                   '--output', str(args.output), '--network-scale', str(args.network_scale),
                   '--segment-s', str(args.segment_s), '--window-steps', str(args.window_steps),
                   '--threads', str(args.threads), '--seed', str(args.seed)]
        if args.compact_artifacts:
            command += ['--compact-artifacts']
        if args.full_history:
            command += ['--full-history']
        with (args.output/f'{phase}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    with np.load(args.output/'uninterrupted.npz') as baseline, np.load(args.output/'restore.npz') as replay:
        if set(baseline.files) != set(replay.files):
            raise AssertionError('recovered fields differ')
        for name in baseline.files:
            # Byte comparison includes signed zero and dtype, not just values.
            if (baseline[name].dtype != replay[name].dtype or
                    baseline[name].shape != replay[name].shape or
                    baseline[name].tobytes() != replay[name].tobytes()):
                raise AssertionError(f'checkpoint replay differs in {name}')
        fields = baseline.files
    phases = {phase:json.loads((args.output/f'{phase}.json').read_text())
              for phase in ['uninterrupted', 'save', 'restore']}
    if len({phase['pid'] for phase in phases.values()}) != 3:
        raise AssertionError('three fresh process identities required')
    for phase, values in phases.items():
        expected_time = args.segment_s*(1 if phase == 'save' else 2)
        if not np.isclose(values['time_s'], expected_time, rtol=0, atol=1e-12):
            raise AssertionError(f'checkpoint phase clock differs: {phase}')
    report = {'backend': args.backend, 'fresh_process_replay_exact': True, 'fields': fields,
              'recording_window_steps': None if args.full_history else args.window_steps,
              'comparison_scope': 'within-backend uninterrupted versus fresh-process restore; backend RNG streams differ',
              'checkpoint_bytes': (args.output/'midpoint.pkl').stat().st_size,
              'phases': phases, 'cpp_store_restore': 'unsupported'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
