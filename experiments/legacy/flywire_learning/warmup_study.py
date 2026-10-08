"""Fixed, event-matched warmup intervention around the unchanged CPU v1 model."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import experiment
from study import compare

HERE = Path(__file__).resolve().parent
SEEDS = [11, 23, 47, 83, 131]
ARMS = [100, 10100]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run_case(args):
    base = args.baseline / f'seed-{args.seed}-frozen'
    cfg = json.loads((base / 'configuration.json').read_text())
    assert cfg['scope'] == 'full' and cfg['duration_ms'] == 19100
    assert cfg['experiment_sha256'] == sha(HERE / 'experiment.py')
    assert cfg['rule_sha256'] == sha(HERE / 'spiking.py')
    assert cfg['prepared_manifest_sha256'] == sha(args.prepared / 'manifest.json')
    with np.load(base / 'input.npz') as saved:
        original_inputs = {k: saved[k] for k in saved.files}
    digest = hashlib.sha256()
    for key, value in sorted(original_inputs.items()):
        digest.update(key.encode() + np.asarray(value).tobytes())
    assert digest.hexdigest() == cfg['input_sha256'], 'baseline input identity mismatch'
    delta = args.warmup_ms - 100
    offset = delta * 10
    old_schedule, old_poisson = experiment.schedule, experiment.poisson_schedule

    def schedule(seed, condition, protocol):
        assert seed == args.seed and condition == 'frozen' and protocol == 'formal'
        trials, stages, _, _, duration = old_schedule(seed, condition, protocol)
        trials = [{**t, 'start_ms': t['start_ms'] + delta, 'training': False, 'reward': 0}
                  for t in trials]
        stages = [{**s, 'end_ms': s['end_ms'] + delta, 'training': False} for s in stages]
        ticks = int(round((duration + delta) * 10))
        return trials, stages, np.zeros(ticks), np.zeros(ticks), duration + delta

    def poisson(n, rate, dt, duration, seed, start=0., end=None):
        if seed == args.seed + 30000:
            assert n == 512 and rate == 300 and dt == .1 and start == 0 and end is None
            if offset:
                prefix_i, prefix_t = old_poisson(n, rate, dt, delta, args.seed + 90000)
            else:
                prefix_i = prefix_t = np.array([], dtype=np.int32)
            return (np.r_[prefix_i, original_inputs['background_ids']],
                    np.r_[prefix_t, original_inputs['background_ticks'] + offset])
        return old_poisson(n, rate, dt, duration, seed, start=start, end=end)

    experiment.schedule, experiment.poisson_schedule = schedule, poisson
    try:
        experiment.run(SimpleNamespace(prepared=args.prepared, output=args.output,
                       backend='aot', scope='full', seed=args.seed, condition='frozen',
                       protocol='formal', amplitude=40., threads=4, unsplit=False,
                       resume=None, single_run=True, save_after_acquisition=False))
    finally:
        experiment.schedule, experiment.poisson_schedule = old_schedule, old_poisson
    with np.load(args.output / 'input.npz') as inputs:
        for name in ['mapping', 'stimulus_ids', 'stimulus_targets']:
            np.testing.assert_array_equal(inputs[name], original_inputs[name])
        np.testing.assert_array_equal(inputs['stimulus_ticks'] - offset, original_inputs['stimulus_ticks'])
        matched = inputs['background_ticks'] >= offset
        np.testing.assert_array_equal(inputs['background_ticks'][matched] - offset, original_inputs['background_ticks'])
        np.testing.assert_array_equal(inputs['background_ids'][matched], original_inputs['background_ids'])
        assert not inputs['teaching'].any() and not inputs['learning'].any()
    with np.load(args.output / 'snapshot.npz') as state:
        np.testing.assert_array_equal(state['gain'], np.ones(2560))
        assert len(state['v']) == 139255
    save(args.output / 'warmup_identity.json', dict(warmup_ms=args.warmup_ms, shift_ticks=offset,
         matched_inputs_exact=True, all_gates_zero=True, all_gains_one=True,
         baseline_input_sha256=sha(base / 'input.npz'),
         baseline_configuration_sha256=sha(base / 'configuration.json'),
         wrapper_sha256=sha(__file__), protocol_sha256=sha(HERE / 'WARMUP_PROTOCOL.md'),
         initial_state='Unchanged CPU v1 seed initialization; extra background precedes the matched segment.'))


def main(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    identity = {str(p): sha(p) for p in [Path(__file__), HERE / 'experiment.py', HERE / 'spiking.py',
                HERE / 'WARMUP_PROTOCOL.md', args.prepared / 'manifest.json',
                args.baseline / 'study_identity.json',
                experiment.ROOT / 'python/brian2_rust/device.py',
                experiment.ROOT / 'python/brian2_rust/export.py',
                experiment.ROOT / 'target/release/b2-runner']}
    identity_path = out / 'study_identity.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity, 'changed study identity'
    else:
        save(identity_path, identity)
    progress_path = out / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    for seed in SEEDS:
        for warmup in ARMS:
            name = f'seed-{seed}-warmup-{warmup}'
            folder = out / name
            entry = progress.get(name, {})
            if entry.get('status') == 'complete':
                for file in ['snapshot.npz', 'report.json', 'warmup_identity.json']:
                    assert (folder / file).exists(), f'missing completed {name}/{file}'
            else:
                if folder.exists() or entry:
                    raise RuntimeError(f'Inspect previous process and artifacts before retrying {name}')
                cmd = [sys.executable, str(Path(__file__).resolve()), '--case', '--seed', str(seed),
                       '--warmup-ms', str(warmup), '--prepared', str(args.prepared),
                       '--baseline', str(args.baseline), '--output', str(folder)]
                started = time.time()
                print(f'START {name}', flush=True)
                with (out / f'{name}.log').open('w') as log:
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                    progress[name] = dict(status='running', pid=proc.pid, command=cmd, started=started)
                    save(progress_path, progress)
                    code = proc.wait()
                progress[name].update(status='complete' if code == 0 else 'failed', exit_code=code,
                                      wall_seconds=time.time() - started)
                save(progress_path, progress)
                if code:
                    raise RuntimeError(f'{name} failed; inspect its log')
                print(f'DONE {name} {time.time()-started:.1f}s', flush=True)
            if seed == 11 and warmup == 100:
                errors = compare(folder, args.baseline / 'seed-11-frozen', exact=True)
                save(out / 'short_reference_equivalence.json', errors)
                print('GATE: short seed 11 matches original frozen snapshot exactly', flush=True)
    print('WARMUP RUNS COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case', action='store_true')
    p.add_argument('--seed', type=int, choices=SEEDS)
    p.add_argument('--warmup-ms', type=int, choices=ARMS)
    a = p.parse_args()
    a.prepared, a.baseline, a.output = a.prepared.resolve(), a.baseline.resolve(), a.output.resolve()
    if a.case:
        if a.seed is None or a.warmup_ms is None:
            p.error('--case requires seed and warmup-ms')
        run_case(a)
    else:
        main(a)
