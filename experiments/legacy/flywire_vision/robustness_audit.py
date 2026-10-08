"""Frozen P4 decoder: new grouped distribution shifts and temporal controls.

No weights or hyperparameters are selected in this audit. Static/scrambled
controls measure recovery of the source clip's label, not valid motion accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .stimuli import movie, MovieConfig
from .simulation import SimulationConfig
from .refinement import encode_variant, linear_scores
from .pilot import DIRECTIONS, event_features, scores
from .direction_study import balanced_groups, evaluate, polarity
from .run_experiment import save

CONDITIONS = {
    'reference': 'Original distribution; new trajectories',
    'slow': 'Travel 0.4 instead of 0.8; speed and path length both halved',
    'fast': 'Travel 1.0 instead of 0.8; speed and path length both x1.25',
    'position': 'Center on radius 0.30 ring instead of central +/-0.15 square',
    'texture': 'Shared static smooth visual background, amplitude <=0.10',
    'gray_low': 'Visual background 0.40 instead of 0.50; object contrast unchanged',
    'gray_high': 'Visual background 0.60 instead of 0.50; object contrast unchanged',
    'static_first': 'Repeat first frame: source-label recovery, no actual motion',
    'static_last': 'Repeat last frame: source-label recovery, no actual motion',
    'scrambled_middle': 'Keep endpoints; permute 38 middle frames with a shared group permutation',
}


def audit_movie(kind, seed, condition):
    if condition not in CONDITIONS:
        raise ValueError('unknown audit condition')
    if condition == 'position' and kind in ('left', 'down'):
        return audit_movie({'left': 'right', 'down': 'up'}[kind], seed, condition)[::-1].copy()
    config = MovieConfig(travel={'slow': .4, 'fast': 1.0}.get(condition, .8))
    original = movie(kind, seed, config)
    if condition in ('reference', 'slow', 'fast'):
        return original
    if condition == 'static_first':
        return np.repeat(original[:1], 40, axis=0)
    if condition == 'static_last':
        return np.repeat(original[-1:], 40, axis=0)
    if condition == 'scrambled_middle':
        permutation = np.random.default_rng(np.random.SeedSequence([seed, 926])).permutation(np.arange(1, 39))
        return original[np.r_[0, permutation, 39]].copy()
    if condition in ('gray_low', 'gray_high'):
        # +/-0.1 plus the existing <=0.4 object excursion remains in [0,1].
        return (original + (-.1 if condition == 'gray_low' else .1)).clip(0, 1).astype(np.float32)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 927]))
    if condition == 'texture':
        axis = np.linspace(-1, 1, 48)
        x, y = np.meshgrid(axis, axis)
        phase = rng.uniform(-np.pi, np.pi, 2)
        texture = .05 * (np.sin(2*np.pi*x + phase[0]) + np.sin(2*np.pi*y + phase[1]))
        return (original + texture[None]).clip(0, 1).astype(np.float32)
    # Reconstruct only the center parameter, preserving all other nuisance draws.
    original_rng = np.random.default_rng(seed)
    original_rng.uniform(-.15, .15, 2)
    width = original_rng.uniform(.10, .16)
    sign = original_rng.choice([-1, 1])
    angle = rng.uniform(-np.pi, np.pi)
    cx, cy = .30*np.cos(angle), .30*np.sin(angle)
    axis = np.linspace(-1, 1, 48)
    x, y = np.meshgrid(axis, axis)
    t = np.linspace(-.5, .5, 40)
    dx, dy = {'right': (1, 0), 'left': (-1, 0), 'up': (0, -1), 'down': (0, 1)}[kind]
    px, py = cx + .8*t*dx, cy + .8*t*dy
    field = np.exp(-((x[None]-px[:, None, None])**2 + (y[None]-py[:, None, None])**2)/(2*width**2))
    return (.5 + sign*.4*field).astype(np.float32)


def centroid_rule(frames, last=False):
    """Fixed shortcut rule, no fitting: dominant endpoint displacement from origin."""
    frame = frames[-1 if last else 0]
    mass = np.abs(frame.astype(float) - .5)
    axis = np.linspace(-1, 1, frame.shape[0])
    x = float((mass.sum(0)*axis).sum()/max(mass.sum(), 1e-15))
    y = float((mass.sum(1)*axis).sum()/max(mass.sum(), 1e-15))
    if last:
        x, y = -x, -y
    return (0 if x < 0 else 1) if abs(x) >= abs(y) else (2 if y > 0 else 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from flywire_mnist.backends.frozen_cpu import FrozenCPU
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    parent = read(args.study/'protocol.json')
    report = read(args.study/'report.json')
    cfg = SimulationConfig(**parent['config'])
    if (cfg.frames, cfg.warmup_ms, cfg.tail_ms, cfg.dt_ms, cfg.frame_ms) != (40, 100., 100., .1, 10.):
        raise ValueError('audit requires original feature timing')
    channels = read(args.artifact/'channels.json')
    cells = np.asarray(parent['readout_indices'])
    inputs = np.asarray(parent['input_indices'])
    models = {}
    for name in ('neural', 'input_neurons', 'encoded_input'):
        path = args.study/(name+'-readout.npz')
        if sha(path) != report['readouts'][name]['readout_sha256']:
            raise ValueError('decoder artifact changed')
        with np.load(path, allow_pickle=False) as m:
            models[name] = dict(m)
    protocol = {'schema': 'flywire-robustness-audit-v1', 'parent_study': str(args.study),
                'parent_report_sha256': sha(args.study/'report.json'),
                'groups': balanced_groups(80000, 8), 'conditions': CONDITIONS,
                'clips_per_condition': 32, 'directions': DIRECTIONS, 'config': parent['config'],
                'base_sha256': parent['base_sha256'], 'binary_sha256': parent['binary_sha256'],
                'input_mode': parent['input_mode'], 'readout_indices': cells.tolist(), 'input_indices': inputs.tolist(),
                'readouts_sha256': {n: report['readouts'][n]['readout_sha256'] for n in models},
                'selection': 'All P4 readouts frozen; no training, model selection or test-based promotion',
                'scope': 'Exploratory eight-group paired audit, not a reliable population estimate; neural background remains fixed',
                'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in ('robustness_audit.py', 'stimuli.py', 'simulation.py', 'pilot.py', 'refinement.py')}}
    save(args.output/'protocol.json', protocol)
    locked = sha(args.output/'protocol.json')
    print(json.dumps({'stage': 'audit_locked_before_execution', 'protocol_sha256': locked, 'trials': 320}), flush=True)
    template = read(args.artifact/'model.json')
    runner = FrozenCPU(template, args.artifact/'compile/native', args.output/'execution', population='visual_input', threads=4)
    if runner.base_hash != parent['base_sha256'] or runner.binary_hash != parent['binary_sha256']:
        raise ValueError('neural model mismatch')
    ni = next(i for i, p in enumerate(template['definition']['populations']) if p['name'] == 'flywire_neurons')
    neurons = template['definition']['populations'][ni]['count']
    features = {n: [] for n in models}
    rows, labels, results = [], [], {}
    started = time.perf_counter()
    for condition in CONDITIONS:
        initial = len(rows)
        shortcuts = {'first_frame_rule': [], 'last_frame_rule': []}
        for group in protocol['groups']:
            for label, kind in enumerate(DIRECTIONS):
                frames = audit_movie(kind, group, condition)
                i, t, _ = encode_variant(frames, channels, cfg, protocol['input_mode'])
                key = f'{condition}-g{group}-{kind}'
                result = runner.run(i, t, key)
                pop = result['populations'][ni]
                neural = event_features(pop['indices'], pop['spike_ticks'], cells, neurons)
                values = {'neural': neural, 'input_neurons': event_features(pop['indices'], pop['spike_ticks'], inputs, neurons),
                          'encoded_input': event_features(i, t, np.arange(len(channels)), len(channels))}
                for name in models:
                    features[name].append(values[name])
                for name, last in (('first_frame_rule', False), ('last_frame_rule', True)):
                    shortcuts[name].append(centroid_rule(frames, last))
                rows.append({'group': group, 'kind': kind, 'condition': condition, 'polarity': polarity(group),
                             'input_sha256': hashlib.sha256(np.stack([t, i], 1).astype('<i8').tobytes()).hexdigest(),
                             'events_sha256': hashlib.sha256(np.stack([pop['spike_ticks'], pop['indices']], 1).astype('<i8').tobytes()).hexdigest(),
                             'movie_sha256': hashlib.sha256(frames.tobytes()).hexdigest(),
                             'first_rule': shortcuts['first_frame_rule'][-1], 'last_rule': shortcuts['last_frame_rule'][-1],
                             'external_spikes': len(i), 'seconds': result['wall_seconds']})
                labels.append(label)
                del pop, result
                shutil.rmtree(runner.directory/key)
                (runner.directory/(key+'.spikes')).unlink()
            print(json.dumps({'condition': condition, 'completed': len(rows), 'total': 320, 'seconds': time.perf_counter()-started}), flush=True)
        metrics = {}
        for name, model in models.items():
            x = np.asarray(features[name][initial:])
            pred = (linear_scores(model, x) if 'coefficients' in model else scores(model, x)).argmax(1)
            metrics[name] = evaluate(pred, labels[initial:], rows[initial:])
        for name, pred in shortcuts.items():
            metrics[name] = evaluate(pred, labels[initial:], rows[initial:])
        results[condition] = {'description': CONDITIONS[condition], 'source_label_control': condition.startswith('static') or condition == 'scrambled_middle', 'readouts': metrics}
        save(args.output/'rows.json', rows)
        save(args.output/'partial-results.json', results)
        np.savez_compressed(args.output/'features.npz', **{n: np.asarray(v) for n, v in features.items()}, labels=labels)
    if sha(args.output/'protocol.json') != locked:
        raise ValueError('audit protocol changed')
    for name, h in protocol['readouts_sha256'].items():
        if sha(args.study/(name+'-readout.npz')) != h:
            raise ValueError('frozen readout changed')
    final = {'schema': protocol['schema'], 'status': 'complete', 'protocol_sha256': locked,
             'features_sha256': sha(args.output/'features.npz'), 'conditions': results,
             'groups_per_condition': 8, 'clips_per_condition': 32, 'new_simulations': len(rows),
             'seconds': time.perf_counter()-started,
             'limits': ['Exploratory small sample; paired direction and condition variants share trajectory groups',
                        'No refitting or selection; all conditions reported; static/scrambled metrics are source-label recovery',
                        'Visual background changes only; fixed neural background seed; altered travel also changes path length']}
    save(args.output/'report.json', final)
    print(json.dumps({'status': 'complete', 'downstream': {c: r['readouts']['neural']['accuracy'] for c, r in results.items()}}), flush=True)


if __name__ == '__main__':
    main()
