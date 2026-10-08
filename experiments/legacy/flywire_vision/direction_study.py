"""Frozen-model direction diagnosis with polarity-balanced, grouped holdout.

The protocol is written before simulation. Readouts are selected using fit and
validation only, saved and hashed before any held-out trial is executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .pilot import DIRECTIONS, event_features, fit, scores
from .simulation import SimulationConfig, READOUT_TYPES, encode_movie
from .stimuli import movie
from .run_experiment import save

FEATURES = ('encoded_input', 'input_neurons', 'neural', 'neural_types', 'neural_time_sum')


def polarity(seed):
    return 'bright' if movie('right', seed).mean() > .5 else 'dark'


def balanced_groups(start, count):
    if count < 2 or count % 2:
        raise ValueError('group count must be positive and even')
    selected = {'bright': [], 'dark': []}
    seed = start
    while any(len(v) < count // 2 for v in selected.values()):
        bucket = selected[polarity(seed)]
        if len(bucket) < count // 2:
            bucket.append(seed)
        seed += 1
    return sorted(selected['bright'] + selected['dark'])


def pool_types(neural, sizes):
    bins = np.asarray(neural).reshape(8, sum(sizes))
    cuts = np.cumsum([0] + list(sizes))
    return np.stack([bins[:, a:b].sum(1) for a, b in zip(cuts[:-1], cuts[1:])], 1).ravel()


def evaluate(predicted, labels, rows):
    predicted, labels = np.asarray(predicted), np.asarray(labels)
    confusion = np.zeros((4, 4), dtype=int)
    np.add.at(confusion, (labels, predicted), 1)
    groups = sorted({r['group'] for r in rows})
    correctness = predicted == labels
    per_group = np.array([correctness[[r['group'] == g for r in rows]].mean() for g in groups])
    rng = np.random.default_rng(9102026)
    boot = rng.choice(per_group, size=(10000, len(groups)), replace=True).mean(1)
    precision = np.diag(confusion) / np.maximum(confusion.sum(0), 1)
    recall = np.diag(confusion) / np.maximum(confusion.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {'accuracy': float(correctness.mean()), 'correct': int(correctness.sum()),
            'clips': len(labels), 'groups': len(groups), 'macro_f1': float(f1.mean()),
            'group_bootstrap_95_interval': np.quantile(boot, [.025, .975]).tolist(),
            'confusion': confusion.tolist(), 'predictions': predicted.tolist(),
            'by_polarity': {p: float(correctness[[r['polarity'] == p for r in rows]].mean())
                            for p in ('bright', 'dark')},
            'group_accuracy': {str(g): float(a) for g, a in zip(groups, per_group)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from flywire_mnist.backends.frozen_cpu import FrozenCPU
    args.output.mkdir(parents=True, exist_ok=False)
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = read(args.artifact / 'manifest.json')
    cfg = SimulationConfig(**manifest['config'])
    if (cfg.frames, cfg.warmup_ms, cfg.tail_ms, cfg.dt_ms, cfg.frame_ms) != (40, 100., 100., .1, 10.):
        raise ValueError('study requires original P1 timing')
    model = read(args.artifact / 'model.json')
    channels = read(args.artifact / 'channels.json')
    groups = read(args.artifact / 'groups.json')
    cells = np.concatenate([groups[k] for k in READOUT_TYPES])
    inputs = np.array([r['index'] for r in channels])
    sizes = [len(groups[k]) for k in READOUT_TYPES]
    ni = next(i for i, p in enumerate(model['definition']['populations']) if p['name'] == 'flywire_neurons')
    runner = FrozenCPU(model, args.artifact / 'compile/native', args.output / 'execution',
                       population='visual_input', threads=manifest['threads'])
    original = read(args.artifact / 'intact/identity.json')
    if runner.base_hash != original['base_sha256'] or runner.binary_hash != original['binary_sha256']:
        raise ValueError('study must retain original P1 model and executable')
    protocol = {'schema': 'flywire-direction-study-v1', 'scope': 'small grouped in-distribution holdout; fixed speed, contrast and background',
                'primary_readout': 'neural', 'directions': DIRECTIONS, 'alphas': [.01, .1, 1., 10.],
                'fit_groups': balanced_groups(10000, 32), 'validation_groups': balanced_groups(20000, 16),
                'test_groups': balanced_groups(30000, 16), 'polarity_rule': 'first seeds with equal bright/dark groups; no outcome selection',
                'features': FEATURES, 'additional_control': 'fit labels independently permuted within each group, seed 910; neural features',
                'selection_rule': 'highest validation accuracy, ties choose larger alpha; never refit on validation/test',
                'test_rule': 'select and hash all readouts before executing test; report every predeclared readout, primary remains neural',
                'config': manifest['config'], 'readout_indices': cells.tolist(), 'input_indices': inputs.tolist(),
                'readout_type_sizes': dict(zip(READOUT_TYPES, sizes)),
                'base_sha256': runner.base_hash, 'binary_sha256': runner.binary_hash,
                'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in ('direction_study.py', 'pilot.py', 'simulation.py', 'stimuli.py')}}
    save(args.output / 'protocol.json', protocol)
    protocol_hash = sha(args.output / 'protocol.json')
    features = {name: [] for name in FEATURES}
    rows, labels, readouts, selected = [], [], {}, {}
    started = time.perf_counter()

    def collect(split):
        for group in protocol[split + '_groups']:
            group_polarity = polarity(group)
            for label, kind in enumerate(DIRECTIONS):
                frames = movie(kind, group)
                i, t, _ = encode_movie(frames, channels, cfg)
                key = f'g{group}-{kind}'
                result = runner.run(i, t, key)
                pop = result['populations'][ni]
                neural = event_features(pop['indices'], pop['spike_ticks'], cells, manifest['neurons'])
                input_neural = event_features(pop['indices'], pop['spike_ticks'], inputs, manifest['neurons'])
                values = {'encoded_input': event_features(i, t, np.arange(len(channels)), len(channels)),
                          'input_neurons': input_neural, 'neural': neural,
                          'neural_types': pool_types(neural, sizes),
                          'neural_time_sum': neural.reshape(8, len(cells)).sum(0)}
                for name, value in values.items():
                    features[name].append(value)
                labels.append(label)
                rows.append({'group': group, 'kind': kind, 'split': split, 'polarity': group_polarity,
                             'input_sha256': hashlib.sha256(np.stack([t, i], 1).astype('<i8').tobytes()).hexdigest(),
                             'events_sha256': hashlib.sha256(np.stack([pop['spike_ticks'], pop['indices']], 1).astype('<i8').tobytes()).hexdigest(),
                             'external_spikes': len(i), 'input_cell_spikes': int(input_neural.sum()),
                             'downstream_spikes': int(neural.sum()),
                             'active_input_cells': int(np.count_nonzero(input_neural.reshape(8, -1).sum(0))),
                             'active_downstream_cells': int(np.count_nonzero(neural.reshape(8, -1).sum(0))),
                             'seconds': result['wall_seconds']})
                del pop, result
                # Only this run's own temporary native outputs; features and hashes remain.
                shutil.rmtree(args.output / 'execution' / key)
                (args.output / 'execution' / (key + '.spikes')).unlink()
            print(json.dumps({'split': split, 'completed_clips': len(rows), 'total_clips': 256,
                              'seconds': time.perf_counter() - started}), flush=True)
        save(args.output / 'rows.json', rows)
        np.savez_compressed(args.output / 'features.npz', **{k: np.asarray(v) for k, v in features.items()}, labels=labels)

    collect('fit')
    collect('validation')
    nfit, nval = 128, 64
    y = np.asarray(labels)
    shuffled = y[:nfit].reshape(-1, 4).copy()
    rng = np.random.default_rng(910)
    for group_labels in shuffled:
        rng.shuffle(group_labels)
    for name in (*FEATURES, 'labels_shuffled'):
        x = np.asarray(features['neural' if name == 'labels_shuffled' else name], dtype=np.float32)
        fit_labels = shuffled.ravel() if name == 'labels_shuffled' else y[:nfit]
        candidates = []
        for alpha in protocol['alphas']:
            m = fit(x[:nfit], fit_labels, alpha)
            pred = scores(m, x[nfit:]).argmax(1)
            candidates.append((float(np.mean(pred == y[nfit:])), alpha, m, pred))
        accuracy, alpha, m, pred = max(candidates, key=lambda a: (a[0], a[1]))
        path = args.output / (name + '-readout.npz')
        np.savez_compressed(path, **m)
        readouts[name] = m
        selected[name] = {'selected_alpha': alpha, 'development_validation_accuracy': accuracy,
                          'features': x.shape[1], 'readout_sha256': sha(path),
                          'validation': evaluate(pred, y[nfit:], rows[nfit:]),
                          'fit_accuracy': float(np.mean(scores(m, x[:nfit]).argmax(1) == fit_labels))}
    save(args.output / 'selection.json', {'protocol_sha256': protocol_hash, 'test_clips_executed': 0, 'readouts': selected})
    selection_hash = sha(args.output / 'selection.json')
    print(json.dumps({'stage': 'selection_locked_before_test', 'selection_sha256': selection_hash,
                      'validation_accuracy': {n: v['development_validation_accuracy'] for n, v in selected.items()}}), flush=True)
    collect('test')
    assert sha(args.output / 'protocol.json') == protocol_hash
    assert sha(args.output / 'selection.json') == selection_hash
    y = np.asarray(labels)
    for name, values in selected.items():
        assert sha(args.output / (name + '-readout.npz')) == values['readout_sha256']
        x = np.asarray(features['neural' if name == 'labels_shuffled' else name], dtype=np.float32)
        pred = scores(readouts[name], x[nfit + nval:]).argmax(1)
        values['test'] = evaluate(pred, y[nfit + nval:], rows[nfit + nval:])
    # Independent trial groups, not four derived clips, are the bootstrap units.
    primary = np.asarray(selected['neural']['test']['predictions']) == y[nfit + nval:]
    direct = np.asarray(selected['encoded_input']['test']['predictions']) == y[nfit + nval:]
    delta = (primary.astype(float) - direct).reshape(-1, 4).mean(1)
    boot = np.random.default_rng(9102026).choice(delta, (10000, len(delta)), replace=True).mean(1)
    report = {'schema': protocol['schema'], 'status': 'complete', 'scope': protocol['scope'],
              'fit_clips': nfit, 'validation_clips': nval, 'test_clips': 64,
              'independent_validation_groups': 16, 'independent_test_groups': 16,
              'primary_readout': 'neural', 'protocol_sha256': protocol_hash,
              'selection_sha256': selection_hash, 'readouts': selected,
              'neural_minus_encoded_input': {'test_accuracy_difference': float(delta.mean()),
                                            'paired_group_bootstrap_95_interval': np.quantile(boot, [.025, .975]).tolist()},
              'limits': ['fixed speed, contrast and background; no OOD evaluation',
                         'artificial Mi1/Tm1 drive; no natural retina model',
                         'no rewired topology control or approach classifier',
                         'small holdout; intervals descriptive and unadjusted for multiple readouts'],
              'features_sha256': sha(args.output / 'features.npz'), 'seconds': time.perf_counter() - started}
    save(args.output / 'report.json', report)
    print(json.dumps({'status': 'complete', 'test_accuracy': {n: r['test']['accuracy'] for n, r in selected.items()},
                      'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
