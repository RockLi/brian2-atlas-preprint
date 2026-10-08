"""Refine downstream cell selection on development data, then a new holdout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .pilot import DIRECTIONS, fit, scores, event_features
from .simulation import SimulationConfig
from .refinement import gain_model, encode_variant, linear_scores
from .direction_study import balanced_groups, evaluate, polarity, pool_types
from .stimuli import movie
from .run_experiment import save

COUNTS = (32, 64, 128, 256, 512, 1024, 2048, 6142)
SCALINGS = ('raw', 'floor0.25', 'standard')
ALPHAS = (.001, .01, .1, 1., 10.)


def fit_sparse(raw, labels, count, scaling, alpha):
    raw = np.asarray(raw, dtype=float)
    n, dimension = raw.shape
    ncells = dimension // 8
    variance = raw.var(0)
    between = np.stack([raw[labels == i].mean(0) for i in range(4)]).var(0)
    statistic = (between / np.maximum(variance - between, 1e-3)).reshape(8, ncells).sum(0)
    selected = np.argsort(statistic)[::-1][:min(count, ncells)]
    indices = (np.arange(8)[:, None] * ncells + selected).ravel()
    x = raw[:, indices]
    mean = x.mean(0)
    if scaling == 'raw':
        scale = np.ones(len(indices))
    elif scaling == 'floor0.25':
        scale = np.sqrt(np.maximum(x.var(0), .25))
    elif scaling == 'standard':
        scale = x.std(0)
    else:
        raise ValueError('unknown scaling')
    scale[scale < 1e-8] = 1
    z = (x - mean) / scale
    dual = np.linalg.solve(z @ z.T / z.shape[1] + alpha * np.eye(n), np.eye(4)[labels])
    compact = z.T @ dual / z.shape[1] / scale[:, None]
    coefficient = np.zeros((dimension, 4))
    coefficient[indices] = compact
    model = {'coefficients': coefficient, 'bias': -mean @ compact, 'alpha': np.array(alpha),
             'selected_cells': selected, 'scaling': np.array(scaling),
             'recipe': np.array(f'fisher{len(selected)}_{scaling}'), 'format': np.array('linear_raw_counts_v1')}
    np.testing.assert_allclose(linear_scores(model, raw), z @ (z.T @ dual) / z.shape[1], rtol=1e-10, atol=1e-10)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from flywire_mnist.backends.frozen_cpu import FrozenCPU
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    parent = read(args.previous / 'protocol.json')
    old_report = read(args.previous / 'report.json')
    if sha(args.previous / 'features.npz') != old_report['features_sha256']:
        raise ValueError('parent features changed')
    if sha(args.previous / 'neural-readout.npz') != old_report['readouts']['neural']['readout_sha256']:
        raise ValueError('parent decoder changed')
    cfg = SimulationConfig(**parent['config'])
    cells = np.asarray(parent['readout_indices'])
    inputs = np.asarray(parent['input_indices'])
    channels = read(args.artifact / 'channels.json')
    size_values = list(parent['readout_type_sizes'].values())
    protocol = {'schema': 'flywire-direction-sparse-v3', 'parent_study': str(args.previous),
                'parent_report_sha256': sha(args.previous / 'report.json'),
                'fit_groups': parent['fit_groups'], 'validation_groups': parent['validation_groups'],
                'test_groups': balanced_groups(60000, 32), 'counts': COUNTS, 'scalings': SCALINGS,
                'alphas': ALPHAS, 'directions': DIRECTIONS, 'config': parent['config'],
                'input_mode': parent['input_mode'], 'selected_variant': parent['selected_variant'],
                'readout_indices': cells.tolist(), 'input_indices': inputs.tolist(),
                'readout_type_sizes': parent['readout_type_sizes'],
                'base_sha256': parent['base_sha256'], 'binary_sha256': parent['binary_sha256'],
                'selection_rule': '128 fit / 64 validation only; highest validation accuracy, then fewer cells, raw scaling, larger alpha; lock before new test',
                'scope': 'same 32 mV brain; change only downstream selection/scaling; all previous test rows excluded',
                'sources_sha256': {n: sha(Path(__file__).with_name(n)) for n in ('sparse_refinement.py', 'refinement.py', 'simulation.py', 'stimuli.py', 'pilot.py', 'direction_study.py')}}
    save(args.output / 'protocol.json', protocol)
    with np.load(args.previous / 'features.npz', allow_pickle=False) as f:
        development = {k: f['development_' + k].copy() for k in ('neural', 'input_neurons', 'encoded_input', 'neural_types', 'neural_time_sum')}
        y = f['development_labels'].copy()
    development_rows = read(args.previous / 'rows.json')[:192]
    candidates, best = [], None
    for count in COUNTS:
        for rank, scaling in enumerate(SCALINGS):
            for alpha in ALPHAS:
                model = fit_sparse(development['neural'][:128], y[:128], count, scaling, alpha)
                prediction = linear_scores(model, development['neural'][128:]).argmax(1)
                accuracy = float(np.mean(prediction == y[128:]))
                candidates.append({'cells': count, 'scaling': scaling, 'alpha': alpha, 'validation_accuracy': accuracy})
                score = (accuracy, -count, -rank, alpha)
                if best is None or score > best[0]:
                    best = score, model
    modes = {'neural': best[1]}
    for name, source in (('old_neural', 'neural'), ('encoded_input', 'encoded_input'), ('input_neurons', 'input_neurons'),
                         ('neural_types', 'neural_types'), ('neural_time_sum', 'neural_time_sum')):
        with np.load(args.previous / (source + '-readout.npz'), allow_pickle=False) as m:
            modes[name] = dict(m)
    shuffled = y[:128].reshape(-1, 4).copy()
    rng = np.random.default_rng(910)
    for group in shuffled:
        rng.shuffle(group)
    selected = modes['neural']
    modes['labels_shuffled'] = fit_sparse(development['neural'][:128], shuffled.ravel(), len(selected['selected_cells']), str(selected['scaling']), float(selected['alpha']))
    def predict(m, x):
        return (linear_scores(m, x) if 'coefficients' in m else scores(m, x)).argmax(1)
    reports = {}
    for name, m in modes.items():
        path = args.output / (name + '-readout.npz')
        np.savez_compressed(path, **m)
        x = development['neural' if name in ('old_neural', 'labels_shuffled') else name]
        validation = evaluate(predict(m, x[128:]), y[128:], development_rows[128:])
        reports[name] = {'readout_sha256': sha(path), 'recipe': str(m.get('recipe', 'standardized_ridge')),
                         'selected_alpha': float(m['alpha']), 'validation': validation,
                         'development_validation_accuracy': validation['accuracy'],
                         'fit_accuracy': float(np.mean(predict(m, x[:128]) == (shuffled.ravel() if name == 'labels_shuffled' else y[:128])))}
    selection = {'protocol_sha256': sha(args.output / 'protocol.json'), 'test_clips_executed': 0,
                 'selected_cells': len(selected['selected_cells']), 'readouts': reports, 'candidates': candidates}
    save(args.output / 'selection.json', selection)
    selection_hash = sha(args.output / 'selection.json')
    print(json.dumps({'stage': 'locked_before_new_test', 'cells': len(selected['selected_cells']),
                      'validation_accuracy': reports['neural']['development_validation_accuracy'], 'selection_sha256': selection_hash}), flush=True)
    template = gain_model(read(args.artifact / 'model.json'), cfg.input_weight_mv)
    ni = next(i for i, p in enumerate(template['definition']['populations']) if p['name'] == 'flywire_neurons')
    n = template['definition']['populations'][ni]['count']
    runner = FrozenCPU(template, args.artifact / 'compile/native', args.output / 'execution', population='visual_input', threads=4)
    if runner.base_hash != parent['base_sha256'] or runner.binary_hash != parent['binary_sha256']:
        raise ValueError('brain changed')
    test = {k: [] for k in development}
    rows, labels = [], []
    started = time.perf_counter()
    for group in protocol['test_groups']:
        for label, kind in enumerate(DIRECTIONS):
            frames = movie(kind, group)
            i, t, _ = encode_variant(frames, channels, cfg, protocol['input_mode'])
            key = f'g{group}-{kind}'
            result = runner.run(i, t, key)
            pop = result['populations'][ni]
            neural = event_features(pop['indices'], pop['spike_ticks'], cells, n)
            data = {'neural': neural, 'input_neurons': event_features(pop['indices'], pop['spike_ticks'], inputs, n),
                    'encoded_input': event_features(i, t, np.arange(len(channels)), len(channels)),
                    'neural_types': pool_types(neural, size_values), 'neural_time_sum': neural.reshape(8, -1).sum(0)}
            for name in test:
                test[name].append(data[name])
            rows.append({'group': group, 'kind': kind, 'polarity': polarity(group), 'split': 'test',
                         'input_sha256': hashlib.sha256(np.stack([t, i], 1).astype('<i8').tobytes()).hexdigest(),
                         'events_sha256': hashlib.sha256(np.stack([pop['spike_ticks'], pop['indices']], 1).astype('<i8').tobytes()).hexdigest(),
                         'seconds': result['wall_seconds']})
            labels.append(label)
            del pop, result
            shutil.rmtree(runner.directory / key)
            (runner.directory / (key + '.spikes')).unlink()
        print(json.dumps({'test_groups': len(rows) // 4, 'total_groups': 32, 'seconds': time.perf_counter() - started}), flush=True)
    for name, m in modes.items():
        assert sha(args.output / (name + '-readout.npz')) == reports[name]['readout_sha256']
        x = np.asarray(test['neural' if name in ('old_neural', 'labels_shuffled') else name])
        reports[name]['test'] = evaluate(predict(m, x), labels, rows)
    assert sha(args.output / 'selection.json') == selection_hash
    new = np.asarray(reports['neural']['test']['predictions']) == labels
    old = np.asarray(reports['old_neural']['test']['predictions']) == labels
    difference = (new.astype(float) - old).reshape(-1, 4).mean(1)
    bootstrap = np.random.default_rng(9102026).choice(difference, (10000, 32), replace=True).mean(1)
    np.savez_compressed(args.output / 'features.npz', **{'development_' + k: v for k, v in development.items()},
                        **{'test_' + k: np.asarray(v) for k, v in test.items()}, development_labels=y, test_labels=labels)
    save(args.output / 'rows.json', development_rows + rows)
    report = {'schema': protocol['schema'], 'status': 'complete', 'scope': 'fresh grouped holdout, same 32 mV input brain, only downstream sparse linear readout refined; fixed speed/contrast/background',
              'fit_clips': 128, 'validation_clips': 64, 'test_clips': 128, 'independent_validation_groups': 16,
              'independent_test_groups': 32, 'selected_variant': parent['selected_variant'], 'input_mode': parent['input_mode'],
              'input_weight_mv': cfg.input_weight_mv, 'primary_readout': 'neural', 'selected_cells': len(selected['selected_cells']),
              'readouts': reports, 'selection_sha256': selection_hash, 'protocol_sha256': sha(args.output / 'protocol.json'),
              'paired_improvement': {'old_neural': {'accuracy_difference': float(difference.mean()), 'group_bootstrap_95_interval': np.quantile(bootstrap, [.025, .975]).tolist()}},
              'comparison_labels': {'old_neural': '上一版 · 128 细胞', 'neural': f'本轮 · {len(selected["selected_cells"])} 细胞'},
              'features_sha256': sha(args.output / 'features.npz'), 'new_simulations': 128,
              'seconds': time.perf_counter() - started,
              'limits': ['small in-distribution holdout; no OOD or real-topology benefit claim', 'prior holdouts used only to assess earlier versions; never used to select this decoder']}
    save(args.output / 'report.json', report)
    print(json.dumps({'status': 'complete', 'test': {k: v['test']['accuracy'] for k, v in reports.items()}}), flush=True)


if __name__ == '__main__':
    main()
