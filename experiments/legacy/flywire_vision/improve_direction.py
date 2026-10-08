"""Development-only drive/readout search, then locked paired fresh holdout."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .simulation import SimulationConfig, READOUT_TYPES
from .stimuli import movie
from .pilot import DIRECTIONS, event_features, fit, scores
from .direction_study import balanced_groups, polarity, evaluate, pool_types
from .refinement import encode_variant, gain_model, anatomical_map, spatial_projection, fit_linear, linear_scores
from .run_experiment import save

RECIPES = ('raw_standard', 'spatial12_standard', 'fisher128_raw')
ALPHAS = (.001, .01, .1, 1., 10.)
VARIANTS = {'contrast16': ('contrast', 16.), 'contrast32': ('contrast', 32.),
            'contrast64': ('contrast', 64.), 'temporal32': ('temporal_difference_x4', 32.)}


def choose_readout(raw, y, nfit, projection):
    choices = []
    for rank, recipe in enumerate(RECIPES):
        for alpha in ALPHAS:
            model = fit_linear(raw[:nfit], y[:nfit], recipe, alpha, projection)
            prediction = linear_scores(model, raw[nfit:]).argmax(1)
            choices.append((float(np.mean(prediction == y[nfit:])), -rank, alpha, model, prediction))
    best = max(choices, key=lambda c: c[:3])
    return best[3], best[4], [{'recipe': RECIPES[-rank], 'alpha': alpha, 'validation_accuracy': accuracy}
                             for accuracy, rank, alpha, _, _ in choices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from flywire_mnist.graph import load_graph
    from flywire_mnist.backends.frozen_cpu import FrozenCPU
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = read(args.artifact / 'manifest.json')
    base = read(args.artifact / 'model.json')
    channels = read(args.artifact / 'channels.json')
    groups = read(args.artifact / 'groups.json')
    cfg = SimulationConfig(**manifest['config'])
    cells = np.concatenate([groups[k] for k in READOUT_TYPES])
    inputs = np.array([c['index'] for c in channels])
    sizes = [len(groups[k]) for k in READOUT_TYPES]
    ni = next(i for i, p in enumerate(base['definition']['populations']) if p['name'] == 'flywire_neurons')
    graph = load_graph(args.graph)
    if graph.identity != manifest['graph_identity']:
        raise ValueError('graph identity changed')
    mapping = anatomical_map(graph, channels, groups)
    projection = spatial_projection(mapping)
    np.savez_compressed(args.output / 'spatial-map.npz', **mapping)
    old_protocol = read(args.previous / 'protocol.json')
    old_report = read(args.previous / 'report.json')
    if sha(args.previous / 'features.npz') != old_report['features_sha256']:
        raise ValueError('previous feature artifact changed')
    if sha(args.previous / 'neural-readout.npz') != old_report['readouts']['neural']['readout_sha256']:
        raise ValueError('original readout artifact changed')
    old_rows = read(args.previous / 'rows.json')
    old_identity = read(args.artifact / 'intact/identity.json')
    if old_protocol['base_sha256'] != old_identity['base_sha256']:
        raise ValueError('previous study belongs to another model')
    # Only old fit/validation rows are imported. Its revealed test is excluded.
    cache = {}
    with np.load(args.previous / 'features.npz') as f:
        for i, row in enumerate(old_rows):
            if row['split'] not in ('fit', 'validation'):
                continue
            cache['contrast16', row['group'], row['kind']] = (
                {k: f[k][i].copy() for k in ('neural', 'input_neurons', 'encoded_input')}, dict(row))
    def subset(seed_groups, n):
        buckets = {p: [g for g in seed_groups if polarity(g) == p][:n // 2] for p in ('bright', 'dark')}
        return sorted(buckets['bright'] + buckets['dark'])
    protocol = {'schema': 'flywire-direction-improvement-v2', 'directions': DIRECTIONS,
                'fit_groups': old_protocol['fit_groups'], 'validation_groups': old_protocol['validation_groups'],
                'screen_fit_groups': subset(old_protocol['fit_groups'], 16),
                'screen_validation_groups': subset(old_protocol['validation_groups'], 8),
                'test_groups': balanced_groups(50000, 32), 'variants': VARIANTS,
                'recipes': RECIPES, 'alphas': ALPHAS,
                'selection_rule': 'screen on 64 fit/32 validation clips; choose highest validation, ties prefer earlier variant; expand winner to 128/64 and select readout on validation; hash before fresh test',
                'old_test_excluded': True, 'test_comparison': 'selected drive and original drive on same 32 new groups; original frozen and readout-only baselines',
                'spatial_map': 'positive-contact-weighted Mi1->T4 and Tm1->T5 coordinates, 12x12 bilinear pooling; external engineered readout',
                'mapped_cells': int(mapping['valid'].sum()), 'spatial_map_sha256': sha(args.output / 'spatial-map.npz'),
                'input_indices': inputs.tolist(), 'readout_indices': cells.tolist(), 'readout_type_sizes': dict(zip(READOUT_TYPES, sizes)),
                'parent_base_sha256': old_identity['base_sha256'], 'binary_sha256': old_identity['binary_sha256'],
                'parent_study': str(args.previous), 'config': manifest['config'],
                'sources_sha256': {name: sha(Path(__file__).with_name(name)) for name in
                                  ('improve_direction.py', 'refinement.py', 'simulation.py', 'stimuli.py', 'pilot.py', 'direction_study.py')}}
    save(args.output / 'protocol.json', protocol)
    save(args.output / 'initial-protocol.json', protocol)
    runners, models, executions = {}, {}, []

    def get_runner(variant):
        if variant not in runners:
            mode, gain = VARIANTS[variant]
            model = gain_model(base, gain)
            models[variant] = model
            runners[variant] = FrozenCPU(model, args.artifact / 'compile/native', args.output / ('execution-' + variant),
                                         population='visual_input', threads=manifest['threads'])
            if runners[variant].binary_hash != old_identity['binary_sha256']:
                raise ValueError('native executable changed')
            if variant == 'contrast16' and runners[variant].base_hash != old_identity['base_sha256']:
                raise ValueError('original baseline snapshot changed')
        return runners[variant]

    def collect(variant, seeds, split):
        values, rows, labels = {k: [] for k in ('neural', 'input_neurons', 'encoded_input')}, [], []
        mode, gain = VARIANTS[variant]
        for group in seeds:
            for label, kind in enumerate(DIRECTIONS):
                key = variant, group, kind
                if key not in cache:
                    runner = get_runner(variant)
                    frames = movie(kind, group)
                    i, t, _ = encode_variant(frames, channels, replace(cfg, input_weight_mv=gain), mode)
                    run_key = f'g{group}-{kind}'
                    result = runner.run(i, t, run_key)
                    pop = result['populations'][ni]
                    data = {'neural': event_features(pop['indices'], pop['spike_ticks'], cells, manifest['neurons']),
                            'input_neurons': event_features(pop['indices'], pop['spike_ticks'], inputs, manifest['neurons']),
                            'encoded_input': event_features(i, t, np.arange(len(channels)), len(channels))}
                    row = {'group': group, 'kind': kind, 'polarity': polarity(group), 'variant': variant,
                           'input_spikes': len(i), 'neural_stimulus_spikes': int(data['neural'].sum()),
                           'input_sha256': hashlib.sha256(np.stack([t, i], 1).astype('<i8').tobytes()).hexdigest(),
                           'events_sha256': hashlib.sha256(np.stack([pop['spike_ticks'], pop['indices']], 1).astype('<i8').tobytes()).hexdigest(),
                           'seconds': result['wall_seconds']}
                    cache[key] = data, row
                    executions.append(row)
                    del pop, result
                    shutil.rmtree(runner.directory / run_key)
                    (runner.directory / (run_key + '.spikes')).unlink()
                data, row = cache[key]
                for name in values:
                    values[name].append(data[name])
                rows.append({**row, 'split': split})
                labels.append(label)
            print(json.dumps({'stage': split, 'variant': variant, 'groups_done': len(rows) // 4,
                              'groups_total': len(seeds), 'new_simulations': len(executions),
                              'seconds': time.perf_counter() - started}), flush=True)
        return {k: np.asarray(v) for k, v in values.items()}, rows, np.array(labels)

    screen = {}
    screen_seeds = protocol['screen_fit_groups'] + protocol['screen_validation_groups']
    for variant in VARIANTS:
        data, rows, y = collect(variant, screen_seeds, 'screen')
        m, pred, choices = choose_readout(data['neural'], y, 64, projection)
        screen[variant] = {'validation_accuracy': float(np.mean(pred == y[64:])), 'recipe': str(m['recipe']),
                           'alpha': float(m['alpha']), 'candidates': choices}
        save(args.output / 'screen.json', screen)
        np.savez_compressed(args.output / (variant + '-screen.npz'), **data, labels=y)
        print(json.dumps({'screen_result': variant, **{k: v for k, v in screen[variant].items() if k != 'candidates'}}), flush=True)
    winner = max(VARIANTS, key=lambda v: screen[v]['validation_accuracy'])
    development_seeds = protocol['fit_groups'] + protocol['validation_groups']
    development, development_rows, y = collect(winner, development_seeds, 'development')
    selected_model, prediction, choices = choose_readout(development['neural'], y, 128, projection)
    old_development, _, _ = collect('contrast16', development_seeds, 'baseline-development')
    readout_only, _, _ = choose_readout(old_development['neural'], y, 128, projection)
    with np.load(args.previous / 'neural-readout.npz') as m:
        old_readout = dict(m)
    modes = {'neural': selected_model, 'readout_only': readout_only, 'old_neural': old_readout}
    def expand(data):
        return {**data, 'neural_types': np.stack([pool_types(v, sizes) for v in data['neural']]),
                'neural_time_sum': data['neural'].reshape(len(data['neural']), 8, -1).sum(1)}
    development = expand(development)
    for name in ('encoded_input', 'input_neurons', 'neural_types', 'neural_time_sum'):
        x = development[name]
        candidates = []
        for alpha in ALPHAS:
            m = fit(x[:128], y[:128], alpha)
            acc = float(np.mean(scores(m, x[128:]).argmax(1) == y[128:]))
            candidates.append((acc, alpha, m))
        modes[name] = max(candidates, key=lambda a: a[:2])[2]
    shuffled = y[:128].reshape(-1, 4).copy()
    rng = np.random.default_rng(910)
    for group in shuffled:
        rng.shuffle(group)
    modes['labels_shuffled'] = fit_linear(development['neural'][:128], shuffled.ravel(), str(selected_model['recipe']), float(selected_model['alpha']), projection)
    def predict(m, x):
        return (linear_scores(m, x) if 'coefficients' in m else scores(m, x)).argmax(1)
    reports = {}
    for name, m in modes.items():
        x = old_development['neural'] if name in ('old_neural', 'readout_only') else development['neural' if name == 'labels_shuffled' else name]
        path = args.output / (name + '-readout.npz')
        np.savez_compressed(path, **m)
        val = evaluate(predict(m, x[128:]), y[128:], development_rows[128:])
        reports[name] = {'readout_sha256': sha(path), 'selected_alpha': float(m['alpha']),
                         'recipe': str(m.get('recipe', 'original_standardized_ridge')),
                         'development_validation_accuracy': val['accuracy'], 'validation': val,
                         'fit_accuracy': float(np.mean(predict(m, x[:128]) == (shuffled.ravel() if name == 'labels_shuffled' else y[:128])))}
    get_runner(winner)
    protocol.update(selected_variant=winner, input_mode=VARIANTS[winner][0],
                    config={**manifest['config'], 'input_weight_mv': VARIANTS[winner][1]},
                    base_sha256=runners[winner].base_hash)
    save(args.output / 'protocol.json', protocol)
    selection = {'selected_variant': winner, 'readouts': reports, 'development_candidates': choices,
                 'test_clips_executed': 0, 'protocol_sha256': sha(args.output / 'protocol.json')}
    save(args.output / 'selection.json', selection)
    selection_hash = sha(args.output / 'selection.json')
    print(json.dumps({'stage': 'selection_locked_before_fresh_test', 'variant': winner,
                      'validation_accuracy': reports['neural']['development_validation_accuracy'],
                      'recipe': reports['neural']['recipe'], 'selection_sha256': selection_hash}), flush=True)
    test, test_rows, test_y = collect(winner, protocol['test_groups'], 'test')
    original_test, _, _ = collect('contrast16', protocol['test_groups'], 'paired-original-test')
    test = expand(test)
    for name, m in modes.items():
        assert sha(args.output / (name + '-readout.npz')) == reports[name]['readout_sha256']
        x = original_test['neural'] if name in ('old_neural', 'readout_only') else test['neural' if name == 'labels_shuffled' else name]
        reports[name]['test'] = evaluate(predict(m, x), test_y, test_rows)
    assert sha(args.output / 'selection.json') == selection_hash
    paired = {}
    for name in ('old_neural', 'readout_only', 'encoded_input'):
        a = np.asarray(reports['neural']['test']['predictions']) == test_y
        b = np.asarray(reports[name]['test']['predictions']) == test_y
        difference = (a.astype(float) - b).reshape(-1, 4).mean(1)
        bootstrap = np.random.default_rng(9102026).choice(difference, (10000, len(difference)), replace=True).mean(1)
        paired[name] = {'accuracy_difference': float(difference.mean()), 'group_bootstrap_95_interval': np.quantile(bootstrap, [.025, .975]).tolist()}
    np.savez_compressed(args.output / 'features.npz', **{'development_' + k: v for k, v in development.items()},
                        **{'test_' + k: v for k, v in test.items()}, original_test_neural=original_test['neural'],
                        original_development_neural=old_development['neural'], development_labels=y, test_labels=test_y)
    save(args.output / 'rows.json', development_rows + test_rows)
    save(args.output / 'executions.json', executions)
    report = {'schema': protocol['schema'], 'status': 'complete', 'scope': 'fresh grouped direction holdout after development-only input/readout selection; fixed speed, contrast and background',
              'fit_clips': 128, 'validation_clips': 64, 'test_clips': 128, 'independent_validation_groups': 16,
              'independent_test_groups': 32, 'selected_variant': winner, 'input_mode': protocol['input_mode'],
              'input_weight_mv': protocol['config']['input_weight_mv'], 'primary_readout': 'neural',
              'readouts': reports, 'paired_improvement': paired, 'selection_sha256': selection_hash,
              'protocol_sha256': sha(args.output / 'protocol.json'), 'features_sha256': sha(args.output / 'features.npz'),
              'new_simulations': len(executions), 'seconds': time.perf_counter() - started,
              'limits': ['fixed speed/contrast/background; no OOD claim', 'input gain/encoding and external readout may change; internal graph unchanged',
                         'development search explored multiple settings; fresh test untouched until selection lock',
                         'no real-topology advantage claim; no approach classifier']}
    save(args.output / 'report.json', report)
    print(json.dumps({'status': 'complete', 'variant': winner, 'test': {k: v['test']['accuracy'] for k, v in reports.items()}, 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
