"""Rebuild frozen speed-bank weights, input schedules, features and statistics."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from .motion_refinement import read, sha, load_npz
from .motion_readout import fit_readout, predict, transform
from .multispeed_readout import candidates, bank_features
from .multispeed_data import speed_movie, digest
from .multispeed_study import grids_for, features_for, CONDITIONS, MODELS
from .motion_stress import phase_for
from .motion_challenge import parameters
from .motion_resolution import temporal_counts
from .refinement import encode_variant
from .simulation import SimulationConfig
from .pilot import DIRECTIONS
from .direction_study import evaluate
from .causal_motion import paired_interval
from .run_experiment import save


def verify(root, artifact):
    p, s, r = [read(root/(n+'.json')) for n in ('protocol', 'selection', 'report')]
    rows = read(root/'rows.json'); f = load_npz(root/'features.npz'); arrays = load_npz(root/'transform.npz')
    parent = Path(p['parent']); pp = read(parent/'protocol.json')
    checks = {'complete': r['status'] == 'complete',
        'protocol': sha(root/'protocol.json') == s['protocol_sha256'] == r['protocol_sha256'],
        'selection': sha(root/'selection.json') == r['selection_sha256'] and s['test_clips_executed'] == 0,
        'parent': sha(parent/'protocol.json') == p['parent_protocol_sha256'] and sha(parent/'selection.json') == p['parent_selection_sha256'],
        'sources': all(sha(Path(__file__).with_name(n)) == h for n, h in p['source_sha256'].items()),
        'data': sha(root/'features.npz') == r['features_sha256'] and sha(root/'rows.json') == r['rows_sha256'],
        'transform': sha(root/'transform.npz') == p['transform_sha256'],
        'new_groups': not set(p['test_groups']) & set(pp['fit_groups']+pp['validation_groups']+pp['test_groups']),
        'rows': len(rows) == 576 and len({(v['condition'], v['group'], *v['phase_index'], v['kind']) for v in rows}) == 576,
        'polarity': all(v['polarity'] == ('bright' if parameters(v['group'])['polarity'] > 0 else 'dark') for v in rows)}
    original_arrays = load_npz(parent/'transform.npz')
    checks['original_mapping_blank'] = all(np.array_equal(v, arrays[k]) for k, v in original_arrays.items())
    models = {n: load_npz(root/(n+'.npz')) for n in MODELS}
    checks['models'] = all(sha(root/(n+'.npz')) == p['readouts'][n] for n in MODELS)
    checks['p7_model_exact'] = all(np.array_equal(v, models['p7'][k]) for k, v in load_npz(parent/'neural_motion-readout.npz').items())
    for name in ('intact', 'cut_input', 'matched_cut'):
        identity = read(root/name/'identity.json'); old = read(parent/name/'identity.json')
        checks[name+'_base'] = sha(root/name/'base.bin') == identity['base_sha256'] == old['base_sha256']
        checks[name+'_binary'] = sha(artifact/'compile/native/b2-native') == identity['binary_sha256'] == old['binary_sha256']
        checks[name+'_locked_before_test'] = (root/'selection.json').stat().st_mtime_ns < (root/name/'base.bin').stat().st_mtime_ns
    devrows, devgrids = [], []
    checks['development_hashes'] = True
    channels = read(artifact/'channels.json'); cfg = SimulationConfig(**pp['config'])
    checks['development_inputs'] = True
    for speed, d in enumerate(p['development'], 1):
        directory = Path(d['directory'])
        checks['development_hashes'] &= all(sha(directory/(name+'.json')) == d[name+'_sha256'] for name in ('report', 'rows')) and sha(directory/'features.npz') == d['features_sha256']
        dd = load_npz(directory/'features.npz'); rr = read(directory/'rows.json')
        for idx, row in enumerate(rr):
            frames = speed_movie(row['kind'], row['group'], row['phase_index'], speed)
            ii, tt, _ = encode_variant(frames, channels, cfg, pp['input_mode'])
            checks['development_inputs'] &= digest(np.stack([tt, ii], 1).astype('<i8')) == row['input_sha256'] and np.array_equal(dd['encoded'][idx], temporal_counts(ii, tt, np.arange(len(channels)), len(channels)))
        devgrids.append(grids_for(dd['neural'], arrays, p['grid_recipe']))
        devrows.extend([{**v, 'speed': speed} for v in rr])
    grids = np.concatenate(devgrids); labels = np.array([DIRECTIONS.index(v['kind']) for v in devrows])
    train = np.array([v['split'] == 'fit' for v in devrows]); masks = [np.array([v['split'] == 'validation' and v['speed'] == speed for v in devrows]) for speed in (1, 2)]
    checks['development_split'] = len(devrows) == 384 and train.sum() == 256 and all(set(v['group'] for v in devrows if v['split'] == split) == set(p[key]) for split, key in (('fit', 'fit_groups'), ('validation', 'validation_groups')))
    regenerated = []
    for recipe in candidates():
        x = bank_features(grids, recipe)
        for alpha in (.01, .1, 1., 10., 100.):
            m = fit_readout(x[train], labels[train], alpha, True)
            acc = [float(np.mean(predict(m, x[mask]).argmax(1) == labels[mask])) for mask in masks]
            regenerated.append({'recipe': recipe, 'alpha': alpha, 'validation_by_speed': acc, 'worst_speed': min(acc), 'mean': float(np.mean(acc)), 'features': x.shape[1]})
    key = lambda v: (v['worst_speed'], v['mean'], -v['features'], v['alpha'])
    checks['all_search_results'] = regenerated == s['candidates']
    checks['selection_rule'] = max(regenerated, key=key) == p['chosen'] and max([v for v in regenerated if v['recipe'] == candidates()[0]], key=key) == p['fixed_retrained']
    for name in MODELS:
        choice = p['chosen'] if name == 'multispeed' else p['fixed_retrained']
        x = bank_features(grids, choice['recipe'])
        if name != 'p7':
            m = fit_readout(x[train], labels[train], choice['alpha'], True)
            checks[name+'_fit_only'] = all(np.array_equal(v, models[name][k]) for k, v in m.items())
        checks[name+'_validation'] = all(evaluate(predict(models[name], x[mask]).argmax(1), labels[mask], [v for v, keep in zip(devrows, mask) if keep]) == s['validation'][name][str(speed)] for speed, mask in zip((1, 2), masks))
    print({'development_verified': all(checks.values())}, flush=True)
    lookup = {(v['condition'], v['group'], *v['phase_index'], v['kind']): i for i, v in enumerate(rows)}
    blanks = load_npz(root/'blanks.npz'); endpoints = {}; static = {}
    checks.update({k: True for k in ('movies', 'inputs', 'features', 'shuffle', 'cut_blank', 'lesion_same_input', 'static_same', 'complete_groups', 'original_p7_features', 'original_p7_predictions')})
    for idx, row in enumerate(rows):
        c = row['condition']; frames = speed_movie(row['kind'], row['group'], row['phase_index'], row['speed'])
        if c == 'static_first':frames = np.repeat(frames[:1], 40, axis=0)
        checks['movies'] &= digest(frames) == row['movie_sha256'] and np.array_equal(frames[0], frames[-1])
        endpoints.setdefault((c, row['group']), {k: Counter() for k in DIRECTIONS})[row['kind']][digest(frames[0])] += 1
        ii, tt, _ = encode_variant(frames, channels, cfg, pp['input_mode'])
        checks['inputs'] &= digest(np.stack([tt, ii], 1).astype('<i8')) == row['input_sha256'] and len(ii) == row['external_spikes'] and np.array_equal(f['encoded'][idx], temporal_counts(ii, tt, np.arange(len(channels)), len(channels)))
        ff = features_for(f['neural'][idx], arrays, p)
        checks['features'] &= all(np.allclose(ff[n], f[n][idx], atol=1e-12, rtol=1e-12) for n in MODELS)
        original = transform(f['neural'][idx:idx+1], arrays['blank'], arrays, arrays['weights'], p['grid_recipe'])
        checks['original_p7_features'] &= np.allclose(original[0], f['p7'][idx], atol=1e-12, rtol=1e-12)
        if c.startswith('speed_'):
            checks['original_p7_predictions'] &= int(predict(models['p7'], original).argmax(1)[0]) == int(predict(models['p7'], f['p7'][idx:idx+1]).argmax(1)[0])
        order = np.arange(60); order[10:50] = np.random.default_rng(np.random.SeedSequence([row['group'], *row['phase_index'], 814])).permutation(order[10:50])
        checks['shuffle'] &= np.allclose(features_for(f['neural'][idx], arrays, p, order)['multispeed'], f['shuffled'][idx], atol=1e-12, rtol=1e-12)
        if c in ('cut_speed_2', 'matched_speed_2'):
            ref = rows[lookup['speed_2', row['group'], *row['phase_index'], row['kind']]]
            checks['lesion_same_input'] &= ref['input_sha256'] == row['input_sha256'] and tuple(row['phase_index']) == phase_for(row['kind'])
        if c == 'cut_speed_2':checks['cut_blank'] &= np.array_equal(f['neural'][idx], blanks['cut_input'])
        if c == 'static_first':
            h = row['movie_sha256']
            if h in static:checks['static_same'] &= np.array_equal(f['neural'][idx], f['neural'][static[h]])
            else:static[h] = idx
    checks['endpoint_balance'] = len(endpoints) == 48 and all(all(v == next(iter(d.values())) for v in d.values()) for d in endpoints.values())
    checks['complete_groups'] = all(sum(v['condition'] == c and v['group'] == g for v in rows) == (4 if c in ('cut_speed_2', 'matched_speed_2') else 16) for c in CONDITIONS for g in p['test_groups'])
    restored = read(root/'restoration.json')
    checks['reset'] = all(rows[0][k] == restored[k] for k in ('events_sha256', 'states_sha256', 'input_sha256')) and r['reset_exact']
    checks['native_count'] = r['native_runs'] == sum(not v['reused'] for v in rows)+4
    truth = np.array([DIRECTIONS.index(v['kind']) for v in rows]); pred = {n: predict(m, f[n]).argmax(1) for n, m in models.items()}
    stats = {}
    for c in CONDITIONS:
        mask = np.array([v['condition'] == c for v in rows]); rr = [v for v, keep in zip(rows, mask) if keep]
        stats[c] = all(evaluate(y[mask], truth[mask], rr) == r['conditions'][c][n] for n, y in pred.items())
        stats[c] &= paired_interval(pred['multispeed'][mask] == truth[mask], pred['p7'][mask] == truth[mask], rr) == r['paired_vs_p7'][c]
        if c in ('static_first', 'cut_speed_2'):stats[c] &= all(v['accuracy'] == .25 for v in r['conditions'][c].values())
    mask = np.array([v['condition'] in ('speed_1', 'speed_2') for v in rows]); rr = [v for v, keep in zip(rows, mask) if keep]
    stats['primary'] = all(evaluate(y[mask], truth[mask], rr) == r['primary'][n] for n, y in pred.items()) and paired_interval(pred['multispeed'][mask] == truth[mask], pred['p7'][mask] == truth[mask], rr) == r['primary_gain']
    subset = np.array([v['condition'] == 'speed_2' and tuple(v['phase_index']) == phase_for(v['kind']) for v in rows]); rr = [v for v, keep in zip(rows, subset) if keep]
    stats['subset'] = all(evaluate(y[subset], truth[subset], rr) == r['lesion_subset_baseline'][n] for n, y in pred.items())
    for c in ('cut_speed_2', 'matched_speed_2'):
        mask = np.array([v['condition'] == c for v in rows])
        stats[c+'_effect'] = paired_interval(pred['multispeed'][subset] == truth[subset], pred['multispeed'][mask] == truth[mask], rr) == r['lesion_effects'][c]
    for speed in (1, 2, 3):
        mask = np.array([v['condition'] == f'speed_{speed}' for v in rows]); rr = [v for v, keep in zip(rows, mask) if keep]
        sp = predict(models['multispeed'], f['shuffled'][mask]).argmax(1)
        stats[f'shuffle_{speed}'] = evaluate(sp, truth[mask], rr) == r['time_shuffle'][str(speed)]['result'] and paired_interval(pred['multispeed'][mask] == truth[mask], sp == truth[mask], rr) == r['time_shuffle'][str(speed)]['intact_minus_shuffle']
    result = {'all_passed': bool(all(checks.values()) and all(stats.values())), 'report_sha256': sha(root/'report.json'),
              'checks': {k: bool(v) for k, v in checks.items()}, 'statistics': {k: bool(v) for k, v in stats.items()}}
    save(root/'verification.json', result); print(result, flush=True)
    if not result['all_passed']:raise ValueError('multispeed verification failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path); parser.add_argument('--artifact', type=Path, required=True)
    args = parser.parse_args(); verify(args.root, args.artifact)
