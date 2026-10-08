"""Development-selected speed-bank readout, then frozen new-orbit evaluation."""
import argparse
from pathlib import Path
import time
import numpy as np
from .motion_refinement import read, sha, load_npz
from .motion_readout import pooled_grids, fit_readout, predict
from .multispeed_readout import bank_features, candidates
from .multispeed_data import speed_movie, digest, NativeRecorder
from .motion_challenge import balanced_orbit_seeds, iter_orbit
from .motion_stress import phase_for
from .causal_motion import intervene, paired_interval
from .pilot import DIRECTIONS
from .direction_study import evaluate
from .run_experiment import save

SOURCES = ('multispeed_study.py', 'multispeed_readout.py', 'multispeed_data.py',
           'motion_readout.py', 'motion_challenge.py', 'motion_stress.py', 'motion_refinement.py',
           'motion_resolution.py', 'refinement.py', 'simulation.py', 'pilot.py', 'direction_study.py', 'causal_motion.py')
CONDITIONS = ('speed_1', 'speed_2', 'speed_3', 'static_first', 'cut_speed_2', 'matched_speed_2')
MODELS = ('p7', 'fixed_retrained', 'multispeed')


def grids_for(neural, arrays, recipe, order=None):
    blank = arrays['blank']
    if order is not None:
        neural, blank = neural[:, order], blank[order]
    return pooled_grids(neural, blank, arrays, arrays['weights'], recipe)


def select(args):
    args.output.mkdir(parents=True, exist_ok=False)
    parent = read(args.parent/'protocol.json')
    arrays = load_npz(args.parent/'transform.npz')
    recipe = parent['chosen']['recipe']
    if recipe != dict(feature='spectral', separation='family', start=10, stop=50, width=1, weight='none', transform='signed'):
        raise ValueError('requires P7 selected family spectral recipe')
    rows, grids, records = [], [], []
    for speed, directory in enumerate(args.development, 1):
        report = read(directory/'report.json')
        if report['status'] != 'complete' or sha(directory/'features.npz') != report['features_sha256']:
            raise ValueError('development data changed')
        rr = read(directory/'rows.json')
        if len(rr) != 192 or any(r['split'] != ('fit' if i < 128 else 'validation') for i, r in enumerate(rr)):
            raise ValueError('development-only groups required')
        nn = load_npz(directory/'features.npz')['neural']
        grids.append(grids_for(nn, arrays, recipe))
        rows.extend([{**r, 'speed': speed} for r in rr])
        records.append({'directory': str(directory), 'report_sha256': sha(directory/'report.json'),
                        'rows_sha256': sha(directory/'rows.json'), 'features_sha256': sha(directory/'features.npz')})
    if len(grids) != 2:
        raise ValueError('one- and two-speed development required')
    grids = np.concatenate(grids)
    labels = np.array([DIRECTIONS.index(r['kind']) for r in rows])
    train = np.array([r['split'] == 'fit' for r in rows])
    masks = [np.array([r['split'] == 'validation' and r['speed'] == s for r in rows]) for s in (1, 2)]
    options = []
    for candidate in candidates():
        x = bank_features(grids, candidate)
        for alpha in (.01, .1, 1., 10., 100.):
            model = fit_readout(x[train], labels[train], alpha, True)
            results = [float(np.mean(predict(model, x[m]).argmax(1) == labels[m])) for m in masks]
            options.append({'recipe': candidate, 'alpha': alpha, 'validation_by_speed': results,
                            'worst_speed': min(results), 'mean': float(np.mean(results)), 'features': x.shape[1]})
    # Fixed in code before evaluating development: maximize the weaker speed,
    # then balanced mean; prefer fewer features and stronger regularization.
    key = lambda r: (r['worst_speed'], r['mean'], -r['features'], r['alpha'])
    chosen = max(options, key=key)
    fixed = max([r for r in options if r['recipe'] == candidates()[0]], key=key)
    models = {'p7': load_npz(args.parent/'neural_motion-readout.npz')}
    for name, choice in (('multispeed', chosen), ('fixed_retrained', fixed)):
        x = bank_features(grids, choice['recipe'])
        models[name] = fit_readout(x[train], labels[train], choice['alpha'], True)
    for name, model in models.items():np.savez_compressed(args.output/(name+'.npz'), **model)
    np.savez_compressed(args.output/'transform.npz', **arrays)
    validation = {}
    for name, model in models.items():
        choice = chosen if name == 'multispeed' else fixed
        x = bank_features(grids, choice['recipe'])
        validation[name] = {str(s): evaluate(predict(model, x[m]).argmax(1), labels[m], [r for r, keep in zip(rows, m) if keep]) for s, m in zip((1, 2), masks)}
    groups = balanced_orbit_seeds(1220000, 8)
    if set(groups) & set(parent['fit_groups']+parent['validation_groups']+parent['test_groups']):
        raise ValueError('new test group overlap')
    protocol = {'schema': 'flywire-multispeed-v1', 'parent': str(args.parent),
                'parent_protocol_sha256': sha(args.parent/'protocol.json'),
                'parent_selection_sha256': sha(args.parent/'selection.json'),
                'development': records, 'chosen': chosen, 'fixed_retrained': fixed,
                'grid_recipe': recipe, 'test_groups': groups, 'conditions': CONDITIONS, 'models': MODELS,
                'selection_rule': 'max worst speed accuracy, mean, fewer features, stronger alpha; fit feature-reversal augmentation',
                'fit_groups': parent['fit_groups'], 'validation_groups': parent['validation_groups'],
                'test_scope': '128 clips per speed/static, eight complete 2x2-phase orbits; 32 shared-start clips per lesion, paired to speed-2 subset',
                'speeds_fit': [1, 2], 'speeds_test': [1, 2, 3], 'primary': 'balanced accuracy across speeds 1 and 2, grouped by whole orbit',
                'readouts': {n: sha(args.output/(n+'.npz')) for n in MODELS},
                'transform_sha256': sha(args.output/'transform.npz'),
                'source_sha256': {n: sha(Path(__file__).with_name(n)) for n in SOURCES},
                'control_scope': 'static actual simulations reused for identical images; temporal shuffle is recorded-count ablation; target and matched cuts on shared-start subset'}
    save(args.output/'protocol.json', protocol)
    save(args.output/'selection.json', {'test_clips_executed': 0, 'protocol_sha256': sha(args.output/'protocol.json'),
         'candidates': options, 'validation': validation, 'chosen': chosen, 'fixed_retrained': fixed})
    print({'chosen': chosen, 'fixed_retrained': fixed, 'validation': {n: {s: v['accuracy'] for s, v in d.items()} for n, d in validation.items()}}, flush=True)


def features_for(neural, arrays, protocol, order=None):
    grids = grids_for(neural[None], arrays, protocol['grid_recipe'], order)
    fixed = bank_features(grids, {'mode': 'concat', 'speeds': [1]})[0]
    return {'p7': fixed, 'fixed_retrained': fixed, 'multispeed': bank_features(grids, protocol['chosen']['recipe'])[0]}


def statistics(rows, features, models, shuffled):
    labels = np.array([DIRECTIONS.index(r['kind']) for r in rows])
    preds = {n: predict(m, features[n]).argmax(1) for n, m in models.items()}
    results, gain = {}, {}
    for c in CONDITIONS:
        mask = np.array([r['condition'] == c for r in rows])
        rr = [r for r, keep in zip(rows, mask) if keep]
        results[c] = {n: evaluate(pred[mask], labels[mask], rr) for n, pred in preds.items()}
        gain[c] = paired_interval(preds['multispeed'][mask] == labels[mask], preds['p7'][mask] == labels[mask], rr)
    primary = np.array([r['condition'] in ('speed_1', 'speed_2') for r in rows])
    rr = [r for r, keep in zip(rows, primary) if keep]
    primary_results = {n: evaluate(pred[primary], labels[primary], rr) for n, pred in preds.items()}
    primary_gain = paired_interval(preds['multispeed'][primary] == labels[primary], preds['p7'][primary] == labels[primary], rr)
    subset = np.array([r['condition'] == 'speed_2' and tuple(r['phase_index']) == phase_for(r['kind']) for r in rows])
    rr = [r for r, keep in zip(rows, subset) if keep]
    subset_results = {n: evaluate(pred[subset], labels[subset], rr) for n, pred in preds.items()}
    effects = {}
    for c in ('cut_speed_2', 'matched_speed_2'):
        mask = np.array([r['condition'] == c for r in rows])
        effects[c] = paired_interval(preds['multispeed'][subset] == labels[subset], preds['multispeed'][mask] == labels[mask], rr)
    time_results = {}
    for speed in (1, 2, 3):
        mask = np.array([r['condition'] == f'speed_{speed}' for r in rows])
        indices = np.flatnonzero(mask); rr = [rows[i] for i in indices]
        sp = predict(models['multispeed'], shuffled[indices]).argmax(1)
        time_results[str(speed)] = {'result': evaluate(sp, labels[mask], rr),
            'intact_minus_shuffle': paired_interval(preds['multispeed'][mask] == labels[mask], sp == labels[mask], rr)}
    return {'conditions': results, 'paired_vs_p7': gain, 'primary': primary_results, 'primary_gain': primary_gain,
            'lesion_subset_baseline': subset_results, 'lesion_effects': effects, 'time_shuffle': time_results}


def run(args):
    root = args.output; p = read(root/'protocol.json'); parent = Path(p['parent'])
    if read(root/'selection.json')['protocol_sha256'] != sha(root/'protocol.json'):
        raise ValueError('protocol changed')
    if any(sha(Path(__file__).with_name(n)) != h for n, h in p['source_sha256'].items()):
        raise ValueError('frozen sources changed')
    if any(sha(root/(n+'.npz')) != h for n, h in p['readouts'].items()) or sha(root/'transform.npz') != p['transform_sha256']:
        raise ValueError('frozen transforms changed')
    arrays = load_npz(root/'transform.npz'); models = {n: load_npz(root/(n+'.npz')) for n in MODELS}
    pp = read(parent/'protocol.json'); template = read(args.artifact/'model.json')
    runners = {'intact': NativeRecorder(args.artifact, root/'intact', parent)}
    for name, mask in (('cut_input', pp['input_indices']), ('matched_cut', pp['matched_cut_indices'])):
        runners[name] = NativeRecorder(args.artifact, root/name, parent, intervene(template, mask), name)
    blank = np.full((40,48,48), .5, dtype=np.float32)
    blanks = {n: r.run(blank, 'blank') for n, r in runners.items()}
    np.testing.assert_array_equal(blanks['intact'][0], arrays['blank'])
    np.savez_compressed(root/'blanks.npz', **{n: v[0] for n, v in blanks.items()})
    save(root/'blank-events.json', {n: v[2] for n, v in blanks.items()})
    neural, inputs, rows, shuffled = [], [], [], []
    features = {n: [] for n in MODELS}; cache = {}; audits = []; start = time.perf_counter()
    for condition in CONDITIONS:
        speed = int(condition[-1]) if condition.startswith('speed_') else 2
        runner_name = {'cut_speed_2': 'cut_input', 'matched_speed_2': 'matched_cut'}.get(condition, 'intact')
        for group in p['test_groups']:
            endpoints = {k: [] for k in DIRECTIONS}
            for metadata, _ in iter_orbit(group, 2.):
                if runner_name != 'intact' and tuple(metadata['phase_index']) != phase_for(metadata['kind']):continue
                frames = speed_movie(metadata['kind'], group, metadata['phase_index'], speed)
                if condition == 'static_first':frames = np.repeat(frames[:1], 40, axis=0)
                if not np.array_equal(frames[0], frames[-1]):raise ValueError('cycle not closed')
                endpoints[metadata['kind']].append(digest(frames[0]))
                key = f'{condition}-{group}-{metadata["phase_index"][0]}-{metadata["phase_index"][1]}-{metadata["kind"]}'
                cache_key = (runner_name, digest(frames))
                reused = cache_key in cache
                if reused:nn, ee, rr = cache[cache_key]
                else:
                    nn, ee, rr = runners[runner_name].run(frames, key)
                    cache[cache_key] = (nn, ee, rr)
                if runner_name == 'cut_input':np.testing.assert_array_equal(nn, blanks['cut_input'][0])
                ff = features_for(nn, arrays, p)
                for n in MODELS:features[n].append(ff[n])
                order = np.arange(60)
                order[10:50] = np.random.default_rng(np.random.SeedSequence([group, *metadata['phase_index'], 814])).permutation(order[10:50])
                shuffled.append(features_for(nn, arrays, p, order)['multispeed'])
                neural.append(nn); inputs.append(ee)
                rows.append({**metadata, **rr, 'condition': condition, 'speed': speed, 'reused': reused, 'runner': runner_name})
            if any(sorted(v) != sorted(endpoints['right']) for v in endpoints.values()):raise ValueError('endpoint multiset differs')
            audits.append({'condition': condition, 'group': group, 'endpoint_equal': True})
            save(root/'rows.json', rows)
            np.savez_compressed(root/'features.npz', neural=np.array(neural), encoded=np.array(inputs),
                                shuffled=np.array(shuffled), **{n: np.array(v) for n, v in features.items()})
            print({'condition': condition, 'rows': len(rows), 'native_runs': sum(r.runs for r in runners.values()), 'seconds': time.perf_counter()-start}, flush=True)
    # Actual repeat/restoration, including all network events and state hashes.
    first = rows[0]
    _, _, restored = runners['intact'].run(speed_movie(first['kind'], first['group'], first['phase_index'], 1), 'restoration')
    reset = all(first[k] == restored[k] for k in ('events_sha256', 'states_sha256', 'input_sha256'))
    if not reset:raise ValueError('restoration failed')
    save(root/'restoration.json', restored)
    save(root/'endpoint-audit.json', audits)
    summary = statistics(rows, {n: np.array(v) for n, v in features.items()}, models, np.array(shuffled))
    report = {'schema': p['schema'], 'status': 'complete', 'protocol_sha256': sha(root/'protocol.json'),
              'selection_sha256': sha(root/'selection.json'), 'features_sha256': sha(root/'features.npz'),
              'rows_sha256': sha(root/'rows.json'), 'native_runs': sum(r.runs for r in runners.values()),
              'reset_exact': reset, 'target_cut_equals_blank': True, 'seconds': time.perf_counter()-start, **summary}
    save(root/'report.json', report)
    print({'primary': {n: r['accuracy'] for n, r in report['primary'].items()}, 'gain': report['primary_gain'],
           'speeds': {c: {n: r['accuracy'] for n, r in d.items()} for c, d in report['conditions'].items()}}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    select_parser = sub.add_parser('select')
    for name in ('parent', 'output'):select_parser.add_argument('--'+name, type=Path, required=True)
    select_parser.add_argument('--development', type=Path, nargs=2, required=True)
    run_parser = sub.add_parser('test')
    for name in ('output', 'artifact'):run_parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    (select if args.stage == 'select' else run)(args)
