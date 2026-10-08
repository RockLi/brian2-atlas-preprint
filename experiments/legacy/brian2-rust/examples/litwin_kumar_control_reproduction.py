"""Export and reproduce complete control evidence after performance measurements finish."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

CONDITIONS = {'full': None, 'no_stimulation': 'stimulation',
              'no_istdp': 'inhibitory_plasticity', 'no_normalization': 'normalization'}
SEEDS = [20260906, 20260907, 20260908]
INITIAL_FIELDS = ['ee_i', 'ee_j', 'ee_initial', 'membership_e', 'membership_i']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def job_map(path):
    jobs = json.loads(path.read_text())
    keys = [(j['seed'], j['condition']) for j in jobs]
    if len(keys) != len(set(keys)) or set(keys) != {(s, c) for s in SEEDS for c in CONDITIONS}:
        raise ValueError('complete unique twelve-condition manifest required')
    if any(Path(j['label']).name != j['label'] for j in jobs):
        raise ValueError('job labels must be directory names')
    return {(j['seed'], j['condition']): j['label'] for j in jobs}


def export(science_root, output):
    results = science_root/'results'
    suite, analysis = results/'science', results/'science-analysis'
    jobs = job_map(suite/'science_jobs.json')
    proof_path = results/'control-initial-structure-review.json'
    proof = json.loads(proof_path.read_text())
    checks = {(r['seed'], r['condition']): r for r in proof['checks']}
    if proof.get('all_passed') is not True or set(checks) != set(jobs) or len(proof['checks']) != 12:
        raise ValueError('complete raw initial-structure proof required')
    for key, label in jobs.items():
        if digest(suite/label/'result.json') != checks[key]['result_sha256']:
            raise ValueError('run report changed since raw structure audit')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(suite/'science_jobs.json', output/'science_jobs.json')
    shutil.copy2(proof_path, output/'initial-structure.json')
    shutil.copy2(analysis/'report.json', output/'analysis-report.json')
    shutil.copy2(analysis/'seed_metrics.csv', output/'seed_metrics.csv')
    for key, label in jobs.items():
        destination = output/label
        destination.mkdir()
        shutil.copy2(suite/label/'result.json', destination/'result.json')
        for name in ['activity_source.npz', 'connectivity_source.npz', 'weight_trajectory.csv']:
            shutil.copy2(analysis/label/name, destination/name)
        with np.load(suite/label/'activity.npz') as source, np.load(suite/label/'state.npz') as state:
            arrays = {k: source[k] for k in [*INITIAL_FIELDS, 'ee_final']}
            arrays['ie_final'] = state['ie_w']
            np.savez_compressed(destination/'raw_weights.npz', **arrays)
        episodes = analysis/f'reactivation-seed-{key[0]}-{key[1]}.json'
        shutil.copy2(episodes, destination/'episodes.json')
    manifest = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(dict(files=manifest,
        scope='Original run reports; analyzed activity arrays; raw EE topology/initial/final weights, final IE weights and memberships. Long voltage histories excluded.'), indent=2)+'\n')
    return {'exported_runs': len(jobs), 'files': len(manifest)}


def verify(evidence, output):
    from litwin_kumar_analysis import (population_conditioned_activity,
                                      reactivation_episodes, selective_activity)

    manifest = json.loads((evidence/'manifest.json').read_text())['files']
    required = {'science_jobs.json', 'initial-structure.json', 'analysis-report.json', 'seed_metrics.csv'}
    if not required <= set(manifest):
        raise ValueError('manifest omits required root evidence')
    for relative, expected in manifest.items():
        path = evidence/relative
        if not path.resolve().is_relative_to(evidence.resolve()) or digest(path) != expected:
            raise ValueError(f'source checksum mismatch: {relative}')
    jobs = job_map(evidence/'science_jobs.json')
    required.update(f'{label}/{name}' for label in jobs.values() for name in
                    ['result.json', 'activity_source.npz', 'connectivity_source.npz',
                     'weight_trajectory.csv', 'raw_weights.npz', 'episodes.json'])
    if set(manifest) != required:
        raise ValueError('manifest must cover every required source file')
    report = json.loads((evidence/'analysis-report.json').read_text())
    expected_rows = {(r['seed'], r['condition']): r for r in report['seed_metrics']}
    if set(expected_rows) != set(jobs) or len(report['seed_metrics']) != 12:
        raise ValueError('complete unique source metric rows required')
    proof = json.loads((evidence/'initial-structure.json').read_text())
    initial = {(r['seed'], r['condition']): r for r in proof['checks']}
    if proof.get('all_passed') is not True or len(proof['checks']) != 12 or set(initial) != set(jobs):
        raise ValueError('complete raw structure proof required')
    checks = []
    for seed in SEEDS:
        runs, data = {}, {}
        for condition in CONDITIONS:
            key = seed, condition
            path = evidence/jobs[key]
            if digest(path/'result.json') != initial[key]['result_sha256']:
                raise ValueError('archived run report differs from structure proof')
            runs[condition] = json.loads((path/'result.json').read_text())
            with np.load(path/'activity_source.npz') as source:
                data[condition] = {k: source[k] for k in
                    ['time_s', 'coverage', 'assembly_rates_hz', 'population_rate_hz']}
        config = runs['full']['configuration']
        if any(config.get(k) != v for k, v in dict(seed=seed, scale=1, mode='learn',
                ne=4000, ni=1000, dt_ms=.1, duration_s=2610, warmup_s=10,
                training_s=1600, spontaneous_s=1000).items()):
            raise ValueError('declared full-scale protocol required')
        time = data['full']['time_s']
        if any(not np.array_equal(d['time_s'], time) for d in data.values()):
            raise ValueError('control time grids differ')
        common = np.logical_and.reduce([d['coverage'] for d in data.values()])
        baseline = common & (time >= min(1., config['warmup_s']/2)) & (time < config['warmup_s'])
        spontaneous = common & (time >= config['warmup_s']+config['training_s'])
        bin_s = float(time[1]-time[0])
        for condition, toggle in CONDITIONS.items():
            key = seed, condition
            run, item, path = runs[condition], data[condition], evidence/jobs[key]
            expected_config = dict(config)
            if toggle:
                expected_config[toggle] = False
            if (run['configuration'] != expected_config or run['backend'] != 'rust'
                    or run['biological_seconds'] != 2610 or not run['complete_training_protocol']):
                raise ValueError('unmatched completed control protocol')
            with np.load(path/'raw_weights.npz') as source:
                raw = {k: source[k] for k in source.files}
            for field in INITIAL_FIELDS:
                a = raw[field]
                actual = dict(shape=list(a.shape), dtype=str(a.dtype),
                              sha256=hashlib.sha256(a.tobytes()).hexdigest())
                if actual != initial[key]['initial_arrays'][field]:
                    raise ValueError('raw initial array differs from remote byte comparison')
            membership, pre, post, weights = (raw[k] for k in ['membership_e', 'ee_i', 'ee_j', 'ee_final'])
            within = np.zeros(len(weights), dtype=bool)
            for members in membership:
                within |= members[pre] & members[post]
            within_mean, between_mean = float(weights[within].mean()), float(weights[~within].mean())
            for name, value in [('within_mean_pf', within_mean), ('between_mean_pf', between_mean)]:
                np.testing.assert_allclose(value, run['weights'][name], rtol=1e-12, atol=1e-12)
            adjusted, fractions = population_conditioned_activity(item['assembly_rates_hz'],
                item['population_rate_hz'], membership.sum(axis=1), config['ne'], bin_s, baseline)
            conditioned = selective_activity(adjusted, np.zeros_like(time), baseline, spontaneous)
            events, episode_metrics = reactivation_episodes(adjusted, np.zeros_like(time),
                time, baseline, spontaneous, common)
            saved_events = json.loads((path/'episodes.json').read_text())
            if events != saved_events['episodes']:
                raise ValueError('episode reconstruction differs from archived events')
            np.testing.assert_allclose(fractions, saved_events['baseline_spike_fractions'], rtol=1e-12, atol=1e-12)
            observed = dict(within_between_ratio=within_mean/between_mean,
                upper_bound_fraction=float(np.mean(weights >= 21.4-1e-12)),
                inhibitory_upper_fraction=float(np.mean(raw['ie_final'] >= 243.-1e-12)),
                common_baseline_seconds=float(baseline.sum()*bin_s),
                common_spontaneous_seconds=float(spontaneous.sum()*bin_s),
                population_conditioned_excess_change=conditioned['selective_excess_change_hz'],
                population_conditioned_occupancy=conditioned['selective_occupancy'],
                **episode_metrics, **selective_activity(item['assembly_rates_hz'], item['population_rate_hz'], baseline, spontaneous))
            if set(observed) != set(expected_rows[key])-{'seed', 'condition'}:
                raise ValueError('metric reconstruction omits report fields')
            errors = []
            for name, value in observed.items():
                expected = expected_rows[key][name]
                if value is None or expected is None:
                    if value is not expected:
                        raise ValueError('missing metric differs from source')
                else:
                    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12,
                                               err_msg=f'{key}/{name}')
                    errors.append(abs(value-expected))
            checks.append(dict(seed=seed, condition=condition, metrics=len(observed),
                               maximum_absolute_error=max(errors), episode_count=len(events)))
    result = dict(all_passed=True, checks=checks, source_manifest_sha256=digest(evidence/'manifest.json'),
        scope='Metrics and episodes recalculated from archived activity using the analysis functions; EE means and saturation fractions recalculated from raw final weights. This is source-data reproduction, not an independent scientific model implementation.')
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return {'runs': len(checks), 'metrics': sum(r['metrics'] for r in checks),
            'maximum_absolute_error': max(r['maximum_absolute_error'] for r in checks)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('export')
    p.add_argument('--science-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('verify')
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args())
    print(json.dumps(globals()[args.pop('command')](**args)))
