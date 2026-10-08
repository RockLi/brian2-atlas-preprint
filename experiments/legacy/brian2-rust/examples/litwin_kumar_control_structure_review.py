"""Check raw initial structure across the complete four-condition LK cohort."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def review(suite, output):
    conditions = {'full': None, 'no_stimulation': 'stimulation',
                  'no_istdp': 'inhibitory_plasticity', 'no_normalization': 'normalization'}
    jobs = json.loads((suite/'science_jobs.json').read_text())
    keys = [(j['seed'], j['condition']) for j in jobs]
    seeds = [20260906, 20260907, 20260908]
    required = {(s, c) for s in seeds for c in conditions}
    if len(keys) != len(set(keys)) or set(keys) != required:
        raise ValueError('complete unique three-seed four-condition manifest required')
    paths = {(j['seed'], j['condition']): suite/j['label'] for j in jobs}
    runs = {key: json.loads((path/'result.json').read_text()) for key, path in paths.items()}
    checks = []
    for seed in seeds:
        baseline = runs[seed, 'full']
        config = baseline['configuration']
        if (config['seed'] != seed or config['scale'] != 1 or config['mode'] != 'learn'
                or config['duration_s'] != 2610 or config['ne'] != 4000 or config['ni'] != 1000
                or any(config[k] is not True for k in conditions.values() if k)):
            raise ValueError('full-scale full-protocol baseline with all mechanisms required')
        reference_path = paths[seed, 'full']/'activity.npz'
        # Read only raw structure members, not retained spike/voltage histories.
        with np.load(reference_path) as reference:
            for condition, toggle in conditions.items():
                key = seed, condition
                run, path = runs[key], paths[key]
                expected = dict(config)
                if toggle:
                    expected[toggle] = False
                if (run['configuration'] != expected or run['backend'] != 'rust'
                        or run['biological_seconds'] != 2610
                        or run['complete_training_protocol'] is not True
                        or run['topology_edges'] != baseline['topology_edges']):
                    raise ValueError(f'unmatched or incomplete protocol: {key}')
                hashes = {}
                with np.load(path/'activity.npz') as candidate:
                    for field in ['ee_i', 'ee_j', 'ee_initial', 'membership_e', 'membership_i']:
                        a, b = reference[field], candidate[field]
                        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                            raise ValueError(f'initial structure differs: {key}/{field}')
                        hashes[field] = dict(shape=list(a.shape), dtype=str(a.dtype),
                                             sha256=hashlib.sha256(a.tobytes()).hexdigest())
                checks.append(dict(seed=seed, condition=condition,
                    result_sha256=hashlib.sha256((path/'result.json').read_bytes()).hexdigest(),
                    initial_arrays=hashes, all_projection_edge_counts=run['topology_edges']))
    result = dict(all_passed=True, checks=checks,
        scope='Raw EE edge and initial-weight arrays and both membership arrays are byte exact across conditions within each seed; other projection edge counts match. Final states are intentionally not compared for equality.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    result = review(**vars(parser.parse_args()))
    print(json.dumps({'all_passed': result['all_passed'], 'runs_checked': len(result['checks'])}))
