"""Create the frozen E1-large common arrays only on 100.90.28.27.

This generator never creates a reduced batch/time variant. Existing arrays are
authoritative when the manifest and every file digest match the frozen rule.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 23, 37, 51, 71)
RULE = dict(sizes=[512, 1024, 1024, 20], B=32, T=128,
            input_values=[0, 1, 2], input_probabilities=[.9, .095, .005],
            weight_distribution='independent Normal(0,sqrt(2/fan_in)) per bank',
            labels='arange(B)%C', beta=.95, theta=1., seeds=list(SEEDS),
            precision='arrays cast to IEEE f64 in compute; integer count JSON is lossless',
            numpy_rng='default_rng PCG64; draw inputs then banks in network order')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')


def prepare():
    if socket.gethostname() != 'rock-mac-studio-1.local':
        raise RuntimeError('E1-large generation is restricted to the user-selected 100.90.28.27 host')
    import numpy as np
    directory = ROOT/'fixtures/e1-large'
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['rule'] != RULE or set(manifest['files']) != {f'seed-{seed}.json' for seed in SEEDS}:
            raise ValueError('Existing E1-large rule or seed denominator differs; do not overwrite')
        for name, identity in manifest['files'].items():
            if digest(directory/name) != identity['sha256']:
                raise ValueError(f'Existing common array checksum differs: {name}')
        return manifest_path
    if any(directory.iterdir()):
        raise FileExistsError('Incomplete prior E1-large preparation must be preserved and inspected')
    manifest = dict(schema='e1-large-common-arrays-v1', rule=RULE, numpy_version=np.__version__,
                    generator_sha256=digest(__file__), files={})
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        sizes = RULE['sizes']
        inputs = rng.choice(RULE['input_values'], size=(RULE['B'], RULE['T'], sizes[0]), p=RULE['input_probabilities'])
        weights = [rng.normal(0., np.sqrt(2/a), size=(a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        case = dict(id=f'E1-large-seed-{seed}', seed=seed, sizes=sizes, inputs=inputs.tolist(),
                    labels=(np.arange(RULE['B']) % sizes[-1]).tolist(), weights=[w.ravel().tolist() for w in weights],
                    beta=RULE['beta'], theta=RULE['theta'])
        path = directory/f'seed-{seed}.json'
        write_json(path, case)
        manifest['files'][path.name] = dict(sha256=digest(path), bytes=path.stat().st_size)
        print(json.dumps(dict(seed=seed, **manifest['files'][path.name])), flush=True)
        del inputs, weights, case
    write_json(manifest_path, manifest)
    return manifest_path


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    path = prepare()
    print(json.dumps(dict(manifest=str(path), sha256=digest(path))), flush=True)
