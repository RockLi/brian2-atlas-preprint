"""Frozen R-case fixture preparation, one case/seed on the authorized host."""
import argparse
import json
from pathlib import Path
import socket
from generate_dense_large import ROOT, SEEDS, digest, write_json

CONTRACT = ROOT/'protocol/recurrent-fixture-r1.json'
CASE_SOURCE = ROOT/'protocol/finite-engine-cases.json'


def cases():
    selected = [row for row in json.loads(CASE_SOURCE.read_text())['cases'] if row['case_id'].startswith('R-')]
    if len(selected) != 9:
        raise ValueError('Frozen recurrent denominator must contain exactly nine cases')
    return {row['case_id']: row for row in selected}


def paths(case_id, seed):
    directory = ROOT/'fixtures/recurrent-r1'/case_id
    return directory/f'seed-{seed}.json', directory/f'seed-{seed}.manifest.json'


def contract():
    value = json.loads(CONTRACT.read_text())
    if value['status'] != 'frozen_before_generation_and_any_R_execution':
        raise ValueError('Recurrent rules are not frozen for execution')
    return value


def prepare(case_id, seed):
    if socket.gethostname() != 'rock-mac-studio-1.local':
        raise RuntimeError('Recurrent fixture generation is restricted to 100.90.28.27')
    import numpy as np
    policy = contract()
    spec = cases()[case_id]
    path, manifest_path = paths(case_id, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = dict(contract_sha256=digest(CONTRACT), case_source_sha256=digest(CASE_SOURCE), generator_sha256=digest(__file__))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity'] != identity or manifest['case'] != spec or digest(path) != manifest['arrays_sha256']:
            raise ValueError('Existing fixture identity changed; preserve and investigate it')
        return manifest_path
    if path.exists():
        raise FileExistsError('A partial fixture without manifest exists; preserve it rather than silently regenerating')
    seed_rng = lambda *parts: np.random.default_rng(np.random.SeedSequence([seed, *parts]))
    n, b, ticks, nin, nout, density = (spec[key] for key in ('hidden', 'B', 'T', 'input', 'output', 'density'))
    # Per-sample streams preserve prefixes when T changes, independent of N/p.
    inputs = [seed_rng(101, sample).choice([0, 1, 2], size=(ticks, nin), p=[.9, .095, .005]).tolist() for sample in range(b)]
    w0 = seed_rng(303, n).normal(0., np.sqrt(2/nin), size=(nin, n)).ravel().tolist()
    w1 = seed_rng(304, n).normal(0., np.sqrt(2/n), size=(n, nout)).ravel().tolist()
    projections = []
    for layer, (source_count, target_count) in enumerate(((nin, n), (n, nout))):
        projections.append(dict(source_layer=layer, target_layer=layer+1, parameter_count=source_count*target_count,
                                sources=[i for j in range(target_count) for i in range(source_count)],
                                targets=[j for j in range(target_count) for i in range(source_count)],
                                parameter_ids=[i*target_count+j for j in range(target_count) for i in range(source_count)]))
    sources, targets, recurrent = [], [], []
    indegrees, self_edges = [], 0
    scale = np.sqrt(2/max(1, density*n))
    for target in range(n):
        mask = seed_rng(202, n, target).random(n) < density
        selected = np.flatnonzero(mask)
        candidates = seed_rng(305, n, target).normal(0., 1., size=n)
        sources.extend(selected.tolist())
        targets.extend([target]*len(selected))
        recurrent.extend((candidates[mask]*scale).tolist())
        indegrees.append(len(selected))
        self_edges += int(mask[target])
    edges = len(recurrent)
    projections.append(dict(source_layer=1, target_layer=1, parameter_count=edges,
                            sources=sources, targets=targets, parameter_ids=list(range(edges))))
    case = dict(id=case_id, seed=seed, sizes=[nin, n, nout], inputs=inputs,
                labels=(np.arange(b)%nout).tolist(), weights=[w0, w1, recurrent], projections=projections,
                beta=.95, theta=1., workload_ids=spec['workload_ids'])
    write_json(path, case)
    manifest = dict(schema='recurrent-common-arrays-v1', seed=seed, case=spec, identity=identity,
                    numpy_version=np.__version__, arrays_sha256=digest(path), arrays_bytes=path.stat().st_size,
                    recurrent_edges=edges, recurrent_self_edges=self_edges, actual_density=edges/(n*n),
                    indegrees=indegrees, projection_edges=[len(p['sources']) for p in projections],
                    parameter_banks=[len(bank) for bank in case['weights']],
                    shared_array_scope='identical case JSON for every engine; temporal prefix/network coupling is fixed by contract',
                    max_qualification_rss_bytes=policy['queue']['supervisor_rss_guard_bytes'])
    write_json(manifest_path, manifest)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-id', choices=tuple(cases()), required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    manifest = prepare(args.case_id, args.seed)
    print(json.dumps(dict(manifest=str(manifest), sha256=digest(manifest))), flush=True)


if __name__ == '__main__':
    main()
