"""Post-hoc descriptive audit of all frozen CPU v1 cases; no parameter fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SEEDS = [11, 23, 47, 83, 131]
CONDITIONS = ['paired', 'frozen', 'teaching_off', 'shuffled_reward', 'reversal']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def diagnose(study, prepared, output):
    output.mkdir(parents=True, exist_ok=False)
    with np.load(prepared / 'plastic.npz') as edges:
        edge_sources = edges['source_index'].copy()
    projecting = np.unique(edge_sources)
    rows, memberships, provenance = [], [], []
    for seed in SEEDS:
        for condition in CONDITIONS:
            folder = study / f'seed-{seed}-{condition}'
            cfg = json.loads((folder / 'configuration.json').read_text())
            report = json.loads((folder / 'report.json').read_text())
            assert cfg['seed'] == seed and cfg['condition'] == condition
            assert cfg['scope'] == 'full' and cfg['duration_ms'] == 19100
            selected = [cfg['recorded_indices'].index(i) for i in cfg['mbon_indices']]
            with np.load(folder / 'snapshot.npz') as snapshot:
                spike_t, spike_i = snapshot['spike_t'], snapshot['spike_i']
                trace_t = snapshot['trace_t']
                trace = snapshot['neuron_trace'][:, selected, :]
            assert np.all(np.diff(spike_t) >= 0)
            spike_tick = np.rint(spike_t / .0001).astype(np.int64)
            trace_tick = np.rint(trace_t / .0001).astype(np.int64)
            activity = {block: np.zeros((2, cfg['neurons']), dtype=np.int64)
                        for block in ['acquisition', 'second']}
            for trial_id, (trial, old) in enumerate(zip(cfg['trials'], report['trials'], strict=True)):
                onset = int(round((trial['start_ms'] + 50) / .1))
                row = dict(seed=seed, condition=condition, trial_id=trial_id, **trial)
                for name, begin, end in [('baseline', onset - 500, onset),
                                         ('response', onset, onset + 2500)]:
                    left, right = np.searchsorted(spike_tick, [begin, end])
                    counts = np.bincount(spike_i[left:right], minlength=cfg['neurons'])
                    seconds = (end - begin) * .0001
                    rates = {}
                    for group, indices in [('mbon', cfg['mbon_indices']),
                                           ('kc', cfg['kc_indices']),
                                           ('input', cfg['stimulus_groups'][trial['cue']]),
                                           ('projecting_kc', projecting)]:
                        rates[group + '_hz'] = float(counts[indices].sum() / (len(indices) * seconds))
                        if name == 'response' and group != 'projecting_kc':
                            assert np.isclose(rates[group + '_hz'], old[group + '_hz'], rtol=0, atol=1e-12)
                    l, r = np.searchsorted(trace_tick, [begin, end])
                    v, ge, gi = trace[:, :, l:r]
                    rates.update(v_mean_mv=float(v.mean() * 1000), v_max_mv=float(v.max() * 1000),
                                 ge_mean=float(ge.mean()), gi_mean=float(gi.mean()),
                                 excitation_term_mv_per_s=float((-ge * v / .020).mean() * 1000),
                                 inhibition_term_mv_per_s=float((-gi * (v + .070) / .020).mean() * 1000))
                    row[name] = rates
                    if name == 'response' and trial['block'] in activity:
                        activity[trial['block']][trial['cue']] += counts
                rows.append(row)
            for block, counts in activity.items():
                active_a, active_b = counts[0] > 0, counts[1] > 0
                with np.load(folder / f'{block}-plastic.npz') as state:
                    gains = state['gain']
                for label, mask in [('A_only', active_a & ~active_b),
                                    ('B_only', active_b & ~active_a),
                                    ('both', active_a & active_b),
                                    ('neither', ~active_a & ~active_b)]:
                    edge_mask = mask[edge_sources]
                    memberships.append(dict(seed=seed, condition=condition, block=block, category=label,
                                            projecting_kc_count=int(mask[projecting].sum()),
                                            edge_count=int(edge_mask.sum()),
                                            gain_mean=float(gains[edge_mask].mean()) if edge_mask.any() else None))
            provenance.append(dict(seed=seed, condition=condition,
                                   files={name: sha(folder / name) for name in
                                          ['configuration.json', 'report.json', 'snapshot.npz',
                                           'acquisition-plastic.npz', 'second-plastic.npz']}))
            print(f'checked {seed} {condition}', flush=True)
    summary = []
    for condition in CONDITIONS:
        for block in ['pre', 'acquisition', 'post', 'second', 'final']:
            subset = [r for r in rows if r['condition'] == condition and r['block'] == block]
            summary.append(dict(condition=condition, block=block, trials=len(subset),
                                **{window: {key: float(np.mean([r[window][key] for r in subset]))
                                            for key in subset[0][window]} for window in ['baseline', 'response']}))
    result = dict(scope='Post-hoc descriptive diagnostics of all CPU v1 cases; no causal attribution or parameter tuning.',
                  projecting_kc_count=len(projecting), checked_trials=len(rows),
                  original_readouts_recounted_exact_within_1e_12=True,
                  source_sha256=sha(Path(__file__)), plan_sha256=sha(Path(__file__).with_name('DIAGNOSTIC_PLAN.md')),
                  prepared_manifest_sha256=sha(prepared / 'manifest.json'),
                  rows=rows, summary=summary, projection_activity=memberships, provenance=provenance)
    (output / 'diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Complete: {len(rows)} trials', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    diagnose(a.study.resolve(), a.prepared.resolve(), a.output.resolve())
