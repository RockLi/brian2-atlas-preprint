"""Recompute frozen-decoder audit scores and verify all planned conditions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .pilot import DIRECTIONS, scores
from .refinement import linear_scores
from .run_experiment import save
from .robustness_audit import audit_movie, centroid_rule
from .direction_study import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    root = parser.parse_args().directory
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    p, report, rows = [read(root/(n+'.json')) for n in ('protocol', 'report', 'rows')]
    parent = Path(p['parent_study'])
    pp = read(parent/'protocol.json')
    prior_groups = set()
    path = parent
    while (path/'protocol.json').exists():
        previous = read(path/'protocol.json')
        for name in ('fit_groups', 'validation_groups', 'test_groups'):
            prior_groups.update(previous.get(name, []))
        if not previous.get('parent_study'):
            break
        path = Path(previous['parent_study'])
    checks = {
        'complete': report['status'] == 'complete',
        'protocol_hash': sha(root/'protocol.json') == report['protocol_sha256'],
        'features_hash': sha(root/'features.npz') == report['features_sha256'],
        'parent_report_hash': sha(parent/'report.json') == p['parent_report_sha256'],
        'no_previous_group_reused': not set(p['groups']) & prior_groups,
        'frozen_model_and_encoder': all(p[k] == pp[k] for k in ('base_sha256', 'binary_sha256', 'config', 'input_mode', 'readout_indices', 'input_indices')),
        'sources_unchanged': all(sha(Path(__file__).with_name(n)) == h for n, h in p['sources_sha256'].items()),
        'every_condition_reported': set(report['conditions']) == set(p['conditions']),
        'row_count': len(rows) == len(p['conditions'])*p['clips_per_condition'] == report['new_simulations'],
        'no_duplicate_trials': len({(r['condition'], r['group'], r['kind']) for r in rows}) == len(rows),
        'all_groups_paired': all([r['kind'] for r in rows if r['condition'] == c and r['group'] == g] == list(DIRECTIONS) for c in p['conditions'] for g in p['groups']),
        'readout_excludes_inputs': not set(p['readout_indices']) & set(p['input_indices']),
    }
    identity = read(root/'execution/identity.json')
    checks['actual_runner_identity'] = all(identity[k] == p[k] for k in ('base_sha256', 'binary_sha256')) and identity['mutable_population'] == 'visual_input'
    checks['movies_and_shortcuts_regenerated'] = True
    for row in rows:
        frames = audit_movie(row['kind'], row['group'], row['condition'])
        checks['movies_and_shortcuts_regenerated'] &= (hashlib.sha256(frames.tobytes()).hexdigest() == row['movie_sha256'] and
            centroid_rule(frames) == row['first_rule'] and centroid_rule(frames, last=True) == row['last_rule'])
    models = {}
    for name, expected in p['readouts_sha256'].items():
        path = parent/(name+'-readout.npz')
        checks[name+'_unchanged'] = sha(path) == expected
        checks[name+'_weights_precede_audit'] = path.stat().st_mtime_ns < (root/'protocol.json').stat().st_mtime_ns
        with np.load(path, allow_pickle=False) as m:
            models[name] = dict(m)
    details = {}
    with np.load(root/'features.npz', allow_pickle=False) as features:
        labels = np.asarray([DIRECTIONS.index(r['kind']) for r in rows])
        checks['labels_from_rows'] = np.array_equal(features['labels'], labels)
        checks['valid_counts'] = all(np.isfinite(features[n]).all() and np.all(features[n] >= 0) for n in models)
        for condition, result in report['conditions'].items():
            mask = np.array([r['condition'] == condition for r in rows])
            selected_rows = [r for r in rows if r['condition'] == condition]
            details[condition] = {
                'group_balance': len(selected_rows) == p['clips_per_condition'] and sum(r['polarity'] == 'bright' for r in selected_rows) == len(selected_rows)//2,
                'control_label_explicit': result['source_label_control'] == (condition.startswith('static') or condition == 'scrambled_middle'),
                'all_five_readouts': set(result['readouts']) == {'neural', 'input_neurons', 'encoded_input', 'first_frame_rule', 'last_frame_rule'},
            }
            for name, metric in result['readouts'].items():
                if name in models:
                    m, x = models[name], features[name][mask]
                    pred = (linear_scores(m, x) if 'coefficients' in m else scores(m, x)).argmax(1)
                else:
                    pred = np.asarray([r['first_rule' if name == 'first_frame_rule' else 'last_rule'] for r in selected_rows])
                confusion = np.bincount(labels[mask]*4 + pred, minlength=16).reshape(4, 4)
                details[condition][name+'_prediction'] = pred.tolist() == metric['predictions']
                details[condition][name+'_confusion'] = confusion.tolist() == metric['confusion']
                details[condition][name+'_accuracy'] = float(np.mean(pred == labels[mask])) == metric['accuracy']
                details[condition][name+'_all_statistics'] = evaluate(pred, labels[mask], selected_rows) == metric
    passed = all(checks.values()) and all(all(v.values()) for v in details.values())
    save(root/'verification.json', {'all_passed': passed, 'checks': checks, 'conditions': details})
    print(json.dumps({'all_passed': passed, 'checks': checks}))
    if not passed:
        raise RuntimeError('robustness audit verification failed')


if __name__ == '__main__':
    main()
