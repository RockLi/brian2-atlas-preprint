"""Independently recompute sparse-study predictions and audit holdout provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .pilot import DIRECTIONS, scores
from .refinement import linear_scores
from .run_experiment import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    root = parser.parse_args().directory
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    protocol, selection, report, rows = [read(root / (n + '.json')) for n in ('protocol', 'selection', 'report', 'rows')]
    previous = Path(protocol['parent_study'])
    parent = read(previous / 'protocol.json')
    ancestor = read(Path(parent['parent_study']) / 'protocol.json')
    prior_groups = set(sum((p[k] for p in (parent, ancestor) for k in ('fit_groups', 'validation_groups', 'test_groups')), []))
    checks = {
        'complete': report['status'] == 'complete',
        'no_test_at_selection': selection['test_clips_executed'] == 0,
        'protocol_hash': sha(root / 'protocol.json') == report['protocol_sha256'] == selection['protocol_sha256'],
        'selection_hash': sha(root / 'selection.json') == report['selection_sha256'],
        'feature_hash': sha(root / 'features.npz') == report['features_sha256'],
        'parent_hash': sha(previous / 'report.json') == protocol['parent_report_sha256'],
        'brain_and_binary_unchanged': all(protocol[k] == parent[k] for k in ('base_sha256', 'binary_sha256', 'config', 'input_mode')),
        'test_groups_exclude_all_prior_cohorts': not set(protocol['test_groups']) & prior_groups,
        'source_hashes': all(sha(Path(__file__).with_name(n)) == h for n, h in protocol['sources_sha256'].items()),
        'row_count': len(rows) == 320,
        'test_rows_match_protocol': sorted({r['group'] for r in rows[192:]}) == protocol['test_groups'],
        'balanced_test_polarity': sum(r['polarity'] == 'bright' for r in rows[192:]) == 64,
        'readout_excludes_inputs': not set(protocol['readout_indices']) & set(protocol['input_indices']),
        'complete_direction_groups': all([r['kind'] for r in rows if r['group'] == g] == list(DIRECTIONS)
            for g in protocol['fit_groups'] + protocol['validation_groups'] + protocol['test_groups']),
    }
    details = {}
    with np.load(root / 'features.npz', allow_pickle=False) as features, np.load(previous / 'features.npz', allow_pickle=False) as prior:
        checks['development_features_unchanged'] = all(np.array_equal(features[k], prior[k]) for k in features.files if k.startswith('development_'))
        for name, expected in report['readouts'].items():
            path = root / (name + '-readout.npz')
            result = {'weights_hash': sha(path) == expected['readout_sha256'],
                      'selection_fields_unchanged': all(expected[k] == v for k, v in selection['readouts'][name].items()),
                      'weights_saved_before_test_lock': path.stat().st_mtime_ns <= (root / 'selection.json').stat().st_mtime_ns}
            with np.load(path, allow_pickle=False) as model:
                if name == 'neural':
                    used = np.flatnonzero(np.any(model['coefficients'].reshape(8, -1, 4), axis=(0, 2)))
                    result['selected_support_only'] = set(used) <= set(model['selected_cells']) and len(model['selected_cells']) == report['selected_cells']
                if name == 'old_neural':
                    with np.load(previous / 'neural-readout.npz', allow_pickle=False) as old:
                        result['baseline_arrays_unchanged'] = all(np.array_equal(model[k], old[k]) for k in old.files)
                for split in ('validation', 'test'):
                    prefix = 'development' if split == 'validation' else 'test'
                    key = prefix + '_' + ('neural' if name in ('old_neural', 'labels_shuffled') else name)
                    x = features[key][128:] if split == 'validation' else features[key]
                    labels = features['development_labels'][128:] if split == 'validation' else features['test_labels']
                    prediction = (linear_scores(model, x) if 'coefficients' in model else scores(model, x)).argmax(1)
                    result[split + '_predictions'] = prediction.tolist() == expected[split]['predictions']
                    result[split + '_accuracy'] = float(np.mean(prediction == labels)) == expected[split]['accuracy']
            details[name] = result
    passed = all(checks.values()) and all(all(v.values()) for v in details.values())
    save(root / 'verification.json', {'all_passed': passed, 'checks': checks, 'readouts': details})
    print(json.dumps({'all_passed': passed, 'checks': checks, 'readouts': details}))
    if not passed:
        raise RuntimeError('sparse verification failed')


if __name__ == '__main__':
    main()
