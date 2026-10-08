"""Check refinement selection, new holdout separation and saved predictions."""
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
    args = parser.parse_args()
    root = args.directory
    read = lambda name: json.loads((root / name).read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    protocol, initial, selection, report, rows = [read(n + '.json') for n in ('protocol', 'initial-protocol', 'selection', 'report', 'rows')]
    previous = Path(protocol['parent_study'])
    old_protocol = json.loads((previous / 'protocol.json').read_text())
    checks = {'complete': report['status'] == 'complete', 'no_test_at_selection': selection['test_clips_executed'] == 0,
              'protocol_hash': sha(root / 'protocol.json') == report['protocol_sha256'] == selection['protocol_sha256'],
              'selection_hash': sha(root / 'selection.json') == report['selection_sha256'],
              'feature_hash': sha(root / 'features.npz') == report['features_sha256'],
              'initial_test_groups_unchanged': initial['test_groups'] == protocol['test_groups'],
              'new_test_groups_not_previously_used': not set(protocol['test_groups']) & set(old_protocol['fit_groups'] + old_protocol['validation_groups'] + old_protocol['test_groups']),
              'sources': all(sha(Path(__file__).with_name(n)) == h for n, h in protocol['sources_sha256'].items()),
              'spatial_map_hash': sha(root / 'spatial-map.npz') == protocol['spatial_map_sha256'],
              'row_count': len(rows) == 320,
              'test_rows_match_planned_groups': sorted({r['group'] for r in rows[192:]}) == protocol['test_groups'],
              'balanced_test_polarity': sum(r['polarity'] == 'bright' for r in rows[192:]) == 64}
    checks['readout_excludes_input_neurons'] = not set(protocol['readout_indices']) & set(protocol['input_indices'])
    if protocol['input_mode'] == 'contrast':
        executions = read('executions.json')
        lookup = {(r['variant'], r['group'], r['kind']): r for r in executions}
        checks['paired_test_input_events_identical'] = all(
            lookup[protocol['selected_variant'], g, kind]['input_sha256'] == lookup['contrast16', g, kind]['input_sha256']
            for g in protocol['test_groups'] for kind in DIRECTIONS)
    # Each group has exactly the four directions in frozen order.
    checks['complete_direction_groups'] = all([r['kind'] for r in rows if r['group'] == g] == list(DIRECTIONS)
                                               for g in protocol['fit_groups'] + protocol['validation_groups'] + protocol['test_groups'])
    details = {}
    with np.load(root / 'features.npz', allow_pickle=False) as f:
        for name, expected in report['readouts'].items():
            path = root / (name + '-readout.npz')
            result = {'weights_hash': sha(path) == expected['readout_sha256'],
                      'pretest_selection_fields': all(expected[k] == v for k, v in selection['readouts'][name].items()),
                      'saved_before_test_lock': path.stat().st_mtime_ns <= (root / 'selection.json').stat().st_mtime_ns}
            with np.load(path, allow_pickle=False) as model:
                if name == 'neural' and str(model.get('recipe', '')) == 'fisher128_raw':
                    used = np.any(model['coefficients'].reshape(8, -1, 4) != 0, axis=(0, 2))
                    result['at_most_128_selected_downstream_cells'] = int(used.sum()) <= 128
                if name == 'old_neural':
                    with np.load(previous / 'neural-readout.npz', allow_pickle=False) as old:
                        result['original_baseline_arrays_unchanged'] = all(np.array_equal(model[k], old[k]) for k in old.files)
                for split in ('validation', 'test'):
                    prefix = 'development' if split == 'validation' else 'test'
                    key = ('original_' + prefix + '_neural') if name in ('old_neural', 'readout_only') else prefix + '_' + ('neural' if name == 'labels_shuffled' else name)
                    x = f[key][128:] if split == 'validation' else f[key]
                    labels = f['development_labels'][128:] if split == 'validation' else f['test_labels']
                    prediction = (linear_scores(model, x) if 'coefficients' in model else scores(model, x)).argmax(1)
                    result[split + '_prediction'] = prediction.tolist() == expected[split]['predictions']
                    result[split + '_accuracy'] = float(np.mean(prediction == labels)) == expected[split]['accuracy']
            details[name] = result
    passed = all(checks.values()) and all(all(r.values()) for r in details.values())
    save(root / 'verification.json', {'all_passed': passed, 'checks': checks, 'readouts': details})
    print(json.dumps({'all_passed': passed, 'checks': checks, 'readouts': details}))
    if not passed:
        raise RuntimeError('improvement verification failed')


if __name__ == '__main__':
    main()
