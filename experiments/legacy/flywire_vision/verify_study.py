"""Recompute frozen study predictions and validate provenance/split boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .pilot import DIRECTIONS
from .run_experiment import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory
    read = lambda name: json.loads((directory / name).read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    protocol, selection, report, rows = [read(n + '.json') for n in ('protocol', 'selection', 'report', 'rows')]
    checks = {'status_complete': report['status'] == 'complete',
              'selection_precedes_test': selection['test_clips_executed'] == 0,
              'protocol_hash': sha(directory / 'protocol.json') == report['protocol_sha256'] == selection['protocol_sha256'],
              'selection_hash': sha(directory / 'selection.json') == report['selection_sha256'],
              'feature_hash': sha(directory / 'features.npz') == report['features_sha256'],
              'direction_order': tuple(protocol['directions']) == DIRECTIONS,
              'source_hashes': all(sha(Path(__file__).with_name(n)) == h for n, h in protocol['sources_sha256'].items())}
    splits = [protocol[name + '_groups'] for name in ('fit', 'validation', 'test')]
    checks['disjoint_groups'] = len(set(sum(splits, []))) == sum(map(len, splits)) == 64
    checks['row_counts'] = len(rows) == 256 and all(sum(r['split'] == name for r in rows) == count
                                                  for name, count in (('fit', 128), ('validation', 64), ('test', 64)))
    checks['balanced_complete_groups'] = all(
        [r['kind'] for r in rows if r['group'] == group] == list(DIRECTIONS)
        and {r['split'] for r in rows if r['group'] == group} == {name}
        for name in ('fit', 'validation', 'test') for group in protocol[name + '_groups'])
    checks['polarity_balance'] = all(sum(r['polarity'] == 'bright' for r in rows if r['split'] == name) == count // 2
                                     for name, count in (('fit', 128), ('validation', 64), ('test', 64)))
    details = {}
    with np.load(directory / 'features.npz', allow_pickle=False) as features:
        labels = features['labels']
        checks['labels_match_stimulus_rows'] = np.array_equal(labels, [DIRECTIONS.index(r['kind']) for r in rows])
        for name, frozen in selection['readouts'].items():
            expected = report['readouts'][name]
            model_path = directory / (name + '-readout.npz')
            detail = {'selection_fields_unchanged': all(expected[k] == v for k, v in frozen.items()),
                      'model_hash': sha(model_path) == frozen['readout_sha256'],
                      'model_saved_before_selection': model_path.stat().st_mtime_ns <= (directory / 'selection.json').stat().st_mtime_ns}
            x = features['neural' if name == 'labels_shuffled' else name]
            with np.load(model_path, allow_pickle=False) as m:
                detail['fit_only_mean'] = np.array_equal(m['mean'], x[:128].mean(0, dtype=np.float64))
                scale = x[:128].std(0, dtype=np.float64)
                scale[scale < 1e-8] = 1
                detail['fit_only_scale'] = np.array_equal(m['scale'], scale)
                for split, a, b in (('validation', 128, 192), ('test', 192, 256)):
                    z = ((x[a:b] - m['mean']) / m['scale']).astype(np.float32)
                    predictions = ((z @ m['train'].T) / m['train'].shape[1] @ m['weights']).argmax(1)
                    metrics = expected[split]
                    detail[split + '_predictions_recomputed'] = predictions.tolist() == metrics['predictions']
                    detail[split + '_accuracy_recomputed'] = float(np.mean(predictions == labels[a:b])) == metrics['accuracy']
            details[name] = detail
    success = all(checks.values()) and all(all(d.values()) for d in details.values())
    save(directory / 'verification.json', {'all_passed': success, 'checks': checks, 'readouts': details})
    print(json.dumps({'all_passed': success, 'checks': checks, 'readouts': details}))
    if not success:
        raise RuntimeError('study verification failed')


if __name__ == '__main__':
    main()
