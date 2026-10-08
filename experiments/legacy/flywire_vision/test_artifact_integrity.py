"""Reject mismatched readout provenance and replay caches before displaying them."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .decoding import Decoder, sparse_feature
from .pilot import DIRECTIONS, fit
from .serve import load_audit


class ArtifactIntegrityChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='flywire-integrity-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.protocol = {'schema': 'flywire-direction-sparse-v3', 'base_sha256': 'brain',
                         'binary_sha256': 'binary', 'input_mode': 'contrast',
                         'directions': DIRECTIONS, 'readout_indices': [3, 1, 7],
                         'input_indices': [0], 'config': {'frames': 40, 'warmup_ms': 100.,
                         'tail_ms': 100., 'dt_ms': .1, 'frame_ms': 10.}}
        coefficients = np.zeros((24, 4))
        coefficients[0, 0], coefficients[1, 1] = .2, .3
        self.model = {'coefficients': coefficients, 'bias': np.zeros(4),
                      'format': np.array('linear_raw_counts_v1')}
        self.write_fixture()

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def sha(self, name):
        return hashlib.sha256((self.root / name).read_bytes()).hexdigest()

    def write_fixture(self, legacy=False):
        protocol = dict(self.protocol)
        if legacy:
            protocol.pop('schema', None)
        self.write('protocol.json', protocol)
        np.savez_compressed(self.root / 'neural-readout.npz', **self.model)
        locked = {'readout_sha256': self.sha('neural-readout.npz')}
        report = {'status': 'complete', 'scope': 'synthetic test', 'readouts': {'neural': locked}}
        if not legacy:
            self.write('selection.json', {'protocol_sha256': self.sha('protocol.json'),
                       'test_clips_executed': 0, 'readouts': {'neural': locked}})
            report.update(schema=protocol['schema'], protocol_sha256=self.sha('protocol.json'),
                          selection_sha256=self.sha('selection.json'))
        self.write('report.json', report)

    def decoder(self):
        return Decoder(self.root, 'brain', 10)

    def entry(self, decoder):
        feature = np.zeros(24, dtype=np.float32)
        feature[:2] = [4, 1]
        return {'events_sha256': 'events', 'classification': decoder.predict_feature(feature),
                'neural_feature': sparse_feature(feature)}

    def test_protocol_reordering_is_rejected_by_hash_chain(self):
        self.protocol['readout_indices'].reverse()
        self.write('protocol.json', self.protocol)
        with self.assertRaisesRegex(ValueError, 'protocol or selection'):
            self.decoder()

    def test_selection_and_weight_changes_are_rejected(self):
        selection = json.loads((self.root / 'selection.json').read_text())
        selection['test_clips_executed'] = 1
        self.write('selection.json', selection)
        with self.assertRaisesRegex(ValueError, 'protocol or selection'):
            self.decoder()
        self.write_fixture()
        self.model['bias'][0] = 1
        np.savez_compressed(self.root / 'neural-readout.npz', **self.model)
        with self.assertRaisesRegex(ValueError, 'readout artifact changed'):
            self.decoder()

    def test_legacy_pilot_still_loads_with_valid_dimensions(self):
        self.model = fit(np.arange(4*24, dtype=float).reshape(4, 24), np.arange(4), .1)
        self.write_fixture(legacy=True)
        decoder = self.decoder()
        entry = self.entry(decoder)
        entry.pop('neural_feature')
        self.assertEqual(decoder.cached_prediction(entry, 'events')['direction'], entry['classification']['direction'])

    def test_invalid_indices_are_rejected_even_for_legacy_pilot(self):
        for indices in ([1, 1, 2], [-1, 1, 2], [1, 2, 10], [1., 2., 3.], [0, 1, 2]):
            with self.subTest(indices=indices):
                self.protocol['readout_indices'] = indices
                self.write_fixture(legacy=True)
                with self.assertRaises(ValueError):
                    self.decoder()

    def test_valid_hash_cannot_hide_invalid_weight_shape_or_values(self):
        for coefficient in (np.zeros((23,4)), np.full((24,4), np.nan)):
            with self.subTest(shape=coefficient.shape):
                self.model['coefficients'] = coefficient
                self.write_fixture()
                with self.assertRaisesRegex(ValueError, 'invalid readout parameter'):
                    self.decoder()

    def test_incompatible_timing_rejected(self):
        self.protocol['config']['dt_ms'] = .2
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'stimulus window'):
            self.decoder()

    def test_cache_is_recomputed_from_counts(self):
        decoder = self.decoder()
        entry = self.entry(decoder)
        self.assertEqual(decoder.cached_prediction(entry, 'events'), entry['classification'])
        entry['classification']['scores'][0] += .1  # Still the same argmax and readout hash.
        with self.assertRaisesRegex(ValueError, 'recomputed neural readout'):
            decoder.cached_prediction(entry, 'events')

    def test_stale_invalid_or_mislabeled_predictions_are_rejected(self):
        decoder = self.decoder()
        valid = self.entry(decoder)
        for key, value in [('readout_sha256', 'old-model'), ('scores', [np.nan,0,0,0]),
                           ('direction', 'down'), ('directions', list(reversed(DIRECTIONS)))]:
            with self.subTest(key=key):
                entry = copy.deepcopy(valid)
                entry['classification'][key] = value
                with self.assertRaisesRegex(ValueError, 'cached prediction'):
                    decoder.cached_prediction(entry, 'events')

    def test_cache_requires_matching_events_and_checked_features(self):
        decoder = self.decoder()
        entry = self.entry(decoder)
        with self.assertRaisesRegex(ValueError, 'displayed neural events'):
            decoder.cached_prediction(entry, 'different-events')
        entry['neural_feature']['counts'][0] += 1
        with self.assertRaisesRegex(ValueError, 'features changed'):
            decoder.cached_prediction(entry, 'events')
        entry.pop('neural_feature')
        with self.assertRaisesRegex(ValueError, 'regenerate viewer predictions'):
            decoder.cached_prediction(entry, 'events')

    def test_empty_event_features_remain_valid(self):
        decoder = self.decoder()
        feature = decoder.features({'indices': np.array([], dtype=int),
                                    'spike_ticks': np.array([], dtype=int)}, 10)
        entry = {'events_sha256': 'empty', 'classification': decoder.predict_feature(feature),
                 'neural_feature': sparse_feature(feature)}
        self.assertEqual(decoder.cached_prediction(entry, 'empty'), entry['classification'])

    def audit_fixture(self):
        report = json.loads((self.root/'report.json').read_text())
        for name in ('input_neurons','encoded_input'):
            (self.root/(name+'-readout.npz')).write_bytes((self.root/'neural-readout.npz').read_bytes())
            report['readouts'][name] = dict(report['readouts']['neural'])
        self.write('report.json', report)
        root = self.root/'audit'
        root.mkdir(exist_ok=True)
        protocol = {k: self.protocol[k] for k in ('base_sha256','binary_sha256','config','input_mode','readout_indices','input_indices','directions')}
        protocol.update(schema='flywire-robustness-audit-v1', parent_study=str(self.root),
                        parent_report_sha256=self.sha('report.json'), conditions={'reference': 'new trajectories'},
                        groups=[80000,80001], clips_per_condition=8,
                        readouts_sha256={k: v['readout_sha256'] for k,v in report['readouts'].items()})
        self.write('audit/protocol.json', protocol)
        (root/'features.npz').write_bytes(b'fixture features')
        report = {'schema': protocol['schema'], 'status': 'complete',
                  'protocol_sha256': self.sha('audit/protocol.json'),
                  'features_sha256': self.sha('audit/features.npz'), 'groups_per_condition': 2,
                  'clips_per_condition': 8, 'conditions': {'reference': {}}}
        self.write('audit/report.json', report)
        return root, protocol, report

    def test_optional_audit_must_match_parent_and_current_decoder(self):
        root, protocol, report = self.audit_fixture()
        decoder = self.decoder()
        self.assertEqual(load_audit(root, decoder, self.root), report)
        for key, value in [('parent_report_sha256','other-parent'),
                           ('readouts_sha256',{**protocol['readouts_sha256'], 'neural': 'old-readout'}),
                           ('readout_indices',list(reversed(protocol['readout_indices']))),
                           ('input_indices',[9])]:
            with self.subTest(key=key):
                changed = {**protocol, key: value}
                self.write('audit/protocol.json', changed)
                self.write('audit/report.json', {**report, 'protocol_sha256': self.sha('audit/protocol.json')})
                with self.assertRaises(ValueError):
                    load_audit(root, decoder, self.root)

    def test_partial_or_changed_audit_is_rejected(self):
        root, protocol, report = self.audit_fixture()
        decoder = self.decoder()
        self.write('audit/report.json', {**report, 'status': 'running'})
        with self.assertRaisesRegex(ValueError, 'completed robustness report'):
            load_audit(root, decoder, self.root)
        self.write('audit/report.json', report)
        (root/'features.npz').write_bytes(b'changed features')
        with self.assertRaisesRegex(ValueError, 'audit features changed'):
            load_audit(root, decoder, self.root)

    def test_audit_schema_cannot_be_loaded_as_direction_decoder(self):
        root, _, _ = self.audit_fixture()
        with self.assertRaisesRegex(ValueError, 'schema mismatch'):
            Decoder(root, 'brain', 10)


if __name__ == '__main__':
    unittest.main()
