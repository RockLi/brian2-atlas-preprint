import tempfile
import unittest
from pathlib import Path
import numpy as np
from .generalization_replay import GeneralizationReplay
from .motion_refinement import sha
from .generalization_study import CONDITIONS, MODELS
from .generalization_data import generalization_movie, digest
from .motion_stress import phase_for
from .pilot import DIRECTIONS, fit
from .direction_study import evaluate
from .run_experiment import save


class GeneralizationReplayChecks(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root = Path(temp.name)
        root = self.root; rows = []
        for c in CONDITIONS:
            for g in range(8):
                for i in range(2):
                    for j in range(2):
                        for k in DIRECTIONS:
                            if c.startswith(('cut', 'matched')) and (i, j) != phase_for(k):continue
                            rows.append(dict(condition=c, group=g, phase_index=[i,j], kind=k, speed=2,
                                             polarity='bright' if g < 4 else 'dark'))
        rows[0]['movie_sha256'] = digest(generalization_movie('right', 0, (0,0), 2))
        model = fit(np.zeros((4,24)), np.arange(4), 1.)
        for n in MODELS:np.savez_compressed(root/(n+'.npz'), **model)
        save(root/'protocol.json', {'test_groups': list(range(8)), 'chosen': {}, 'readouts': {n: sha(root/(n+'.npz')) for n in MODELS}})
        save(root/'selection.json', {}); save(root/'rows.json', rows)
        np.savez_compressed(root/'features.npz', neural=np.zeros((len(rows),60,1),dtype=np.uint16), **{n: np.zeros((len(rows),24)) for n in MODELS})
        self.report = {'schema': 'flywire-speed-generalization-v1', 'status': 'complete',
            **{n+'_sha256': sha(root/(n+'.json')) for n in ('protocol', 'selection', 'rows')},
            'features_sha256': sha(root/'features.npz'), 'conditions': {}}
        for c in CONDITIONS:
            rr = [v for v in rows if v['condition'] == c]; y = [DIRECTIONS.index(v['kind']) for v in rr]
            self.report['conditions'][c] = {n: evaluate(np.zeros(len(y),dtype=int), y, rr) for n in MODELS}
        self.save_verified()

    def save_verified(self):
        save(self.root/'report.json', self.report)
        save(self.root/'verification.json', {'all_passed': True, 'report_sha256': sha(self.root/'report.json')})

    def test_actual_movie_and_unknown_row(self):
        replay = GeneralizationReplay(self.root)
        self.assertEqual(len(replay.lookup), 576)
        self.assertEqual(replay.trial('speed_2',0,0,0,'right')['predictions']['generalized']['direction'], 'right')
        with self.assertRaisesRegex(ValueError, 'unknown multispeed'):replay.trial('speed_2',999,0,0,'right')

    def test_rejects_changed_verification_and_statistics(self):
        self.report['conditions']['speed_2']['generalized']['accuracy'] = 1.
        save(self.root/'report.json', self.report)
        with self.assertRaisesRegex(ValueError, 'independently verified'):GeneralizationReplay(self.root)
        self.save_verified()
        with self.assertRaisesRegex(ValueError, 'predictions changed'):GeneralizationReplay(self.root)

    def test_rejects_changed_weights(self):
        np.savez_compressed(self.root/'generalized.npz', invented=np.ones(1))
        with self.assertRaisesRegex(ValueError, 'weights changed'):GeneralizationReplay(self.root)

    def test_rejects_changed_input_audit(self):
        save(self.root/'input-audit.json',{'features_sha256':self.report['features_sha256']})
        save(self.root/'input-audit-verification.json',{'all_passed':True,'audit_sha256':sha(self.root/'input-audit.json')})
        self.assertIsNotNone(GeneralizationReplay(self.root).index()['input_audit'])
        save(self.root/'input-audit.json',{'features_sha256':'modified'})
        with self.assertRaisesRegex(ValueError,'input audit not verified'):GeneralizationReplay(self.root)

    def save_totals_audit(self, **changes):
        np.savez_compressed(self.root/'totals-audit-model.npz', coefficients=np.zeros((1,4)))
        save(self.root/'totals-audit.json',{'features_sha256':self.report['features_sha256'],
            'report_sha256':sha(self.root/'report.json'),'model_sha256':sha(self.root/'totals-audit-model.npz'),**changes})
        save(self.root/'totals-audit-verification.json',{'all_passed':True,'audit_sha256':sha(self.root/'totals-audit.json')})

    def test_rejects_totals_audit_from_other_data(self):
        self.save_totals_audit()
        self.assertIsNotNone(GeneralizationReplay(self.root).index()['totals_audit'])
        self.save_totals_audit(features_sha256='other dataset')
        with self.assertRaisesRegex(ValueError,'totals audit not verified'):GeneralizationReplay(self.root)

    def test_rejects_changed_totals_audit_model(self):
        self.save_totals_audit()
        np.savez_compressed(self.root/'totals-audit-model.npz', coefficients=np.ones((1,4)))
        with self.assertRaisesRegex(ValueError,'totals audit not verified'):GeneralizationReplay(self.root)


if __name__ == '__main__':unittest.main()
