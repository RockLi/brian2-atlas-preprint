import tempfile
import unittest
from pathlib import Path
import numpy as np
from .speed_replay import SpeedReplay
from .motion_refinement import sha
from .multispeed_study import CONDITIONS, MODELS
from .multispeed_data import speed_movie, digest
from .motion_stress import phase_for
from .pilot import DIRECTIONS, fit
from .direction_study import evaluate
from .run_experiment import save


class SpeedReplayChecks(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root = Path(temp.name)
        root = self.root; rows = []
        for c in CONDITIONS:
            for g in range(8):
                for i in range(2):
                    for j in range(2):
                        for k in DIRECTIONS:
                            if c.startswith(('cut', 'matched')) and (i, j) != phase_for(k):continue
                            rows.append(dict(condition=c, group=g, phase_index=[i,j], kind=k, speed=1,
                                             polarity='bright' if g < 4 else 'dark'))
        rows[0]['movie_sha256'] = digest(speed_movie('right', 0, (0,0), 1))
        model = fit(np.zeros((4,24)), np.arange(4), 1.)
        for n in MODELS:np.savez_compressed(root/(n+'.npz'), **model)
        save(root/'protocol.json', {'test_groups': list(range(8)), 'chosen': {}, 'readouts': {n: sha(root/(n+'.npz')) for n in MODELS}})
        save(root/'selection.json', {}); save(root/'rows.json', rows)
        np.savez_compressed(root/'features.npz', neural=np.zeros((len(rows),60,1),dtype=np.uint16), **{n: np.zeros((len(rows),24)) for n in MODELS})
        self.report = {'schema': 'flywire-multispeed-v1', 'status': 'complete',
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
        replay = SpeedReplay(self.root)
        self.assertEqual(len(replay.lookup), 576)
        self.assertEqual(replay.trial('speed_1',0,0,0,'right')['predictions']['multispeed']['direction'], 'right')
        with self.assertRaisesRegex(ValueError, 'unknown multispeed'):replay.trial('speed_1',999,0,0,'right')

    def test_rejects_changed_verification_and_statistics(self):
        self.report['conditions']['speed_1']['multispeed']['accuracy'] = 1.
        save(self.root/'report.json', self.report)
        with self.assertRaisesRegex(ValueError, 'independently verified'):SpeedReplay(self.root)
        self.save_verified()
        with self.assertRaisesRegex(ValueError, 'predictions changed'):SpeedReplay(self.root)

    def test_rejects_changed_weights(self):
        np.savez_compressed(self.root/'multispeed.npz', invented=np.ones(1))
        with self.assertRaisesRegex(ValueError, 'weights changed'):SpeedReplay(self.root)


if __name__ == '__main__':unittest.main()
