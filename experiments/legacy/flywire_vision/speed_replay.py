"""Expose independently verified, saved multispeed trials without retraining."""
from pathlib import Path
import numpy as np
from .motion_refinement import read, sha, load_npz
from .motion_readout import predict
from .multispeed_study import MODELS, CONDITIONS
from .multispeed_data import speed_movie, digest
from .pilot import DIRECTIONS
from .direction_study import evaluate
from .run_experiment import array64


class SpeedReplay:
    schema = "flywire-multispeed-v1"
    model_names = MODELS
    condition_names = CONDITIONS
    movie_factory = staticmethod(speed_movie)

    def __init__(self, directory):
        root = Path(directory)
        self.report = read(root/'report.json'); self.protocol = read(root/'protocol.json')
        if self.report['status'] != 'complete' or self.report['schema'] != self.schema:
            raise ValueError('completed multispeed study required')
        verified = read(root/'verification.json')
        if not verified['all_passed'] or sha(root/'report.json') != verified['report_sha256']:
            raise ValueError('multispeed report not independently verified')
        for name in ('protocol', 'selection', 'rows'):
            if sha(root/(name+'.json')) != self.report[name+'_sha256']:raise ValueError('multispeed records changed')
        if sha(root/'features.npz') != self.report['features_sha256']:raise ValueError('multispeed features changed')
        self.rows = read(root/'rows.json'); self.models = {}
        for n in self.model_names:
            if sha(root/(n+'.npz')) != self.protocol['readouts'][n]:raise ValueError('multispeed weights changed')
            self.models[n] = load_npz(root/(n+'.npz'))
        with np.load(root/'features.npz', allow_pickle=False) as f:
            self.features = {n: f[n] for n in self.model_names}
            raw = f['neural']
            self.counts = raw[:, 10:50].reshape(len(raw), 8, 5, raw.shape[-1]).sum((2, 3))
        self.lookup = {(v['condition'], v['group'], *v['phase_index'], v['kind']): i for i, v in enumerate(self.rows)}
        labels = np.array([DIRECTIONS.index(v['kind']) for v in self.rows])
        for c in self.condition_names:
            mask = np.array([v['condition'] == c for v in self.rows]); rr = [v for v, keep in zip(self.rows, mask) if keep]
            for n, m in self.models.items():
                if evaluate(predict(m, self.features[n][mask]).argmax(1), labels[mask], rr) != self.report['conditions'][c][n]:
                    raise ValueError('multispeed predictions changed')

    def index(self):
        return {'report': self.report, 'chosen': self.protocol['chosen'], 'groups': self.protocol['test_groups'],
                'conditions': list(self.condition_names), 'directions': list(DIRECTIONS)}

    def trial(self, condition, group, i, j, kind):
        key = (condition, group, i, j, kind)
        if key not in self.lookup:raise ValueError('unknown multispeed trial')
        idx = self.lookup[key]; row = self.rows[idx]
        frames = self.movie_factory(kind, group, (i, j), row['speed'])
        if condition == 'static_first':frames = np.repeat(frames[:1], 40, axis=0)
        if digest(frames) != row['movie_sha256']:raise ValueError('multispeed movie changed')
        predictions = {}
        for n, m in self.models.items():
            scores = predict(m, self.features[n][idx:idx+1])[0]
            predictions[n] = {'direction': DIRECTIONS[int(scores.argmax())], 'scores': scores.tolist()}
        return {'row': row, 'frames_u8': array64(np.rint(frames*255), 'u1'),
                'predictions': predictions, 'counts': self.counts[idx].tolist()}
