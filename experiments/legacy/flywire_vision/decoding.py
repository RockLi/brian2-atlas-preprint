"""Attach an explicitly labelled pilot readout to genuine neural spike events."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .pilot import DIRECTIONS, event_features, scores
from .run_experiment import save


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _indices(values, upper=None):
    values = np.asarray(values)
    if (values.ndim != 1 or not len(values) or values.dtype.kind not in 'iu' or
            np.any(values < 0) or len(np.unique(values)) != len(values) or
            (upper is not None and np.any(values >= upper))):
        raise ValueError('invalid or duplicated readout cell indices')
    return values.astype(np.int64)


def sparse_feature(feature):
    """Small replay cache of the exact counts used by the external readout."""
    feature = np.asarray(feature, dtype='<f4')
    indices = np.flatnonzero(feature)
    return {'size': len(feature), 'indices': indices.tolist(),
            'counts': feature[indices].astype(np.int64).tolist(),
            'sha256': hashlib.sha256(feature.tobytes()).hexdigest()}


class Decoder:
    def __init__(self,directory,base_hash,neuron_count=None):
        directory=Path(directory)
        self.protocol=json.loads((directory/'protocol.json').read_text())
        self.report=json.loads((directory/'report.json').read_text())
        if self.report['status']!='complete' or self.protocol['base_sha256']!=base_hash:
            raise ValueError('pilot does not match this frozen neural model')
        if tuple(self.protocol['directions'])!=DIRECTIONS:raise ValueError('unexpected readout labels')
        self.checked_study = any(k in self.protocol or k in self.report for k in ('schema','protocol_sha256','selection_sha256'))
        if self.checked_study:
            schema = self.protocol.get('schema')
            if not isinstance(schema, str) or not schema.startswith('flywire-direction-') or schema != self.report.get('schema'):
                raise ValueError('readout protocol/report schema mismatch')
            selection = json.loads((directory/'selection.json').read_text())
            if (_sha(directory/'protocol.json') != self.report.get('protocol_sha256') or
                    _sha(directory/'protocol.json') != selection.get('protocol_sha256') or
                    _sha(directory/'selection.json') != self.report.get('selection_sha256')):
                raise ValueError('readout protocol or selection artifact changed')
            if selection.get('test_clips_executed') != 0:
                raise ValueError('readout selection must precede the holdout')
            locked = selection.get('readouts', {}).get('neural', {})
            if not locked or any(self.report['readouts']['neural'].get(k) != v for k, v in locked.items()):
                raise ValueError('readout no longer matches the locked selection')
        config = self.protocol.get('config', {})
        if tuple(config.get(k) for k in ('frames','warmup_ms','tail_ms','dt_ms','frame_ms')) != (40,100.,100.,.1,10.):
            raise ValueError('readout requires the original 8 x 50 ms stimulus window')
        path=directory/'neural-readout.npz'
        self.readout_hash = self.report['readouts']['neural']['readout_sha256']
        if _sha(path)!=self.readout_hash:
            raise ValueError('readout artifact changed')
        with np.load(path,allow_pickle=False) as data:self.model=dict(data)
        self.cells=_indices(self.protocol['readout_indices'], neuron_count)
        if set(self.cells) & set(self.protocol.get('input_indices', [])):
            raise ValueError('downstream readout overlaps externally driven input cells')
        self.dimension = 8 * len(self.cells)
        m = self.model
        if 'coefficients' in m:
            shapes = {'coefficients': (self.dimension,4), 'bias': (4,)}
            if str(m.get('format', '')) != 'linear_raw_counts_v1':
                raise ValueError('unknown linear readout format')
            if 'selected_cells' in m:
                selected = _indices(m['selected_cells'], len(self.cells))
                if m['coefficients'].shape == shapes['coefficients']:
                    used = np.flatnonzero(np.any(m['coefficients'].reshape(8,-1,4), axis=(0,2)))
                    if not set(used) <= set(selected):
                        raise ValueError('readout uses cells outside its selected support')
        else:
            if 'train' not in m or m['train'].ndim != 2 or not len(m['train']):
                raise ValueError('invalid dual readout training matrix')
            shapes = {'mean': (self.dimension,), 'scale': (self.dimension,),
                      'train': (len(m['train']),self.dimension), 'weights': (len(m['train']),4)}
        for key, shape in shapes.items():
            if key not in m or m[key].shape != shape or m[key].dtype.kind not in 'fiu' or not np.isfinite(m[key]).all():
                raise ValueError('invalid readout parameter: '+key)
        if 'scale' in shapes and np.any(m['scale'] <= 0):
            raise ValueError('readout scale must be positive')

    def features(self,population,neuron_count):
        # Only neural activity enters the predictor; no stimulus ID, movie or label.
        _indices(self.cells, neuron_count)
        indices, ticks = np.asarray(population['indices']), np.asarray(population['spike_ticks'])
        if (indices.ndim != 1 or ticks.shape != indices.shape or indices.dtype.kind not in 'iu' or
                ticks.dtype.kind not in 'iu' or np.any(indices < 0) or np.any(indices >= neuron_count)):
            raise ValueError('invalid neural spike events')
        return event_features(indices,ticks,self.cells,neuron_count)

    def predict_feature(self,feature):
        feature = np.asarray(feature, dtype=float)
        if (feature.shape != (self.dimension,) or not np.isfinite(feature).all() or
                np.any(feature < 0) or np.any(feature != np.floor(feature))):
            raise ValueError('invalid neural count features')
        if 'coefficients' in self.model:
            from .refinement import linear_scores
            values=linear_scores(self.model,feature[None])[0]
        else:
            values=scores(self.model,feature[None])[0]
        return {'direction':DIRECTIONS[int(values.argmax())],'directions':list(DIRECTIONS),
                'scores':values.tolist(),'scope':self.report['scope'],
                'readout_sha256':self.readout_hash}

    def predict(self,population,neuron_count):
        return self.predict_feature(self.features(population,neuron_count))

    def cached_prediction(self,entry,event_hash):
        if entry.get('events_sha256') != event_hash:
            raise ValueError('prediction does not match the displayed neural events')
        cached = entry.get('classification', {})
        values = np.asarray(cached.get('scores', []), dtype=float)
        if (cached.get('readout_sha256') != self.readout_hash or
                cached.get('directions') != list(DIRECTIONS) or values.shape != (4,) or
                not np.isfinite(values).all() or cached.get('direction') != DIRECTIONS[int(values.argmax())]):
            raise ValueError('cached prediction does not match this readout')
        encoded = entry.get('neural_feature')
        if encoded is None:
            if self.checked_study:
                raise ValueError('regenerate viewer predictions to include checked neural count features')
            return {**cached, 'scope': self.report['scope']}
        if encoded.get('size') != self.dimension:
            raise ValueError('cached neural feature size changed')
        positions = np.asarray(encoded.get('indices', []))
        counts = np.asarray(encoded.get('counts', []))
        if (positions.ndim != 1 or counts.shape != positions.shape or
                (len(positions) and (positions.dtype.kind not in 'iu' or np.any(positions < 0) or
                np.any(positions >= self.dimension) or len(np.unique(positions)) != len(positions))) or
                counts.dtype.kind not in 'fiu' or not np.isfinite(counts).all() or
                np.any(counts <= 0) or np.any(counts != np.floor(counts))):
            raise ValueError('invalid cached neural count features')
        feature = np.zeros(self.dimension, dtype='<f4')
        feature[positions.astype(int)] = counts
        if hashlib.sha256(feature.tobytes()).hexdigest() != encoded.get('sha256'):
            raise ValueError('cached neural count features changed')
        actual = self.predict_feature(feature)
        if actual['direction'] != cached['direction'] or not np.allclose(actual['scores'], values, rtol=1e-10, atol=1e-10):
            raise ValueError('cached prediction differs from recomputed neural readout')
        return actual


def main():
    parser=argparse.ArgumentParser(description='Decode saved P1 events with the pilot external readout')
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--pilot',type=Path,required=True)
    args=parser.parse_args()
    from brian2_rust.results import load_results
    identity=json.loads((args.artifact/'intact/identity.json').read_text())
    decoder=Decoder(args.pilot,identity['base_sha256'])
    models={c:json.loads((args.artifact/n).read_text()) for c,n in (('intact','model.json'),('cut','model-cut.json'))}
    trials=json.loads((args.artifact/'viewer-data.json').read_text())['trials']
    result={}
    for trial in trials:
        model=models[trial['condition']]
        ni=next(i for i,p in enumerate(model['definition']['populations']) if p['name']=='flywire_neurons')
        population=load_results(model,args.artifact/trial['condition']/trial['id'])['populations'][ni]
        actual=hashlib.sha256(np.stack([population['spike_ticks'],population['indices']],axis=1).astype('<i8').tobytes()).hexdigest()
        if actual!=trial['summary']['neural_event_sha256']:raise ValueError('saved neural activity changed')
        feature=decoder.features(population,model['definition']['populations'][ni]['count'])
        result[trial['id']]={'events_sha256':actual,'classification':decoder.predict_feature(feature),
                             'neural_feature':sparse_feature(feature)}
    save(args.pilot/'viewer-predictions.json',result)
    print(json.dumps({'decoded_real_trials':len(result)}))


if __name__=='__main__':main()
