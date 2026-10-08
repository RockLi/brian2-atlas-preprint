"""Checked replay of the new periodic benchmark; no live or synthetic scores."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .pilot import DIRECTIONS, scores
from .refinement import linear_scores
from .verify_causal_motion import trial_movie
from .run_experiment import array64
from .direction_study import evaluate


class CausalReplay:
    def __init__(self, directory):
        root=Path(directory);read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        self.protocol=read(root/'protocol.json');self.report=read(root/'report.json')
        selection=read(root/'selection.json')
        if self.report.get('schema')!='flywire-causal-motion-v1' or self.protocol.get('schema')!=self.report['schema'] or self.report.get('status')!='complete':
            raise ValueError('requires completed causal motion study')
        if (sha(root/'protocol.json')!=self.report['protocol_sha256'] or selection['protocol_sha256']!=self.report['protocol_sha256'] or
                sha(root/'selection.json')!=self.report['selection_sha256'] or sha(root/'features.npz')!=self.report['features_sha256']):
            raise ValueError('causal study artifacts changed')
        self.rows=read(root/'rows.json');self.models={}
        for name,expected in selection['readouts'].items():
            path=root/(name+'-readout.npz')
            if sha(path)!=expected['readout_sha256'] or self.report['readouts'][name]!=expected:raise ValueError('causal readout changed')
            with np.load(path,allow_pickle=False) as m:self.models[name]=dict(m)
        with np.load(root/'features.npz',allow_pickle=False) as f:self.features=dict(f)
        labels=np.asarray([DIRECTIONS.index(row['kind']) for row in self.rows])
        if not np.array_equal(labels,self.features['labels']):raise ValueError('causal labels and rows disagree')
        for name,dimension in (('neural_linear',8*len(self.protocol['readout_indices'])),('neural_motion',24),('input_motion',24)):
            if self.features[name].shape!=(len(self.rows),dimension) or not np.isfinite(self.features[name]).all():raise ValueError('invalid causal features')
        self.lookup={(row['condition'],row['group'],*row['phase_index'],row['kind']):i for i,row in enumerate(self.rows) if row['split']=='test'}
        if len(self.lookup)!=self.report['test_clips_per_condition']*len(self.report['conditions']):raise ValueError('incomplete causal replay rows')
        # Independently bind each displayed prediction to the report and counts.
        for condition,results in self.report['conditions'].items():
            selected=[i for i,row in enumerate(self.rows) if row['split']=='test' and row['condition']==condition]
            for name,model in self.models.items():
                x=self.features['neural_motion' if name=='labels_shuffled' else name][selected]
                prediction=(linear_scores(model,x) if name=='neural_linear' else scores(model,x)).argmax(1)
                if evaluate(prediction,labels[selected],[self.rows[i] for i in selected])!=results[name]:raise ValueError('causal prediction or statistic mismatch')

    def index(self):
        return {'report':self.report,'groups':self.protocol['test_groups'],'directions':list(DIRECTIONS),
                'conditions':list(self.report['conditions']),'phases':[[i,j] for i in range(2) for j in range(2)]}

    def trial(self, condition, group, i, j, kind):
        key=(condition,group,i,j,kind)
        if key not in self.lookup:raise ValueError('unknown recorded causal trial')
        index=self.lookup[key];row=self.rows[index]
        frames=trial_movie(row,self.protocol['travel'])
        if hashlib.sha256(frames.tobytes()).hexdigest()!=row['movie_sha256']:raise ValueError('causal movie changed')
        predictions={}
        for name,model in self.models.items():
            feature=self.features['neural_motion' if name=='labels_shuffled' else name][index:index+1]
            values=(linear_scores(model,feature) if name=='neural_linear' else scores(model,feature))[0]
            predictions[name]={'direction':DIRECTIONS[int(values.argmax())],'scores':values.tolist()}
        intact=self.lookup['intact',group,i,j,kind]
        return {'row':row,'frames_u8':array64(np.rint(frames*255),'u1'),'predictions':predictions,
                'downstream_counts':self.features['neural_linear'][index].reshape(8,-1).sum(1).tolist(),
                'intact_counts':self.features['neural_linear'][intact].reshape(8,-1).sum(1).tolist(),
                'readout_cells':len(self.protocol['readout_indices'])}
