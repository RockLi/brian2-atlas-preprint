"""Verified saved-trial replay of the refined external motion decoder."""
import hashlib
import numpy as np
from .motion_refinement import read, sha, load_npz, MODELS, model_scores, transformed_frames
from .motion_challenge import movie
from .pilot import DIRECTIONS
from .direction_study import evaluate
from .run_experiment import array64
from pathlib import Path


class MotionReplay:
    def __init__(self,directory):
        root=Path(directory);self.protocol=read(root/'protocol.json');self.report=read(root/'report.json');s=read(root/'selection.json')
        if self.report['status']!='complete' or self.report['schema']!='flywire-motion-refinement-v1':raise ValueError('completed refinement required')
        if (sha(root/'protocol.json')!=self.report['protocol_sha256'] or s['protocol_sha256']!=self.report['protocol_sha256'] or
            sha(root/'selection.json')!=self.report['selection_sha256'] or sha(root/'features.npz')!=self.report['features_sha256']):raise ValueError('refinement provenance mismatch')
        self.models={}
        for n in MODELS:
            if sha(root/(n+'-readout.npz'))!=s['readouts'][n]:raise ValueError('refinement weights changed')
            self.models[n]=load_npz(root/(n+'-readout.npz'))
        self.stress=None
        if (root/'robustness/report.json').exists():
            verified=read(root/'robustness/verification.json')
            if not verified['all_passed'] or sha(root/'robustness/report.json')!=verified['report_sha256']:raise ValueError('stress report not verified')
            self.stress=read(root/'robustness/report.json')
        self.rows=read(root/'rows.json')
        with np.load(root/'features.npz',allow_pickle=False) as f:
            self.features={n:f[n] for n in MODELS if n!='labels_shuffled'};labels=f['labels']
        if not np.array_equal(labels,[DIRECTIONS.index(r['kind']) for r in self.rows]):raise ValueError('refinement labels changed')
        self.lookup={(r['condition'],r['group'],*r['phase_index'],r['kind']):i for i,r in enumerate(self.rows)}
        if len(self.lookup)!=768:raise ValueError('incomplete refinement replay')
        for c,result in self.report['conditions'].items():
            mask=np.array([r['condition']==c for r in self.rows]);rows=[r for r in self.rows if r['condition']==c]
            for name,m in self.models.items():
                x=self.features['neural_motion' if name=='labels_shuffled' else name][mask]
                if not np.isfinite(x).all() or len(x)!=128:raise ValueError('invalid refinement features')
                if evaluate(model_scores(name,m,x).argmax(1),labels[mask],rows)!=result[name]:raise ValueError('refinement statistics changed')

    def index(self):
        p=self.protocol;chosen=p['chosen'];recipe=chosen['recipe']
        frontend='因果帧差 ON/OFF' if p['input_mode']=='temporal_difference_x4' else '原始亮暗对比度'
        feature={'correlation':'时空相关','spectral':'方向运动能量','hybrid':'时空相关与方向运动能量'}[recipe.get('feature','correlation')]
        return {'report':self.report,'stress':self.stress,'groups':p['test_groups'],'directions':list(DIRECTIONS),'conditions':list(self.report['conditions']),
                'phases':[[i,j] for i in range(2) for j in range(2)],
                'summary':f"{frontend}输入；{recipe['width']*10} ms 计数窗口，读取 {recipe['start']*10}–{recipe['stop']*10} ms 响应；{feature}外部读出。频谱运动能量按本任务的固定速度设置，跨速度与背景复测见下方，滤波器仍按训练速度设置。",
                'model_labels':{'previous_motion':'上一轮运动读出 · 同输入','neural_motion':'改进运动读出 · 神经活动','input_motion':'实际输入 · 运动读出','neural_linear':'原始神经计数 · 线性','labels_shuffled':'训练标签打乱'}}

    def trial(self,condition,group,i,j,kind):
        key=(condition,group,i,j,kind)
        if key not in self.lookup:raise ValueError('unknown refinement trial')
        index=self.lookup[key];row=self.rows[index]
        frames=transformed_frames(row,movie(kind,group,2.,(i,j)),condition)
        if hashlib.sha256(frames.tobytes()).hexdigest()!=row['movie_sha256']:raise ValueError('refinement movie changed')
        predictions={}
        for name,m in self.models.items():
            values=model_scores(name,m,self.features['neural_motion' if name=='labels_shuffled' else name][index:index+1])[0]
            predictions[name]={'direction':DIRECTIONS[int(values.argmax())],'scores':values.tolist()}
        intact=self.lookup['intact',group,i,j,kind]
        return {'row':{**row,'reused_identical_static':row['condition']=='static_first' and row['reused_identical']},
                'frames_u8':array64(np.rint(frames*255),'u1'),'predictions':predictions,
                'downstream_counts':self.features['neural_linear'][index].reshape(8,-1).sum(1).tolist(),
                'intact_counts':self.features['neural_linear'][intact].reshape(8,-1).sum(1).tolist(),
                'readout_cells':len(self.protocol['readout_indices'])}
