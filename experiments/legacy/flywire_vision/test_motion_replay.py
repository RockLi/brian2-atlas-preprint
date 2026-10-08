import tempfile
import unittest
from pathlib import Path
import numpy as np
from .motion_replay import MotionReplay
from .motion_refinement import CONDITIONS, MODELS, sha
from .pilot import DIRECTIONS, fit
from .direction_study import evaluate
from .run_experiment import save


class MotionReplayChecks(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        protocol={'schema':'flywire-motion-refinement-v1','readout_indices':[3],'test_groups':list(range(8)),
                  'input_mode':'contrast','chosen':{'recipe':dict(start=10,stop=50,width=1,separation='family',weight='none',transform='signed',feature='spectral')}}
        save(self.root/'protocol.json',protocol)
        self.rows=[dict(condition=c,group=g,phase_index=[i,j],kind=k,polarity='bright' if g<4 else 'dark') for c in CONDITIONS for g in range(8) for i in range(2) for j in range(2) for k in DIRECTIONS]
        save(self.root/'rows.json',self.rows);labels=np.array([DIRECTIONS.index(r['kind']) for r in self.rows]);count=len(labels)
        np.savez_compressed(self.root/'features.npz',neural_linear=np.zeros((count,8)),previous_motion=np.zeros((count,24)),neural_motion=np.zeros((count,24)),input_motion=np.zeros((count,24)),labels=labels)
        self.models={n:fit(np.zeros((4,24)),np.arange(4),.1) for n in MODELS if n!='neural_linear'}
        self.models['neural_linear']={'coefficients':np.zeros((8,4)),'bias':np.zeros(4)}
        for n,m in self.models.items():np.savez_compressed(self.root/(n+'-readout.npz'),**m)
        selection={'protocol_sha256':sha(self.root/'protocol.json'),'readouts':{n:sha(self.root/(n+'-readout.npz')) for n in MODELS}}
        save(self.root/'selection.json',selection)
        self.report={'schema':protocol['schema'],'status':'complete','protocol_sha256':sha(self.root/'protocol.json'),'selection_sha256':sha(self.root/'selection.json'),'features_sha256':sha(self.root/'features.npz'),'conditions':{}}
        for c in CONDITIONS:
            rr=[r for r in self.rows if r['condition']==c];y=[DIRECTIONS.index(r['kind']) for r in rr]
            self.report['conditions'][c]={n:evaluate(np.zeros(128,dtype=int),y,rr) for n in MODELS}
        save(self.root/'report.json',self.report)

    def test_accepts_verified_statistics_and_rejects_modified_result(self):
        replay=MotionReplay(self.root)
        self.assertEqual(len(replay.lookup),768)
        self.assertIn('previous_motion',replay.index()['model_labels'])
        self.report['conditions']['intact']['neural_motion']['accuracy']=1
        save(self.root/'report.json',self.report)
        with self.assertRaisesRegex(ValueError,'statistics changed'):MotionReplay(self.root)

    def test_rejects_stress_report_changed_after_verification(self):
        root=self.root/'robustness';root.mkdir()
        save(root/'report.json',{'status':'complete','conditions':{}})
        save(root/'verification.json',{'all_passed':True,'report_sha256':sha(root/'report.json')})
        self.assertEqual(MotionReplay(self.root).stress['status'],'complete')
        save(root/'report.json',{'status':'complete','conditions':{'invented':1}})
        with self.assertRaisesRegex(ValueError,'stress report not verified'):MotionReplay(self.root)

    def test_rejects_modified_weights(self):
        self.models['neural_linear']['bias'][0]=1
        np.savez_compressed(self.root/'neural_linear-readout.npz',**self.models['neural_linear'])
        with self.assertRaisesRegex(ValueError,'weights changed'):MotionReplay(self.root)

    def test_rejects_row_label_change_and_unknown_trial(self):
        replay=MotionReplay(self.root)
        with self.assertRaisesRegex(ValueError,'unknown refinement trial'):replay.trial('intact',999,0,0,'right')
        self.rows[0]['kind']='left';save(self.root/'rows.json',self.rows)
        with self.assertRaisesRegex(ValueError,'labels changed'):MotionReplay(self.root)


if __name__=='__main__':unittest.main()
