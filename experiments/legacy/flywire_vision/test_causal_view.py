import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .causal_view import CausalReplay
from .direction_study import evaluate
from .pilot import DIRECTIONS, fit
from .run_experiment import save


class CausalReplayChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        conditions=['intact','cut_input','matched_cut','static_first','scrambled']
        self.protocol={'schema':'flywire-causal-motion-v1','readout_indices':[3],'test_groups':[123], 'travel':2.0}
        save(self.root/'protocol.json',self.protocol)
        self.rows=[{'split':'test','condition':c,'group':123,'phase_index':[i,j],'kind':k,'polarity':'bright' if i==0 else 'dark'} for c in conditions for i in range(2) for j in range(2) for k in DIRECTIONS]
        save(self.root/'rows.json',self.rows)
        labels=np.array([DIRECTIONS.index(r['kind']) for r in self.rows]);count=len(labels)
        np.savez_compressed(self.root/'features.npz',neural_linear=np.zeros((count,8)),neural_motion=np.zeros((count,24)),input_motion=np.zeros((count,24)),labels=labels)
        self.models={'neural_linear':{'coefficients':np.zeros((8,4)),'bias':np.zeros(4)},'neural_motion':fit(np.zeros((4,24)),np.arange(4),.1),'input_motion':fit(np.zeros((4,24)),np.arange(4),.1),'labels_shuffled':fit(np.zeros((4,24)),np.arange(4),.1)}
        selected={}
        for name,m in self.models.items():
            np.savez_compressed(self.root/(name+'-readout.npz'),**m)
            selected[name]={'readout_sha256':self.sha(name+'-readout.npz')}
        save(self.root/'selection.json',{'protocol_sha256':self.sha('protocol.json'),'readouts':selected,'test_clips_executed':0})
        self.report={'schema':self.protocol['schema'],'status':'complete','protocol_sha256':self.sha('protocol.json'),'selection_sha256':self.sha('selection.json'),'features_sha256':self.sha('features.npz'),'readouts':selected,'test_clips_per_condition':16,'conditions':{}}
        for c in conditions:
            rows=[r for r in self.rows if r['condition']==c];y=[DIRECTIONS.index(r['kind']) for r in rows]
            self.report['conditions'][c]={n:evaluate(np.zeros(16,dtype=int),y,rows) for n in self.models}
        save(self.root/'report.json',self.report)

    def sha(self,name):return hashlib.sha256((self.root/name).read_bytes()).hexdigest()

    def test_accepts_recomputed_scores_and_rejects_changed_statistic(self):
        self.assertEqual(len(CausalReplay(self.root).lookup),80)
        self.report['conditions']['intact']['neural_motion']['accuracy']=1.
        save(self.root/'report.json',self.report)
        with self.assertRaisesRegex(ValueError,'statistic mismatch'):CausalReplay(self.root)

    def test_rejects_row_label_mismatch(self):
        self.rows[0]['kind']='left';save(self.root/'rows.json',self.rows)
        with self.assertRaisesRegex(ValueError,'labels and rows'):CausalReplay(self.root)

    def test_rejects_changed_weights(self):
        self.models['neural_linear']['bias'][0]=1
        np.savez_compressed(self.root/'neural_linear-readout.npz',**self.models['neural_linear'])
        with self.assertRaisesRegex(ValueError,'readout changed'):CausalReplay(self.root)


if __name__=='__main__':unittest.main()
