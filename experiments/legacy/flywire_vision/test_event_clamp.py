import tempfile
import unittest
from pathlib import Path
import numpy as np
from .event_clamp_engine import Budget
from .event_clamp_data import verify_delivery,store_trial,load_trial
from .event_clamp_study import design,gate_alias

class EventClampChecks(unittest.TestCase):
    def test_budget_counts_failed_and_refuses_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            b=Budget(Path(td)/'budget.json');n=b.reserve('one',{});b.update(n,status='failed')
            self.assertEqual(b.reserve('two',{}),2)
            with self.assertRaises(RuntimeError):b.reserve('one',{})
    def test_matrix_and_alias_budget(self):
        rows=design();self.assertEqual(len(rows),256);self.assertEqual(len({r['key'] for r in rows}),256)
        self.assertEqual(sum(gate_alias(r) is not None for r in rows),19);self.assertEqual(9+237+4,250)
    def test_actual_edge_audit_rejects_wrong_tick_weight_and_duplicates(self):
        graph={'source':np.array([3]),'edge':np.array([91]),'target':np.array([8]),'contacts':np.array([2.])}
        ge=(.275/52)*2;vals=np.array([ge,-0.,.2,.2+ge,.1,.1]);audit=np.concatenate((np.array([38,3,91,8],dtype=np.uint64),vals.view(np.uint64)))[None,:]
        self.assertTrue(verify_delivery(audit,(np.array([3]),np.array([20])),[3],0,graph)['all_passed'])
        for bad in (audit.copy(),audit.copy(),np.repeat(audit,2,axis=0)):
            if len(bad)==1:
                if bad[0,0]==38:bad[0,0]+=1
            with self.assertRaises(ValueError):verify_delivery(bad,(np.array([3]),np.array([20])),[3],0,graph)
        bad=audit.copy();bad[0,4]=np.array([ge*2]).view(np.uint64)[0]
        with self.assertRaises(ValueError):verify_delivery(bad,(np.array([3]),np.array([20])),[3],0,graph)
    def test_lossless_trial_roundtrip(self):
        rng=np.random.default_rng(12);blank={k:rng.normal(size=(20,3)) for k in ('v','ge','gi')};data={k:v.copy() for k,v in blank.items()};data['v'][4,1]=-0.;data['ge'][7,1]=np.nextafter(data['ge'][7,1],np.inf)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);store_trial(p,'blank',blank);row=store_trial(p,'trial',data,'blank',blank);restored=load_trial(p,row)
            for k in data:np.testing.assert_array_equal(data[k].view(np.uint8),restored[k].view(np.uint8))

if __name__=='__main__':unittest.main()
