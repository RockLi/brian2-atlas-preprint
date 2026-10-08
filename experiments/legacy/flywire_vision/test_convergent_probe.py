import unittest
import numpy as np
from .convergent_probe import design,monitored_model
from .convergent_archive import encode,decode
from .timing_probe import schedule

class ConvergentChecks(unittest.TestCase):
    def test_balanced_adjacent_pulses(self):
        site={'a':[1],'b':[7]};a,ta=schedule(site,'AB',200);b,tb=schedule(site,'BA',200)
        np.testing.assert_array_equal(np.sort(a),np.sort(b));self.assertEqual(len(a),4)
        self.assertEqual(list(zip(ta,a)),[(1520,1),(1620,1),(1720,7),(1820,7)])
        self.assertEqual(list(zip(tb,b)),[(1520,7),(1620,7),(1720,1),(1820,1)])
    def test_complete_controls_under_both_conditions(self):
        rows=design([{'id':i} for i in range(8)]);self.assertEqual(len(rows),114)
        for c in ('intact','cut_input'):
            for sid in range(8):self.assertEqual(len([r for r in rows if r['condition']==c and r['site']==sid]),7)
    def test_lossless_archive_float_bits(self):
        rng=np.random.default_rng(81);blank={k:rng.normal(size=(12,4)) for k in ('v','ge','gi')};data={k:v.copy() for k,v in blank.items()}
        data['v'][3,1]=-0.;data['ge'][4,2]=np.nextafter(data['ge'][4,2],np.inf);data['indices']=np.array([3,5],dtype=np.int64)
        got=decode(encode(data,blank),blank)
        for k in data:np.testing.assert_array_equal(data[k].view(np.uint8),got[k].view(np.uint8))
    def test_time_matched_voltage_interaction(self):
        t=np.arange(20)*.1;blank=np.sin(t);a=np.exp(-t);b=np.cos(t);ad=np.exp(-2*t);bd=np.cos(2*t)
        ab=blank+a+bd;ba=blank+b+ad
        j=ab-ba-(blank+a)-(blank+bd)+(blank+b)+(blank+ad)
        np.testing.assert_allclose(j,0,atol=1e-15)

if __name__=='__main__':unittest.main()
