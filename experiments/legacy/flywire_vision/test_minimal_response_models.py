import unittest
import numpy as np
from .minimal_response_models import linear,local,schedule,weights_for,GAIN,N

class Tests(unittest.TestCase):
    def test_latency(self):
        impulse=np.zeros(N);impulse[100]=1.;x=linear(impulse)
        self.assertTrue(np.all(x[:102]==0));self.assertAlmostEqual(x[102],.005*.052*1000)
    def test_linear_closed_form(self):
        a=np.zeros(N);a[12]=2.;x=linear(a);k=np.arange(N)-12;expected=np.zeros(N);mask=k>=2;expected[mask]=2*.005*.052*1000*(.995**(k[mask]-1)-.98**(k[mask]-1))/.015
        np.testing.assert_allclose(x,expected,atol=1e-12,rtol=1e-12)
    def test_balance_swap_preserve_total(self):
        pair=(3.,11.);original=linear(schedule(0,'AB',pair))-linear(schedule(0,'BA',pair))
        for name in ('balanced','swapped'):
            w=weights_for(pair,name);self.assertEqual(sum(w),sum(pair));out=linear(schedule(0,'AB',w))-linear(schedule(0,'BA',w));np.testing.assert_array_equal(out,0*original if name=='balanced' else -original)
    def test_local_reset(self):
        v,s=local(np.full(100,10.),np.zeros(100),-.046,np.zeros(100));self.assertGreater(len(s),2);self.assertTrue(np.all(np.diff(s)>=22));self.assertEqual(v[s[0]+1],-52.)
    def test_linear_full_integral_cancels(self):
        x=linear(schedule(0,'AB',(3.,11.)))-linear(schedule(0,'BA',(3.,11.)))
        self.assertLess(abs(x.sum()),1e-6);self.assertGreater(abs(x[1538:2038].sum()),1.)

if __name__=='__main__':unittest.main()
