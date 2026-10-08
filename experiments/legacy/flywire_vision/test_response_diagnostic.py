import unittest
import numpy as np
from .response_diagnostic import currents,J_COEF,L_COEF
from .onset_diagnostic import DELTA,BACKGROUNDS

class DiagnosticTests(unittest.TestCase):
    def trial(self,spike=False):
        n=60;v=np.zeros((n,1));v[0,0]=-.046 if spike else -.052;ge=np.full((n,1),10. if spike else .1);gi=np.full((n,1),.03);spikes=[];available=0
        for t in range(n-1):
            value=v[t,0]
            if t>=available:
                value+=.005*(-(value+.052)-ge[t,0]*value-gi[t,0]*(value+.070))
                if value>-.045:spikes.append(t);available=t+22;value=-.052
            v[t+1,0]=value
        return {'v':v,'ge':ge,'gi':gi,'cells':np.array([3]),'indices':np.full(len(spikes),3),'ticks':np.array(spikes,dtype=int)}
    def test_subthreshold_accounting(self):
        d=self.trial();c,e,s=currents(d);self.assertLess(e,1e-15);np.testing.assert_allclose(sum(c.values()),(d['v'][:,0]-d['v'][0,0])*1000,atol=1e-12)
    def test_reset_and_refractory_accounting(self):
        d=self.trial(True);c,e,s=currents(d);self.assertGreater(len(s),1);self.assertTrue(np.any(c['reset']));np.testing.assert_allclose(sum(c.values()),(d['v'][:,0]-d['v'][0,0])*1000,atol=1e-12)
    def test_corrupt_nonspike_step_rejected(self):
        d=self.trial();d['v'][10,0]+=.001
        with self.assertRaisesRegex(ValueError,'accounting'):currents(d)
    def test_decomposition_coefficients(self):
        for k in set(J_COEF)|set(L_COEF):self.assertEqual(J_COEF.get(k,0)+L_COEF.get(k,0),1 if k==('AB',200) else -1 if k==('BA',200) else 0)
    def test_bounded_single_factor(self):self.assertEqual((DELTA,BACKGROUNDS),(500,[785,787]))

if __name__=='__main__':unittest.main()
