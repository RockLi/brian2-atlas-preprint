import unittest
import numpy as np
from .timing_probe import schedule, design, event_counts, interaction, DELAYS, ONSET_TICK


class TimingProbeChecks(unittest.TestCase):
    site={'id':0,'a':[0,1,2,3],'b':[4,5,6,7]}

    def test_counts_identical_for_order_and_delay(self):
        for delay in (0,*DELAYS):
            for kind in ('AB','BA'):
                i,t=schedule(self.site,kind,delay)
                np.testing.assert_array_equal(np.bincount(i,minlength=8),np.full(8,2))
                self.assertEqual(len(set(zip(t,i))),16)
                self.assertEqual(list(zip(t,i)),sorted(zip(t,i)))

    def test_simultaneous_identical(self):
        for x,y in zip(schedule(self.site,'AB',0),schedule(self.site,'BA',0)):
            np.testing.assert_array_equal(x,y)

    def test_single_controls_reproduce_exact_packets(self):
        for d in DELAYS:
            for pair,left,right in [('AB','A0','Bd'),('BA','B0','Ad')]:
                i,t=schedule(self.site,pair,d);i0,t0=schedule(self.site,left,0);i1,t1=schedule(self.site,right,d)
                self.assertEqual(list(zip(t,i)),sorted([*zip(t0,i0),*zip(t1,i1)]))
                self.assertGreaterEqual(t.min(),ONSET_TICK)
                self.assertLess(t.max(),6000)

    def test_linear_time_matched_control_cancels(self):
        blank=np.array([3,7,2]);a0=np.array([2,-1,0]);b0=np.array([4,2,-1]);ad=np.array([1,3,5]);bd=np.array([2,0,4])
        np.testing.assert_array_equal(interaction(blank+a0+bd,blank+b0+ad,blank+a0,blank+b0,blank+ad,blank+bd),np.zeros(3))
        self.assertEqual(interaction(np.array([0],np.uint16),np.array([3],np.uint16),np.array([0]),np.array([0]),np.array([0]),np.array([0]))[0],-3)

    def test_event_time_bins_and_cell_order(self):
        totals,trace,slots,ticks=event_counts(np.array([11,99,10,11]),np.array([9,20,10,5999]),np.array([10,11]),np.array([0,4]))
        np.testing.assert_array_equal(totals,[1,2]);self.assertEqual(trace[0,4],1);self.assertEqual(trace[1,0],1);self.assertEqual(trace[599,4],1)
        np.testing.assert_array_equal(slots,[1,0,1]);self.assertEqual(trace.sum(),3)

    def test_observation_window_excludes_endpoint(self):
        from .timing_probe_windows import count_until
        data={'slots':np.array([1,2,1]),'ticks':np.array([1519,1520,1521])}
        np.testing.assert_array_equal(count_until(data,1520,4),[0,1,0,0])
        np.testing.assert_array_equal(count_until(data,1522,4),[0,2,1,0])

    def test_design_complete_and_unique(self):
        rows=design([{'id':i} for i in range(8)])
        self.assertEqual(len(rows),155)
        self.assertEqual(len({(r['condition'],r['site'],r['kind'],r['delay']) for r in rows}),155)
        self.assertEqual(sum(r['condition']=='intact' for r in rows),121)


if __name__=='__main__':unittest.main()
