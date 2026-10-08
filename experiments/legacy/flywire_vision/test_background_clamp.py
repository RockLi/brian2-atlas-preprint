"""Pre-freeze checks of P12b intervention invariants; no full-graph launches."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .background_clamp import Budget200,events_array,safe_union,shift_for_background,stimulus
from .event_clamp_study import SPECS
from .motion_refinement import read

class BackgroundClampChecks(unittest.TestCase):
    def test_background_conflict_minimal_common_shift(self):
        bg=events_array([(1504,10)])
        self.assertEqual(shift_for_background(bg,[10,11]),6)
        for shift in range(6):
            with self.assertRaises(ValueError):safe_union(bg,events_array([(1520+shift,10)]))
        safe_union(bg,events_array([(1526,10)]))

    def test_all_orders_preserve_exact_background_and_balanced_stimuli(self):
        site={'a':[2],'b':[3],'a_cell':10,'b_cell':11};bg=events_array([(1504,10),(2600,11)]);shift=shift_for_background(bg,[10,11])
        all_events={}
        for kind,delay in SPECS:
            vis,ev=stimulus(site,kind,delay,shift);merged=safe_union(bg,ev)
            self.assertEqual(len(merged[0]),len(bg[0])+len(ev[0]))
            original=set(zip(bg[1],bg[0]));actual=set(zip(merged[1],merged[0]))
            self.assertTrue(original<=actual)
            self.assertEqual(actual-original,set(zip(ev[1],ev[0])))
            np.testing.assert_array_equal(vis[1],ev[1]);all_events[kind,delay]=ev
            for source in (10,11):self.assertTrue(np.all(np.diff(merged[1][merged[0]==source])>=22))
        for key in (('AB',0),('AB',200),('BA',200)):
            self.assertEqual([np.sum(all_events[key][0]==s) for s in (10,11)],[2,2])
        for combined,single,source in [(('AB',200),('A0',0),10),(('AB',200),('Bd',200),11),(('BA',200),('B0',0),11),(('BA',200),('Ad',200),10)]:
            ev=all_events[combined];np.testing.assert_array_equal(ev[1][ev[0]==source],all_events[single][1])

    def test_reject_duplicates_close_spacing_and_unresolvable_background(self):
        bg=events_array([(1520,10)])
        for tick in (1520,1521,1541):
            with self.assertRaises(ValueError):safe_union(bg,events_array([(tick,10)]))
        safe_union(bg,events_array([(1542,10)]))
        with self.assertRaises(ValueError):events_array([(2,3),(2,3)])
        dense=events_array((tick,10) for tick in range(1480,1970,22))
        with self.assertRaises(ValueError):shift_for_background(dense,[10,11])

    def test_empty_background_identity(self):
        empty=events_array([]);ev=events_array([(1520,10),(1620,10)])
        self.assertEqual(shift_for_background(empty,[10,11]),0)
        merged=safe_union(empty,ev)
        for a,b in zip(merged,ev):np.testing.assert_array_equal(a,b)

    def test_failed_attempts_count_and_200_is_hard_cap(self):
        with tempfile.TemporaryDirectory() as td:
            b=Budget200(Path(td)/'budget.json');n=b.reserve('failed',{});b.update(n,status='failed')
            with self.assertRaises(RuntimeError):b.reserve('failed',{})
            for i in range(199):b.reserve(str(i),{})
            with self.assertRaisesRegex(RuntimeError,'200-run'):b.reserve('overflow',{})
            self.assertEqual(len(read(b.path)['attempts']),200)
            self.assertEqual(read(b.path)['attempts'][0]['status'],'failed')

if __name__=='__main__':unittest.main()
