import re
import unittest
from pathlib import Path
import numpy as np
from .repeat_design import change_monitor_only,MONITOR_PATTERN,subset,selection,SPECS,BACKGROUNDS
from .motion_refinement import read
from .background_clamp import events_array,safe_union,shift_for_background,stimulus

class RepeatChecks(unittest.TestCase):
    def test_native_patch_only_replaces_expected_passive_reads(self):
        root=Path('brian2-rust/validation');p=read(root/'flywire-vision-convergent-probe-v1/protocol.json');source=(root/'flywire-vision-event-clamp-v1/validation-v1/native/main.rs').read_text();newcells=list(range(200,224));patched=change_monitor_only(source,p['monitor_cells'],newcells)
        self.assertEqual(re.sub(MONITOR_PATTERN,'MONITOR',source),re.sub(MONITOR_PATTERN,'MONITOR',patched))
        self.assertEqual([int(m[2]) for m in re.findall(MONITOR_PATTERN,patched)][::3],newcells)
        with self.assertRaises(ValueError):change_monitor_only(source,p['monitor_cells'][:-1],newcells)
        with self.assertRaises(ValueError):change_monitor_only(source.replace('p1_samples_0.push','other.push',1),p['monitor_cells'],newcells)
    def test_new_anatomy_excludes_old_cells_and_spatial_blocks(self):
        p=read(Path('brian2-rust/validation/flywire-vision-convergent-probe-v1/protocol.json'));sites,log=selection(Path(p['artifact']),Path(p['parent']),p['sites']);old={s[k] for s in p['sites'] for k in ('target','a_cell','b_cell')};new=[s[k] for s in sites for k in ('target','a_cell','b_cell')]
        self.assertEqual(len(set(new)),24);self.assertFalse(old&set(new));self.assertEqual(len(log),8)
        for s in sites:
            self.assertGreaterEqual(s['old_target_separation'],.15);self.assertTrue(.02<=s['distance']<=.08);np.testing.assert_allclose(np.array(s['b_xy'])-s['a_xy'],s['signed_displacement'])
    def test_scope_and_held_background_order(self):
        self.assertEqual(BACKGROUNDS,(785,786,787,788));self.assertEqual(len(SPECS)*8*len(BACKGROUNDS),256);self.assertEqual(1+4+32+224+4,265)
    def test_background_union_matches_timed_singles(self):
        site={'a':[1],'b':[2],'a_cell':10,'b_cell':11};bg=events_array([(1504,10)]);shift=shift_for_background(bg,[10,11]);self.assertEqual(shift,6)
        for kind,d in SPECS:
            visual,ev=stimulus(site,kind,d,shift);merged=safe_union(bg,ev);self.assertEqual(len(merged[0]),len(ev[0])+1);self.assertIn((1504,10),list(zip(merged[1],merged[0])))
    def test_subset_keeps_exact_cell_order_and_source_audit(self):
        data={k:np.arange(24).reshape(6,4).astype(float) for k in ('v','ge','gi')};data.update(cells=np.array([10,20,30,40]),indices=np.array([10,20,30,40]),ticks=np.arange(4),audit=np.array([[0,10,1,0],[0,20,1,0],[0,30,1,0]],dtype=np.uint64),outgoing_indices=np.array([],dtype=int),outgoing_ticks=np.array([],dtype=int))
        got=subset(data,[40,10,30],[10,30]);np.testing.assert_array_equal(got['v'],data['v'][:,[3,0,2]]);np.testing.assert_array_equal(got['indices'],[10,30,40]);np.testing.assert_array_equal(got['audit'][:,1],[10,30]);got['v'][0,0]=-3;self.assertEqual(data['v'][0,3],3)

if __name__=='__main__':unittest.main()
