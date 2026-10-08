"""Independent checks of the scientific input contract (stdlib unittest)."""
import unittest
import numpy as np
from .mapping import parse_hexels, root_id
from .stimuli import KINDS, MovieConfig, movie, on_off, sample_hexels


class VisualInputContract(unittest.TestCase):
    def test_root_ids_never_round_through_float(self):
        value = "720575940603042273"
        self.assertEqual(root_id(value), value)
        self.assertNotEqual(int(float(value)), int(value))
        for bad in (float(value), "7.20575940603042273e17", "-1", "0", str(2**64)):
            with self.assertRaises(ValueError):
                root_id(bad)

    def test_explicit_coordinate_labels_define_orientation(self):
        rows = parse_hexels("p/q,2,1\n3,720575940603042273,\n1,,720575940603042275\n", "Mi1")
        self.assertEqual([(r["p"],r["q"]) for r in rows], [(3,2),(1,1)])
        self.assertEqual(rows[0]["root_id"], "720575940603042273")

    def test_reject_ambiguous_grids(self):
        for text in ("p/q,1,1\n1,,\n", "p/q,1\n1,\n1,\n", "p/q,1\n1\n"):
            with self.assertRaises(ValueError):
                parse_hexels(text, "Mi1")

    def test_opposite_directions_have_identical_frame_sets(self):
        for a,b in (("right","left"),("up","down"),("looming","receding")):
            np.testing.assert_array_equal(movie(a), movie(b)[::-1])
            np.testing.assert_allclose(movie(a).mean(0),movie(b).mean(0),atol=2e-7)

    def test_motion_labels_match_image_centroids(self):
        y,x = np.indices((48,48))
        for kind,axis,sign in (("right",x,1),("left",x,-1),("up",y,-1),("down",y,1)):
            contrast = np.abs(movie(kind)-.5)
            centers = (contrast*axis).sum((1,2))/contrast.sum((1,2))
            self.assertTrue(np.all(np.diff(centers)*sign > 0),kind)

    def test_radial_and_flicker_controls(self):
        for kind,sign in (("looming",1),("receding",-1)):
            area = np.abs(movie(kind)-.5).sum((1,2))
            self.assertTrue(np.all(np.diff(area)*sign > 0))
        static = movie("static")
        np.testing.assert_array_equal(static, np.broadcast_to(static[0],static.shape))
        flicker = movie("flicker")
        self.assertTrue(np.all(flicker[4:8] == .5))

    def test_on_off_preserves_changes_and_explicit_onset(self):
        frames = np.array([[[.75,.25]],[[.25,.75]],[[.5,.5]]],dtype=np.float32)
        on,off = on_off(frames)
        np.testing.assert_array_equal(on[0], [[.25,0]])
        np.testing.assert_array_equal(off[0], [[0,.25]])
        np.testing.assert_array_equal(.5+np.cumsum(on-off,axis=0),frames)
        self.assertFalse(np.any((on>0)&(off>0)))

    def test_hexel_sampling_recovers_affine_image(self):
        axis = np.linspace(-1,1,11)
        x,y = np.meshgrid(axis,axis)
        frames = np.stack([.5+.1*x+.2*y, .5-.1*x])
        # Origin; +p moves anterior and dorsal; +q moves posterior and dorsal.
        pq = np.array([[19,17],[20,17],[19,18]])
        values = sample_hexels(frames,pq)
        h = np.array([0,-np.sqrt(3)/44,np.sqrt(3)/44])
        v = np.array([0,-1/44,-1/44])
        np.testing.assert_allclose(values, np.stack([.5+.1*h+.2*v,.5-.1*h]),atol=3e-8)

    def test_reproducibility_and_range(self):
        for kind in KINDS:
            a=movie(kind)
            np.testing.assert_array_equal(a,movie(kind))
            self.assertEqual(a.shape,(40,48,48))
            self.assertTrue(np.all((a>=0)&(a<=1)))
        self.assertFalse(np.array_equal(movie("right",1),movie("right",2)))

    def test_invalid_parameters_and_outside_field_fail(self):
        for kwargs in ({"dt_ms":0},{"frame_ms":.15},{"contrast":float("nan")},{"frames":3},{"size":8.5}):
            with self.assertRaises(ValueError):
                MovieConfig(**kwargs)
        with self.assertRaises(ValueError):
            sample_hexels(movie("right"),[[100,100]])
        with self.assertRaises(ValueError):
            on_off(np.array([[[2.0]]]))


if __name__ == "__main__":
    unittest.main()
