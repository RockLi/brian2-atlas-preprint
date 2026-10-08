import unittest
import hashlib
import numpy as np
from .motion_stress import stimulus,CONDITIONS
from .pilot import DIRECTIONS


class MotionStressChecks(unittest.TestCase):
    def test_all_conditions_share_closed_endpoints_across_directions(self):
        for seed in (1120000,1120001):
            for condition in CONDITIONS:
                hashes=[]
                for kind in DIRECTIONS:
                    frames=stimulus(kind,seed,condition)
                    np.testing.assert_array_equal(frames[0],frames[-1])
                    self.assertEqual(frames.shape,(40,48,48))
                    self.assertTrue(np.isfinite(frames).all() and frames.min()>=0 and frames.max()<=1)
                    hashes.append(hashlib.sha256(frames[0].tobytes()).hexdigest())
                self.assertEqual(len(set(hashes)),1)

    def test_double_speed_opposites_are_exact_reversals(self):
        for a,b in [('right','left'),('up','down')]:
            np.testing.assert_array_equal(stimulus(a,1120000,'speed_2x')[::-1],stimulus(b,1120000,'speed_2x'))
        self.assertFalse(np.array_equal(stimulus('right',1120000,'baseline'),stimulus('right',1120000,'speed_2x')))

    def test_static_texture_is_direction_independent(self):
        # Entire static endpoint images, including texture, are identical for all labels.
        np.testing.assert_array_equal(stimulus('up',1120002,'texture')[0],stimulus('right',1120002,'texture')[0])
        self.assertFalse(np.array_equal(stimulus('right',1120002,'texture'),stimulus('right',1120002,'baseline')))


if __name__=='__main__':unittest.main()
