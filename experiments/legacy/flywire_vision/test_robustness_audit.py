import unittest
import numpy as np
from .robustness_audit import audit_movie, centroid_rule, CONDITIONS
from .pilot import DIRECTIONS
from .stimuli import movie


class RobustnessAuditChecks(unittest.TestCase):
    def test_physical_shifts_preserve_reversal_and_valid_luminance(self):
        for seed in (80000, 80001):
            for condition in CONDITIONS:
                frames = audit_movie('right', seed, condition)
                self.assertEqual(frames.shape, (40, 48, 48))
                self.assertTrue(np.isfinite(frames).all())
                self.assertGreaterEqual(frames.min(), 0)
                self.assertLessEqual(frames.max(), 1)
                if condition not in ('static_first', 'static_last', 'scrambled_middle'):
                    np.testing.assert_array_equal(frames[::-1], audit_movie('left', seed, condition))

    def test_temporal_controls_preserve_the_claimed_information(self):
        original = movie('up', 80002)
        for condition, frame in (('static_first', original[0]), ('static_last', original[-1])):
            np.testing.assert_array_equal(audit_movie('up', 80002, condition), np.repeat(frame[None], 40, axis=0))
        shuffled = audit_movie('up', 80002, 'scrambled_middle')
        np.testing.assert_array_equal(shuffled[[0, -1]], original[[0, -1]])
        self.assertFalse(np.array_equal(shuffled[1:-1], original[1:-1]))
        self.assertEqual(sorted(map(bytes, shuffled)), sorted(map(bytes, original)))

    def test_original_endpoint_shortcut_is_reproduced_without_training(self):
        for seed in range(70000, 70010):
            for label, kind in enumerate(DIRECTIONS):
                self.assertEqual(centroid_rule(movie(kind, seed)), label)
                self.assertEqual(centroid_rule(movie(kind, seed), last=True), label)


if __name__ == '__main__':
    unittest.main()
