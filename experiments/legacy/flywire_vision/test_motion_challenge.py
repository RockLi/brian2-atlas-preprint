import unittest

import numpy as np

from .motion_challenge import (
    KINDS, balanced_orbit_seeds, centers, diagnose, endpoint_hashes,
    first_frame_rule, iter_orbit, lattice, movie, parameters, render,
)


class MotionChallengeChecks(unittest.TestCase):
    def test_opposites_are_exact_reversals_with_valid_luminance(self):
        for seed in (13, 198, 907):
            for travel in (.4, .8, 1.2):
                right = movie('right', seed, travel)
                up = movie('up', seed, travel)
                np.testing.assert_array_equal(movie('left', seed, travel), right[::-1])
                np.testing.assert_array_equal(movie('down', seed, travel), up[::-1])
                self.assertEqual(right.shape, (40, 48, 48))
                self.assertEqual(right.dtype, np.float32)
                self.assertTrue(np.isfinite(right).all())
                self.assertGreaterEqual(float(right.min()), .099999)
                self.assertLessEqual(float(right.max()), .900001)
                np.testing.assert_array_equal(right[:, 0], right[:, -1])
                np.testing.assert_array_equal(right[:, :, 0], right[:, :, -1])

    def test_wrapping_preserves_motion_sign_and_constant_velocity(self):
        crossed_seam = False
        for seed in range(20):
            for kind, expected in zip(KINDS, ((1, 0), (-1, 0), (0, -1), (0, 1))):
                position = centers(kind, seed)
                self.assertTrue(np.all((position >= -1) & (position < 1)))
                delta = np.diff(position, axis=0)
                crossed_seam |= bool(np.any(np.abs(delta) > 1))
                unwrapped_delta = (delta + 1) % 2 - 1
                np.testing.assert_allclose(unwrapped_delta,
                    np.broadcast_to(np.array(expected) * .8 / 39, (39, 2)), atol=1e-14)
        self.assertTrue(crossed_seam)

    def test_endpoint_images_have_identical_direction_multisets(self):
        for seed, travel in ((918, .8), (712, .4), (301, 1.2)):
            counts = endpoint_hashes(seed, travel)
            for endpoint in ('first', 'last'):
                self.assertTrue(all(counts[endpoint][kind] == counts[endpoint]['right'] for kind in KINDS))
                self.assertEqual(sum(counts[endpoint]['right'].values()), lattice(travel)[1] ** 2)

    def test_orbit_labels_share_nuisances_and_pixels_encode_no_static_label(self):
        rows, endpoint_predictions = [], []
        for row, frames in iter_orbit(71):
            rows.append(row)
            endpoint_predictions.append(int(first_frame_rule(frames[:1])[0]))
        self.assertEqual(len(rows), 100)
        self.assertEqual({r['group'] for r in rows}, {71})
        self.assertEqual(len({r['width'] for r in rows}), 1)
        self.assertEqual(len({r['polarity'] for r in rows}), 1)
        for phase_start in range(0, len(rows), 4):
            group = rows[phase_start:phase_start + 4]
            self.assertEqual([r['kind'] for r in group], list(KINDS))
            self.assertEqual(len({tuple(r['phase_index']) for r in group}), 1)
        self.assertEqual(np.mean(np.array(endpoint_predictions) == np.tile(np.arange(4), 25)), .25)

    def test_fresh_seed_static_rule_and_polarity_balancing(self):
        report = diagnose(start=930_000, seeds=512)
        self.assertLess(abs(report['uniform_phase_probe']['accuracy'] - .25), .05)
        self.assertTrue(all(r['first'] and r['last'] for r in report['strict_orbit_endpoint_multiset_checks']))
        seeds = balanced_orbit_seeds(960_000, 8)
        self.assertEqual(len(set(seeds)), 8)
        self.assertEqual(sum(parameters(s)['polarity'] == 1 for s in seeds), 4)

    def test_rejects_unsupported_parameters(self):
        for travel in (0, -1, float('nan'), .3333333333333, 2.1):
            with self.assertRaises(ValueError):
                movie('right', travel=travel)
        for phase in ((1.5, 0), (5, 0), (-1, 0), (1,)):
            with self.assertRaises(ValueError):
                movie('right', phase=phase)
        with self.assertRaises(ValueError):
            movie('looming')
        with self.assertRaises(ValueError):
            balanced_orbit_seeds(1, 3)
        with self.assertRaises(ValueError):
            render(np.array([[float('nan'), 0]]), 1)


if __name__ == '__main__':
    unittest.main()
