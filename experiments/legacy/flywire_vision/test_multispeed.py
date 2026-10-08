import unittest
from collections import Counter
import numpy as np
from .multispeed_data import speed_movie, digest
from .multispeed_readout import bank_features, candidates
from .motion_readout import spectral_energy


class MultispeedChecks(unittest.TestCase):
    def test_closed_balanced_and_exact_reversal(self):
        for speed in (1, 2, 3):
            starts = {k: Counter() for k in ('right', 'left', 'up', 'down')}
            for i in range(2):
                for j in range(2):
                    movies = {k: speed_movie(k, 783, (i, j), speed) for k in starts}
                    for k, frames in movies.items():
                        np.testing.assert_array_equal(frames[0], frames[-1])
                        starts[k][digest(frames[0])] += 1
                    np.testing.assert_array_equal(movies['right'][::-1], movies['left'])
                    np.testing.assert_array_equal(movies['up'][::-1], movies['down'])
            self.assertTrue(all(v == starts['right'] for v in starts.values()))

    def test_one_speed_bank_reproduces_p7_and_reverse(self):
        grids = np.random.default_rng(711).normal(size=(3, 2, 40, 12, 12))
        x = bank_features(grids, {'mode': 'concat', 'speeds': [1]})
        np.testing.assert_allclose(x, spectral_energy(grids, {'transform': 'signed', 'width': 1}), atol=1e-13)
        for recipe in candidates():
            y = bank_features(grids, recipe)
            np.testing.assert_allclose(bank_features(grids[:, :, ::-1], recipe), -y, atol=1e-12)
            np.testing.assert_array_equal(bank_features(np.zeros_like(grids), recipe), np.zeros_like(y))

    def test_speed_two_matches_previous_stress_stimulus(self):
        from .motion_stress import stimulus, phase_for
        for kind in ('right', 'left', 'up', 'down'):
            np.testing.assert_array_equal(speed_movie(kind, 783, phase_for(kind), 2), stimulus(kind, 783, 'speed_2x'))

    def test_traveling_wave_direction_and_broadband_speed(self):
        for speed in (1, 2, 3):
            wave = np.cos(2*np.pi*(np.arange(12)[None]/12-speed*np.arange(40)[:, None]/40))
            grids = np.broadcast_to(wave[None, None, :, None, :], (1, 2, 40, 12, 12))
            x = bank_features(grids, {'mode': 'broadband', 'upper': 12})
            self.assertTrue(np.all(x[0, :2] < -.99))
        with self.assertRaises(ValueError):speed_movie('right', 1, (0, 0), 1.5)


if __name__ == '__main__':unittest.main()
