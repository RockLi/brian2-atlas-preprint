import unittest
import numpy as np
from .refinement import spatial_projection, fit_linear, linear_scores, encode_variant
from .simulation import SimulationConfig
from .sparse_refinement import fit_sparse


class RefinementChecks(unittest.TestCase):
    def test_sparse_selection_and_prediction_do_not_depend_on_test_batch(self):
        rng = np.random.default_rng(42)
        labels = np.tile(np.arange(4), 8)
        raw = rng.poisson(.2, (32, 8 * 48)).astype(float)
        raw[:, 7] += labels * 5
        for scaling in ('raw', 'floor0.25', 'standard'):
            model = fit_sparse(raw, labels, 8, scaling, .1)
            self.assertEqual(len(model['selected_cells']), 8)
            self.assertIn(7, model['selected_cells'])
            support = np.flatnonzero(np.any(model['coefficients'].reshape(8, 48, 4), axis=(0, 2)))
            self.assertEqual(set(support), set(model['selected_cells']))
            before = model['coefficients'].copy()
            alone = linear_scores(model, raw[:1])
            together = linear_scores(model, np.vstack([raw[:1], np.full((1, 384), 1e6)]))[:1]
            np.testing.assert_allclose(alone, together, atol=1e-10)
            np.testing.assert_array_equal(before, model['coefficients'])

    def test_spatial_pool_preserves_counts_for_mapped_cells(self):
        mapping = {'xy': np.array([[-1., -1.], [0., 0.], [1., 1.], [.2, -.3]]),
                   'valid': np.array([True, True, True, False])}
        projection = spatial_projection(mapping, 4)
        np.testing.assert_allclose(np.asarray(projection.sum(1)).ravel(), [1, 1, 1, 0])
        self.assertEqual(float((np.array([2, 3, 4, 99]) @ projection).sum()), 9)

    def test_linear_conversion_and_batch_independence(self):
        rng = np.random.default_rng(7)
        x = rng.poisson(.4, (16, 8 * 12)).astype(float)
        labels = np.tile(np.arange(4), 4)
        projection = spatial_projection({'xy': rng.uniform(-1, 1, (12, 2)), 'valid': np.ones(12, dtype=bool)})
        for recipe in ('raw_standard', 'fisher128_raw', 'spatial12_standard'):
            model = fit_linear(x, labels, recipe, .1, projection)
            single = linear_scores(model, x[:1])[0]
            combined = linear_scores(model, np.vstack([x[:1], np.full((1, 96), 1e6)]))[0]
            np.testing.assert_allclose(single, combined, rtol=1e-10, atol=1e-10)

    def test_temporal_encoder_is_causal_and_blank_is_silent(self):
        cfg = SimulationConfig()
        channels = [{'cell_type': 'Mi1', 'p': 19, 'q': 17}, {'cell_type': 'Tm1', 'p': 19, 'q': 17}]
        blank = np.full((40, 48, 48), .5, dtype=np.float32)
        self.assertEqual(len(encode_variant(blank, channels, cfg, 'temporal_difference_x4')[0]), 0)
        a = blank.copy();a[5:15] = .9
        b = a.copy();b[20:] = .1
        i, t, _ = encode_variant(a, channels, cfg, 'temporal_difference_x4')
        j, u, _ = encode_variant(b, channels, cfg, 'temporal_difference_x4')
        np.testing.assert_array_equal(i[t < 3000], j[u < 3000])
        np.testing.assert_array_equal(t[t < 3000], u[u < 3000])
        self.assertTrue(len(i) > 0)


if __name__ == '__main__':
    unittest.main()
