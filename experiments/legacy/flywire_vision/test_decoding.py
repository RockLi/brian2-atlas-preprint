"""Readout timing boundaries and fit-only preprocessing checks."""
import unittest
import numpy as np
from .pilot import event_features, fit, scores


class ReadoutChecks(unittest.TestCase):
    def test_stimulus_window_and_cell_order(self):
        # Exclude warmup, tail and cells outside the frozen readout. Include both
        # edges of every 50 ms bin; preserve explicit, non-sorted cell order.
        indices = np.array([3, 3, 1, 3, 1, 3, 2])
        ticks = np.array([999, 1000, 1499, 1500, 4999, 5000, 1000])
        actual = event_features(indices, ticks, [3, 1], 4).reshape(8, 2)
        expected = np.zeros((8, 2))
        expected[0] = [1, 1]
        expected[1, 0] = 1
        expected[7, 1] = 1
        np.testing.assert_array_equal(actual, expected)

    def test_inference_cannot_refit_scaler_or_depend_on_batch(self):
        x = np.array([[0, 1, 7], [2, 3, 7], [4, 5, 7], [6, 7, 7]], dtype=np.float32)
        model = fit(x, np.arange(4), .1)
        np.testing.assert_array_equal(model['mean'], [3, 4, 7])
        self.assertEqual(model['scale'][2], 1)
        frozen = {key: value.copy() for key, value in model.items()}
        query = np.array([[1., 2., 7.]])
        alone = scores(model, query)
        with_outlier = scores(model, np.concatenate([query, [[1e6, -1e6, 7]]]))
        np.testing.assert_allclose(alone[0], with_outlier[0])
        for key, value in frozen.items():
            np.testing.assert_array_equal(model[key], value)


if __name__ == '__main__':
    unittest.main()
