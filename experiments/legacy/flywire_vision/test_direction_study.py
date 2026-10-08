import unittest
import numpy as np
from .direction_study import balanced_groups, polarity, pool_types, evaluate


class StudyChecks(unittest.TestCase):
    def test_group_selection_balances_polarity_and_keeps_splits_disjoint(self):
        splits = [balanced_groups(start, count) for start, count in ((10000, 32), (20000, 16), (30000, 16))]
        self.assertEqual(len(set(sum(splits, []))), 64)
        for groups in splits:
            self.assertEqual(sum(polarity(g) == 'bright' for g in groups), len(groups) // 2)
        with self.assertRaises(ValueError):
            balanced_groups(1, 3)

    def test_type_pooling_preserves_bins_and_counts(self):
        raw = np.arange(40).reshape(8, 5)
        actual = pool_types(raw.ravel(), [2, 3]).reshape(8, 2)
        np.testing.assert_array_equal(actual[:, 0], raw[:, :2].sum(1))
        np.testing.assert_array_equal(actual[:, 1], raw[:, 2:].sum(1))
        self.assertEqual(raw.sum(), actual.sum())

    def test_metrics_resample_groups_not_individual_clips(self):
        rows = [{'group': g, 'polarity': 'bright' if g == 0 else 'dark'} for g in (0, 1) for _ in range(4)]
        labels = np.tile(np.arange(4), 2)
        predicted = np.concatenate([labels[:4], (labels[4:] + 1) % 4])
        result = evaluate(predicted, labels, rows)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['group_accuracy'], {'0': 1., '1': 0.})
        self.assertEqual(result['group_bootstrap_95_interval'], [0., 1.])
        self.assertEqual(result['by_polarity'], {'bright': 1., 'dark': 0.})


if __name__ == '__main__':
    unittest.main()
