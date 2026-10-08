import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
spec = importlib.util.spec_from_file_location('kernel_readout', ROOT/'validation/flywire_mnist_kernel_readout.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


def test_kernel_solution_matches_independent_intercept_block_system():
    rng = np.random.default_rng(12)
    raw = rng.integers(0, 10, (70, 4, 8)); y = np.arange(70) % 10; n = 50
    fixed = dict(representation='sqrt_sum', gamma=1., alpha_per_sample=.02)
    summary, model = r.search(raw, y, n, fixed)
    x = r.representation(raw, 'sqrt_sum'); d, dv, band = r.distances(x, n)
    k = np.exp(-d/band); v = np.exp(-dv/band)
    # Solve KRR with a genuinely unpenalized intercept, without using centering helper.
    block = np.block([[k+np.eye(n), np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    solution = np.linalg.solve(block, np.vstack([np.eye(10)[y[:n]], np.zeros((1, 10))]))
    expected = v@solution[:n]+solution[n]
    _, centered, _, _ = r.center_kernel(k, v)
    actual = centered@model['coefficients']+model['bias']
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert summary['selected']['alpha'] == 1.


def test_bandwidth_and_fixed_model_do_not_use_validation_labels_or_values():
    rng = np.random.default_rng(8)
    raw = rng.integers(0, 7, (50, 4, 6)); labels = np.arange(50) % 10
    fixed = dict(representation='sqrt_time', gamma=.25, alpha_per_sample=.1)
    _, first = r.search(raw, labels, 30, fixed)
    changed_labels = labels.copy(); changed_labels[30:] = 9
    _, second = r.search(raw, changed_labels, 30, fixed)
    np.testing.assert_array_equal(first['predicted'], second['predicted'])
    raw[30:] *= 100
    _, third = r.search(raw, changed_labels, 30, fixed)
    np.testing.assert_array_equal(first['coefficients'], third['coefficients'])
    assert first['bandwidth'] == third['bandwidth']


def test_constant_activity_is_chance_and_paired_identical_has_zero_difference():
    labels = np.arange(70) % 10
    summary, model = r.search(np.zeros((70, 4, 2)), labels, 50,
                             dict(representation='sqrt_sum', gamma=1., alpha_per_sample=.1))
    assert summary['selected']['validation_accuracy'] == .1
    comparison = r.paired(labels[50:], model['predicted'], model['predicted'])
    assert comparison['difference'] == 0
    assert comparison['paired_bootstrap_95'] == [0., 0.]
    assert comparison['mcnemar_exact_p'] == 1.


def test_multi_seed_interval_resamples_images_together():
    spec = importlib.util.spec_from_file_location('readout_summary', ROOT/'validation/flywire_mnist_readout_summary.py')
    summary = importlib.util.module_from_spec(spec); spec.loader.exec_module(summary)
    neural = np.tile(np.arange(100) % 2 == 0, (3, 1))
    direct = np.zeros_like(neural)
    result = summary.joint_comparison(neural, direct)
    # Identical seeds give exactly the same interval as a single paired experiment.
    labels = np.zeros(100, int)
    single = r.paired(labels, np.where(neural[0], 0, 1), np.ones(100, int))
    assert result['paired_image_bootstrap_95'] == single['paired_bootstrap_95']
    assert result['mean_difference'] == .5
