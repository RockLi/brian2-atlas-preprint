import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from mam_paper_interarea import functional_connectivity, synaptic_area_inputs


def test_fc_against_independent_pearson():
    rng = np.random.default_rng(778)
    x = rng.normal(size=(5, 10000))
    x[1] += x[0] * 2
    got, flat = functional_connectivity(x)
    np.testing.assert_allclose(got, np.corrcoef(x), rtol=2e-14, atol=2e-14)
    assert len(flat) == 0
    changed, _ = functional_connectivity(x * np.arange(1, 6)[:, None] + 11)
    np.testing.assert_allclose(got, changed, rtol=2e-14, atol=2e-14)


def test_constant_rows_are_explicit():
    matrix, flat = functional_connectivity(np.array([np.ones(20), np.arange(20)]))
    assert flat.tolist() == [0]
    assert np.isnan(matrix[0, 1]) and np.isnan(matrix[1, 0])
    assert np.array_equal(np.diag(matrix), [1., 1.])


def test_impulse_filter_alignment_and_absolute_inhibition():
    rates = np.zeros((2, 40)); rates[0, 20] = 1; rates[1, 20] = 2
    w = np.array([[2., -3.], [1., -5.]])
    k = np.array([[4., 6.], [2., 7.]])
    got = synaptic_area_inputs(rates, w, k, [1.5, 2.5], [0, 0], .5)
    expected = np.zeros(40)
    # A full-convolution impulse at 20 is sliced from offset (20-1)//2 = 9.
    amplitude = (1.5 * (8 + 36) + 2.5 * (2 + 70)) / 4
    expected[11:31] = amplitude * np.exp(-np.arange(20) / .5)
    np.testing.assert_allclose(got[0], expected, rtol=2e-15, atol=1e-20)
    assert got[0, 11] > 0 and got[0, 10] == 0


@pytest.mark.parametrize('shape', [(0, 20), (33, 20), (2, 19), (2, 100001)])
def test_fc_shape_limits(shape):
    with pytest.raises(ValueError):functional_connectivity(np.zeros(shape))


def test_nonfinite_rejected():
    x = np.zeros((2, 20)); x[0, 3] = np.nan
    with pytest.raises(ValueError):functional_connectivity(x)


@pytest.mark.parametrize('field,value', [('weights', [[float('nan')]]),
    ('indegrees', [[-1.]]), ('neuron_numbers', [0]), ('area_indices', [2]),
    ('tau_syn_ms', 0), ('population_rates', -np.ones((1, 20)))])
def test_invalid_synaptic_inputs(field, value):
    args = dict(population_rates=np.ones((1, 20)), weights=[[1.]], indegrees=[[1.]],
                neuron_numbers=[1.], area_indices=[0], tau_syn_ms=.5)
    args[field] = value
    with pytest.raises(ValueError):synaptic_area_inputs(**args)


def test_reference_catalog_rejects_corruption_and_symlinks(tmp_path):
    import hashlib
    from audit_mam_original_interarea import read_checked
    path = tmp_path / 'reference.npy'; path.write_bytes(b'original')
    catalog = {'files': {'reference.npy': {'bytes': 8,
        'sha256': hashlib.sha256(b'original').hexdigest()}}}
    assert read_checked(tmp_path, catalog, path.name) == path
    path.write_bytes(b'modified')
    with pytest.raises(ValueError):read_checked(tmp_path, catalog, path.name)
    path.unlink(); other = tmp_path / 'other'; other.write_bytes(b'original')
    path.symlink_to(other)
    with pytest.raises(ValueError):read_checked(tmp_path, catalog, path.name)
    with pytest.raises(ValueError):read_checked(tmp_path, catalog, '../reference.npy')
