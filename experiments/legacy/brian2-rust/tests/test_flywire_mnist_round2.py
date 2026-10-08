import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'python')]
spec = importlib.util.spec_from_file_location('round2', ROOT/'validation/flywire_mnist_round2.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


def test_confirmation_validation_is_disjoint_and_balanced():
    labels = np.tile(np.arange(10), 800)
    split = {'fit_ids': list(range(6000)), 'validation_ids': list(range(6000, 8000))}
    s = r.make_splits(split, labels)
    assert len(s['a_fit']) == 1000 and len(s['b_fit']) == 5000
    assert len(s['a_val']) == 500 and len(s['b_val']) == 1000
    assert not set(s['a_val']) & set(s['b_val'])
    for name, ids in s.items():
        assert np.all(np.bincount(labels[ids]) == len(ids)//10)
        assert not set(ids) & set(split['validation_ids'] if name.endswith('fit') else split['fit_ids'])


def test_candidate_budget_preserves_legacy_calibration():
    cases = r.candidates()
    assert len(cases) == len({x['name'] for x in cases}) == 14
    assert {(x['rate'], round(3*x['gain'])) for x in cases[:6]} == {(rate, amp) for rate in [20,60,120] for amp in [1,3]}
    assert all(x['mapping_seed'] == x['encoder_seed'] == 783 for x in cases)
    assert not any(x['cut_alpn'] for x in cases)


def test_fit_matches_existing_solver_and_constant_cut():
    from flywire_mnist.readout import fit, predict
    rng = np.random.default_rng(28)
    raw = rng.integers(0, 6, (90, 100)).astype(float); raw[:, 0] = 0; raw[:, 1] = 2
    labels = np.tile(np.arange(10), 9)
    summary, _, pred = r.fit_readout(raw, labels, 60, [100.])
    expected = predict(fit(raw[:60], labels[:60], 100.), raw[60:])
    np.testing.assert_array_equal(pred, expected)
    assert summary['active_fit_features'] == 98
    summary, _, pred = r.fit_readout(np.zeros((90, 12)), labels, 60, [500.])
    assert summary['selected']['validation_accuracy'] == .1
    assert summary['active_fit_features'] == 0
    assert summary['trials'][0]['alpha'] == 500.


def test_corrupt_shards_are_not_reused(tmp_path):
    ids = np.array([3, 7], dtype=np.int64)
    x = np.zeros((2, 4, 8), dtype=np.uint16); p = np.ones((2, 4, 3), dtype=np.uint16)
    path = tmp_path/'shard.npz'
    def save():
        np.savez(path, ids=ids, features=x, projected=p, case_id='case', payload_sha256=r.payload_hash(ids, x, p))
    save(); r.load_shard(path, ids, 'case', 8, 3)
    with pytest.raises(ValueError, match='corrupt'):
        r.load_shard(path, ids[::-1], 'case', 8, 3)
    with np.load(path) as f:
        data = dict(f)
    data['features'] = x.copy(); data['features'][0, 0, 0] = 1
    np.savez(path, **data)
    with pytest.raises(ValueError, match='corrupt'):
        r.load_shard(path, ids, 'case', 8, 3)
