from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'validation'),str(ROOT/'experiments')]
import flywire_mnist_nystrom as n
import flywire_mnist_kernel_readout as exact


def test_all_landmarks_match_exact_kernel_with_unpenalized_intercept():
    rng=np.random.default_rng(4);raw=rng.integers(0,10,(90,4,12));labels=np.arange(90)%10
    model,report=n.fit(raw,labels,np.arange(60),landmarks=60,gamma=1.,alpha=.3,batch_size=13)
    x=exact.representation(raw,'sqrt_sum');d,dv,band=exact.distances(x,60)
    k,v,_,_=exact.center_kernel(np.exp(-d/band),np.exp(-dv/band))
    target=np.eye(10)[labels[:60]];bias=target.mean(0)
    expected=v@np.linalg.solve(k+np.eye(60)*.3,target-bias)+bias
    np.testing.assert_allclose(n.scores(model,raw,np.arange(60,90)),expected,rtol=1e-8,atol=1e-9)
    assert report['landmarks']==report['effective_rank']==60


def test_landmarks_bandwidth_and_fit_exclude_validation():
    rng=np.random.default_rng(12);raw=rng.random((80,3,5));labels=np.arange(80)%10
    rows=np.arange(0,60,2)
    first,_=n.fit(raw,labels,rows,landmarks=12,gamma=1.,alpha=1.,batch_size=7)
    others=np.setdiff1d(np.arange(80),rows);raw[others]*=1e6;labels[others]=9
    second,_=n.fit(raw,labels,rows,landmarks=12,gamma=1.,alpha=1.,batch_size=7)
    assert set(first['landmark_rows']).issubset(rows)
    for key in ['reference','coefficients','intercept','bandwidth']:
        np.testing.assert_array_equal(first[key],second[key])


def test_constant_activity_and_saved_model_roundtrip(tmp_path):
    raw=np.ones((70,4,3));labels=np.arange(70)%10
    model,report=n.fit(raw,labels,np.arange(50),landmarks=20,gamma=1.,alpha=1.,batch_size=11)
    assert report['effective_rank']==1
    prediction=n.scores(model,raw,np.arange(50,70)).argmax(1)
    assert np.all(prediction==prediction[0]) and np.mean(prediction==labels[50:])==.1
    path=tmp_path/'model.npz';np.savez_compressed(path,**model)
    with np.load(path,allow_pickle=False) as saved:
        np.testing.assert_array_equal(n.scores(dict(saved),raw),n.scores(model,raw))


def test_batch_bound_and_invalid_training_indices():
    raw=np.arange(600).reshape(100,2,3);labels=np.arange(100)%10
    _,report=n.fit(raw,labels,np.arange(80),landmarks=16,gamma=1.,alpha=1.,batch_size=8)
    assert report['normal_matrix_bytes']<=16*16*8
    assert report['max_kernel_block_bytes']==8*16*8
    with pytest.raises(ValueError,match='training'):
        n.fit(raw,labels,np.array([0,0,1]),landmarks=2,gamma=1.,alpha=1.)


def test_full_training_requires_exact_official_ids():
    from flywire_mnist_full_readout import training_rows
    ids=np.arange(60000)[::-1]
    np.testing.assert_array_equal(training_rows(ids),ids)
    for invalid in [np.arange(59999),np.r_[np.arange(59999),0],np.arange(60000)+1]:
        with pytest.raises(ValueError,match='each official training ID'):
            training_rows(invalid)
