"""Fit-only standardization and regularized ten-output least-squares readout."""
import numpy as np
from scipy.linalg import solve


def fit(x, labels, alpha):
    x, labels = np.asarray(x, dtype=float), np.asarray(labels)
    if x.ndim != 2 or len(x) != len(labels) or not np.isfinite(x).all():
        raise ValueError("invalid training features")
    if not np.isin(labels, np.arange(10)).all() or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("invalid labels or regularization")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    z = (x-mean)/scale
    y = np.eye(10)[labels.astype(int)]
    bias = y.mean(axis=0)
    y -= bias
    # Dual solve keeps development subsets inexpensive; full training uses primal.
    if len(z) < z.shape[1]:
        weights = z.T @ solve(z@z.T + alpha*np.eye(len(z)), y, assume_a="pos")
    else:
        weights = solve(z.T@z + alpha*np.eye(z.shape[1]), z.T@y, assume_a="pos")
    return dict(mean=mean, scale=scale, weights=weights, bias=bias, alpha=np.array(alpha))


def predict(model, x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(model["mean"]) or not np.isfinite(x).all():
        raise ValueError("invalid inference features")
    return (((x-model["mean"])/model["scale"])@model["weights"] + model["bias"]).argmax(axis=1)


def metrics(labels, predicted):
    labels, predicted = np.asarray(labels, dtype=int), np.asarray(predicted, dtype=int)
    if labels.shape != predicted.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("invalid evaluation labels")
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    return {"count": len(labels), "accuracy": float(np.mean(labels == predicted)),
            "confusion_matrix": confusion.tolist()}


def select(x, y, vx, vy, alphas=(.1, 1., 10., 100.)):
    trials = []
    best = None
    for alpha in alphas:
        model = fit(x, y, alpha)
        score = metrics(vy, predict(model, vx))
        trials.append({"alpha": alpha, **score})
        if best is None or score["accuracy"] > best[0]:
            best = (score["accuracy"], model)
    return best[1], trials


def ridge_path(x, labels, alphas=(.1,1.,10.,100.), block_size=2048):
    """Full-dataset primal fits share one Gram matrix; bounded row temporaries."""
    x=np.asarray(x);labels=np.asarray(labels)
    if x.ndim!=2 or len(x)!=len(labels) or not np.isin(labels,np.arange(10)).all():
        raise ValueError('invalid training data')
    if not len(x) or not np.isfinite(x).all():raise ValueError('invalid training features')
    mean=x.mean(axis=0,dtype=np.float64);scale=x.std(axis=0,dtype=np.float64)
    scale[scale<1e-12]=1
    bias=np.bincount(labels.astype(int),minlength=10)/len(labels)
    gram=np.zeros((x.shape[1],x.shape[1]));rhs=np.zeros((x.shape[1],10))
    for start in range(0,len(x),block_size):
        stop=min(start+block_size,len(x));z=(x[start:stop]-mean)/scale
        y=np.eye(10)[labels[start:stop].astype(int)]-bias
        gram+=z.T@z;rhs+=z.T@y
    for alpha in alphas:
        if not np.isfinite(alpha) or alpha<=0:raise ValueError('invalid regularization')
        regularized=gram.copy();regularized.flat[::len(gram)+1]+=alpha
        yield dict(mean=mean,scale=scale,weights=solve(regularized,rhs,assume_a='pos'),
                   bias=bias,alpha=np.array(alpha))
