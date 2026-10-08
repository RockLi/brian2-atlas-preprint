"""Noise-scaled, anatomically separated external motion readout of neural events.

The inference API receives only recorded downstream counts and a frozen model.
No pixels, input spikes, direction, seed or phase enter its feature transform.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from .refinement import spatial_projection
from .pilot import fit, scores


def binned_residual(raw, blank, recipe):
    start, stop, width = (int(recipe[k]) for k in ('start', 'stop', 'width'))
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != np.shape(blank) or raw.shape[1] != 60:
        raise ValueError('expected batch of 60-bin downstream neural counts')
    if not (0 <= start < stop <= 60 and width > 0 and (stop-start) % width == 0):
        raise ValueError('invalid aligned feature window')
    return (raw[:, start:stop]-blank[start:stop]).reshape(len(raw), (stop-start)//width, width, raw.shape[2]).sum(2)


def cell_weights(raw, blank, recipe):
    residual = binned_residual(raw, blank, recipe)
    if recipe['weight'] == 'none':
        return np.ones(raw.shape[2])
    if recipe['weight'] != 'std':
        raise ValueError('unknown neural weighting')
    return 1 / np.maximum(residual.std((0,1)), .2)


def pooled_grids(raw, blank, mapping, weights, recipe):
    residual = binned_residual(raw, blank, recipe)
    group_key = recipe['separation']
    if group_key not in ('family', 'subtype'):
        raise ValueError('unknown anatomical separation')
    count = 2 if group_key == 'family' else 8
    projections = [spatial_projection({**mapping, 'valid': mapping['valid'] & (mapping[group_key] == family)}).multiply(weights[:,None]) for family in range(count)]
    return np.stack([(residual.reshape(-1, residual.shape[2]) @ p).reshape(len(raw), residual.shape[1], 12, 12) for p in projections], 1)


def correlations(grids, recipe):
    transform = recipe['transform']
    if transform == 'signed':
        channels = (np.maximum(grids,0), np.maximum(-grids,0))
    elif transform == 'both':
        channels = (grids, np.maximum(grids,0))
    else:
        raise ValueError('unknown motion transform')
    width = int(recipe['width'])
    lags = {1:(1,2,4,8), 2:(1,2,4), 5:(1,2)}[width]
    values = []
    for data in channels:
        data = gaussian_filter(data, sigma=(0,0,0,.65,.65), mode='wrap')
        for lag in lags:
            a,b = data[:,:,:-lag],data[:,:,lag:]
            denominator = np.maximum(np.sqrt(np.sum(a*a,axis=(2,3,4))*np.sum(b*b,axis=(2,3,4))),1e-12)
            for axis in (4,3):
                for shift in (1,2,3):
                    values.append(np.sum(b*np.roll(a,shift,axis)-a*np.roll(b,shift,axis),axis=(2,3,4))/denominator)
    return np.concatenate(values,axis=1)


def spectral_energy(grids, recipe):
    """External opposite-direction traveling-wave energy; fixed training scale.

    Spatial frequencies 1/2/3 and matching temporal frequencies describe a
    full cycle per 400 ms. This is a speed-tuned engineered visual readout.
    Preserve orthogonal spatial locations until after taking energy so that
    unrelated local phases cannot cancel before motion measurement.
    """
    channels = ((grids, np.maximum(grids,0)) if recipe['transform']=='both' else
                (np.maximum(grids,0), np.maximum(-grids,0)))
    values=[]
    for grid in channels:
        for axis in (4,3):
            for k in (1,2,3):
                spatial=np.exp(-2j*np.pi*k*np.arange(12)/12)
                z=(grid*spatial).sum(4) if axis==4 else (grid*spatial[None,None,None,:,None]).sum(3)
                phase=np.exp(-2j*np.pi*k*np.arange(grid.shape[2])/(40/int(recipe['width'])))[:,None]
                positive=(np.abs((z*phase).sum(2))**2).sum(2)
                negative=(np.abs((z*np.conj(phase)).sum(2))**2).sum(2)
                values.append((positive-negative)/np.maximum(positive+negative,1e-12))
    return np.concatenate(values,axis=1)


def grid_features(grids, recipe):
    mode=recipe.get('feature','correlation')
    if mode=='correlation':return correlations(grids,recipe)
    if mode=='spectral':return spectral_energy(grids,recipe)
    if mode=='hybrid':return np.concatenate([correlations(grids,recipe),spectral_energy(grids,recipe)],axis=1)
    raise ValueError('unknown motion feature')


def transform(raw, blank, mapping, weights, recipe):
    return grid_features(pooled_grids(raw, blank, mapping, weights, recipe), recipe)


def fit_readout(features, labels, alpha, reverse=False):
    if reverse:
        features = np.concatenate([features,-features])
        labels = np.concatenate([labels, np.array([1,0,3,2])[labels]])
    return fit(features,labels,alpha)


def predict(model, features):
    return scores(model,features)
