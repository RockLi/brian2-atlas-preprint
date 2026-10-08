"""External neural motion energy across temporal frequencies, without speed input.

Only downstream neural grids enter inference. Neither stimulus speed nor labels
select a filter. All family/axis/harmonic channels use the same frozen bank.
"""
import numpy as np


def energy_pairs(grids, frequencies, harmonics=(1, 2, 3)):
    grids = np.asarray(grids, dtype=float)
    if grids.ndim != 5 or grids.shape[2:] != (40, 12, 12):
        raise ValueError('expected batch x family x 40 x 12 x 12 grids')
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 1 or np.any(frequencies <= 0) or np.any(frequencies >= 20):
        raise ValueError('temporal frequencies must be within the Nyquist limit')
    temporal = np.exp(-2j*np.pi*np.arange(40)[:, None]*frequencies[None]/40)
    positive, negative = [], []
    for grid in (np.maximum(grids, 0), np.maximum(-grids, 0)):
        for axis in (4, 3):
            for k in harmonics:
                phase = np.exp(-2j*np.pi*k*np.arange(12)/12)
                z = (grid*phase).sum(4) if axis == 4 else (grid*phase[None, None, None, :, None]).sum(3)
                positive.append((np.abs(np.einsum('bfto,ts->bfos', z, temporal))**2).sum(2))
                negative.append((np.abs(np.einsum('bfto,ts->bfos', z, temporal.conj()))**2).sum(2))
    # Channels are ordered exactly as P7: sign, axis, harmonic, family.
    return np.concatenate(positive, axis=1), np.concatenate(negative, axis=1)


def bank_features(grids, recipe):
    frequencies = np.arange(1, 13, dtype=float)
    pos, neg = energy_pairs(grids, frequencies)
    if recipe['mode'] == 'broadband':
        mask = frequencies <= recipe['upper']
        p, n = pos[..., mask].sum(-1), neg[..., mask].sum(-1)
        return (p-n)/np.maximum(p+n, 1e-12)
    speeds = recipe['speeds']
    family_count = grids.shape[1]
    harmonics = np.tile(np.repeat([1, 2, 3], family_count), 4)
    pp = np.stack([pos[:, np.arange(pos.shape[1]), (harmonics*s-1).astype(int)] for s in speeds], -1)
    nn = np.stack([neg[:, np.arange(neg.shape[1]), (harmonics*s-1).astype(int)] for s in speeds], -1)
    if recipe['mode'] == 'concat':
        return ((pp-nn)/np.maximum(pp+nn, 1e-12)).reshape(len(grids), -1)
    if recipe['mode'] == 'sum':
        pp, nn = pp.sum(-1), nn.sum(-1)
    elif recipe['mode'] == 'max':
        which = np.argmax(pp+nn, axis=-1)[..., None]
        pp, nn = np.take_along_axis(pp, which, -1)[..., 0], np.take_along_axis(nn, which, -1)[..., 0]
    else:
        raise ValueError('unknown bank mode')
    return (pp-nn)/np.maximum(pp+nn, 1e-12)


def candidates():
    return ([{'mode': 'concat', 'speeds': [1]}] +
            [{'mode': mode, 'speeds': speeds} for speeds in ([1, 2], [1, 2, 3], [1, 2, 3, 4])
             for mode in ('concat', 'sum', 'max')] +
            [{'mode': 'broadband', 'upper': upper} for upper in (3, 6, 9, 12)])
