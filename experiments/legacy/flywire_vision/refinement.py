"""Explicit visual-drive variants and linear readouts of real downstream spikes."""
import copy
import struct
import numpy as np
from scipy.sparse import csr_matrix
from .simulation import encode_movie, READOUT_TYPES


def encode_variant(frames, channels, config, mode='contrast'):
    if mode == 'contrast':
        return encode_movie(frames, channels, config)
    if mode != 'temporal_difference_x4':
        raise ValueError('unknown visual encoding')
    frames = np.asarray(frames)
    if frames.shape != (config.frames, 48, 48) or not np.isfinite(frames).all() or np.any((frames < 0) | (frames > 1)):
        raise ValueError('expected finite luminance movie')
    delta = np.diff(frames, axis=0, prepend=np.full_like(frames[:1], .5))
    # Causal frame differences; fixed gain, no label or per-clip normalization.
    drive = (.5 + np.clip(4 * delta, -.5, .5)).astype(np.float32)
    return encode_movie(drive, channels, config)


def gain_model(template, gain):
    from brian2_rust.protocol import attach_protocol
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('expected positive input gain')
    model = copy.deepcopy(template)
    index = next(i for i, s in enumerate(model['definition']['synapses']) if s['name'] == 'visual_connections')
    params = model['instance']['synapses'][index]['parameters']
    params['amplitude'] = [struct.pack('>d', gain / 52).hex()] * len(params['amplitude'])
    attach_protocol(model)
    return model


def anatomical_map(graph, channels, groups):
    from brian2_rust.binary_topology import inspect_csr, csr_arrays
    cells = np.concatenate([groups[t] for t in READOUT_TYPES])
    lookup = np.full(len(graph.root_ids), -1, dtype=int)
    lookup[cells] = np.arange(len(cells))
    family = np.concatenate([np.full(len(groups[t]), i // 4) for i, t in enumerate(READOUT_TYPES)])
    offsets, target, values = csr_arrays(inspect_csr(graph.csr))
    xy, weight = np.zeros((len(cells), 2)), np.zeros(len(cells))
    for row in channels:
        a, b = int(offsets[row['index']]), int(offsets[row['index'] + 1])
        slots = lookup[target[a:b]]
        valid = slots >= 0
        slots, contacts = slots[valid], values[0, a:b][valid]
        valid = (family[slots] == (0 if row['cell_type'] == 'Mi1' else 1)) & (contacts > 0)
        slots, contacts = slots[valid], contacts[valid]
        p, q = row['p'] - 19, row['q'] - 17
        position = np.array([np.sqrt(3) / 2 * (q - p), -(p + q) / 2]) / 22
        np.add.at(weight, slots, contacts)
        np.add.at(xy, slots, contacts[:, None] * position)
    valid = weight > 0
    xy[valid] /= weight[valid, None]
    return {'xy': xy, 'valid': valid, 'cells': cells, 'family': family}


def spatial_projection(mapping, size=12):
    xy, valid = mapping['xy'], mapping['valid']
    if np.any(np.abs(xy[valid]) > 1 + 1e-10):
        raise ValueError('spatial map outside field')
    points = (xy[valid] + 1) * (size - 1) / 2
    lower = np.floor(points).astype(int)
    fraction = points - lower
    selected = np.flatnonzero(valid)
    rows, columns, weights = [], [], []
    for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
        x = np.minimum(lower[:, 0] + dx, size - 1)
        y = np.minimum(lower[:, 1] + dy, size - 1)
        w = (fraction[:, 0] if dx else 1 - fraction[:, 0]) * (fraction[:, 1] if dy else 1 - fraction[:, 1])
        rows.extend(selected)
        columns.extend(y * size + x)
        weights.extend(w)
    return csr_matrix((weights, (rows, columns)), shape=(len(xy), size * size))


def fit_linear(raw, labels, recipe, alpha, projection):
    """Fit-only selection/scaling, collapsed to a linear map of raw spike counts.

    Anatomical pooling is a fixed external readout operation. Only neural spike
    features enter this function; no movie pixels or input spike train is used.
    """
    raw = np.asarray(raw, dtype=float)
    n, dimension = raw.shape
    cells = dimension // 8
    indices = None
    if recipe == 'spatial12_standard':
        x = (raw.reshape(-1, cells) @ projection).reshape(n, -1)
    elif recipe == 'fisher128_raw':
        variance = raw.var(0)
        between = np.stack([raw[labels == i].mean(0) for i in range(4)]).var(0)
        statistic = between / np.maximum(variance - between, 1e-3)
        selected = np.argsort(statistic.reshape(8, cells).sum(0), kind='stable')[::-1][:min(128, cells)]
        indices = (np.arange(8)[:, None] * cells + selected).ravel()
        x = raw[:, indices]
    elif recipe == 'raw_standard':
        x = raw
    else:
        raise ValueError('unknown readout recipe')
    mean = x.mean(0)
    scale = np.ones(x.shape[1]) if recipe == 'fisher128_raw' else x.std(0)
    scale[scale < 1e-8] = 1
    z = (x - mean) / scale
    dual = np.linalg.solve(z @ z.T / z.shape[1] + alpha * np.eye(n), np.eye(4)[labels])
    coef = (z.T @ dual) / z.shape[1] / scale[:, None]
    bias = -mean @ coef
    if recipe == 'spatial12_standard':
        coefficients = np.stack([projection @ c for c in coef.reshape(8, -1, 4)]).reshape(dimension, 4)
    elif indices is not None:
        coefficients = np.zeros((dimension, 4))
        coefficients[indices] = coef
    else:
        coefficients = coef
    # Conversion is algebraically equivalent to the training representation.
    np.testing.assert_allclose(raw @ coefficients + bias, z @ (z.T @ dual) / z.shape[1], atol=1e-10, rtol=1e-10)
    return {'coefficients': coefficients, 'bias': bias, 'alpha': np.array(alpha),
            'format': np.array('linear_raw_counts_v1'), 'recipe': np.array(recipe)}


def linear_scores(model, raw):
    return np.asarray(raw, dtype=float) @ model['coefficients'] + model['bias']
