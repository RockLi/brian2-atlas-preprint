"""Artificial ALPN input: no labels enter any function in this module."""
import hashlib
import numpy as np


def keyed_rng(seed, stream, sample_id=0):
    key = f"flywire-mnist-v1/{seed}/{stream}/{sample_id}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:16], "little"))


def schedule(rates, start, stop, dt_ms, rng):
    rates = np.asarray(rates, dtype=float)
    probabilities = rates * dt_ms / 1000
    if rates.ndim != 1 or not np.isfinite(rates).all() or np.any(probabilities < 0) or np.any(probabilities >= 1):
        raise ValueError("invalid event rates")
    indices, ticks = [], []
    for channel, probability in enumerate(probabilities):
        if probability == 0:
            continue
        tick = start - 1
        while True:
            tick += int(rng.geometric(probability))
            if tick >= stop:
                break
            indices.append(channel); ticks.append(tick)
    indices, ticks = np.asarray(indices, dtype=np.int32), np.asarray(ticks, dtype=np.int64)
    order = np.lexsort((indices, ticks))
    return indices[order], ticks[order]


def encode(image, sample_id, config):
    image = np.asarray(image)
    if image.shape != (28, 28) or image.dtype != np.uint8:
        raise ValueError("input must be a 28×28 uint8 image")
    if type(sample_id) not in (int, np.int64) or sample_id < 0:
        raise ValueError("sample ID must be a nonnegative integer")
    rates = image.reshape(-1).astype(float) / 255 * config.input_rate_hz
    return schedule(rates, config.edges[0], config.edges[-1], config.dt_ms,
                    keyed_rng(config.seed, "pixels", sample_id))


def projection(targets, fanout, seed):
    targets = np.asarray(targets, dtype=np.int32)
    if len(np.unique(targets)) != len(targets) or not 1 <= fanout <= len(targets):
        raise ValueError("fanout exceeds distinct annotated input cells")
    rng = keyed_rng(seed, "projection")
    return np.repeat(np.arange(784, dtype=np.int32), fanout), np.concatenate([
        rng.choice(targets, fanout, replace=False) for _ in range(784)])
