"""Periodic, endpoint-balanced motion benchmark; incompatible with legacy movies.

An independent statistical group is an entire seed/orbit, not one phase or clip.
See motion_challenge_design.md before allocating simulation or training budgets.
"""
import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


SCHEMA = 'flywire-periodic-motion-v1'
KINDS = ('right', 'left', 'up', 'down')
SIZE = 48
FRAMES = 40
FRAME_MS = 10.0
CONTRAST = .8


def lattice(travel=.8):
    """Return half-travel in torus lattice units (stride, order).

    The torus has circumference 2. Thus half-travel / circumference =
    travel / 4, and its denominator determines the finite phase orbit.
    Restrict denominators to prevent an accidental enormous simulation set.
    """
    if not np.isfinite(travel) or not 0 < travel <= 2:
        raise ValueError('travel must be finite and in (0, 2]')
    fraction = Fraction(str(float(travel))) / 4
    if fraction.denominator > 20:
        raise ValueError('travel / 4 must have an exact decimal-rational denominator <= 20')
    return fraction.numerator, fraction.denominator


def parameters(seed):
    """Nuisance parameters are independent of kind and common to the whole orbit."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-1., 1., 2)
    width = float(rng.uniform(.10, .16))
    polarity = int(rng.choice([-1, 1]))
    return {'phase': phase, 'width': width, 'polarity': polarity}


def balanced_orbit_seeds(start, count):
    """Choose an even number of whole orbits, equally bright and dark."""
    if type(count) is not int or count <= 0 or count % 2:
        raise ValueError('count must be a positive even integer')
    by_polarity = {-1: [], 1: []}
    seed = int(start)
    while min(map(len, by_polarity.values())) < count // 2:
        bucket = by_polarity[parameters(seed)['polarity']]
        if len(bucket) < count // 2:
            bucket.append(seed)
        seed += 1
    return sorted(by_polarity[-1] + by_polarity[1])


def _point(base_phase, index, order):
    # Reducing integer indices BEFORE float arithmetic is essential: all labels
    # then share bit-identical endpoint coordinates, including across a seam.
    return (base_phase + 2. * (np.asarray(index, dtype=int) % order) / order + 1.) % 2. - 1.


def centers(kind, seed=783, travel=.8, phase=(0, 0)):
    """Forty wrapped display coordinates; positive display y points downwards."""
    if kind not in KINDS:
        raise ValueError(f'unknown motion kind {kind}')
    stride, order = lattice(travel)
    index = np.asarray(phase)
    if index.shape != (2,) or not np.issubdtype(index.dtype, np.integer):
        raise ValueError('phase must contain two integer lattice indices')
    if np.any(index < 0) or np.any(index >= order):
        raise ValueError(f'phase indices must be in [0, {order})')
    if kind in ('left', 'down'):
        return centers({'left': 'right', 'down': 'up'}[kind], seed, travel, phase)[::-1].copy()
    base = parameters(seed)['phase']
    position = np.broadcast_to(_point(base, index, order), (FRAMES, 2)).copy()
    dimension, sign = (0, 1) if kind == 'right' else (1, -1)
    position[:, dimension] += sign * travel * np.linspace(-.5, .5, FRAMES)
    position = (position + 1.) % 2. - 1.
    for frame, step in ((0, -1), (-1, 1)):
        endpoint = index.copy()
        endpoint[dimension] += step * sign * stride
        position[frame] = _point(base, endpoint, order)
    return position


def render(position, seed):
    """Render minimum-image periodic Gaussian fields on the legacy display grid.

    The -1 and +1 boundary samples represent the same torus location. Keeping
    these repeated edges preserves the existing bilinear display calibration.
    """
    position = np.asarray(position, dtype=float)
    if position.ndim != 2 or position.shape[1] != 2 or not np.isfinite(position).all():
        raise ValueError('expected finite frame x 2 positions')
    nuisance = parameters(seed)
    x, y = np.meshgrid(np.linspace(-1., 1., SIZE), np.linspace(-1., 1., SIZE))
    dx = (x[None] - position[:, 0, None, None] + 1.) % 2. - 1.
    dy = (y[None] - position[:, 1, None, None] + 1.) % 2. - 1.
    field = np.exp(-(dx * dx + dy * dy) / (2 * nuisance['width'] ** 2))
    return (.5 + nuisance['polarity'] * .5 * CONTRAST * field).astype(np.float32)


def movie(kind, seed=783, travel=.8, phase=(0, 0)):
    """Generate one clip. Only complete iter_orbit collections are exactly balanced."""
    if kind in ('left', 'down'):
        return movie({'left': 'right', 'down': 'up'}[kind], seed, travel, phase)[::-1].copy()
    return render(centers(kind, seed, travel, phase), seed)


def iter_orbit(seed, travel=.8):
    """Yield (metadata, movie) for a complete atomic split group.

    Metadata must not be supplied as classifier input. Seeds are the grouping
    keys for split allocation and confidence-interval resampling.
    """
    _, order = lattice(travel)
    nuisance = parameters(seed)
    for i in range(order):
        for j in range(order):
            for kind in KINDS:
                yield {
                    'schema': SCHEMA, 'group': int(seed), 'seed': int(seed),
                    'phase_index': [i, j], 'phase_order': order, 'kind': kind,
                    'travel': float(travel), 'width': nuisance['width'],
                    'polarity': 'bright' if nuisance['polarity'] > 0 else 'dark',
                    'frames': FRAMES, 'size': SIZE, 'frame_ms': FRAME_MS,
                }, movie(kind, seed, travel, (i, j))


def endpoint_hashes(seed, travel=.8):
    """Exact image-multiset audit; render just endpoints to keep checks cheap."""
    _, order = lattice(travel)
    counts = {endpoint: {kind: Counter() for kind in KINDS} for endpoint in ('first', 'last')}
    for i in range(order):
        for j in range(order):
            for kind in KINDS:
                frames = render(centers(kind, seed, travel, (i, j))[[0, -1]], seed)
                for endpoint, frame in zip(counts, frames):
                    counts[endpoint][kind][sha256(frame.tobytes()).hexdigest()] += 1
    return counts


def first_frame_rule(frames):
    """A static position shortcut: use the strongest pixel's major-axis offset.

    This rule recognizes legacy start-position biases. It receives pixels only,
    never seed, phase, direction metadata, or frames after the first one.
    """
    pixels = np.asarray(frames)
    if pixels.ndim != 3 or pixels.shape[1:] != (SIZE, SIZE):
        raise ValueError('expected batch of first frames')
    maxima = np.argmax(np.abs(pixels - .5).reshape(len(pixels), -1), axis=1)
    iy, ix = np.divmod(maxima, SIZE)
    axis = np.linspace(-1., 1., SIZE)
    x, y = axis[ix], axis[iy]
    return np.where(np.abs(x) >= np.abs(y), np.where(x < 0, 0, 1), np.where(y > 0, 2, 3))


def diagnose(start=900_000, seeds=4096, travel=.8):
    """Pixel-only static rule on new uniform phases, plus strict orbit proofs."""
    if seeds < 1:
        raise ValueError('seeds must be positive')
    correct = np.zeros(4, dtype=int)
    for seed in range(start, start + seeds):
        first = np.concatenate([render(centers(kind, seed, travel)[:1], seed) for kind in KINDS])
        correct += first_frame_rule(first) == np.arange(4)
    audit_seeds = list(range(start + seeds, start + seeds + 8))
    audits = []
    for seed in audit_seeds:
        counts = endpoint_hashes(seed, travel)
        audits.append({'seed': seed, **{
            endpoint: all(histogram == by_kind[KINDS[0]] for histogram in by_kind.values())
            for endpoint, by_kind in counts.items()
        }})
    _, order = lattice(travel)
    return {
        'schema': SCHEMA, 'travel': travel, 'phase_order': order,
        'source_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'numpy_version': np.__version__,
        'clips_per_orbit': 4 * order ** 2,
        'uniform_phase_probe': {
            'first_seed': start, 'independent_seeds': seeds, 'clips': seeds * 4,
            'rule': 'strongest-pixel major-axis position; no fitting',
            'correct': int(correct.sum()),
            'accuracy': float(correct.sum() / (seeds * 4)),
            'accuracy_by_direction': dict(zip(KINDS, (correct / seeds).tolist())),
            'interpretation': 'Monte Carlo check only; single-phase sampling has no exact finite-sample guarantee',
        },
        'strict_orbit_endpoint_multiset_checks': audits,
        'strict_orbit_single_endpoint_accuracy_upper_bound': .25,
        'upper_bound_assumptions': 'deterministic classifier receives only one endpoint image; complete equally weighted orbit',
        'not_tested': ['connectome simulation', 'direction readout training', 'background or speed generalization'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=900_000)
    parser.add_argument('--seeds', type=int, default=4096)
    parser.add_argument('--travel', type=float, default=.8)
    args = parser.parse_args()
    result = diagnose(args.start, args.seeds, args.travel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
