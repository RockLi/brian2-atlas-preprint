"""Deterministic diagnostic movies, not a frozen train/test benchmark."""
from dataclasses import dataclass
import numpy as np

KINDS = ("right", "left", "up", "down", "looming", "receding", "static", "flicker")


@dataclass(frozen=True)
class MovieConfig:
    size: int = 48
    frames: int = 40
    frame_ms: float = 10.0
    dt_ms: float = 0.1
    contrast: float = 0.8
    travel: float = 0.8  # displacement in normalized display coordinates

    def __post_init__(self):
        if type(self.size) is not int or type(self.frames) is not int or self.size < 8 or self.frames < 4:
            raise ValueError("size and frames must be integers >= 8 and >= 4")
        if not np.isfinite([self.frame_ms, self.dt_ms, self.contrast, self.travel]).all():
            raise ValueError("non-finite movie parameter")
        if self.dt_ms <= 0 or self.frame_ms <= 0 or not 0 < self.contrast <= 1 or not 0 < self.travel <= 1:
            raise ValueError("invalid movie parameter range")
        if not np.isclose(self.frame_ms / self.dt_ms, round(self.frame_ms / self.dt_ms), rtol=0, atol=1e-9):
            raise ValueError("frame boundaries must be integral simulation ticks")
        if self.frame_ms < self.dt_ms:
            raise ValueError("frame must span at least one tick")


def movie(kind, seed=783, config=MovieConfig()):
    if kind not in KINDS:
        raise ValueError(f"unknown stimulus {kind}")
    # Paired labels share nuisance parameters and the exact same frames in reverse.
    if kind in ("left", "down", "receding"):
        base = {"left": "right", "down": "up", "receding": "looming"}[kind]
        return movie(base, seed, config)[::-1].copy()
    rng = np.random.default_rng(seed)
    cx, cy = rng.uniform(-0.15, 0.15, 2)
    width = rng.uniform(0.10, 0.16)
    polarity = rng.choice([-1, 1])
    axis = np.linspace(-1, 1, config.size)
    x, y = np.meshgrid(axis, axis)  # display y increases downwards
    t = np.linspace(-0.5, 0.5, config.frames)
    if kind in ("right", "up"):
        px = cx + (config.travel*t if kind == "right" else np.zeros_like(t))
        py = cy - (config.travel*t if kind == "up" else np.zeros_like(t))
        field = np.exp(-((x[None]-px[:, None, None])**2 +
                         (y[None]-py[:, None, None])**2)/(2*width**2))
    else:
        radius = np.linspace(0.08, 0.60, config.frames) if kind == "looming" else np.full(config.frames, .3)
        distance = np.hypot(x-cx, y-cy)
        field = np.clip((radius[:, None, None]-distance[None])/0.035 + .5, 0, 1)
        if kind == "flicker":
            field *= (np.arange(config.frames) % 8 < 4)[:, None, None]
    return (0.5 + polarity*0.5*config.contrast*field).astype(np.float32)


def on_off(frames):
    """Positive/negative frame differences from a uniform 0.5 prestimulus field.

    This engineering encoder is not a photoreceptor or lamina physiological model.
    """
    frames = np.asarray(frames)
    if frames.ndim != 3 or not all(frames.shape) or not np.isfinite(frames).all():
        raise ValueError("expected finite nonempty time x height x width movie")
    if np.any(frames < 0) or np.any(frames > 1):
        raise ValueError("luminance outside [0,1]")
    difference = np.diff(frames.astype(np.float32), axis=0,
                         prepend=np.full_like(frames[:1], .5, dtype=np.float32))
    return np.maximum(difference, 0), np.maximum(-difference, 0)


def sample_hexels(frames, coordinates):
    """Bilinear sampling of right-eye p/q locations using an ideal planar hex grid.

    x = sqrt(3)/2*(q-p) (posterior); y = -(p+q)/2 (ventral).
    Origin (19,17) and scale 22 are explicit engineering display calibration.
    This is not a calibrated angular retinal projection.
    """
    frames = np.asarray(frames)
    if frames.ndim != 3 or min(frames.shape) < 1 or not np.isfinite(frames).all():
        raise ValueError("invalid movie")
    pq = np.asarray(coordinates, dtype=float)
    if pq.ndim != 2 or pq.shape[1] != 2 or len(pq) == 0 or not np.isfinite(pq).all():
        raise ValueError("expected finite N x 2 p/q coordinates")
    p, q = (pq - [19, 17]).T
    x, y = np.sqrt(3)/2*(q-p)/22, -(p+q)/2/22
    if np.any(np.abs(x) > 1) or np.any(np.abs(y) > 1):
        raise ValueError("hexel outside calibrated display; do not silently clip")
    h, w = frames.shape[1:]
    u, v = (x+1)*(w-1)/2, (y+1)*(h-1)/2
    i, j = np.floor(u).astype(int), np.floor(v).astype(int)
    du, dv = u-i, v-j
    i1, j1 = np.minimum(i+1, w-1), np.minimum(j+1, h-1)
    return (frames[:, j, i]*(1-du)*(1-dv) + frames[:, j, i1]*du*(1-dv) +
            frames[:, j1, i]*(1-du)*dv + frames[:, j1, i1]*du*dv).astype(np.float32)
