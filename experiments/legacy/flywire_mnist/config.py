from dataclasses import asdict, dataclass
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    seed: int = 783
    warmup_ms: float = 100.0
    stimulus_ms: float = 200.0
    tail_ms: float = 50.0
    dt_ms: float = 0.1
    bins: int = 4
    input_rate_hz: float = 60.0
    input_weight_mv: float = 3.0
    fanout: int = 4
    kc_count: int = 512
    background_channels: int = 512
    background_rate_hz: float = 300.0
    background_weight_mv: float = 3.5

    def __post_init__(self):
        for key in ("seed", "bins", "fanout", "kc_count", "background_channels"):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ValueError(f"invalid {key}")
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {key}")
        if self.dt_ms != 0.1 or self.stimulus_ms <= 0:
            raise ValueError("keep the existing FlyWire dt=0.1 ms and positive stimulus duration")
        for value in (self.warmup_ms, self.stimulus_ms / self.bins, self.tail_ms):
            if not math.isclose(value / self.dt_ms, round(value / self.dt_ms), abs_tol=1e-9):
                raise ValueError("windows must be integer ticks")
        if max(self.input_rate_hz, self.background_rate_hz) * self.dt_ms / 1000 >= 1:
            raise ValueError("Bernoulli-per-tick input probability must be below one")

    @property
    def ticks(self):
        return round((self.warmup_ms + self.stimulus_ms + self.tail_ms) / self.dt_ms)

    @property
    def edges(self):
        first = round(self.warmup_ms / self.dt_ms)
        width = round(self.stimulus_ms / self.bins / self.dt_ms)
        return [first + k * width for k in range(self.bins + 1)]

    @property
    def identity(self):
        return digest(asdict(self))
