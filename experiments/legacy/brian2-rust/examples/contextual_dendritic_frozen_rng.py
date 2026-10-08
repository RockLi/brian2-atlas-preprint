"""Shared stochastic-input contract for the contextual-dendritic model.

The paper model uses Brian's runtime ``rand``/``randn`` functions. Brian Cython
and brian2-rust intentionally use different random-number generators, so a
seed alone cannot make their scientific trajectories comparable. This module
provides named counter functions and constructor adapters whose draws are a
pure function of ``(seed, stream, absolute tick, logical neuron index)``.

The Cython and Rust AOT implementations use the same unsigned-64-bit mixing
and Marsaglia polar arithmetic. No timing or benchmark policy lives here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from typing import Iterator

import brian2 as b
from brian2.codegen.generators.cython_generator import CythonCodeGenerator
import numpy as np


_MASK64 = (1 << 64) - 1
_U53_SCALE = 1.0 / 9_007_199_254_740_992.0
_ORIGINAL_NEURON_GROUP = b.NeuronGroup
_ORIGINAL_POISSON_GROUP = b.PoissonGroup


_CYTHON_UNIFORM = r"""
from libc.math cimport log, sqrt
from libc.stdint cimport int64_t, uint64_t

cdef inline uint64_t _b2_mix64(uint64_t value) noexcept nogil:
    value = (value ^ (value >> 30)) * <uint64_t>0xbf58476d1ce4e5b9
    value = (value ^ (value >> 27)) * <uint64_t>0x94d049bb133111eb
    return value ^ (value >> 31)

cdef inline double _b2_counter_uniform_draw(
        int64_t seed, int64_t stream, int64_t tick, int64_t index,
        uint64_t draw) noexcept nogil:
    cdef uint64_t counter = (
        <uint64_t>seed
        ^ (<uint64_t>stream * <uint64_t>0x9e3779b97f4a7c15)
        ^ (<uint64_t>tick * <uint64_t>0xd1b54a32d192ed03)
        ^ (<uint64_t>index * <uint64_t>0x94d049bb133111eb)
        ^ (draw * <uint64_t>0x369dea0f31a53f85)
    )
    return <double>(_b2_mix64(counter) >> 11) * (1.0 / 9007199254740992.0)

cdef inline double _b2_counter_uniform(
        int64_t seed, int64_t stream, int64_t tick,
        int64_t index) noexcept nogil:
    return _b2_counter_uniform_draw(seed, stream, tick, index, 0)
"""


_CYTHON_NORMAL = r"""
cdef inline double _b2_counter_normal(
        int64_t seed, int64_t stream, int64_t tick,
        int64_t index) noexcept nogil:
    cdef int64_t pair = index // 2
    cdef uint64_t draw = 0
    cdef double x1
    cdef double x2
    cdef double radius
    cdef double factor
    while True:
        x1 = 2.0 * _b2_counter_uniform_draw(seed, stream, tick, pair, draw) - 1.0
        x2 = 2.0 * _b2_counter_uniform_draw(seed, stream, tick, pair, draw + 1) - 1.0
        radius = x1*x1 + x2*x2
        if radius < 1.0 and radius != 0.0:
            factor = sqrt(-2.0 * log(radius) / radius)
            if index & 1 == 0:
                return factor*x1
            return factor*x2
        draw = draw + 2
"""


_C_UNIFORM = r"""
#include <stdint.h>

static uint64_t b2_mix64(uint64_t value) {
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static double b2_counter_uniform_draw(
        int64_t seed, int64_t stream, int64_t tick, int64_t index,
        uint64_t draw) {
    uint64_t counter = (uint64_t)seed
        ^ (uint64_t)stream * UINT64_C(0x9e3779b97f4a7c15)
        ^ (uint64_t)tick * UINT64_C(0xd1b54a32d192ed03)
        ^ (uint64_t)index * UINT64_C(0x94d049bb133111eb)
        ^ draw * UINT64_C(0x369dea0f31a53f85);
    return (double)(b2_mix64(counter) >> 11)
        * (1.0 / 9007199254740992.0);
}

double b2_counter_uniform(
        int64_t seed, int64_t stream, int64_t tick, int64_t index) {
    return b2_counter_uniform_draw(seed, stream, tick, index, UINT64_C(0));
}
"""


_C_NORMAL = r"""
#include <math.h>
#include <stdint.h>

static uint64_t b2_normal_mix64(uint64_t value) {
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static double b2_normal_uniform_draw(
        int64_t seed, int64_t stream, int64_t tick, int64_t index,
        uint64_t draw) {
    uint64_t counter = (uint64_t)seed
        ^ (uint64_t)stream * UINT64_C(0x9e3779b97f4a7c15)
        ^ (uint64_t)tick * UINT64_C(0xd1b54a32d192ed03)
        ^ (uint64_t)index * UINT64_C(0x94d049bb133111eb)
        ^ draw * UINT64_C(0x369dea0f31a53f85);
    return (double)(b2_normal_mix64(counter) >> 11)
        * (1.0 / 9007199254740992.0);
}

double b2_counter_normal(
        int64_t seed, int64_t stream, int64_t tick, int64_t index) {
    int64_t pair = index / 2;
    uint64_t draw = UINT64_C(0);
    for (;;) {
        double x1 = 2.0 * b2_normal_uniform_draw(
            seed, stream, tick, pair, draw) - 1.0;
        double x2 = 2.0 * b2_normal_uniform_draw(
            seed, stream, tick, pair, draw + UINT64_C(1)) - 1.0;
        double radius = x1*x1 + x2*x2;
        if (radius < 1.0 && radius != 0.0) {
            double factor = sqrt(-2.0 * log(radius) / radius);
            return factor * ((index & 1) == 0 ? x1 : x2);
        }
        draw += UINT64_C(2);
    }
}
"""


def _mix64_scalar(value: int) -> int:
    value &= _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _uniform_scalar(seed: int, stream: int, tick: int, index: int, draw: int = 0) -> float:
    counter = (
        (seed & _MASK64)
        ^ ((stream * 0x9E3779B97F4A7C15) & _MASK64)
        ^ ((tick * 0xD1B54A32D192ED03) & _MASK64)
        ^ ((index * 0x94D049BB133111EB) & _MASK64)
        ^ ((draw * 0x369DEA0F31A53F85) & _MASK64)
    )
    return (_mix64_scalar(counter) >> 11) * _U53_SCALE


def _broadcast_apply(function, *arguments):
    values = np.broadcast_arrays(*[np.asarray(value) for value in arguments])
    output = np.empty(values[0].shape, dtype=np.float64)
    flat = [value.reshape(-1) for value in values]
    for at in range(output.size):
        output.reshape(-1)[at] = function(*[int(value[at]) for value in flat])
    return output.item() if output.ndim == 0 else output


@b.implementation("b2ir-c-abi-v1", _C_UNIFORM, name="b2_counter_uniform")
@b.implementation(
    CythonCodeGenerator,
    code=_CYTHON_UNIFORM,
    name="_b2_counter_uniform",
)
@b.check_units(seed=1, stream=1, tick=1, index=1, result=1)
@b.declare_types(
    seed="integer", stream="integer", tick="integer", index="integer", result="float"
)
def b2_counter_uniform(seed, stream, tick, index):
    return _broadcast_apply(_uniform_scalar, seed, stream, tick, index)


def _normal_scalar(seed: int, stream: int, tick: int, index: int) -> float:
    pair = index // 2
    draw = 0
    while True:
        x1 = 2.0 * _uniform_scalar(seed, stream, tick, pair, draw) - 1.0
        x2 = 2.0 * _uniform_scalar(seed, stream, tick, pair, draw + 1) - 1.0
        radius = x1*x1 + x2*x2
        if radius < 1.0 and radius != 0.0:
            factor = np.sqrt(-2.0 * np.log(radius) / radius)
            return float(factor * (x1 if index & 1 == 0 else x2))
        draw = (draw + 2) & _MASK64


@b.implementation("b2ir-c-abi-v1", _C_NORMAL, name="b2_counter_normal")
@b.implementation(
    CythonCodeGenerator,
    code=_CYTHON_NORMAL,
    name="_b2_counter_normal",
    dependencies={"b2_counter_uniform": b2_counter_uniform},
)
@b.check_units(seed=1, stream=1, tick=1, index=1, result=1)
@b.declare_types(
    seed="integer", stream="integer", tick="integer", index="integer", result="float"
)
def b2_counter_normal(seed, stream, tick, index):
    return _broadcast_apply(_normal_scalar, seed, stream, tick, index)


@dataclass
class FrozenRandomContract:
    seed: int
    streams: dict[str, int] = field(default_factory=dict)

    def stream(self, role: str, name: str) -> int:
        key = f"{role}:{name}"
        if key not in self.streams:
            encoded = hashlib.sha256(key.encode()).digest()[:8]
            self.streams[key] = int.from_bytes(encoded, "little") & ((1 << 63) - 1)
        return self.streams[key]

    @property
    def function_namespace(self) -> dict[str, object]:
        return {
            "b2_counter_uniform": b2_counter_uniform,
            "b2_counter_normal": b2_counter_normal,
        }

    def poisson_group(self, n, rates, *, name: str, **kwargs):
        stream = self.stream("bernoulli", name)
        namespace = dict(kwargs.pop("namespace", {}) or {})
        namespace.update(self.function_namespace)
        if isinstance(rates, str):
            # brian2-rust populations intentionally require at least one
            # mutable state.  Brian's PoissonGroup has none, so retain one
            # inert state and inline the rate expression in the threshold.
            model = "frozen_rng_state : integer"
            threshold = (
                f"b2_counter_uniform({self.seed}, {stream}, timestep(t, dt), i) "
                f"< ({rates})*dt"
            )
            group = _ORIGINAL_NEURON_GROUP(
                n,
                model=model,
                threshold=threshold,
                namespace=namespace,
                name=name,
                **kwargs,
            )
            group.frozen_rng_state = 0
        else:
            threshold = (
                f"b2_counter_uniform({self.seed}, {stream}, timestep(t, dt), i) "
                "< rates*dt"
            )
            group = _ORIGINAL_NEURON_GROUP(
                n,
                model="rates : Hz",
                threshold=threshold,
                namespace=namespace,
                name=name,
                **kwargs,
            )
            group.rates = rates
        return group

    def _neuron_group(self, *args, **kwargs):
        threshold = kwargs.get("threshold")
        name = kwargs.get("name", "unnamed_neuron_group")
        if isinstance(threshold, str) and "rand()" in threshold:
            stream = self.stream("bernoulli", name)
            draw = (
                f"b2_counter_uniform({self.seed}, {stream}, timestep(t, dt), i)"
            )
            kwargs["threshold"] = threshold.replace("rand()", draw)
            namespace = dict(kwargs.get("namespace", {}) or {})
            namespace.update(self.function_namespace)
            kwargs["namespace"] = namespace
        return _ORIGINAL_NEURON_GROUP(*args, **kwargs)

    @contextmanager
    def patch_area_constructors(self) -> Iterator[None]:
        previous_neuron = b.NeuronGroup
        previous_poisson = b.PoissonGroup
        b.NeuronGroup = self._neuron_group
        b.PoissonGroup = self.poisson_group
        try:
            yield
        finally:
            b.NeuronGroup = previous_neuron
            b.PoissonGroup = previous_poisson

    def soma_noise_expression(self, soma_name: str) -> str:
        stream = self.stream("normal", soma_name)
        return (
            f"noise_soma = b2_counter_normal({self.seed}, {stream}, "
            "timestep(t, dt), i)"
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "contextual-dendritic-frozen-rng-v1",
            "algorithm": "splitmix64-counter-v1-marsaglia-polar",
            "seed": self.seed,
            "identity": ["seed", "named_stream", "absolute_tick", "logical_index"],
            "streams": dict(sorted(self.streams.items())),
        }
