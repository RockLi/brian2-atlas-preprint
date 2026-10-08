"""Correctness-only checks for the paper reproduction's shared RNG contract."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402, F401
from contextual_dendritic_frozen_rng import (  # noqa: E402
    FrozenRandomContract,
    b2_counter_normal,
    b2_counter_uniform,
)
from contextual_dendritic_reproduction import source_tree_digest  # noqa: E402


def _expected(function, stream: int) -> np.ndarray:
    return np.asarray(
        [[function(11, stream, tick, index) for tick in range(5)] for index in range(4)]
    )


def test_counter_reference_vectors_and_named_streams_are_stable():
    assert b2_counter_uniform(11, 22, 33, 44) == 0.2421522378054266
    assert b2_counter_normal(11, 22, 33, 44) == -0.9267051344621895
    contract = FrozenRandomContract(11)
    assert contract.stream("bernoulli", "inputs_1_to_area_A") == 173509963571379373
    assert contract.stream("normal", "somas_A") == 4905697467824426076
    assert contract.manifest()["streams"] == {
        "bernoulli:inputs_1_to_area_A": 173509963571379373,
        "normal:somas_A": 4905697467824426076,
    }


def test_source_tree_digest_uses_relative_contents_and_ignores_local_metadata(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "src" / "model_specs").mkdir(parents=True)
        (root / "src" / "area.py").write_text("MODEL = 1\n")
        (root / "src" / "model_specs" / "equations.txt").write_text("[Soma]\nV : 1\n")
    (first / "src" / "._area.py").write_text("appledouble noise")
    (second / "src" / "__pycache__").mkdir()
    (second / "src" / "__pycache__" / "area.pyc").write_bytes(b"cache noise")

    first_digest, first_count = source_tree_digest(first)
    second_digest, second_count = source_tree_digest(second)
    assert first_count == second_count == 2
    assert first_digest == second_digest

    (second / "src" / "area.py").write_text("MODEL = 2\n")
    assert source_tree_digest(second)[0] != first_digest


def test_cython_counter_functions_match_python_reference(tmp_path):
    previous_device = b.get_device()
    previous_target = b.prefs.codegen.target
    previous_cache = b.prefs.codegen.runtime.cython.cache_dir
    try:
        b.set_device("runtime")
        b.start_scope()
        b.prefs.codegen.target = "cython"
        b.prefs.codegen.runtime.cython.cache_dir = str(tmp_path / "cython-cache")
        group = b.NeuronGroup(
            4,
            "u = b2_counter_uniform(11, 22, timestep(t, dt), i) : 1\n"
            "z = b2_counter_normal(11, 23, timestep(t, dt), i) : 1",
            dt=0.1 * b.ms,
            namespace={
                "b2_counter_uniform": b2_counter_uniform,
                "b2_counter_normal": b2_counter_normal,
            },
        )
        monitor = b.StateMonitor(group, ("u", "z"), record=True, when="end")
        b.Network(group, monitor).run(0.5 * b.ms)
        np.testing.assert_array_equal(np.asarray(monitor.u), _expected(b2_counter_uniform, 22))
        np.testing.assert_allclose(
            np.asarray(monitor.z),
            _expected(b2_counter_normal, 23),
            rtol=1e-15,
            atol=1e-15,
        )
    finally:
        b.set_device(previous_device)
        b.prefs.codegen.target = previous_target
        b.prefs.codegen.runtime.cython.cache_dir = previous_cache


def test_rust_aot_counter_functions_match_python_reference(tmp_path):
    runner = Path(
        os.environ.get("B2_TEST_RUNNER", ROOT / "target" / "release" / "b2-runner")
    )
    if not runner.is_file():
        raise AssertionError(f"build the correctness runner first: {runner}")
    previous_device = b.get_device()
    device = None
    try:
        b.start_scope()
        b.set_device(
            "rust_standalone",
            runner=runner,
            directory=tmp_path / "rust-aot",
            engine="aot",
        )
        device = b.get_device()
        # The registered Brian device is process-global.  Re-activate it so a
        # runner cached by another test/session cannot override this fixture.
        device.reinit()
        device.activate(
            runner=runner,
            directory=tmp_path / "rust-aot",
            engine="aot",
        )
        group = b.NeuronGroup(
            4,
            "u : 1\nz : 1",
            dt=0.1 * b.ms,
            namespace={
                "b2_counter_uniform": b2_counter_uniform,
                "b2_counter_normal": b2_counter_normal,
            },
        )
        group.run_regularly(
            "u = b2_counter_uniform(11, 22, timestep(t, dt), i)\n"
            "z = b2_counter_normal(11, 23, timestep(t, dt), i)",
            when="start",
            order=-1,
        )
        monitor = b.StateMonitor(group, ("u", "z"), record=True, when="start")
        b.Network(group, monitor).run(0.5 * b.ms)
        np.testing.assert_array_equal(np.asarray(monitor.u), _expected(b2_counter_uniform, 22))
        np.testing.assert_allclose(
            np.asarray(monitor.z),
            _expected(b2_counter_normal, 23),
            rtol=1e-15,
            atol=1e-15,
        )
    finally:
        b.set_device(previous_device)
        if device is not None:
            device.reinit()
