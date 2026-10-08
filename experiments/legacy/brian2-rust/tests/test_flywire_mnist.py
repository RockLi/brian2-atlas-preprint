import gzip
import json
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"python"), str(ROOT/"experiments")]
from flywire_mnist.config import Config
from flywire_mnist.dataset import read_idx, split
from flywire_mnist.encoding import encode, projection
from flywire_mnist.graph import load_graph
from flywire_mnist.model import RUNNER, build, features, instance
from flywire_mnist.backends import CPU, GPU
from flywire_mnist.readout import fit, predict


def test_idx_and_split(tmp_path):
    path = tmp_path/"sample.gz"
    path.write_bytes(gzip.compress(b"\0\0\x08\x01"+struct.pack(">I", 3)+bytes([1,2,3])))
    np.testing.assert_array_equal(read_idx(path), [1,2,3])
    path.write_bytes(gzip.compress(b"\0\0\x08\x01"+struct.pack(">I", 4)+bytes([1,2,3])))
    with pytest.raises(ValueError, match="payload"):
        read_idx(path)
    labels = np.tile(np.arange(10), 20)
    train, val = split(labels, validation=50)
    assert len(train) == 150 and len(val) == 50
    assert not np.intersect1d(train, val).size
    np.testing.assert_array_equal(np.bincount(labels[val]), np.full(10, 5))
    np.testing.assert_array_equal(split(labels, validation=50)[0], train)


def test_encoding_is_order_independent_and_label_free():
    c = Config()
    image = np.full((28,28), 255, dtype=np.uint8)
    a = encode(image, 123, c)
    encode(image, 456, c)
    b = encode(image, 123, c)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert len(a[0]) > 0 and a[1].min() >= c.edges[0] and a[1].max() < c.edges[-1]
    assert not len(encode(np.zeros_like(image), 123, c)[0])
    src, dst = projection(np.arange(8), 4, 783)
    assert all(len(set(dst[src == i])) == 4 for i in range(784))


def test_readout_fits_training_statistics_only():
    x = np.eye(10)
    model = fit(x, np.arange(10), .1)
    np.testing.assert_array_equal(predict(model, x), np.arange(10))
    np.testing.assert_allclose(model["mean"], .1)
    predict(model, x*1000)
    np.testing.assert_allclose(model["mean"], .1)


def model_fixture():
    c = Config(warmup_ms=10, stimulus_ms=20, tail_ms=5, fanout=2, input_rate_hz=120)
    g = load_graph(ROOT/"wasm/flywire-circuit.json")
    return c, g, build(g, c, monitor=True)


def reference(model, directory):
    from brian2_rust.results import load_results
    path = directory.with_suffix(".json")
    path.write_text(json.dumps(model))
    subprocess.run([str(RUNNER), str(path), str(directory)], check=True, capture_output=True)
    return load_results(model, directory)


def check_counters(model, result, graph, c):
    p = next(i for i,p in enumerate(model["definition"]["populations"]) if p["name"] == "flywire_neurons")
    pop = result["populations"][p]
    spikes, ticks = pop["indices"], pop["spike_ticks"]
    assert len(spikes) > 0  # the counter comparison must exercise actual spikes
    for k, (left, right) in enumerate(zip(c.edges, c.edges[1:])):
        expected = np.bincount(spikes[(ticks >= left) & (ticks < right)], minlength=len(graph.root_ids))
        np.testing.assert_array_equal(pop["states"][f"count{k}"], expected)


def test_aot_instance_reuse_reset_and_reference(tmp_path):
    c, g, template = model_fixture()
    a = instance(template, np.full((28,28), 255, dtype=np.uint8), 7, c)
    b = instance(template, np.zeros((28,28), dtype=np.uint8), 8, c)
    executor = CPU(template, tmp_path/"cpu")
    first, _middle, last = [executor.run(m, key) for m, key in [(a,"a"),(b,"b"),(a,"a-again")]]
    expected = reference(a, tmp_path/"reference")
    for p, q, r in zip(first["populations"], last["populations"], expected["populations"]):
        for state in p["states"]:
            np.testing.assert_array_equal(p["states"][state], q["states"][state])
            np.testing.assert_array_equal(p["states"][state], r["states"][state])
        # Reference exports only requested spike monitors; AOT also retains
        # internal generator histories needed for event delivery.
        if p["states"]:
            np.testing.assert_array_equal(p["spike_ticks"], r["spike_ticks"])
            np.testing.assert_array_equal(p["indices"], r["indices"])
    check_counters(a, first, g, c)
    plain = instance(build(g, c, monitor=True, counters=False),
                     np.full((28,28), 255, dtype=np.uint8), 7, c)
    observed = reference(plain, tmp_path/"without-counters")
    for p, q in zip(expected["populations"], observed["populations"]):
        for key in q["states"]:
            np.testing.assert_array_equal(p["states"][key], q["states"][key])
        np.testing.assert_array_equal(p["spike_ticks"], q["spike_ticks"])
        np.testing.assert_array_equal(p["indices"], q["indices"])
    # A dynamic-length loader must retain validation at the replacement boundary.
    from brian2_rust import write_compatible_instance, ArtifactCompatibilityError
    import copy
    invalid = copy.deepcopy(a)
    p = next(i for i,p in enumerate(a["definition"]["populations"]) if p["name"] == "pixels")
    invalid["instance"]["populations"][p]["spike_generator"]["spike_indices"][0] = 784
    with pytest.raises(ArtifactCompatibilityError, match="invalid"):
        write_compatible_instance(invalid, executor.native, tmp_path/"invalid.bin")
    assert not (tmp_path/"invalid.bin").exists()


@pytest.mark.skipif(__import__('os').environ.get("FLYWIRE_MNIST_TEST_METAL") != "1",
                    reason="explicit native Metal device test")
def test_metal_reset_counters_and_cpu_f32(tmp_path):
    from brian2_rust.metal import MetalExecutor
    c, g, template = model_fixture()
    a = instance(template, np.full((28,28), 255, dtype=np.uint8), 7, c)
    b = instance(template, np.zeros((28,28), dtype=np.uint8), 8, c)
    executor = GPU(template, tmp_path/"metal", backend="metal")
    try:
        first = executor.run(a, "a")
        executor.run(b, "b")
        last = executor.run(a, "a-again")
        # Compiled CPU-f32 mirror controls GPU lowering and floating-point profile.
        control = executor.previous.run(compute="cpu-f32")
        for p, q, r in zip(first["populations"], last["populations"], control["populations"]):
            for name in p["states"]:
                np.testing.assert_array_equal(p["states"][name], q["states"][name])
                np.testing.assert_array_equal(p["states"][name], r["states"][name])
        check_counters(a, first, g, c)
    finally:
        executor.close()


@pytest.mark.skipif(not __import__('os').environ.get("FLYWIRE_MNIST_MPI_SOURCE"),
                    reason="requires explicit MPI checkout and local launcher")
def test_mpi_two_ranks_match_cpu(tmp_path):
    import os
    from flywire_mnist.backends.mpi import MPI
    c, g, template = model_fixture()
    model = instance(template, np.full((28,28), 255, dtype=np.uint8), 7, c)
    mpi = MPI(model, tmp_path/"mpi", source=os.environ["FLYWIRE_MNIST_MPI_SOURCE"], ranks=2)
    actual = mpi.run(model, "white")
    expected = CPU(model, tmp_path/"cpu").run(model, "white")
    for p, q in zip(actual["populations"], expected["populations"]):
        for name in p["states"]:
            np.testing.assert_array_equal(p["states"][name], q["states"][name])
