import importlib.util
from pathlib import Path

import pytest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SPEC = importlib.util.spec_from_file_location(
    "performance_suite", EXAMPLES / "performance_suite.py")
performance_suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(performance_suite)


def toolchain(release="1.98.0", llvm="22.0.0",
              host="aarch64-apple-darwin"):
    return {
        "invoked_path": "/usr/bin/rustc",
        "resolved_path": "/toolchains/rust/bin/rustc",
        "release": release,
        "release_tuple": performance_suite.parse_version(release, "rustc"),
        "llvm_version": llvm,
        "llvm_version_tuple": performance_suite.parse_version(llvm, "LLVM"),
        "host": host,
    }


def test_parse_version_accepts_release_suffix():
    assert performance_suite.parse_version("1.99.0-nightly", "rustc") == (1, 99, 0)
    assert performance_suite.parse_version("22.1", "LLVM") == (22, 1, 0)


def test_validate_rustc_accepts_matching_native_toolchain():
    performance_suite.validate_rustc(
        toolchain(), expected_host="aarch64-apple-darwin")


def test_litwin_kumar_report_maps_to_strict_unified_case():
    report = {
        "threads": 8,
        "timings_seconds": {"aot-8": 0.004, "cpp-1": 0.024},
        "aot_parallel_speedup_over_cpp_serial": 6.0,
        "worker_count_exact": True,
        "cpp_close": True,
    }
    assert performance_suite.serial_cases("litwin_kumar", report) == [{
        "scenario": "litwin_kumar",
        "aot_loop_median_ms": 4.0,
        "cpp_loop_median_ms": 24.0,
        "aot_speedup_over_cpp": 6.0,
        "correctness_passed": True,
        "performance_gate_passed": True,
        "comparison": "exact AOT / numeric C++",
    }]


def test_expected_rustc_host_is_exact_for_supported_native_platforms():
    assert performance_suite.expected_rustc_host(
        "arm64", "Darwin") == "aarch64-apple-darwin"
    assert performance_suite.expected_rustc_host(
        "x86_64", "Linux", "glibc") == "x86_64-unknown-linux-gnu"
    assert performance_suite.expected_rustc_host(
        "aarch64", "Linux", "musl") == "aarch64-unknown-linux-musl"


@pytest.mark.parametrize(
    "candidate, message",
    [
        (toolchain(release="1.97.9"), "older than required 1.98.0"),
        (toolchain(llvm="21.9.0"), "older than required 22.0.0"),
        (toolchain(host="x86_64-apple-darwin"), "does not exactly match"),
        (toolchain(host="aarch64-unknown-linux-gnu"), "does not exactly match"),
    ],
)
def test_validate_rustc_rejects_invalid_performance_toolchain(candidate, message):
    with pytest.raises(RuntimeError, match=message):
        performance_suite.validate_rustc(
            candidate, expected_host="aarch64-apple-darwin")
