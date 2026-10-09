"""Summarize matched-input Gate-3-qualified 5,120-neuron CPU replays.

Raw trials and unsuccessful runs remain in the replay schedule. The separate
independent-RNG screen remains visible, but a runtime ratio is qualified by
the deterministic same-event gate and frozen-compiler output audit.
"""

import argparse
import json
from pathlib import Path

from summarize_crosshost import cpu_replays, distribution


def replay_order_strata(replay_path: Path):
    rows = json.loads((replay_path / "schedule.json").read_text())
    measured = [row for row in rows if not row["warmup"]]
    if (len(measured) != 12 or
            any(row["exit_code"] for row in measured)):
        raise RuntimeError("six successful Brian2/Rust replay pairs required")
    strata = {}
    for backend in ("cpp", "rust"):
        positions = {}
        for position in (0, 1):
            trials = [row for row in measured if row["backend"] == backend
                      and row["order"] == position]
            if len(trials) != 3:
                raise RuntimeError("replay schedule must balance backend order")
            values = [json.loads((replay_path / Path(row["output"]).name)
                                 .read_text())["wall_seconds"] for row in trials]
            positions["first_in_pair" if position == 0 else
                      "second_in_pair"] = distribution(values)
        strata[backend] = positions
    return strata


def summary_for_host(gate_path: Path, matched_gate_path: Path,
                     replay_path: Path, compile_path: Path,
                     compiler_audit_path: Path):
    gate = json.loads(gate_path.read_text())
    matched_gate = json.loads(matched_gate_path.read_text())
    compilation = json.loads(compile_path.read_text())
    compiler_audit = json.loads(compiler_audit_path.read_text())
    if gate["network_size"] != 5120 or gate["seeds"] != [31, 32, 33, 34, 35]:
        raise RuntimeError(f"unexpected scientific gate protocol: {gate_path}")
    if (len(gate["metrics"]) != 8 or
            not all(metric in gate["metrics"] for metric in (
                "rate_E_Hz", "rate_I_Hz", "steady_nmda_total_E",
                "steady_voltage_E_V", "steady_voltage_I_V",
                "steady_nmda_current_E_A",
                "final_ee_nmda_x_sample_mean",
                "final_ee_nmda_gate_sample_mean"))):
        raise RuntimeError(f"missing a predeclared metric: {gate_path}")
    if (matched_gate.get("network_size") != 5120 or
            not matched_gate.get("deterministic_core_screen_pass") or
            matched_gate.get("failed_requirements") != []):
        raise RuntimeError(
            f"full-size same-event scientific gate failed: {matched_gate_path}")
    if (compilation["rustc_release"] != "1.98.1" or
            compilation["exit_code"] != 0 or
            "target-cpu=native" not in compilation["command"]):
        raise RuntimeError(f"native Rust 1.98.1 provenance failed: {compile_path}")
    replay = cpu_replays(replay_path, True, 8)
    order_strata = replay_order_strata(replay_path)
    scientific_pass = bool(matched_gate["deterministic_core_screen_pass"])
    compiler_equivalent = bool(compiler_audit[
        "compiler_variants_scientifically_identical"])
    runtime_ratio = (replay["rust_over_cpp_same_host_median_runtime_ratio"]
                     if scientific_pass and compiler_equivalent else None)
    return {
        "scientific_screen_pass": scientific_pass,
        "scientific_gate_kind": "deterministic same external events",
        "raw_matched_input_gate": str(matched_gate_path),
        "independent_rng_screen_pass": bool(
            gate["gate3_statistical_screen_pass"]),
        "failed_independent_rng_metrics": [
            name for name, metric in gate["metrics"].items()
            if not metric["within_reference_variability"]],
        "compiler": compilation,
        "compiler_variants_scientifically_identical": compiler_equivalent,
        "compiler_variants_audit": str(compiler_audit_path),
        "raw_scientific_gate": str(gate_path),
        "raw_replay_schedule": str(replay_path / "schedule.json"),
        "compiled_region": replay,
        "compiled_region_order_strata": order_strata,
        "gate3_qualified_rust_over_cpp_median_runtime_ratio": runtime_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    for host in ("27", "23"):
        parser.add_argument(f"--gate{host}", type=Path, required=True)
        parser.add_argument(f"--matched-gate{host}", type=Path, required=True)
        parser.add_argument(f"--replay{host}", type=Path, required=True)
        parser.add_argument(f"--compile{host}", type=Path, required=True)
        parser.add_argument(f"--compiler-audit{host}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "nmda2025-scale5120-cpu-validation-v1",
        "network_size": 5120,
        "thread_budget": 8,
        "biological_duration_s": 1.0,
        "precision": "float64",
        "scope": "six interleaved compiled standalone replays per route after one warmup, with three first and three second positions for each backend; first-run model construction, IR, compile, monitoring and end-to-end costs are separate",
        "paper_timing_denominator_used": False,
        "host27_native_arm64": summary_for_host(
            args.gate27, args.matched_gate27, args.replay27, args.compile27,
            args.compiler_audit27),
        "host23_native_x86_64_pinned_cpus_0_7": summary_for_host(
            args.gate23, args.matched_gate23, args.replay23, args.compile23,
            args.compiler_audit23),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        host: {
            "scientific_screen_pass": data["scientific_screen_pass"],
            "compiler_variants_scientifically_identical": data[
                "compiler_variants_scientifically_identical"],
            "median_runtime_ratio": data[
                "gate3_qualified_rust_over_cpp_median_runtime_ratio"],
            "independent_rng_screen_pass": data["independent_rng_screen_pass"],
            "failed_independent_rng_metrics": data[
                "failed_independent_rng_metrics"],
        }
        for host, data in report.items() if host.startswith("host")
    }, indent=2))


if __name__ == "__main__":
    main()
