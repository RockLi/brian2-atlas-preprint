"""Validate the deterministic NMDA CUDA diagnostics on L4 and A100."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_archives(reference_path, candidate_path):
    with np.load(reference_path) as reference, np.load(candidate_path) as candidate:
        names_equal = set(reference.files) == set(candidate.files)
        rows = {}
        for name in sorted(set(reference.files) | set(candidate.files)):
            if name not in reference or name not in candidate:
                rows[name] = {"present_in_both": False}
                continue
            left, right = np.asarray(reference[name]), np.asarray(candidate[name])
            row = {
                "present_in_both": True,
                "reference_shape": list(left.shape),
                "candidate_shape": list(right.shape),
                "shape_equal": left.shape == right.shape,
                "reference_dtype": str(left.dtype),
                "candidate_dtype": str(right.dtype),
            }
            if left.shape == right.shape:
                row["exact"] = bool(np.array_equal(left, right))
                if np.issubdtype(left.dtype, np.number):
                    a, b = left.astype(np.float64), right.astype(np.float64)
                    delta = np.abs(a - b)
                    scale = float(np.max(np.abs(a), initial=0.0))
                    row.update(max_abs=float(delta.max(initial=0.0)),
                               mean_abs=float(delta.mean()) if delta.size else 0.0,
                               reference_dynamic_scale=scale,
                               normalized_max_abs=(float(delta.max(initial=0.0)) / scale
                                                   if scale else 0.0))
            rows[name] = row
    return {"names_equal": names_equal, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l4", type=Path, required=True)
    parser.add_argument("--a100", type=Path, required=True)
    parser.add_argument("--f32-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {name: json.loads((path / "report.json").read_text())
               for name, path in (("l4", args.l4), ("a100", args.a100),
                                  ("f32_control", args.f32_control))}
    if {reports[name]["shared_input_sha256"] for name in ("l4", "a100")} != {
            "002b363ca9633f9356c06cfc1d5605aceb237e97f1977fc3c240bf83b102cce9"}:
        raise RuntimeError("full-size diagnostics did not use the expected shared input")

    within = {host: compare_archives(path / "cpp.npz", path / "cuda.npz")
              for host, path in (("l4", args.l4), ("a100", args.a100))}
    cross = {backend: compare_archives(args.l4 / f"{backend}.npz",
                                       args.a100 / f"{backend}.npz")
             for backend in ("cpp", "cuda")}
    exact_fields = {
        "rate_E_Hz", "rate_I_Hz", "rate_E_t_s", "rate_I_t_s",
        "lastspike_E", "lastspike_I", "not_refractory_E", "not_refractory_I",
        "label_E", "NMDA_EE_i", "NMDA_EE_j", "NMDA_EI_i", "NMDA_EI_j",
    }
    # Four RK4 stages and four principal storage/reduction layers give a
    # precision-normalized 16-epsilon screen. It applies to each field's full
    # reference dynamic range; discrete event and topology fields remain exact.
    normalized_tolerance = 16 * float(np.finfo(np.float32).eps)
    gates = {}
    for host, comparison in within.items():
        rows = comparison["rows"]
        continuous = [row for name, row in rows.items()
                      if name not in exact_fields and "normalized_max_abs" in row]
        gates[host] = {
            "field_names_and_shapes": (comparison["names_equal"] and
                                        all(row.get("shape_equal", False)
                                            for row in rows.values())),
            "exact_discrete_events_times_and_topology": all(
                rows[name].get("exact", False) for name in exact_fields),
            "continuous_state_within_normalized_tolerance": all(
                row["normalized_max_abs"] <= normalized_tolerance
                for row in continuous),
            "maximum_continuous_normalized_error": max(
                row["normalized_max_abs"] for row in continuous),
        }
        gates[host]["passed"] = all(gates[host][name] for name in (
            "field_names_and_shapes",
            "exact_discrete_events_times_and_topology",
            "continuous_state_within_normalized_tolerance",
        ))
    control = reports["f32_control"]["runs"]["cuda"]["summary"][
        "cuda_vs_cpu_f32_control"]
    cross_exact = {backend: (comparison["names_equal"] and
                             all(row.get("exact", False)
                                 for row in comparison["rows"].values()))
                   for backend, comparison in cross.items()}
    output = {
        "schema": "nmda-skaar-2025-cuda-shared-input-gate-v1",
        "status": "pass" if (all(gate["passed"] for gate in gates.values()) and
                              all(cross_exact.values()) and control["all_exact"])
                  else "fail",
        "scope": "deterministic 640-neuron, 10,000-step fixed-input CUDA float32 scientific screen; the unchanged publication performance baseline remains float64",
        "upstream_commit": "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e",
        "tolerance_contract": {
            "discrete_events_times_and_topology": "byte exact",
            "continuous_state": "max_abs / max_abs(reference) <= 16 * float32 epsilon",
            "normalized_tolerance": normalized_tolerance,
            "rationale": "four RK4 stages times four principal state/reduction/current/monitor rounding layers; this is a precision-normalized forward-error screen, while spike/event identity is required exactly",
        },
        "gates": gates,
        "cross_architecture_exact": cross_exact,
        "cpu_f32_control": {
            "network_size": 160,
            "steps": 10_000,
            "numeric_array_leaf_count": control["numeric_array_leaf_count"],
            "all_present": control["all_present"],
            "all_shapes_equal": control["all_shapes_equal"],
            "all_exact": control["all_exact"],
            "note": "same IR/plan CUDA versus generic CPU-f32 mirror; exact equality is a device implementation control, not an independent scientific oracle",
        },
        "archive_sha256": {
            host: {backend: sha256(path / f"{backend}.npz")
                   for backend in ("cpp", "cuda")}
            for host, path in (("l4", args.l4), ("a100", args.a100))
        },
        "within_host_details": within,
        "cross_architecture_details": cross,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "gates": gates,
                      "cross_architecture_exact": cross_exact,
                      "cpu_f32_control_exact": control["all_exact"]}, indent=2))
    if output["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
