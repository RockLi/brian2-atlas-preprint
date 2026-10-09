"""Numerical audit for the separately labeled identical-input NMDA fixture."""

import argparse
import json
from pathlib import Path

import numpy as np


VOLTAGE_ATOL_V = 2e-13
NMDA_STATE_ATOL = 2e-12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp", type=Path, required=True)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpp_summary = json.loads(args.cpp.with_suffix(".json").read_text())
    rust_summary = json.loads(args.rust.with_suffix(".json").read_text())
    for name in ("shared_input_sha256", "original_source_sha256",
                 "network_size", "simulation_dt_s", "biological_duration_s",
                 "precision"):
        if cpp_summary[name] != rust_summary[name]:
            raise RuntimeError(f"identical-input fixture does not match on {name}")
    fields = {}
    reference_maximum = {}
    with np.load(args.cpp) as cpp, np.load(args.rust) as rust:
        if sorted(cpp.files) != sorted(rust.files):
            raise RuntimeError("scientific result field sets differ")
        for name in sorted(cpp.files):
            left, right = np.asarray(cpp[name]), np.asarray(rust[name])
            if left.shape != right.shape or left.dtype != right.dtype:
                raise RuntimeError(f"{name} has different shape or dtype")
            difference = np.abs(left.astype(np.float64) -
                                right.astype(np.float64))
            fields[name] = {
                "shape": list(left.shape), "dtype": str(left.dtype),
                "byte_equal": bool(left.tobytes() == right.tobytes()),
                "max_absolute_difference": float(difference.max())
                if difference.size else 0.0,
                "different_elements": int(np.count_nonzero(left != right)),
            }
            reference_maximum[name] = float(np.max(np.abs(left))) if left.size else 0.0
    exact_fields = ("rate_E_Hz", "rate_I_Hz", "rate_E_t_s", "rate_I_t_s",
                    "NMDA_EE_i", "NMDA_EE_j", "NMDA_EI_i", "NMDA_EI_j")
    bounded_fields = {name: VOLTAGE_ATOL_V for name in ("V_E", "V_I")}
    bounded_fields |= {name: NMDA_STATE_ATOL for projection in ("EE", "EI")
                       for name in (f"NMDA_{projection}_x",
                                    f"NMDA_{projection}_s_NMDA")}
    excitatory_count = int(cpp_summary["excitatory_count"])
    unit_roundoff = np.finfo(np.float64).eps / 2
    summation_gamma = ((excitatory_count - 1) * unit_roundoff /
                       (1 - (excitatory_count - 1) * unit_roundoff))
    bounded_fields |= {
        name: excitatory_count * NMDA_STATE_ATOL +
              summation_gamma * reference_maximum[name]
        for name in ("s_NMDA_tot_E", "s_NMDA_tot_I")}
    failures = ([name for name in exact_fields if not fields[name]["byte_equal"]] +
                [name for name, bound in bounded_fields.items()
                 if fields[name]["max_absolute_difference"] > bound])
    report = {
        "protocol": "author recurrent equations/topology/delay/RK4/f64/1s, diagnostic PoissonInput replacement by identical externally generated Bernoulli events at the same synapses schedule; this scientific input protocol differs from unchanged publication and is never used as a performance denominator",
        "network_size": cpp_summary["network_size"],
        "shared_input_sha256": cpp_summary["shared_input_sha256"],
        "original_source_sha256": cpp_summary["original_source_sha256"],
        "deterministic_bounds": {
            "voltage_absolute_V": VOLTAGE_ATOL_V,
            "nmda_x_and_gate_absolute": NMDA_STATE_ATOL,
            "summed_total_unit_roundoff": unit_roundoff,
            "summed_total_gamma_ne_minus_one": summation_gamma,
            "justification": "Reuse the documented float64 RK4 deterministic NMDA-core bounds (2e-13 V voltage and 2e-12 dimensionless x/gate) as a strict screen for the longer recurrent diagnostic. Exact event/rate/connectivity equality is required. The summed-total bound follows from at most NE edge-state deviations of that fixed gate bound plus the standard IEEE sequential-sum gamma_(NE-1) rounding envelope times the reference maximum. Individual raw errors are retained; no threshold is fitted to this run.",
        },
        "exact_field_requirements": exact_fields,
        "bounded_field_requirements": bounded_fields,
        "fields": fields,
        "failed_requirements": failures,
        "deterministic_core_screen_pass": not failures,
        "limitations": "diagnostic changes external-input generator and may use a smaller network size; it does not erase any failed original seeded statistical gate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "deterministic_core_screen_pass": report[
            "deterministic_core_screen_pass"],
        "failed_requirements": failures,
        "voltage_max_V": max(fields[name]["max_absolute_difference"]
                             for name in ("V_E", "V_I")),
        "nmda_gate_max": max(fields[f"NMDA_{projection}_s_NMDA"][
            "max_absolute_difference"] for projection in ("EE", "EI")),
    }, indent=2))


if __name__ == "__main__":
    main()
