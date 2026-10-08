#!/usr/bin/env python3
"""Summarize a completed Fig. 4 32-s backend ablation from frozen data.

This only reads hashed JSON field audits and numerical comparisons. It does
not restore checkpoints, simulate a network, or measure performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HASHES = {
    "reference": "535eadf74bf814f7250d15609409ba182f07a8fb12b8a7730dd89485ce3d49a1",
    "fallback": "18ddb348f6cda612de472bc22636c9e981205973dde018928af2e7b6228584dc",
    "compiled": "7c678e7a39b505868d6fc592705e3e48c0ae868d1e2debb6523a5373e16392ab",
    "fallback_comparison": "af7ebf336de9c6d93e4fb434da73e7ede929b93c1196d5da739ff8110a59cf68",
    "compiled_comparison": "c3b33990a65e3faa66a57f182cf97d054dfebe37d02ad162130294b586ce08ab",
}


def load(path: Path, label: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HASHES[label]:
        raise ValueError(f"frozen {label} report hash mismatch")
    return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    for label in HASHES:
        parser.add_argument(label, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an ablation summary")
    reports = {label: load(getattr(args, label), label) for label in HASHES}
    ref, fallback, compiled = (reports[name] for name in ("reference", "fallback", "compiled"))
    old_compare, new_compare = reports["fallback_comparison"], reports["compiled_comparison"]
    if any(report["checkpoint_time_seconds"] != 32.0 for report in (ref, fallback, compiled)):
        raise ValueError("unexpected checkpoint time")
    if not (ref["rng"] == fallback["rng"] == compiled["rng"]):
        raise ValueError("saved NumPy RNG/index fields are not equal")
    if not (set(ref["components"]) == set(fallback["components"]) == set(compiled["components"])):
        raise ValueError("checkpoint component sets differ")
    changed_by_compilation = []
    still_different_from_reference = []
    for component in sorted(ref["components"]):
        fields = ref["components"][component]
        if not (set(fields) == set(fallback["components"][component])
                == set(compiled["components"][component])):
            raise ValueError(f"field sets differ for {component}")
        for field in sorted(fields):
            name = f"{component}.{field}"
            a = fields[field]
            b = fallback["components"][component][field]
            c = compiled["components"][component][field]
            if b != c:
                changed_by_compilation.append(name)
                if field != "_spikequeue" or c != a or b == a:
                    raise ValueError(f"non-queue or non-restored change: {name}")
            if c != a:
                still_different_from_reference.append(name)
    if (len(changed_by_compilation) != 9 or len(still_different_from_reference) != 30
            or old_compare["different_fields"] != 39
            or new_compare["different_fields"] != 30
            or set(old_compare["differences"]) != set(changed_by_compilation + still_different_from_reference)
            or set(new_compare["differences"]) != set(still_different_from_reference)):
        raise ValueError("ablation field counts disagree with frozen numerical reports")
    result = {
        "schema": "contextual-dendritic-fig4-queue-ablation-summary-v1",
        "purpose": "first_32s_backend_ablation_pure_data_no_simulation_or_timing",
        "input_report_sha256": HASHES,
        "checkpoint_time_seconds": 32.0,
        "changed_by_compiled_queue_fields": changed_by_compilation,
        "changed_by_compiled_queue_count": len(changed_by_compilation),
        "queue_fields_now_equal_official": True,
        "nonqueue_candidate_fields_changed_by_compiled_queue": 0,
        "nonqueue_fields_still_different_from_official": still_different_from_reference,
        "nonqueue_fields_still_different_count": len(still_different_from_reference),
        "saved_numpy_rng_and_cython_buffer_indices_equal": True,
        "unconsumed_cython_buffer_values_compared": False,
        "first_32s_continuous_state_recovered": False,
        "post_32s_scientific_gate_tested": False,
        "fig4_scientific_gate_changed": False,
        "backend_causality_for_fig4_gate_proven": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "changed_by_compiled_queue_count", "nonqueue_fields_still_different_count",
        "first_32s_continuous_state_recovered", "performance_authorized")}, sort_keys=True))


if __name__ == "__main__":
    main()
