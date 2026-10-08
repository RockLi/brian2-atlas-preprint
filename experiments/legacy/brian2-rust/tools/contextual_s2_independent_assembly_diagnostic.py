#!/usr/bin/env python3
"""Diagnose the frozen S2 independent-recall imprint-size failure from JSON only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


GATE_SHA256 = "50e0a0e7529c99a484961618ec97282fb8111d3f75175632ff5f5989cb2fb9ee"
REFERENCE_SHA256 = "cd0b27b5e6821421674263f7f8a77d74ea3034fb520ac556c31377011435bfd9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def ks_two_sample(a: list[int], b: list[int]) -> tuple[float, list[int]]:
    positions = sorted(set(a + b))
    deviations = {x: abs(sum(v <= x for v in a) / len(a)
                         - sum(v <= x for v in b) / len(b)) for x in positions}
    maximum = max(deviations.values())
    return maximum, [x for x in positions if deviations[x] == maximum]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t7-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite frozen S2 diagnostic")
    root = args.t7_root / "full-paper-audit-v1"
    gate_path = root / "figs2-full-recall-science-gates-v1/independent-recall-full-v1.json"
    reference_path = root / "figs2-independent-recall-gate-v1/contextual-s2-independent-recall-reference-v2.json"
    require(sha256(gate_path) == GATE_SHA256, "frozen failed science gate changed")
    require(sha256(reference_path) == REFERENCE_SHA256, "official reference changed")
    gate = json.loads(gate_path.read_text())
    reference = json.loads(reference_path.read_text())
    require(gate["passed"] is False and gate["recall_group_count"] == 840
            and gate["checks"]["assembly_size_ks"] is False
            and sum(not passed for passed in gate["checks"].values()) == 1
            and gate["thresholds"]["assembly_size_ks_maximum"] == 0.4
            and len(reference["seeds"]) == 10,
            "frozen S2 failure signature changed")
    sources = {}
    for source in gate["candidate_sources"]:
        name = Path(source["extract_path"]).name
        seed = int(name.removeprefix("independent-seed").removesuffix(".json"))
        path = root / "figs2-full-recall-science-gates-v1/candidate" / name
        require(sha256(path) == source["extract_sha256"],
                f"archived candidate extract changed for seed {seed}")
        require(seed not in sources, f"duplicate candidate seed {seed}")
        sources[seed] = json.loads(path.read_text())
    require(set(sources) == set(reference["seeds"]),
            "candidate/reference seed cohorts differ")
    rows = []
    for seed in reference["seeds"]:
        official = reference["imprints"][str(seed)]
        candidate = sources[seed]["imprints"][str(seed)]
        official_ids = set(official["assembly_ids"])
        candidate_ids = set(candidate["assembly_ids"])
        require(len(official_ids) == official["assembly_size"]
                and len(candidate_ids) == candidate["assembly_size"],
                f"assembly identity/count contract failed for seed {seed}")
        shared = len(official_ids & candidate_ids)
        rows.append({
            "seed": seed,
            "official_size": official["assembly_size"],
            "candidate_size": candidate["assembly_size"],
            "candidate_minus_official_size": candidate["assembly_size"] - official["assembly_size"],
            "shared_assembly_neuron_ids": shared,
            "union_assembly_neuron_ids": len(official_ids | candidate_ids),
            "jaccard_overlap": shared / len(official_ids | candidate_ids),
            "official_checkpoint_sha256": official["checkpoint_sha256"],
            "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        })
    official_sizes = [row["official_size"] for row in rows]
    candidate_sizes = [row["candidate_size"] for row in rows]
    ks, attainment = ks_two_sample(official_sizes, candidate_sizes)
    require(ks == gate["assembly_size"]["ks"] == 0.5
            and sorted(official_sizes) == sorted(gate["assembly_size"]["reference"])
            and sorted(candidate_sizes) == sorted(gate["assembly_size"]["candidate"]),
            "independent KS recomputation differs from frozen science gate")
    report = {
        "schema": "contextual-s2-independent-recall-imprint-assembly-diagnostic-v1",
        "mode": "mac_frozen_json_only_no_simulation_no_performance",
        "diagnostic_source_sha256": sha256(Path(__file__)),
        "frozen_science_gate_sha256": GATE_SHA256,
        "official_reference_sha256": REFERENCE_SHA256,
        "seed_count": len(rows),
        "recall_groups_in_prior_gate": 840,
        "failed_gate_remains_failed": True,
        "predeclared_max_ks": gate["thresholds"]["assembly_size_ks_maximum"],
        "independent_assembly_size_ks": ks,
        "maximum_cdf_gap_at_assembly_size": attainment,
        "official_size_le20_count": sum(x <= 20 for x in official_sizes),
        "candidate_size_le20_count": sum(x <= 20 for x in candidate_sizes),
        "seed_pairs_with_zero_neuron_identity_overlap": sum(row["shared_assembly_neuron_ids"] == 0
                                                          for row in rows),
        "all_ten_candidate_imprint_checkpoint_hashes_differ_from_official":
            all(row["official_checkpoint_sha256"] != row["candidate_checkpoint_sha256"]
                for row in rows),
        "imprint_assembly_difference_precedes_recall": True,
        "cause_of_imprint_difference_established": False,
        "next_remote_investigation": "source_and_rng_provenance_of_imprint_formation_before_recall",
        "performance_authorized": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"independent_ks": ks,
                      "candidate_size_le20_count": report["candidate_size_le20_count"],
                      "report_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
