#!/usr/bin/env python3
"""Compare a Fig. S3 recurrent cell using its exact save-time neuron order.

The paper saves only a square weight subset without its neuron IDs. A sidecar
captured during the *same* remote run supplies that ordering; this script
reconstructs the full weight matrix and applies the paper's rate-and-weight
assembly-size calculation. It does not simulate a network or report timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
from sklearn.cluster import KMeans  # noqa: E402

from contextual_dendritic_s3_recurrent_semantic_compare import (  # noqa: E402
    assembly_sizes_by_weights,
    firing_rates,
    reconstructed_weights,
    summarize,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attrs_as_json(group: h5py.Group) -> dict:
    return {
        name: np.asarray(value).tolist()
        for name, value in group.attrs.items()
    }


def validate_saved_order(
    sidecar: dict, group: h5py.Group
) -> tuple[np.ndarray, dict[str, bool]]:
    saved = group["weights"]
    n_somas = int(np.asarray(group.attrs["n_somas"]).item())
    ids = np.asarray(sidecar["saved_neuron_ids_in_weight_matrix_order"])
    selected = np.asarray(sidecar["selected_ids_at_save"])
    checks = {
        "sidecar_schema": (
            sidecar.get("schema")
            == "contextual-dendritic-s3-recurrent-saved-neuron-order-v2"
        ),
        "result_group_exact": sidecar.get("hdf5_group") == group.name.rsplit("/", 1)[-1],
        "seed_exact": int(sidecar.get("seed", -1))
        == int(np.asarray(group.attrs["seed"]).item()),
        "saved_shape_exact": list(saved.shape) == sidecar.get("saved_weight_shape"),
        "saved_matrix_square": saved.ndim == 2 and saved.shape[0] == saved.shape[1],
        "ids_integer": ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer),
        "ids_length_exact": ids.ndim == 1 and len(ids) == saved.shape[0],
        "ids_unique": ids.ndim == 1 and len(np.unique(ids)) == len(ids),
        "ids_in_range": ids.ndim == 1 and bool(np.all((ids >= 0) & (ids < n_somas))),
        "selected_ids_integer": (
            selected.ndim == 1 and np.issubdtype(selected.dtype, np.integer)
        ),
        "selected_ids_prefix_exact": (
            ids.ndim == 1
            and selected.ndim == 1
            and len(selected) <= len(ids)
            and np.array_equal(ids[:len(selected)], selected)
        ),
        "saved_dimension_equals_selected_plus_25": (
            selected.ndim == 1 and len(selected) + 25 == saved.shape[0]
        ),
        "sidecar_did_not_modify_model_or_parameters": (
            sidecar.get("paper_model_or_parameters_modified") is False
        ),
        "sidecar_capture_call_count_two": sidecar.get("capture_call_count") == 2,
        "selected_capture_is_final": sidecar.get("selected_capture_index") == 1,
        "capture_selected_counts_match": (
            isinstance(sidecar.get("capture_selected_counts"), list)
            and len(sidecar["capture_selected_counts"]) == 2
            and sidecar["capture_selected_counts"][1] == len(selected)
        ),
    }
    return ids.astype(np.int64, copy=False), checks


def paper_assembly_size_with_full_weights(
    group: h5py.Group, weights: np.ndarray
) -> int:
    """Mirror get_assembly_neuron_ids_by_weight_and_rate from tagged utils.py."""
    n_somas = int(np.asarray(group.attrs["n_somas"]).item())
    if weights.shape != (n_somas, n_somas):
        raise ValueError(f"expected full {n_somas}x{n_somas} weights")
    end_ms = 1000.0 * (
        float(np.asarray(group.attrs["runtime_imprint"]).item())
        + float(np.asarray(group.attrs["runtime_baseline"]).item())
    )
    rates = firing_rates(group, end_ms - 2000.0, end_ms)
    rate_kmeans = KMeans(n_clusters=2, random_state=1992).fit(rates.reshape(-1, 1))
    high_label = int(np.argmax(rate_kmeans.cluster_centers_))
    selected = list(np.where(rate_kmeans.labels_ == high_label)[0])
    selected += [
        int(index) for index in np.argsort(rates) if index not in selected
    ][-10:]
    cut = weights[np.ix_(selected, selected)]
    features = np.hstack((cut, np.sum(cut, axis=0).reshape(-1, 1)))
    weight_kmeans = KMeans(n_clusters=2, random_state=1992).fit(features)
    members = [np.where(weight_kmeans.labels_ == index)[0] for index in range(2)]
    means = [float(np.mean(cut[np.ix_(part, part)])) for part in members]
    return int(len(members[int(np.argmax(means))]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--pinned-reference-sha256", required=True)
    parser.add_argument("--expected-paper-source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    for path in (args.reference, args.candidate, args.sidecar):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    sidecar = json.loads(args.sidecar.read_text())
    candidate_sha = sha256_file(args.candidate)
    sidecar_sha = sha256_file(args.sidecar)
    with h5py.File(args.reference, "r") as reference, h5py.File(
        args.candidate, "r"
    ) as candidate:
        group_name = sidecar["hdf5_group"]
        if group_name not in reference or group_name not in candidate:
            parser.error("sidecar result group missing from reference or candidate")
        ref_group = reference[group_name]
        cand_group = candidate[group_name]
        ids, order_checks = validate_saved_order(sidecar, cand_group)
        checks = {
            **order_checks,
            "candidate_hdf5_sha256_exact": candidate_sha == sidecar.get("hdf5_sha256"),
            "paper_source_revision_exact": (
                sidecar.get("paper_source_revision")
                == args.expected_paper_source_revision
            ),
            "reproduction_id_present": bool(sidecar.get("reproduction_id")),
            "all_result_attributes_exact": (
                attrs_as_json(ref_group) == attrs_as_json(cand_group)
            ),
        }
        reference_summary = summarize(ref_group)
        reference_weights, reference_reconstruction = reconstructed_weights(ref_group)
        reference_algorithm_selftest = (
            paper_assembly_size_with_full_weights(ref_group, reference_weights)
            == reference_summary["assembly_size_by_rate_and_weight"]
        )
        checks["reference_algorithm_selftest"] = reference_algorithm_selftest
        checks["reference_original_loader_contract"] = bool(
            reference_reconstruction["tagged_loader_shape_contract_passed"]
        )
        candidate_paper_size = None
        candidate_weight_components = None
        if all(checks.values()):
            n_somas = int(np.asarray(cand_group.attrs["n_somas"]).item())
            w0 = float(np.asarray(cand_group.attrs["w0"]).item())
            full = np.full((n_somas, n_somas), w0, dtype=np.float64)
            full[np.ix_(ids, ids)] = np.asarray(cand_group["weights"], dtype=np.float64)
            candidate_paper_size = paper_assembly_size_with_full_weights(cand_group, full)
            candidate_weight_components = assembly_sizes_by_weights(cand_group)

    valid = all(checks.values())
    result = {
        "schema": "contextual-dendritic-s3-recurrent-sidecar-comparison-v2",
        "purpose": "scientific_correctness_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "reference_hdf5": str(args.reference.resolve()),
        "reference_hdf5_source_sha256_pinned_not_rehashed": args.pinned_reference_sha256,
        "expected_paper_source_revision": args.expected_paper_source_revision,
        "candidate_hdf5": str(args.candidate.resolve()),
        "candidate_hdf5_sha256": candidate_sha,
        "sidecar": str(args.sidecar.resolve()),
        "sidecar_sha256": sidecar_sha,
        "group": group_name,
        "checks": checks,
        "scientific_identity_valid": valid,
        "reference_paper_assembly_size": reference_summary[
            "assembly_size_by_rate_and_weight"
        ],
        "candidate_paper_assembly_size_with_exact_saved_order": candidate_paper_size,
        "paper_assembly_size_exact": (
            candidate_paper_size == reference_summary["assembly_size_by_rate_and_weight"]
            if valid else None
        ),
        "reference_saved_weight_component_sizes": reference_summary[
            "assembly_sizes_by_weights"
        ],
        "candidate_saved_weight_component_sizes": candidate_weight_components,
        "full_1000_cell_ensemble_gate_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "scientific_identity_valid": valid,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "paper_assembly_size_exact": result["paper_assembly_size_exact"],
    }, indent=2, sort_keys=True))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
