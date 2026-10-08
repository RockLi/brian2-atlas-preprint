#!/usr/bin/env python3
"""Gate the five-seed Figure S2 large-imprint ensemble on paper metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from contextual_dendritic_s2_large_imprint_semantic_compare import summarize


SEEDS = [24, 485, 932, 3523, 63]
THRESHOLDS = {
    "pooled_assembly_size_mean_delta_maximum_neurons": 3.0,
    "pooled_assembly_size_wasserstein_maximum_neurons": 4.0,
    "pooled_assembly_size_ks_maximum": 0.25,
    "pooled_assembly_rate_mean_delta_maximum_hz": 3.0,
    "pooled_assembly_rate_wasserstein_maximum_hz": 4.0,
    "pooled_assembly_rate_ks_maximum": 0.25,
    "seed_final_weight_mean_mae_maximum": 0.20,
    "seed_final_weight_quantile_max_abs_delta_maximum": 0.75,
}


def text(value: Any) -> str:
    value = np.asarray(value).item()
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def imprint_identity(
    h5_path: Path, seed: int, checkpoint_dir: Path | None = None
) -> tuple[str, str]:
    with h5py.File(h5_path, "r") as handle:
        names = [
            name
            for name in handle
            if int(np.asarray(handle[name].attrs.get("seed", -1)).item()) == seed
            and "run_recall_after_imprint" not in handle[name].attrs
        ]
        if not names and checkpoint_dir is not None:
            group_name, prefix, _ = reference_identity(h5_path, checkpoint_dir, seed)
            return group_name, prefix
        if len(names) != 1:
            raise ValueError(
                f"expected one large-imprint group for seed {seed}, found {names}"
            )
        group = handle[names[0]]
        prefix = text(group["filename_for_stored_network"][()])
    return names[0], prefix


def reference_identity(
    h5_path: Path, checkpoint_dir: Path, seed: int
) -> tuple[str, str, str]:
    """Find an imprint source, including the published recall-labelled proxy."""
    with h5py.File(h5_path, "r") as handle:
        seed_names = [
            name for name, group in handle.items()
            if int(np.asarray(group.attrs.get("seed", -1)).item()) == seed
        ]
        direct = [
            name for name in seed_names
            if "run_recall_after_imprint" not in handle[name].attrs
        ]
        if len(direct) == 1:
            prefix = text(handle[direct[0]]["filename_for_stored_network"][()])
            kind = "imprint"
            name = direct[0]
        elif not direct:
            eligible = [
                name for name in seed_names
                if all(
                    (checkpoint_dir / f"stored_imprint_{name}_{index}").is_file()
                    for index in range(20)
                )
            ]
            if len(eligible) != 1:
                raise ValueError(
                    f"expected one checkpoint-backed recall proxy for seed {seed}, found {eligible}"
                )
            name = eligible[0]
            prefix = f"stored_imprint_{name}"
            kind = "recall_proxy"
            preimprint_digests = set()
            for candidate_name in seed_names:
                group = handle[candidate_name]
                times = np.asarray(group["spikes_somas_t_A"], dtype=float)
                indices = np.asarray(group["spikes_somas_i_A"], dtype=np.int64)
                if times.shape != indices.shape:
                    raise ValueError(f"seed {seed} has mismatched spike vectors")
                selected = times < 620_000.0
                digest = hashlib.sha256()
                digest.update(times[selected].tobytes())
                digest.update(indices[selected].tobytes())
                preimprint_digests.add(digest.hexdigest())
            if len(preimprint_digests) != 1:
                raise ValueError(f"seed {seed} recall groups differ before imprint end")
        else:
            raise ValueError(f"ambiguous imprint groups for seed {seed}: {direct}")
    if not all((checkpoint_dir / f"{prefix}_{index}").is_file() for index in range(20)):
        raise ValueError(f"seed {seed} has incomplete official checkpoints")
    return name, prefix, kind


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def parameter_fingerprint(h5_path: Path, group_name: str) -> tuple[str, int]:
    with h5py.File(h5_path, "r") as handle:
        attrs = handle[group_name].attrs
        parameters = {
            key: _plain(attrs[key])
            for key in attrs
            if "recall" not in key.lower()
        }
    encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), len(parameters)


def candidate_spec(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        return int(seed_text), Path(path_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("candidate must be SEED=PAPER_REPOSITORY") from error


def distribution(left: list[float], right: list[float]) -> dict[str, float]:
    reference = np.asarray(left, dtype=float)
    candidate = np.asarray(right, dtype=float)
    return {
        "reference_mean": float(np.mean(reference)),
        "candidate_mean": float(np.mean(candidate)),
        "mean_delta": float(abs(np.mean(candidate) - np.mean(reference))),
        "wasserstein": float(wasserstein_distance(reference, candidate)),
        "ks": float(ks_2samp(reference, candidate).statistic),
    }


def load_compact_reference(path: Path) -> dict[int, dict[str, Any]]:
    envelope = json.loads(path.read_text())
    if envelope.get("schema") != "contextual-dendritic-s2-ensemble-reference-summary-v1":
        raise ValueError("unexpected S2 compact reference schema")
    if envelope.get("reported_timings") is not False:
        raise ValueError("compact reference must not contain performance timings")
    rows = envelope.get("seeds")
    if not isinstance(rows, dict) or set(rows) != {str(seed) for seed in SEEDS}:
        raise ValueError("compact reference does not cover all five official seeds")
    source_sha = envelope.get("source", {}).get("hdf5_sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ValueError("compact reference has no source HDF5 digest")
    for seed in SEEDS:
        row = rows[str(seed)]
        summary = row.get("reference")
        if not isinstance(summary, dict):
            raise ValueError(f"compact reference seed {seed} has no summary")
        if summary.get("h5_sha256") != source_sha:
            raise ValueError(f"compact reference seed {seed} source digest mismatch")
        if summary.get("imprints") != 20 or len(summary.get("assembly_sizes", [])) != 20:
            raise ValueError(f"compact reference seed {seed} has incomplete imprints")
        if len(summary.get("checkpoint_sha256", [])) != 20:
            raise ValueError(f"compact reference seed {seed} has incomplete checkpoint digests")
        if row.get("group") != summary.get("group") or row.get("checkpoint_prefix") != summary.get("checkpoint_prefix"):
            raise ValueError(f"compact reference seed {seed} identity mismatch")
        if row.get("source_kind") not in {"imprint", "recall_proxy"}:
            raise ValueError(f"compact reference seed {seed} source kind missing")
        if not isinstance(row.get("parameter_sha256"), str) or len(row["parameter_sha256"]) != 64:
            raise ValueError(f"compact reference seed {seed} parameter digest missing")
        if row.get("preimprint_spike_group_count") != 41 or row.get("preimprint_spike_unique_digests") != 1:
            raise ValueError(f"compact reference seed {seed} lacks spike equivalence proof")
        if not isinstance(row.get("preimprint_spike_sha256"), str) or len(row["preimprint_spike_sha256"]) != 64:
            raise ValueError(f"compact reference seed {seed} has no preimprint spike digest")
    return {int(seed): row for seed, row in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_repo", type=Path)
    parser.add_argument("--candidate", type=candidate_spec, action="append", default=[])
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    compact_reference = None
    if args.reference_repo.suffix.lower() == ".json":
        if not args.reference_repo.is_file():
            parser.error(f"missing compact reference: {args.reference_repo}")
        compact_reference = load_compact_reference(args.reference_repo)
    elif not args.reference_repo.is_dir():
        parser.error(f"missing reference repository: {args.reference_repo}")
    candidates = dict(args.candidate)
    if len(candidates) != len(args.candidate):
        parser.error("candidate seeds are not unique")
    if set(candidates) - set(SEEDS):
        parser.error("candidate includes a non-official seed")
    if not candidates:
        parser.error("at least one --candidate is required")
    if not args.allow_incomplete and set(candidates) != set(SEEDS):
        parser.error("complete validation requires all five official seeds")

    filename = "data_Fig_S2_large_imprint_single_dendrite.h5"
    reference_h5 = args.reference_repo / "results" / "sim_files" / filename if compact_reference is None else None
    reference_checkpoints = args.reference_repo / "stored_networks" / "Fig_S2" if compact_reference is None else None
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        if seed not in candidates:
            continue
        candidate_repo = candidates[seed].resolve()
        candidate_h5 = candidate_repo / "results" / "sim_files" / filename
        candidate_checkpoints = candidate_repo / "stored_networks" / "Fig_S2"
        if compact_reference is None:
            reference_group, reference_prefix, source_kind = reference_identity(
                reference_h5, reference_checkpoints, seed
            )
            reference = summarize(
                reference_h5,
                reference_checkpoints,
                reference_group,
                reference_prefix,
                20,
            )
        else:
            reference_row = compact_reference[seed]
            reference_group = reference_row["group"]
            reference_prefix = reference_row["checkpoint_prefix"]
            source_kind = reference_row["source_kind"]
            reference = reference_row["reference"]
        candidate_group, candidate_prefix = imprint_identity(
            candidate_h5, seed, candidate_checkpoints
        )
        if source_kind == "imprint" and (
            candidate_group != reference_group or candidate_prefix != reference_prefix
        ):
            raise ValueError(
                f"seed {seed} parameter identity differs: "
                f"{candidate_group}/{candidate_prefix} versus "
                f"{reference_group}/{reference_prefix}"
            )
        reference_parameter_sha, reference_parameter_count = (
            parameter_fingerprint(reference_h5, reference_group)
            if compact_reference is None
            else (reference_row["parameter_sha256"], reference_row["parameter_key_count"])
        )
        candidate_parameter_sha, candidate_parameter_count = parameter_fingerprint(
            candidate_h5, candidate_group
        )
        if (candidate_parameter_sha, candidate_parameter_count) != (
            reference_parameter_sha, reference_parameter_count
        ):
            raise ValueError(f"seed {seed} scientific parameter fingerprint differs")
        candidate = summarize(
            candidate_h5,
            candidate_checkpoints,
            candidate_group,
            candidate_prefix,
            20,
        )
        rows.append(
            {
                "seed": seed,
                "group": reference_group,
                "candidate_group": candidate_group,
                "reference_source_kind": source_kind,
                "checkpoint_prefix": reference_prefix,
                "candidate_checkpoint_prefix": candidate_prefix,
                "scientific_parameter_sha256": reference_parameter_sha,
                "reference": reference,
                "candidate": candidate,
                "assembly_size_mean_delta": abs(
                    np.mean(candidate["assembly_sizes"])
                    - np.mean(reference["assembly_sizes"])
                ),
                "assembly_rate_mean_delta": abs(
                    np.mean(candidate["assembly_rates_hz"])
                    - np.mean(reference["assembly_rates_hz"])
                ),
                "final_weight_mean_delta": abs(
                    candidate["final_weight_mean"] - reference["final_weight_mean"]
                ),
                "final_weight_quantile_max_abs_delta": float(
                    np.max(
                        np.abs(
                            np.asarray(candidate["final_weight_quantiles"])
                            - np.asarray(reference["final_weight_quantiles"])
                        )
                    )
                ),
            }
        )

    reference_sizes = [
        value for row in rows for value in row["reference"]["assembly_sizes"]
    ]
    candidate_sizes = [
        value for row in rows for value in row["candidate"]["assembly_sizes"]
    ]
    reference_rates = [
        value for row in rows for value in row["reference"]["assembly_rates_hz"]
    ]
    candidate_rates = [
        value for row in rows for value in row["candidate"]["assembly_rates_hz"]
    ]
    sizes = distribution(reference_sizes, candidate_sizes)
    rates = distribution(reference_rates, candidate_rates)
    weight_mean_mae = float(np.mean([row["final_weight_mean_delta"] for row in rows]))
    weight_quantile_max = float(
        np.max([row["final_weight_quantile_max_abs_delta"] for row in rows])
    )
    complete = set(candidates) == set(SEEDS)
    checks = {
        "pooled_assembly_size_mean_delta": sizes["mean_delta"]
        <= THRESHOLDS["pooled_assembly_size_mean_delta_maximum_neurons"],
        "pooled_assembly_size_wasserstein": sizes["wasserstein"]
        <= THRESHOLDS["pooled_assembly_size_wasserstein_maximum_neurons"],
        "pooled_assembly_size_ks": sizes["ks"]
        <= THRESHOLDS["pooled_assembly_size_ks_maximum"],
        "pooled_assembly_rate_mean_delta": rates["mean_delta"]
        <= THRESHOLDS["pooled_assembly_rate_mean_delta_maximum_hz"],
        "pooled_assembly_rate_wasserstein": rates["wasserstein"]
        <= THRESHOLDS["pooled_assembly_rate_wasserstein_maximum_hz"],
        "pooled_assembly_rate_ks": rates["ks"]
        <= THRESHOLDS["pooled_assembly_rate_ks_maximum"],
        "seed_final_weight_mean_mae": weight_mean_mae
        <= THRESHOLDS["seed_final_weight_mean_mae_maximum"],
        "seed_final_weight_quantile_max_abs_delta": weight_quantile_max
        <= THRESHOLDS["seed_final_weight_quantile_max_abs_delta_maximum"],
    }
    partial_scientific_passed = all(checks.values())
    report = {
        "schema": "contextual-dendritic-s2-large-imprint-ensemble-comparison-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "official_seeds": SEEDS,
        "observed_seeds": [row["seed"] for row in rows],
        "complete": complete,
        "allow_incomplete": args.allow_incomplete,
        "thresholds_predeclared_in_source": THRESHOLDS,
        "metrics": {
            "pooled_assembly_sizes": sizes,
            "pooled_assembly_rates_hz": rates,
            "seed_final_weight_mean_mae": weight_mean_mae,
            "seed_final_weight_quantile_max_abs_delta": weight_quantile_max,
        },
        "checks": checks,
        "partial_scientific_passed": partial_scientific_passed,
        "rows": rows,
        "passed": complete and partial_scientific_passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not (report["passed"] or (args.allow_incomplete and partial_scientific_passed)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
