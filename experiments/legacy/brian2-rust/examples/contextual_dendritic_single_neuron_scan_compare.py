"""Merge Fig. 2/S1 scan shards and compare them to published HDF5 cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def boundary(rates: np.ndarray, values: np.ndarray) -> float | None:
    positive = np.flatnonzero(values >= 0)
    if not positive.size:
        return None
    return float(rates[int(positive[-1])])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("published_h5", type=Path)
    parser.add_argument("shards", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    generated = np.full((16, 200), np.nan, dtype=np.float64)
    published = np.full((16, 200), np.nan, dtype=np.float64)
    keys = np.full((16, 200), "", dtype="U8")
    coverage = np.zeros((16, 200), dtype=bool)
    seed = None
    figure = None

    reports = []
    with h5py.File(args.published_h5, "r") as reference:
        for shard in args.shards:
            report_path = shard.with_suffix(".report.json")
            report = json.loads(report_path.read_text())
            reports.append(report)
            if seed is None:
                seed = report["seed"]
                figure = report["figure"]
            if report["seed"] != seed or report["figure"] != figure:
                raise ValueError("shards mix seeds or figures")
            with np.load(shard, allow_pickle=False) as data:
                active = np.asarray(data["active_inputs"], dtype=np.int64)
                rates = np.asarray(data["inhibitory_rates_hz"], dtype=np.int64)
                changes = np.asarray(data["all_weight_changes"], dtype=np.float64)
                shard_keys = np.asarray(data["result_keys"])
                if not np.all(changes == changes[:1]):
                    raise ValueError(f"paper broadcast axis differs in {shard}")
                if changes.shape != (10, len(active), len(rates)):
                    raise ValueError(f"unexpected aggregate shape in {shard}")
                for ai, active_inputs in enumerate(active):
                    for ri, inhibitory_rate in enumerate(rates):
                        rate_index = int(inhibitory_rate // 2)
                        index = (int(active_inputs), rate_index)
                        if coverage[index]:
                            raise ValueError(f"overlapping shard cell {index}")
                        key = str(shard_keys[ai, ri])
                        if key not in reference:
                            raise ValueError(f"published cache lacks key {key}")
                        weights = np.asarray(
                            reference[key]["final_silent_weights"],
                            dtype=np.float64,
                        )
                        if weights.shape != (60,):
                            raise ValueError(f"unexpected published shape for {key}")
                        generated[index] = changes[0, ai, ri]
                        published[index] = np.mean(weights[:10] - 5.0)
                        keys[index] = key
                        coverage[index] = True

    selected_generated = generated[coverage]
    selected_published = published[coverage]
    difference = selected_generated - selected_published
    sign_mismatch = np.signbit(selected_generated) != np.signbit(selected_published)
    active_rows = np.flatnonzero(np.any(coverage, axis=1))
    boundary_rows = {}
    for active_inputs in active_rows:
        row_mask = coverage[active_inputs]
        row_rates = np.arange(0, 400, 2, dtype=np.int64)[row_mask]
        boundary_rows[str(int(active_inputs))] = {
            "generated_last_nonnegative_rate_hz": boundary(
                row_rates, generated[active_inputs, row_mask]
            ),
            "published_last_nonnegative_rate_hz": boundary(
                row_rates, published[active_inputs, row_mask]
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged_path = args.output.with_suffix(".npz")
    np.savez_compressed(
        merged_path,
        generated_mean_weight_change=generated,
        published_mean_weight_change=published,
        coverage=coverage,
        result_keys=keys,
    )
    result = {
        "schema": "contextual-dendritic-single-neuron-scan-comparison-v1",
        "purpose": "scientific_reproduction_no_timings",
        "figure": figure,
        "seed": seed,
        "grid_cells_compared": int(coverage.sum()),
        "full_seed_grid_covered": bool(coverage.all()),
        "published_h5": str(args.published_h5.resolve()),
        "published_h5_sha256": digest(args.published_h5),
        "shards": [str(path.resolve()) for path in args.shards],
        "environments": [
            json.loads(environment)
            for environment in sorted({
                json.dumps(report.get("environment"), sort_keys=True)
                for report in reports
                if report.get("environment") is not None
            })
        ],
        "generated_merged_npz": str(merged_path.resolve()),
        "generated_merged_npz_sha256": digest(merged_path),
        "max_abs_mean_weight_change": float(np.max(np.abs(difference))),
        "rmse_mean_weight_change": float(np.sqrt(np.mean(difference**2))),
        "pearson_mean_weight_change": (
            float(np.corrcoef(selected_generated, selected_published)[0, 1])
            if selected_generated.size > 1
            else None
        ),
        "sign_mismatches": int(sign_mismatch.sum()),
        "sign_mismatch_fraction": float(sign_mismatch.mean()),
        "boundary_by_active_input_count": boundary_rows,
        "warmups": 0,
        "timings_reported": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
