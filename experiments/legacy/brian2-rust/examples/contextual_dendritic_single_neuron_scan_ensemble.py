"""Aggregate complete per-seed Fig. 2/S1 scan comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def boundary(rates, values):
    positive = np.flatnonzero(values >= 0)
    return float(rates[int(positive[-1])]) if positive.size else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparisons", type=Path, nargs="+")
    parser.add_argument("--figure", choices=("2", "S1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    generated = []
    published = []
    seeds = []
    inputs = []
    for comparison in args.comparisons:
        report_path = comparison.with_suffix(".json")
        report = json.loads(report_path.read_text())
        if report["figure"] != args.figure:
            raise ValueError(f"figure mismatch in {report_path}")
        if not report["full_seed_grid_covered"]:
            raise ValueError(f"incomplete grid in {report_path}")
        with np.load(comparison, allow_pickle=False) as data:
            coverage = np.asarray(data["coverage"], dtype=bool)
            if not coverage.all():
                raise ValueError(f"incomplete coverage in {comparison}")
            generated.append(
                np.asarray(data["generated_mean_weight_change"], dtype=np.float64)
            )
            published.append(
                np.asarray(data["published_mean_weight_change"], dtype=np.float64)
            )
        seeds.append(int(report["seed"]))
        inputs.append(
            {
                "seed": int(report["seed"]),
                "comparison": str(report_path.resolve()),
                "comparison_sha256": digest(report_path),
                "arrays": str(comparison.resolve()),
                "arrays_sha256": digest(comparison),
            }
        )
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed comparisons")
    order = np.argsort(seeds)
    seeds = [seeds[int(index)] for index in order]
    generated = np.stack([generated[int(index)] for index in order])
    published = np.stack([published[int(index)] for index in order])
    inputs = [inputs[int(index)] for index in order]

    generated_mean = np.mean(generated, axis=0)
    published_mean = np.mean(published, axis=0)
    difference = generated_mean - published_mean
    sign_mismatch = np.signbit(generated_mean) != np.signbit(published_mean)
    rates = np.arange(0, 400, 2, dtype=np.int64)
    boundaries = {
        str(active): {
            "generated_last_nonnegative_rate_hz": boundary(
                rates, generated_mean[active]
            ),
            "published_last_nonnegative_rate_hz": boundary(
                rates, published_mean[active]
            ),
        }
        for active in range(16)
    }
    seedwise = {
        str(seed): {
            "pearson": float(
                np.corrcoef(generated[index].ravel(), published[index].ravel())[0, 1]
            ),
            "rmse": float(
                np.sqrt(np.mean((generated[index] - published[index]) ** 2))
            ),
            "sign_mismatches": int(
                np.count_nonzero(
                    np.signbit(generated[index]) != np.signbit(published[index])
                )
            ),
        }
        for index, seed in enumerate(seeds)
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = args.output.with_suffix(".npz")
    np.savez_compressed(
        arrays_path,
        seeds=np.asarray(seeds, dtype=np.int64),
        generated_by_seed=generated,
        published_by_seed=published,
        generated_ensemble_mean=generated_mean,
        published_ensemble_mean=published_mean,
        difference=difference,
        sign_mismatch=sign_mismatch,
    )
    result = {
        "schema": "contextual-dendritic-single-neuron-scan-ensemble-v1",
        "purpose": "scientific_reproduction_no_timings",
        "figure": args.figure,
        "seeds": seeds,
        "complete_published_seed_ensemble": seeds == list(range(10)),
        "grid_shape_per_seed": [16, 200],
        "input_comparisons": inputs,
        "ensemble_arrays": str(arrays_path.resolve()),
        "ensemble_arrays_sha256": digest(arrays_path),
        "pearson_ensemble_mean": float(
            np.corrcoef(generated_mean.ravel(), published_mean.ravel())[0, 1]
        ),
        "rmse_ensemble_mean": float(np.sqrt(np.mean(difference**2))),
        "max_abs_ensemble_mean": float(np.max(np.abs(difference))),
        "sign_mismatches_ensemble_mean": int(sign_mismatch.sum()),
        "sign_mismatch_fraction_ensemble_mean": float(sign_mismatch.mean()),
        "boundary_by_active_input_count": boundaries,
        "seedwise": seedwise,
        "timings_reported": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
