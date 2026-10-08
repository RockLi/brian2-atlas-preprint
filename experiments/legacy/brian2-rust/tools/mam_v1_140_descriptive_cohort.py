#!/usr/bin/env python3
"""Build one descriptive Rust/native V1 140-cell spectrum cohort on T7.

This is a provenance-bound visualization input, not an equivalence test. No
margin, power calculation, p-value or performance claim is inferred.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


T7 = Path("/Volumes/T7")
ROOT = T7 / "brian2-mpi-20260908"
RUST = (ROOT / "confirmation-retry-v2-seed1751-debug"
        / "confirmation1751-v1-140-raw-v1")
NATIVE = {seed: ROOT / "artifacts" / f"native-v1-140-derived-seed{seed}-v1"
          for seed in (1729, 1730, 1731)}
OUTPUT = ROOT / "artifacts/native-v1-140-descriptive-cohort-v1"
VIEWS = ("modern_equal_cell__declared_boxcar",
         "modern_equal_cell__effective_hann",
         "inferred_full_population_weighted__declared_boxcar",
         "inferred_full_population_weighted__effective_hann")
RATE_VIEWS = ("modern_equal_cell", "inferred_full_population_weighted")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bound_source(root, rust):
    receipt = (root / "control/mam-v1-140-collection-v1.json" if rust else
               root / "collection.json")
    need(receipt.is_file() and not receipt.is_symlink(),
         "source collection receipt absent")
    data = json.loads(receipt.read_text())
    expected_schema = ("b2-mam-v1-140-collection-v1" if rust else
                       "b2-mam-native-v1-140-collection-v1")
    need(data.get("schema") == expected_schema
         and data.get("scientific_acceptance") is False
         and data.get("source_completion_sha256"),
         "source collection contract differs")
    rows = data.get("files")
    need(isinstance(rows, list) and len(rows) >= 19
         and len({row["relative"] for row in rows}) == len(rows),
         "source collection member set differs")
    for row in rows:
        path = root / row["relative"]
        need(path.resolve().is_relative_to(root.resolve())
             and path.is_file() and not path.is_symlink()
             and path.stat().st_size == row["bytes"]
             and sha(path) == row["sha256"],
             "source collection member differs: " + row["relative"])
    completion = (root / "control/v1-140-completion-v1.json" if rust else
                  root / "control/completion.json")
    need(sha(completion) == data["source_completion_sha256"],
         "source completion binding differs")
    return dict(root=str(root), receipt_sha256=sha(receipt),
                completion_sha256=sha(completion),
                spectrum_sha256=sha(root / "analysis/v1-four-views.npz") if rust else
                                sha(root / "output/v1-four-views.npz"))


def probe():
    mounted = T7.is_dir() and T7.stat().st_dev != T7.parent.stat().st_dev
    need(mounted, "T7 is not mounted")
    sources = {"rust_seed1751": bound_source(RUST, True)}
    for seed, root in NATIVE.items():
        sources[f"native_seed{seed}"] = bound_source(root, False)
    pending = list(OUTPUT.parent.glob(".native-v1-140-descriptive-cohort-v1.pending-*"))
    exists = OUTPUT.exists() or OUTPUT.is_symlink()
    free = os.statvfs(T7).f_bavail * os.statvfs(T7).f_frsize
    return dict(schema="b2-mam-v1-140-descriptive-cohort-probe-v1",
                ready=not exists and not pending and free >= 128 * 2**30,
                output_exists=exists, pending=[str(path) for path in pending],
                t7_free_bytes=free, sources=sources)


def build():
    admission = probe()
    need(admission["ready"] is True, "descriptive cohort destination unavailable")
    pending = OUTPUT.with_name("." + OUTPUT.name + ".pending-" + str(os.getpid()))
    pending.mkdir(mode=0o700)
    roots = {"rust_seed1751": RUST,
             **{f"native_seed{seed}": root for seed, root in NATIVE.items()}}
    frequency = None
    powers = {view: [] for view in VIEWS}
    rates = {view: [] for view in RATE_VIEWS}
    summaries = {}
    for label, root in roots.items():
        path = (root / "analysis/v1-four-views.npz" if label.startswith("rust") else
                root / "output/v1-four-views.npz")
        with np.load(path, allow_pickle=False) as data:
            expected = ({"sampled_population_counts", "frequency_hz"}
                        | {"power__" + view for view in VIEWS}
                        | {"rate__" + view for view in RATE_VIEWS})
            need(set(data.files) == expected
                 and data["sampled_population_counts"].shape == (8, 100_000),
                 "V1 spectrum array contract differs")
            grid = data["frequency_hz"]
            need(grid.shape == (513,) and np.isfinite(grid).all()
                 and np.all(np.diff(grid) > 0), "V1 spectrum grid differs")
            if frequency is None:
                frequency = grid.copy()
            else:
                need(np.array_equal(grid, frequency), "V1 frequency axes differ")
            entry = {"sampled_spikes": int(data["sampled_population_counts"].sum())}
            for view in VIEWS:
                power = data["power__" + view]
                need(power.shape == (513,) and np.isfinite(power).all()
                     and np.all(power >= 0), "V1 spectral power differs")
                powers[view].append(power.copy())
                peak = int(np.argmax(power[1:]) + 1)
                entry[view] = dict(peak_nonzero_frequency_hz=float(grid[peak]),
                                   peak_nonzero_power=float(power[peak]))
            for view in RATE_VIEWS:
                rate = data["rate__" + view]
                need(rate.shape == (100_000,) and np.isfinite(rate).all()
                     and np.all(rate >= 0), "V1 rate view differs")
                rates[view].append(rate.copy())
                entry["mean_rate_hz__" + view] = float(rate.mean())
            summaries[label] = entry
    np.savez_compressed(pending / "aligned-views.npz", frequency_hz=frequency,
                        **{"power__" + view: np.stack(values)
                           for view, values in powers.items()},
                        **{"rate__" + view: np.stack(values)
                           for view, values in rates.items()})
    report = dict(schema="b2-mam-v1-140-descriptive-cohort-v1",
                  labels=list(roots), sources=admission["sources"],
                  views=list(VIEWS), rate_views=list(RATE_VIEWS),
                  summaries=summaries, scientific_acceptance=False,
                  rust_native_equivalence=False, paper_equivalence=False,
                  performance_cost_acceptance=False,
                  scope="Descriptive modern-wrapper V1 sample views only; no historical-ID or formal equivalence claim.")
    (pending / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    catalog = {name: {"bytes": (pending / name).stat().st_size,
                      "sha256": sha(pending / name)}
               for name in ("aligned-views.npz", "report.json")}
    (pending / "catalog.json").write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    need(sum(path.stat().st_size for path in pending.iterdir()) <= 256 * 2**20,
         "descriptive cohort output cap exceeded")
    os.rename(pending, OUTPUT)
    return dict(built=True, output=str(OUTPUT), catalog_sha256=sha(OUTPUT / "catalog.json"),
                scientific_acceptance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "build"), required=True)
    args = parser.parse_args()
    print(json.dumps(probe() if args.mode == "probe" else build(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
