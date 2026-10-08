#!/usr/bin/env python3
"""Exact-once six-realization descriptive V1 archive from completed evidence."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

import mam_v1_140_descriptive_cohort as previous


PREP = (Path(__file__).resolve().parents[1] / "mpi-evidence"
        / "confirmation-seed1751-v8-science-identity-preparation")
T7 = previous.T7
OUTPUT = previous.ROOT / "artifacts/rust-native-v1-140-six-realization-v1"
PENDING = OUTPUT.with_name(OUTPUT.name + ".pending")
LABELS = ("rust_seed1729", "rust_seed1750", "rust_seed1751",
          "native_seed1729", "native_seed1730", "native_seed1731")
PRIOR = {seed: previous.ROOT / "artifacts/prior-rust-v1-140-raw-v1" / f"rust{seed}"
         for seed in (1729, 1750)}
RECEIPT_SHA = {
    "rust_seed1729": "930ec7d6f24b78662f86b37028881f1c85cc59cd3e3f0819ef799896690a7e23",
    "rust_seed1750": "a6ae294681f56136cfd7dd5951e162ef4b812ceba60222357f1fe745b50f92ee",
    "rust_seed1751": "a285fba067b6c4e6f38031aa51d141faac11cbc0107c9ae394aedf40c4ccf6c7",
    "native_seed1729": "467136a0397f903f3ea5ca690c501db9bb9dbdf7b119f1c964cf8fa046ba5dc3",
    "native_seed1730": "47a33089dacecb98190f64715328a50192723961ebc14d2bfb50e9a67a63e516",
    "native_seed1731": "0ca684ffd045d87d3bd37f3e39eb6461b3b12054cff48bfd96756e0d8db2f8f7",
}
AUDIT_SHA = {
    1729: "e1ad079cacc096c6516f4b834e94b5788d28689747eb55e69076c6b4be51c8c5",
    1750: "bea0a6f157899e6b545b846dc8c6a366c698ac1978c1cab2e9ee522a2b6d4ccc",
}
OLD = previous.OUTPUT
OLD_SHA = {
    "aligned-views.npz": "7329910c6c354c9aa22c2f1b3e860b7f49af22fb1699dbcb2e716b71d012e1ad",
    "report.json": "b282594001f6ff5a755579437ba0dc8c75dc1a4ff1bf9156a411b93c2eb58f47",
}
SCHEMA = "b2-mam-v1-140-six-realization-descriptive-cohort-v1"
MAX_OUTPUT = 256 * 2**20
need, sha = previous.need, previous.sha


def read(path):
    need(path.is_file() and not path.is_symlink(), "required JSON missing: " + str(path))
    return json.loads(path.read_text())


def bound_prior(seed):
    script = PREP / "audit_prior_rust_v1_140_views.py"
    need(sha(script) == "78eab2faaf0af6c3634cc123abb06763e0273391e748e13924618a92d9fd1c54",
         "independent prior-Rust verifier differs")
    spec = importlib.util.spec_from_file_location("pinned_prior_v1_verifier", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt, files = module.checked_members(PRIOR[seed], seed)
    collection = read(receipt)
    completion = files[f"control/rust{seed}-completion-v1.json"]
    finished = read(completion)
    need(sha(completion) == collection["source_completion_sha256"]
         and finished.get("complete") is True
         and finished.get("seed") == seed and finished.get("attempts") == 1
         and finished.get("automatic_retry") is False
         and finished.get("scientific_acceptance") is False,
         "prior-Rust completion differs")
    audit_path = PREP / f"prior-rust{seed}-v1-140-independent-recompute-v1.json"
    need(sha(audit_path) == AUDIT_SHA[seed], "independent recomputation report differs")
    audit = read(audit_path)
    expected_bindings = {
        "collection_receipt": sha(receipt),
        "selected_cell_counts": sha(files["analysis/selected-cell-counts.npz"]),
        "four_views": sha(files["analysis/v1-four-views.npz"]),
        "identity": sha(files["source/identity-v3.json"]),
        "official_sizes": sha(files["reader/mam-official-analysis-neuron-sizes-v1.json"]),
        "fft_implementation": module.FFT_SOURCE_SHA,
    }
    need(audit.get("schema") == module.SCHEMA and audit.get("seed") == seed
         and audit.get("source_sha256") == expected_bindings
         and audit.get("engineering_recompute_passed") is True
         and audit.get("scientific_acceptance") is False
         and len(audit.get("checks", {})) == 6
         and all(row.get("passed") is True for row in audit["checks"].values()),
         "independent prior-Rust recomputation does not bind current archive")
    return dict(root=str(PRIOR[seed]), receipt_sha256=sha(receipt),
                completion_sha256=sha(completion),
                spectrum_sha256=sha(files["analysis/v1-four-views.npz"]),
                independent_recompute_sha256=sha(audit_path))


def load_views(path):
    with np.load(path, allow_pickle=False) as data:
        expected = ({"sampled_population_counts", "frequency_hz"}
                    | {"power__" + view for view in previous.VIEWS}
                    | {"rate__" + view for view in previous.RATE_VIEWS})
        need(set(data.files) == expected, "V1 archive array membership differs")
        population = data["sampled_population_counts"]
        need(population.shape == (8, 100000) and population.dtype.kind in "iu"
             and np.all(population >= 0), "V1 sampled-count contract differs")
        need(np.array_equal(data["frequency_hz"], np.fft.rfftfreq(1024, d=.001)),
             "V1 frequency axis differs")
        arrays = {"frequency_hz": data["frequency_hz"].copy()}
        summary = {"sampled_spikes": int(population.sum(dtype=np.uint64))}
        for view in previous.VIEWS:
            power = data["power__" + view]
            need(power.shape == (513,) and np.isfinite(power).all()
                 and np.all(power >= 0), "V1 power view invalid")
            arrays["power__" + view] = power.copy()
            peak = int(np.argmax(power[1:]) + 1)
            summary[view] = dict(peak_nonzero_frequency_hz=float(arrays["frequency_hz"][peak]),
                                 peak_nonzero_power=float(power[peak]))
        for view in previous.RATE_VIEWS:
            rate = data["rate__" + view]
            need(rate.shape == (100000,) and np.isfinite(rate).all()
                 and np.all(rate >= 0), "V1 rate view invalid")
            arrays["rate__" + view] = rate.copy()
            summary["mean_rate_hz__" + view] = float(rate.mean())
        return arrays, summary


def assembled():
    need(sha(Path(previous.__file__)) ==
         "552ed213ffe9c5a4454f86f099136add201fb013f9769ba16bda13e026af03d9",
         "previous source binder changed")
    roots = {"rust_seed1729": PRIOR[1729], "rust_seed1750": PRIOR[1750],
             "rust_seed1751": previous.RUST,
             **{f"native_seed{seed}": root for seed, root in previous.NATIVE.items()}}
    sources, arrays, summaries = {}, {}, {}
    for label in LABELS:
        if label in ("rust_seed1729", "rust_seed1750"):
            source = bound_prior(int(label[-4:]))
        else:
            source = previous.bound_source(roots[label], label.startswith("rust"))
        need(source["receipt_sha256"] == RECEIPT_SHA[label],
             "frozen collection receipt differs: " + label)
        path = roots[label] / ("analysis/v1-four-views.npz" if label.startswith("rust")
                               else "output/v1-four-views.npz")
        arrays[label], summaries[label] = load_views(path)
        sources[label] = source
    for name, digest in OLD_SHA.items():
        need(sha(OLD / name) == digest, "previous four-source cohort differs")
    old_report = read(OLD / "report.json")
    need(old_report.get("labels") == list(LABELS[2:])
         and old_report.get("scientific_acceptance") is False,
         "previous cohort axes or interpretation differ")
    with np.load(OLD / "aligned-views.npz", allow_pickle=False) as old:
        for index, label in enumerate(old_report["labels"]):
            for key, value in arrays[label].items():
                need(np.array_equal(old[key] if key == "frequency_hz" else old[key][index], value),
                     "previous cohort overlap differs: " + label + " " + key)
    aligned = {"frequency_hz": arrays[LABELS[0]]["frequency_hz"]}
    for key in arrays[LABELS[0]]:
        if key != "frequency_hz":
            aligned[key] = np.stack([arrays[label][key] for label in LABELS])
    report = dict(schema=SCHEMA, labels=list(LABELS),
                  rust_seeds=[1729, 1750, 1751], native_seeds=[1729, 1730, 1731],
                  independent_seed_axes=True, seeds_paired=False,
                  source_sha256=sources, previous_cohort_sha256=OLD_SHA,
                  previous_cohort_overlap_exact=True,
                  views=list(previous.VIEWS), rate_views=list(previous.RATE_VIEWS),
                  summaries=summaries, scientific_acceptance=False,
                  rust_native_equivalence=False, paper_equivalence=False,
                  performance_cost_acceptance=False,
                  scope="Six exploratory realizations under modern deterministic V1 selection; all four views retained.")
    return aligned, report


def verify_output(arrays, report):
    catalog = read(OUTPUT / "catalog.json")
    need(set(catalog) == {"aligned-views.npz", "report.json"}, "cohort catalog differs")
    for name, row in catalog.items():
        path = OUTPUT / name
        need(path.is_file() and not path.is_symlink()
             and row == {"bytes": path.stat().st_size, "sha256": sha(path)},
             "cohort published member differs")
    need(read(OUTPUT / "report.json") == report, "cohort report differs from sources")
    with np.load(OUTPUT / "aligned-views.npz", allow_pickle=False) as saved:
        need(set(saved.files) == set(arrays)
             and all(np.array_equal(saved[key], value) for key, value in arrays.items()),
             "cohort published arrays differ from sources")
    return sha(OUTPUT / "catalog.json")


def probe():
    need(T7.is_mount(), "T7 is not mounted")
    arrays, report = assembled()
    free = os.statvfs(T7).f_bavail * os.statvfs(T7).f_frsize
    need(not OUTPUT.is_symlink() and not PENDING.exists() and not PENDING.is_symlink(),
         "cohort destination inconsistent or pending attempt exists")
    already = OUTPUT.exists()
    catalog_sha = verify_output(arrays, report) if already else None
    return dict(ready=not already and free >= 128 * 2**30 + MAX_OUTPUT,
                already_built=already, catalog_sha256=catalog_sha,
                output=str(OUTPUT), t7_free_bytes=free, sources=report["source_sha256"])


def build():
    state = probe()
    need(state["ready"] and not state["already_built"], "exact-once cohort build inadmissible")
    arrays, report = assembled()
    PENDING.mkdir(mode=0o700)
    np.savez_compressed(PENDING / "aligned-views.npz", **arrays)
    (PENDING / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True,
                                                    allow_nan=False) + "\n")
    catalog = {name: {"bytes": (PENDING / name).stat().st_size,
                      "sha256": sha(PENDING / name)}
               for name in ("aligned-views.npz", "report.json")}
    (PENDING / "catalog.json").write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    need(sum(row["bytes"] for row in catalog.values()) <= MAX_OUTPUT,
         "cohort output exceeds frozen cap")
    os.rename(PENDING, OUTPUT)
    return dict(built=True, catalog_sha256=verify_output(arrays, report),
                output=str(OUTPUT), scientific_acceptance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "build"), required=True)
    args = parser.parse_args()
    print(json.dumps(probe() if args.mode == "probe" else build(), sort_keys=True,
                     allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
