#!/usr/bin/env python3
"""Extract closed HDF weights and saved MT state for a remote data-only probe."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import pickletools

import h5py
import numpy as np

from contextual_fig6_s6_checkpoint_rng_tail_audit import raw_hash_and_tail


RNG_MARKER = b"\x8c\x17_random_generator_state"
ROOT_REPORT = "fig6-s6-checkpoint-rng-tail-v1/report-v1.json"
FIGURES = {
    "Fig_6": ("fig6-corrected-full-v1", "data_Fig_6.h5", "stored_imprint_44873076"),
    "Fig_S6": ("figs6-corrected-full-v1", "data_Fig_S6.h5", "stored_imprint_c1ef03d5"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def saved_mt(tail: bytes) -> tuple[np.ndarray, int, int, float]:
    if tail.count(RNG_MARKER) != 1:
        raise ValueError("RNG marker not unique in checkpoint tail")
    start = tail.index(RNG_MARKER)
    active = False
    payload = None
    numeric = []
    for op, arg, _ in pickletools.genops(io.BytesIO(tail[start:])):
        if op.name == "SHORT_BINUNICODE" and arg == "numpy_state":
            active = True
        elif active and op.name == "BYTEARRAY8" and payload is None:
            payload = bytes(arg)
        elif active and op.name in ("BININT", "BININT1", "BININT2", "BINFLOAT"):
            numeric.append(arg)
        elif op.name == "SHORT_BINUNICODE" and arg == "rand_buffer_index":
            break
    if payload is None or len(payload) != 2496 or len(numeric) < 3:
        raise ValueError("unexpected NumPy MT19937 checkpoint state structure")
    position, has_gauss, cached_gaussian = numeric[-3:]
    if not (0 <= position <= 624 and has_gauss in (0, 1)):
        raise ValueError("unexpected NumPy MT19937 state metadata")
    return np.frombuffer(payload, dtype="<u4").copy(), position, has_gauss, cached_gaussian


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--figure", choices=FIGURES, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_npz.exists() or args.output_json.exists():
        parser.error("refusing to overwrite an existing output")
    root = args.root.resolve(strict=True)
    family, hdf_name, checkpoint_name = FIGURES[args.figure]
    gate_path = root / family / "science-gate-v1.json"
    gate = json.loads(gate_path.read_text())
    group_name = gate["reference"]["imprint_groups"]["additional"]
    if gate["candidate"]["imprint_groups"]["additional"] != group_name:
        raise ValueError("candidate final-imprint HDF group differs")
    rng_report_path = root / ROOT_REPORT
    rng_report = json.loads(rng_report_path.read_text())
    row = next(row for row in rng_report["pairs"] if row["figure"] == args.figure and row["name"] == checkpoint_name)
    if not (row["numpy_mt19937_state_array_exact"] and row["numpy_state_numeric_fields_exact"]):
        raise ValueError("saved final-imprint MT state does not match")

    arrays = {}
    checkpoint_evidence = {}
    for role in ("reference", "candidate"):
        if role == "reference":
            checkpoint = root / "reference/repository/stored_networks" / args.figure / checkpoint_name
            compressed = False
        else:
            checkpoint = root / family / "checkpoints" / checkpoint_name
            if not checkpoint.exists():
                checkpoint = checkpoint.with_name(checkpoint.name + ".zst")
            compressed = checkpoint.suffix == ".zst"
        raw_hash, raw_bytes, tail = raw_hash_and_tail(checkpoint, compressed)
        if raw_hash != row[("reference_tree" if role == "reference" else "candidate") + "_checkpoint_sha256"]:
            raise ValueError(f"{role} checkpoint hash differs")
        state, position, has_gauss, cached_gaussian = saved_mt(tail)
        rng_key = "reference_tree_rng" if role == "reference" else "candidate_rng"
        if hashlib.sha256(state.tobytes()).hexdigest() != row[rng_key]["payloads"]["numpy_state"]["sha256"]:
            raise ValueError(f"{role} extracted MT payload differs from prior audit")
        arrays[f"{role}_mt"] = state
        arrays[f"{role}_mt_position"] = np.array(position, dtype=np.int64)
        arrays[f"{role}_has_gauss"] = np.array(has_gauss, dtype=np.int64)
        arrays[f"{role}_cached_gaussian"] = np.array(cached_gaussian, dtype=np.float64)
        checkpoint_evidence[role] = {"sha256": raw_hash, "raw_bytes": raw_bytes,
                                     "mt_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                                     "mt_position": position, "has_gauss": has_gauss,
                                     "cached_gaussian": cached_gaussian}
        hdf = (root / "reference/repository/results/sim_files" / hdf_name
               if role == "reference" else root / family / hdf_name)
        if hdf.stat().st_size != gate[role]["bytes"]:
            raise ValueError(f"{role} HDF byte size differs")
        with h5py.File(hdf, "r") as handle:
            for area in ("A", "B", "C"):
                full = handle[group_name][f"{area}_weights"][()]
                if full.shape != (400, 2400) or not np.isfinite(full).all():
                    raise ValueError(f"unexpected {area} weight shape or values")
                for context in (0, 1):
                    name = f"{role}_{area}_c{context}"
                    arrays[name] = np.ascontiguousarray(full[:, context::6])
    if not (np.array_equal(arrays["reference_mt"], arrays["candidate_mt"])
            and np.array_equal(arrays["reference_mt_position"], arrays["candidate_mt_position"])
            and np.array_equal(arrays["reference_has_gauss"], arrays["candidate_has_gauss"])
            and np.array_equal(arrays["reference_cached_gaussian"], arrays["candidate_cached_gaussian"])):
        raise ValueError("reference and candidate saved NumPy RNG states differ")
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **arrays)
    report = {"schema": "contextual-fig6-s6-louvain-input-extract-v1",
              "mode": "mac_low_load_closed_data_extraction_no_simulation_no_performance",
              "figure": args.figure, "imprint_group": group_name,
              "gate_sha256": sha256(gate_path), "rng_tail_report_sha256": sha256(rng_report_path),
              "hdf_sha256_inherited_from_frozen_gate_not_rehashed": {
                  role: gate[role]["sha256"] for role in ("reference", "candidate")},
              "checkpoint": checkpoint_evidence,
              "npz_sha256": sha256(args.output_npz),
              "npz_bytes": args.output_npz.stat().st_size,
              "saved_mt_states_exact": True,
              "actual_rng_state_at_sort_or_recall_measured": False,
              "scientific_gate_changed": False, "performance_authorized": False}
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"figure": args.figure, "npz_bytes": report["npz_bytes"],
                      "saved_mt_states_exact": True}, sort_keys=True))


if __name__ == "__main__":
    main()
