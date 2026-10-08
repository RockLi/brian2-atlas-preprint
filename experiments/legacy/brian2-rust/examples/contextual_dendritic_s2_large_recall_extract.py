#!/usr/bin/env python3
"""Extract the 200 Figure S2 large-network recall points in the official cache.

The official cache contains full-strength recalls for twenty learned patterns
and two contexts in each of five seeds. It does not contain the 11-point
cue-size sweeps. This reader never treats absent official points as evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from contextual_dendritic_s2_recall_extract import activity, scalar, sha256


GATE_SHA256 = "caa0ac215c1827f3b6df82a8a5d6c2bdcae7b3dafe0da10d388354126481fa08"
SEEDS = (24, 485, 932, 3523, 63)


def cue_identity(group: h5py.Group) -> tuple[int, int] | None:
    attrs = group.attrs
    if "all_assembly_ids_for_areas_recall" not in attrs:
        return None
    if "assembly_size_recall" in attrs:
        return None  # Candidate-only cue-size sweep, absent in official cache.
    cue = np.asarray(attrs["all_assembly_ids_for_areas_recall"], dtype=int)
    context = np.asarray(attrs["all_context_ids_for_areas_recall"], dtype=int)
    if cue.shape != (1, 1, 3) or context.shape != (1, 1, 2):
        raise ValueError("wrong S2 large-recall cue/context shape")
    if cue[0, 0, 0] != 0 or cue[0, 0, 2] != -1 or context[0, 0, 0] != 0:
        raise ValueError("wrong S2 large-recall cue/context tuple")
    index = int(cue[0, 0, 1])
    value = int(context[0, 0, 1])
    if index not in range(20) or value not in (0, 1):
        raise ValueError("S2 large-recall cue/context outside paper schedule")
    return index, value


def extract(h5_path: Path, gate_path: Path, seed: int) -> dict:
    if seed not in SEEDS:
        raise ValueError("not a paper S2 large-network seed")
    if sha256(gate_path) != GATE_SHA256:
        raise ValueError("five-seed imprint science gate digest changed")
    gate = json.loads(gate_path.read_text())
    if gate.get("passed") is not True or gate.get("complete") is not True:
        raise ValueError("large-imprint science gate is not complete/passed")
    row = next((item for item in gate["rows"] if item["seed"] == seed), None)
    if row is None:
        raise ValueError(f"missing seed {seed} from imprint gate")
    reference_hash = row["reference"]["h5_sha256"]
    candidate_hash = row["candidate"]["h5_sha256"]
    current_hash = sha256(h5_path)
    side = "reference" if current_hash == reference_hash else "candidate"
    if side == "reference" and current_hash != "14fb9e678f11e91d1c2862a72d76a8a5c477ae66ec2333845c22ebf4e73674e8":
        raise ValueError("official large-network HDF5 digest changed")
    if side == "candidate" and not h5_path.is_file():
        raise ValueError("candidate HDF5 missing")
    assemblies = row[side]["assembly_neuron_ids"]
    if len(assemblies) != 20:
        raise ValueError("wrong number of imprint assemblies")
    original_key = row[side]["group"]
    records: dict[str, dict] = {}
    with h5py.File(h5_path, "r") as h5:
        if original_key not in h5:
            raise ValueError("gated imprint group missing from recall HDF5")
        if int(scalar(h5[original_key].attrs["seed"])) != seed:
            raise ValueError("gated imprint group has wrong seed")
        for key, group in h5.items():
            if key == original_key or int(scalar(group.attrs["seed"])) != seed:
                continue
            identity = cue_identity(group)
            if identity is None:
                continue
            cue, context = identity
            if int(scalar(group.attrs["recall_after_imprint_id"])) != 19:
                continue  # Candidate-only recall-after-earlier-imprint sweep.
            baseline = 1000.0 * scalar(group.attrs["runtime_baseline"])
            imprint = 1000.0 * scalar(group.attrs["runtime_imprint"])
            runtime = 1000.0 * scalar(group.attrs["runtime_recall"])
            start = baseline + 20 * (baseline + imprint)
            times = np.asarray(group["spikes_somas_t_A"], dtype=float)
            indices = np.asarray(group["spikes_somas_i_A"], dtype=np.int64)
            if times.shape != indices.shape:
                raise ValueError(f"mismatched soma spike arrays in {key}")
            selected = np.asarray(assemblies[cue], dtype=np.int64)
            result = activity(indices, times, selected, 400, start, start + runtime)
            label = f"{seed}:{cue}:{context}"
            if label in records:
                raise ValueError(f"duplicate S2 large-recall semantic key {label}")
            records[label] = {"h5_group": key, "seed": seed, "cue": cue,
                              "context": context, **result}
    expected = {f"{seed}:{cue}:{context}" for cue in range(20) for context in (0, 1)}
    if set(records) != expected:
        raise ValueError(f"large-recall grid incomplete: {len(records)}/40")
    return {
        "schema": "contextual-dendritic-s2-large-recall-extract-v1",
        "purpose": "scientific_correctness_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "side": side,
        "seed": seed,
        "five_seed_imprint_gate_sha256": GATE_SHA256,
        "source": {"h5_path": str(h5_path.resolve()), "h5_sha256": current_hash,
                   "h5_bytes": h5_path.stat().st_size, "gated_imprint_group": original_key,
                   "original_imprint_h5_sha256": reference_hash if side == "reference" else candidate_hash},
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5", type=Path)
    parser.add_argument("five_seed_imprint_gate", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    result = extract(args.h5, args.five_seed_imprint_gate, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "seed": args.seed,
                      "side": result["side"], "recalls": len(result["records"])}, sort_keys=True))


if __name__ == "__main__":
    main()
