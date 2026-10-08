"""Freeze the modern V1 sample IDs from accepted seed1751 cell evidence.

The archived cell stage counts [500,100500) ms and records tick 500
separately. The modern V1 wrapper selects on (500,100500] ms. Because the
validated raw event order permits at most one terminal event per cell, an ID
with exactly 56 strict-window spikes is the only possible threshold ambiguity.
This probe admits an ID set only when no such ID precedes the last certainly
eligible ID required in any V1 population. It does not read raw events or
calculate spectra.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mam_v1_140_spectrum import V1_POPULATIONS, V1_SAMPLE_COUNTS


SCIENCE_SHA = "cecddd315f1570a40163c4b984407906c01fa89bca75182e70db3b27602a975c"
RAW_AUDIT_SHA = "b52633c963aff20545b8c749a50694834f5f1c7cf48468f337b9c2d74153097d"
TOTAL_NEURONS = 4_129_924


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def member(directory, catalog, name):
    path = directory / name
    entry = catalog[name]
    if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
        raise ValueError(f"catalog member mismatch: {path}")
    return path


def certain_ids(strict_counts, needed):
    """Infer selected IDs for any possible 0/1 terminal-event vector."""
    counts = np.asarray(strict_counts)
    if (counts.ndim != 1 or counts.dtype.kind not in "iu"
            or np.any(counts < 0) or type(needed) is not int or needed <= 0):
        raise ValueError("invalid strict-window cell counts")
    eligible = np.flatnonzero(counts >= 57)
    if len(eligible) < needed:
        raise ValueError("fewer than required certainly eligible cells")
    selected = eligible[:needed]
    # A terminal tick can promote a cell with 56 strict-window spikes. Any
    # such lower ID would displace the selected highest ID.
    uncertain = np.flatnonzero(counts[:int(selected[-1]) + 1] == 56)
    if len(uncertain):
        raise ValueError("terminal tick can change selected IDs")
    return selected, int(eligible.size), int(np.count_nonzero(counts == 56))


def probe(root):
    control, analysis = root / "control", root / "analysis"
    science = control / "science-completion-v8.json"
    raw_audit = control / "raw-audit-completion-v8.json"
    if digest(science) != SCIENCE_SHA or digest(raw_audit) != RAW_AUDIT_SHA:
        raise ValueError("accepted completion hashes differ")
    completion = json.loads(science.read_text())
    audit = json.loads(raw_audit.read_text())
    if (completion.get("full_descriptive_analysis_complete") is not True
            or completion.get("input_sha256", {}).get("engineering_completion") != RAW_AUDIT_SHA
            or audit.get("engineering_acceptance") is not True):
        raise ValueError("science or raw engineering completion invalid")

    catalogs = {}
    for stage in ("cell", "series"):
        directory = analysis / stage
        catalog_path = directory / "catalog.json"
        if digest(catalog_path) != completion["catalogs"][stage]:
            raise ValueError(f"{stage} catalog differs from accepted completion")
        catalogs[stage] = json.loads(catalog_path.read_text())
    cell_json = member(analysis / "cell", catalogs["cell"], "paper-cell-metrics.json")
    cell_npz = member(analysis / "cell", catalogs["cell"], "cell-metrics.npz")
    series_json = member(analysis / "series", catalogs["series"], "time-series.json")
    cell, series = json.loads(cell_json.read_text()), json.loads(series_json.read_text())
    stage_source = control / "analysis-source-v1"
    attested_code = {}
    for name, relative in (("results.py", "python/brian2_rust/results.py"),
                           ("analyze_mam_paper_cell_metrics.py", "tools/analyze_mam_paper_cell_metrics.py")):
        source = stage_source / relative
        source_hash = digest(source)
        if source_hash != cell["implementation_sha256"][name]:
            raise ValueError(f"accepted cell stage source differs: {name}")
        attested_code[name] = source_hash
    if (cell.get("schema") != "b2-mam-paper-cell-metrics-v1"
            or cell.get("window", {}).get("start_tick") != 5000
            or cell["window"].get("end_tick") != 1005000
            or cell["window"].get("rate_endpoint") != "(start,end)"
            or series.get("schema") != "b2-mam-modern-paper-time-series-v1"
            or series.get("physical_tick_ms") != .1
            or series.get("wrapper_window_ms") != "(500,100500]"
            or series.get("helper_histogram_range_ms") != [500.5, 100500.5]):
        raise ValueError("archived stage observation differs")
    names = series["population_names"]
    rows = cell["populations"]
    if ([row["name"] for row in rows] != names or len(rows) != 254
            or sum(row["neurons"] for row in rows) != TOTAL_NEURONS
            or cell["identity"]["model_sha256"] != series["identity"]["model_sha256"]):
        raise ValueError("population order or model identity differs")
    source_hashes = cell["source_sha256"]
    for name in ("results.bin", "events.bin"):
        matches = [value for path, value in source_hashes.items() if path.endswith("/" + name)]
        if len(matches) != 1 or matches[0] != audit["dump_sha256"][name]:
            raise ValueError(f"archived {name} hash differs from raw audit")
    model_hashes = [value for path, value in source_hashes.items() if path.endswith("/model.json")]
    if model_hashes != [cell["identity"]["model_sha256"]]:
        raise ValueError("archived model hash differs")

    norm_path = control / "analysis-source-v1/normalization/mam-official-analysis-neuron-sizes-v1.json"
    norm_hash = digest(norm_path)
    if norm_hash not in series["source_sha256"].values():
        raise ValueError("unrounded official normalization not bound to series stage")
    norm = json.loads(norm_path.read_text())
    if [row["name"] for row in norm["populations"]] != names:
        raise ValueError("normalization population order differs")
    neurons = np.asarray([norm["populations"][names.index(name)]["official_normalization_neurons"]
                          for name in V1_POPULATIONS], dtype=np.float64)
    if tuple(np.rint(140 * neurons / neurons.sum()).astype(int)) != V1_SAMPLE_COUNTS:
        raise ValueError("official V1 population allocation differs")

    offsets = np.cumsum([0] + [row["neurons"] for row in rows[:-1]])
    groups = []
    with np.load(cell_npz, allow_pickle=False) as arrays:
        for name, required in zip(V1_POPULATIONS, V1_SAMPLE_COUNTS, strict=True):
            i = names.index(name)
            half = arrays[f"p{i}_half_open_cell_counts"]
            lower = arrays[f"p{i}_lower_boundary_cell_counts"]
            if (half.shape != (rows[i]["neurons"],) or lower.shape != half.shape
                    or np.any(lower < 0) or np.any(half < lower)):
                raise ValueError(f"{name} archived cell counts invalid")
            strict = half - lower
            local, certainly_eligible, ambiguous_56 = certain_ids(strict, required)
            groups.append(dict(name=name, population_index=i, simulated_neurons=rows[i]["neurons"],
                               global_offset=int(offsets[i]), sample_count=required,
                               selected_local_ids=local.tolist(),
                               selected_global_ids=(local + offsets[i]).tolist(),
                               selected_strict_spikes_lower_bound=strict[local].tolist(),
                               certainly_eligible_cells=certainly_eligible,
                               total_terminal_ambiguous_56_cells=ambiguous_56,
                               ambiguous_56_at_or_before_last_selected=0))
    all_ids = [global_id for group in groups for global_id in group["selected_global_ids"]]
    if len(all_ids) != 140 or len(set(all_ids)) != 140:
        raise ValueError("V1 selected global IDs are not 140 distinct cells")
    return dict(schema="b2-mam-v1-140-derived-identity-probe-v2",
                source="accepted seed1751 v8 archived cell and series stages",
                historical_paper_sample=False, current_raw_rehashed=False,
                scientific_acceptance=False, selected_ids_certain_for_terminal_0_or_1=True,
                terminal_events_per_cell_upper_bound=1,
                selection_policy="lowest local IDs with >0.56 Hz on (500,100500] ms",
                source_sha256={str(path): digest(path) for path in
                               (science, raw_audit, cell_json, cell_npz, series_json, norm_path)},
                accepted_stage_implementation_sha256=attested_code,
                archived_raw_sha256=audit["dump_sha256"],
                archived_model_sha256=cell["identity"]["model_sha256"],
                groups=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = probe(args.root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.output), groups=len(report["groups"]),
                          selected_ids=140, source_bound_to_archived_stages=True,
                          current_raw_rehashed=False)))


if __name__ == "__main__":
    main()
