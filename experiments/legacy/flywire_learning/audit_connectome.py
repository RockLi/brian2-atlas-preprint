"""Read-only inventory of learning-related projections in the pinned EI graph.

No compartment, receptor, behavioural valence or dopamine action is inferred.
"""
import argparse
import csv
import json
from pathlib import Path
import runpy

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Reuse the existing format validator without registering a Brian Device.
CSR = runpy.run_path(str(ROOT / "python/brian2_rust/binary_topology.py"))
EXPECTED_GRAPH = "b84a19b5c6d899181e6ade3a914eba89d53cb3a4cc36789702785b4cfa35ec38"
EXPECTED_ANNOTATIONS = "30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6"


def audit(graph, annotations, output):
    graph, output = Path(graph), Path(output)
    manifest = json.loads((graph / "manifest.json").read_text())
    digest = CSR["file_hash"](graph / "connectome.b2csr")
    if digest != EXPECTED_GRAPH or manifest["csr_sha256"] != digest:
        raise ValueError("expected pinned full FlyWire EI graph")
    if manifest["annotations_sha256"] != EXPECTED_ANNOTATIONS:
        raise ValueError("unexpected annotation provenance")
    if CSR["file_hash"](annotations) != EXPECTED_ANNOTATIONS:
        raise ValueError("original annotation TSV checksum mismatch")
    info = CSR["inspect_csr"](graph / "connectome.b2csr")
    offsets, targets, values = CSR["csr_arrays"](info)
    ids = np.load(graph / "root_ids.npy", allow_pickle=False)
    with (graph / "neurons.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if (len(rows) != info["source_count"] or info["target_count"] != len(rows)
            or ids.dtype != np.dtype("uint64")
            or not np.array_equal(ids, np.array([int(r["root_id"]) for r in rows], dtype=np.uint64))
            or [int(r["index"]) for r in rows] != list(range(len(rows)))):
        raise ValueError("annotation/graph index or exact root-ID mismatch")
    with Path(annotations).open() as stream:
        original = sorted(csv.DictReader(stream, delimiter="\t"), key=lambda r: int(r["root_id"]))
    if len(original) != len(rows):
        raise ValueError("original annotation count mismatch")
    fields = ("root_id", "cell_class", "cell_type", "hemibrain_type", "known_nt")
    if any(any(r[field] != raw[field] for field in fields)
           or r["annotation_prediction"] != raw["top_nt"] for r, raw in zip(rows, original)):
        raise ValueError("derived cell annotations differ from pinned original TSV")
    del original
    classes = {"kc": "Kenyon_Cell", "mbon": "MBON", "dan": "DAN", "pn": "ALPN"}
    groups = {k: np.array([i for i, r in enumerate(rows) if r["cell_class"] == c], dtype=np.int32)
              for k, c in classes.items()}
    masks = {k: np.isin(np.arange(len(rows)), group) for k, group in groups.items()}
    pairs = [("pn", "kc"), ("kc", "mbon"), ("dan", "kc"),
             ("dan", "mbon"), ("mbon", "dan"), ("kc", "dan")]
    statistics, selected = {}, []
    for pre, post in pairs:
        source_list, edge_list = [], []
        for source in groups[pre]:
            start, stop = int(offsets[source]), int(offsets[source + 1])
            edges = np.flatnonzero(masks[post][targets[start:stop]]) + start
            source_list.extend([int(source)] * len(edges))
            edge_list.extend(edges.tolist())
        edge = np.array(edge_list, dtype=np.int64)
        source = np.array(source_list, dtype=np.int32)
        target = np.asarray(targets[edge])
        weight = np.asarray(values[0, edge])
        statistics[f"{pre}_to_{post}"] = {
            "edges": len(edge), "sources_with_edges": len(np.unique(source)),
            "targets_with_edges": len(np.unique(target)),
            "positive_edges": int(np.sum(weight > 0)),
            "negative_edges": int(np.sum(weight < 0)),
            "zero_fast_edges": int(np.sum(weight == 0)),
            # The signed importer erases multiplicity for zero-fast sources.
            "contacts": int(np.abs(weight).sum()) if np.all(weight != 0) else None,
        }
        if (pre, post) == ("kc", "mbon"):
            selected = dict(edge_id=edge, source_index=source, target_index=target,
                            source_root_id=ids[source], target_root_id=ids[target],
                            signed_contacts=weight)
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "kc_mbon_edges.npz", **selected)
    with (output / "mbon_inventory.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows[i] for i in groups["mbon"])
    count = len(selected["edge_id"])
    report = {
        "schema": "flywire-learning-audit-v1", "source_graph": str(graph.resolve()),
        "csr_sha256": digest, "annotation_source_sha256": EXPECTED_ANNOTATIONS,
        "derived_neurons_csv_sha256": CSR["file_hash"](graph / "neurons.csv"),
        "root_ids_sha256": CSR["file_hash"](graph / "root_ids.npy"),
        "derived_annotations_revalidated_against_original_tsv": True,
        "neurons": len(rows), "whole_graph_edges": info["edge_count"],
        "groups": {k: len(v) for k, v in groups.items()}, "projections": statistics,
        "kc_mbon_fraction_of_graph": count / info["edge_count"],
        "four_extra_f64_states_bytes": count * 4 * 8,
        "limitations": [
            "DAN membership uses cell_class, not all dopamine predictions.",
            "Pair-aggregated edges do not resolve mushroom body compartments or contacts.",
            "Zero-fast edges preserve topology but not original contact multiplicity.",
            "Anatomical DAN edges do not define a synaptic dopamine gating map.",
            "Four-array estimate excludes topology, queues, monitors and runtime copies.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.graph, args.annotations, args.output), indent=2))
