from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from brian2_rust.binary_topology import file_hash, inspect_csr
from .config import digest

FULL_HASH = "b84a19b5c6d899181e6ade3a914eba89d53cb3a4cc36789702785b4cfa35ec38"
ANNOTATION_HASH = "30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6"


@dataclass
class Graph:
    root_ids: np.ndarray
    inputs: np.ndarray
    kc: np.ndarray
    mbon: np.ndarray
    identity: str
    scope: str
    csr: Path | None = None
    source: np.ndarray | None = None
    target: np.ndarray | None = None
    contacts: np.ndarray | None = None
    edge_count: int = 0

    def readouts(self, count, seed):
        ranked = sorted(self.kc, key=lambda i: digest([seed, str(self.root_ids[i])]))
        return np.sort(np.concatenate([np.asarray(ranked[:count], dtype=int), self.mbon]))


def load_graph(path):
    path = Path(path).resolve()
    if path.is_dir():
        manifest = json.loads((path / "manifest.json").read_text())
        csr = path / "connectome.b2csr"
        if file_hash(csr) != FULL_HASH or manifest.get("csr_sha256") != FULL_HASH:
            raise ValueError("expected the pinned full FlyWire EI graph")
        if manifest.get("annotations_sha256") != ANNOTATION_HASH:
            raise ValueError("annotation provenance mismatch")
        info = inspect_csr(csr)
        with np.load(path / "annotations.npz", allow_pickle=False) as data:
            ids, inputs, kc, mbon = [data[k].copy() for k in ("root_ids", "all_pn", "kc", "mbon")]
        identity = digest({"csr": FULL_HASH, "annotations": file_hash(path / "annotations.npz")})
        graph = Graph(ids, inputs, kc, mbon, identity, "full FlyWire v783 EI", csr=csr,
                      edge_count=info["edge_count"])
        if (len(ids) != 139255 or info["edge_count"] != 15091983 or
                (len(inputs), len(kc), len(mbon)) != (685, 5177, 96)):
            raise ValueError("full graph size mismatch")
    else:
        # Bundled induced subgraphs are explicitly diagnostic, never full-brain data.
        data = json.loads(path.read_text())
        if data.get("schema") != "b2-flywire-circuit-v1" or data.get("source_csr_sha256") != FULL_HASH:
            raise ValueError("expected a provenance-bearing FlyWire induced circuit")
        nodes, edges = data["nodes"], data["edges"]
        group = lambda label: np.array([i for i, n in enumerate(nodes) if n["cell_class"] == label], dtype=int)
        graph = Graph(np.array([int(n["root_id"]) for n in nodes], dtype=np.uint64),
                      group("ALPN"), group("Kenyon_Cell"), group("MBON"), file_hash(path),
                      data["scope"], source=np.array([e["source"] for e in edges], dtype=np.int32),
                      target=np.array([e["target"] for e in edges], dtype=np.int32),
                      contacts=np.array([e["signed_contacts"] for e in edges], dtype=float),
                      edge_count=len(edges))
    n = len(graph.root_ids)
    if graph.root_ids.dtype != np.uint64 or len(np.unique(graph.root_ids)) != n:
        raise ValueError("duplicate root IDs")
    for group in (graph.inputs, graph.kc, graph.mbon):
        if (group.dtype.kind not in "iu" or group.ndim != 1 or not len(group) or
                len(np.unique(group)) != len(group) or np.any(group < 0) or np.any(group >= n)):
            raise ValueError("missing/invalid annotated cell set")
    if np.intersect1d(graph.inputs, np.r_[graph.kc, graph.mbon]).size:
        raise ValueError("input and readout cells overlap")
    if np.intersect1d(graph.kc, graph.mbon).size:
        raise ValueError("KC and MBON cells overlap")
    if graph.csr is None:
        if (np.any(graph.source < 0) or np.any(graph.source >= n) or np.any(graph.target < 0)
                or np.any(graph.target >= n) or not np.isfinite(graph.contacts).all()):
            raise ValueError("invalid circuit edge")
    return graph
