"""Source-admission tests for the prior-Rust V1 derived extractor."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np  # Keep the C extension loaded across isolated module imports.


TOOLS = Path(__file__).resolve().parents[1] / "tools"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else
                     (json.dumps(value) + "\n").encode())
    return path


def catalog(directory, names):
    return {name: {"bytes": (directory / name).stat().st_size,
                   "sha256": digest(directory / name)} for name in names}


class PriorRustProbeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        corr = types.ModuleType("mam_correlation_stream")
        selector = types.ModuleType("mam_v1_140_selector")
        selector.select_population = lambda *args, **kwargs: None
        spectrum = types.ModuleType("mam_v1_140_spectrum")
        spectrum.V1_POPULATIONS = (
            "mam_V1_23E", "mam_V1_23I", "mam_V1_4E", "mam_V1_4I",
            "mam_V1_5E", "mam_V1_5I", "mam_V1_6E", "mam_V1_6I")
        spectrum.V1_SAMPLE_COUNTS = (34, 9, 50, 12, 15, 3, 14, 3)
        spectrum.four_views = lambda *args, **kwargs: None
        with patch.dict(sys.modules, {
            "mam_correlation_stream": corr,
            "mam_v1_140_selector": selector,
            "mam_v1_140_spectrum": spectrum,
        }):
            spec = importlib.util.spec_from_file_location(
                "prior_rust_fixture", TOOLS / "mam_v1_140_prior_rust_extract.py")
            self.worker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.worker)
        q = self.worker
        deployment = self.root / "deployment"
        deployment.mkdir()
        source = self.root / "source"
        reader = put(source / "python/brian2_rust/results.py", b"fixture reader")
        helpers = {name: put(source / "tools" / name, name.encode())
                   for name in q.SOURCE_TOOL_SHA}
        norm = put(source / "normalization/mam-official-analysis-neuron-sizes-v1.json", {})
        results = self.root / "run"
        results.mkdir()
        model = put(self.root / "model.json", b"{}")
        result_bin = put(results / "results.bin", b"123")
        event_bin = put(results / "events.bin", b"4567")
        analysis = self.root / "analysis"
        cell = analysis / "cell"
        series = analysis / "series"
        cell.mkdir(parents=True)
        series.mkdir()
        cell_summary = put(cell / "paper-cell-metrics.json", {
            "identity": {"model_sha256": digest(model)}})
        self.cell_counts = put(cell / "cell-metrics.npz", b"cell archive")
        put(cell / "catalog.json", catalog(cell, ("paper-cell-metrics.json", "cell-metrics.npz")))
        put(series / "time-series.json", {
            "identity": {"model_sha256": digest(model)},
            "schema": "b2-mam-modern-paper-time-series-v1",
            "wrapper_window_ms": "(500,100500]", "physical_tick_ms": .1,
            "source_sha256": {"normalization": digest(norm)}})
        put(series / "time-series.npz", b"series archive")
        put(series / "catalog.json", catalog(series, ("time-series.json", "time-series.npz")))
        report = put(analysis / "report.json", {
            "analysis_complete": True,
            "catalogs": {"cell": digest(cell / "catalog.json"),
                         "series": digest(series / "catalog.json")}})
        groups = [{"name": name, "sample_count": count,
                   "selected_global_ids": list(range(count))}
                  for name, count in zip(spectrum.V1_POPULATIONS,
                                         spectrum.V1_SAMPLE_COUNTS, strict=True)]
        identity = put(deployment / "identity-v3.json", {
            "schema": "b2-mam-prior-rust-v1-140-identity-probe-v1",
            "selected_ids_certain_for_terminal_0_or_1": True,
            "runs": [{"seed": 1729, "model_sha256": digest(model),
                      "source_sha256": {"report.json": digest(report),
                                        "catalog.json": digest(cell / "catalog.json"),
                                        "paper-cell-metrics.json": digest(cell_summary),
                                        "cell-metrics.npz": digest(self.cell_counts)},
                      "groups": groups}]})
        q.ROOT = deployment
        q.IDENTITY = identity
        q.IDENTITY_SHA = digest(identity)
        q.SOURCE_ROOT = source
        q.READER_SHA = digest(reader)
        q.TOOL_SHA = {name: digest(TOOLS / name) for name in q.TOOL_SHA}
        q.SOURCE_TOOL_SHA = {name: digest(path) for name, path in helpers.items()}
        q.free_bytes = lambda path: q.MIN_FREE + q.MAX_OUTPUT + 1
        q.BINDINGS = {1729: {
            "model": model, "results": results, "analysis": analysis,
            "report": report, "report_sha256": digest(report),
            "series_catalog_sha256": digest(series / "catalog.json"),
            "sizes": {"model.json": model.stat().st_size,
                      "results.bin": result_bin.stat().st_size,
                      "events.bin": event_bin.stat().st_size}}}

    def test_source_binding_and_tamper_rejection(self):
        self.assertTrue(self.worker.probe(1729)["ready"])
        self.cell_counts.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "stage member mismatch"):
            self.worker.probe(1729)

    def test_unbounded_duration_rejected(self):
        self.assertEqual(self.worker.duration_seconds("3h"), 10800)
        with self.assertRaises(ValueError):
            self.worker.duration_seconds("infinity")


if __name__ == "__main__":
    unittest.main()
