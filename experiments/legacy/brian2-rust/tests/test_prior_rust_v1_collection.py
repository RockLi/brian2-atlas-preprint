"""Exact-once local preservation gates for prior Rust V1 derivations."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SOURCE = (Path(__file__).resolve().parents[1] / "mpi-evidence"
          / "confirmation-seed1751-v8-science-identity-preparation"
          / "collect_prior_rust_v1_140.py")
SPEC = importlib.util.spec_from_file_location("prior_rust_v1_collection", SOURCE)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)
REMOTE_SOURCE = SOURCE.with_name("prior_rust_v1_collection_remote_probe.py")
REMOTE_SPEC = importlib.util.spec_from_file_location(
    "prior_rust_v1_remote_probe", REMOTE_SOURCE)
remote_probe = importlib.util.module_from_spec(REMOTE_SPEC)
REMOTE_SPEC.loader.exec_module(remote_probe)


class CollectionGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.volume = Path(self.temp.name)
        self.destination = self.volume / "artifacts"
        self.patch_t7 = mock.patch.object(collector, "T7", self.volume)
        self.patch_root = mock.patch.object(collector, "T7_ROOT", self.destination)
        self.patch_mount = mock.patch.object(
            collector.Path, "is_mount", lambda path: path == self.volume)
        self.patch_t7.start()
        self.patch_root.start()
        self.patch_mount.start()
        self.addCleanup(self.patch_mount.stop)
        self.addCleanup(self.patch_root.stop)
        self.addCleanup(self.patch_t7.stop)

    def test_incomplete_source_cannot_be_collected(self):
        source = {"ready": False, "remote_receipt_exists": False,
                  "files": [], "seed": 1729}
        with mock.patch.object(collector, "source_probe", return_value=source):
            state = collector.probe(1729)
            self.assertFalse(state["ready"])
            self.assertFalse(state["already_collected"])
            with self.assertRaisesRegex(RuntimeError, "not admissible"):
                collector.collect(1729)
        self.assertFalse(self.destination.exists())

    def test_remote_receipt_without_local_copy_is_inconsistent(self):
        source = {"ready": True, "remote_receipt_exists": True,
                  "total_bytes": 1, "files": [], "seed": 1750}
        with mock.patch.object(collector, "source_probe", return_value=source):
            state = collector.probe(1750)
        self.assertFalse(state["ready"])
        self.assertTrue(state["inconsistent_destinations"])

    def test_existing_receipt_rehashes_every_member(self):
        seed = 1729
        target = collector.paths(seed)
        member = target["destination"] / "analysis" / "result.bin"
        member.parent.mkdir(parents=True)
        member.write_bytes(b"frozen")
        row = {"source": "/node23/result.bin", "relative": "analysis/result.bin",
               "bytes": 6, "sha256": hashlib.sha256(b"frozen").hexdigest()}
        receipt = {"schema": "b2-mam-prior-rust-v1-140-collection-v1",
                   "seed": seed, "automatic_retry": False,
                   "scientific_acceptance": False,
                   "source_completion_sha256": "completion", "files": [row],
                   "file_count": 1, "total_bytes": 6}
        target["local_receipt"].parent.mkdir(parents=True)
        target["local_receipt"].write_text(json.dumps(receipt))
        source = {"ready": True, "remote_receipt_exists": True,
                  "remote_receipt_sha256": collector.sha(target["local_receipt"]),
                  "completion_sha256": "completion", "files": [row],
                  "file_count": 1, "total_bytes": 6, "seed": seed}
        with mock.patch.object(collector, "source_probe", return_value=source):
            state = collector.probe(seed)
            self.assertTrue(state["already_collected"])
            self.assertFalse(state["ready"])
            sidecar = member.with_name("._" + member.name)
            sidecar.write_bytes(b"\x00\x05\x16\x07metadata")
            self.assertTrue(collector.probe(seed)["already_collected"])
            directory_sidecar = target["destination"] / "._analysis"
            directory_sidecar.write_bytes(b"\x00\x05\x16\x07metadata")
            self.assertTrue(collector.probe(seed)["already_collected"])
            sidecar.write_bytes(b"invalid-metadata")
            with self.assertRaisesRegex(RuntimeError, "invalid T7 AppleDouble"):
                collector.probe(seed)
            sidecar.write_bytes(b"\x00\x05\x16\x07metadata")
            member.write_bytes(b"edited")
            with self.assertRaisesRegex(RuntimeError, "existing T7 member differs"):
                collector.probe(seed)
            member.write_bytes(b"frozen")
            (target["destination"] / "analysis" / "extra.bin").write_bytes(b"extra")
            with self.assertRaisesRegex(RuntimeError, "membership differs"):
                collector.probe(seed)

    def test_source_change_rejected_before_publication(self):
        original = {"ready": True, "remote_receipt_exists": False,
                    "completion_sha256": "complete", "files": [],
                    "total_bytes": 0}
        changed = dict(original, completion_sha256="different")
        with mock.patch.object(collector, "source_probe", return_value=changed):
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                collector.same_source(1729, original)


class SelectionBindingTests(unittest.TestCase):
    def test_collected_views_bind_frozen_ids_and_raw_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "rust1729"
            output.mkdir()
            groups = [dict(selected_global_ids=[100 + i],
                           selected_strict_spikes_lower_bound=[57])
                      for i in range(8)]
            identity = {"selected_ids_certain_for_terminal_0_or_1": True,
                        "runs": [{"seed": 1729, "groups": groups,
                                  "archived_raw_sha256": {"events.bin": "raw"}}]}
            identity_path = root / "identity-v3.json"
            identity_path.write_text(json.dumps(identity))
            intent = {"identity_sha256": collector.sha(identity_path),
                      "raw_sha256": {"events.bin": "raw"}}
            analysis = {
                "raw_sha256": {"events.bin": "raw"},
                "selected_global_ids": [[100 + i] for i in range(8)],
                "eligibility_spikes": [[57] for _ in range(8)],
                "rates": ["modern_equal_cell", "inferred_full_population_weighted"],
                "power_views": ["modern_equal_cell__declared_boxcar",
                                "modern_equal_cell__effective_hann",
                                "inferred_full_population_weighted__declared_boxcar",
                                "inferred_full_population_weighted__effective_hann"],
                "historical_paper_sample": False, "native_equivalence": False,
                "paper_equivalence": False}
            (output / "intent.json").write_text(json.dumps(intent))
            (output / "analysis.json").write_text(json.dumps(analysis))
            with (mock.patch.object(remote_probe, "ROOT", root),
                  mock.patch.dict(remote_probe.SOURCE_SHA,
                                  {"identity-v3.json": collector.sha(identity_path)})):
                remote_probe.verify_selection_binding(1729, output)
                analysis["selected_global_ids"][0] = [999]
                (output / "analysis.json").write_text(json.dumps(analysis))
                with self.assertRaisesRegex(ValueError, "source or interpretation"):
                    remote_probe.verify_selection_binding(1729, output)


if __name__ == "__main__":
    unittest.main()
