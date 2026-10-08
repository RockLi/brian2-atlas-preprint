"""Synthetic acceptance and tamper checks for independent V1 recomputation."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SOURCE = (Path(__file__).resolve().parents[1] / "mpi-evidence"
          / "confirmation-seed1751-v8-science-identity-preparation"
          / "audit_prior_rust_v1_140_views.py")
SPEC = importlib.util.spec_from_file_location("prior_v1_audit_fixture", SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
NAMES = ("mam_V1_23E", "mam_V1_23I", "mam_V1_4E", "mam_V1_4I",
         "mam_V1_5E", "mam_V1_5I", "mam_V1_6E", "mam_V1_6I")
COUNTS = (34, 9, 50, 12, 15, 3, 14, 3)


class PriorV1IndependentAuditTest(unittest.TestCase):
    def test_recomputes_and_rejects_numeric_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary)
            for directory in ("analysis", "reader", "source", "control"):
                (archive / directory).mkdir()
            groups, arrays = [], {}
            offset = 0
            for index, (name, count) in enumerate(zip(NAMES, COUNTS, strict=True)):
                local = list(range(count))
                global_ids = list(range(offset, offset + count))
                groups.append(dict(name=name, population_index=index,
                                   sample_count=count, selected_local_ids=local,
                                   selected_global_ids=global_ids,
                                   selected_strict_spikes_lower_bound=[57] * count))
                arrays[f"p{index}_cell_counts"] = np.zeros((count, 100000), dtype=np.uint8)
                arrays[f"p{index}_local_ids"] = np.array(local, dtype=np.int64)
                arrays[f"p{index}_global_ids"] = np.array(global_ids, dtype=np.int64)
                arrays[f"p{index}_eligibility_spikes"] = np.full(count, 57, dtype=np.uint32)
                offset += count
            identity = {"runs": [dict(seed=1729, groups=groups,
                                      archived_raw_sha256={"results.bin": "frozen"})]}
            identity_path = archive / "source/identity-v3.json"
            identity_path.write_text(json.dumps(identity))
            (archive / "reader/mam-official-analysis-neuron-sizes-v1.json").write_text(
                json.dumps({"populations": [dict(name=name,
                                                 official_normalization_neurons=count)
                                            for name, count in zip(NAMES, COUNTS, strict=True)]}))
            (archive / "analysis/analysis.json").write_text(json.dumps({
                "seed": 1729, "current_raw_rehashed": True,
                "scientific_acceptance": False,
                "source_identity_sha256": audit.sha(identity_path),
                "raw_sha256": {"results.bin": "frozen"},
                "selected_global_ids": [row["selected_global_ids"] for row in groups],
                "eligibility_spikes": [[57] * count for count in COUNTS]}))
            (archive / "analysis/intent.json").write_text(
                json.dumps({"raw_sha256": {"results.bin": "frozen"}}))
            np.savez_compressed(archive / "analysis/selected-cell-counts.npz", **arrays)
            population = np.zeros((8, 100000), dtype=np.uint64)
            rates = ("modern_equal_cell", "inferred_full_population_weighted")
            views = {"sampled_population_counts": population,
                     "frequency_hz": np.fft.rfftfreq(1024, d=.001)}
            for label in rates:
                views["rate__" + label] = np.zeros(100000, dtype=float)
                for taper in ("declared_boxcar", "effective_hann"):
                    views[f"power__{label}__{taper}"] = np.zeros(513, dtype=float)
            views_path = archive / "analysis/v1-four-views.npz"
            np.savez_compressed(views_path, **views)

            def receipt():
                members = [path for path in archive.rglob("*") if path.is_file()
                           and not path.is_relative_to(archive / "control")]
                rows = [{"relative": path.relative_to(archive).as_posix(),
                         "bytes": path.stat().st_size, "sha256": audit.sha(path)}
                        for path in members]
                (archive / "control/rust1729-collection-v1.json").write_text(json.dumps({
                    "schema": "b2-mam-prior-rust-v1-140-collection-v1", "seed": 1729,
                    "scientific_acceptance": False, "automatic_retry": False,
                    "file_count": len(rows), "total_bytes": sum(row["bytes"] for row in rows),
                    "files": rows}))

            receipt()
            result = audit.audit(archive, 1729)
            self.assertTrue(result["engineering_recompute_passed"])
            self.assertEqual(len(result["checks"]), 6)
            self.assertFalse(result["scientific_acceptance"])
            views["rate__modern_equal_cell"] = np.ones(100000, dtype=float)
            np.savez_compressed(views_path, **views)
            receipt()
            with self.assertRaisesRegex(ValueError, "rate reconstruction differs"):
                audit.audit(archive, 1729)


if __name__ == "__main__":
    unittest.main()
