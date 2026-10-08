"""No-simulation checks for isolated reuse of validated S2 imprint data."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from contextual_dendritic_fig3_official_job import source_tree_digest  # noqa: E402
from contextual_dendritic_s2_import_large_imprints import (  # noqa: E402
    official_rows,
    import_candidate,
    sha256_file,
    validate_candidate,
)


REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"


class ImprintImportTest(unittest.TestCase):
    def test_validated_copy_is_isolated_and_marked_complete(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            template = root / "template"
            (template / "src").mkdir(parents=True)
            for index in range(16):
                (template / "src" / f"source_{index:02d}.py").write_text(f"x = {index}\n")
            manifest, count = source_tree_digest(template)
            self.assertEqual(count, 16)
            original = root / "original"
            (original / "src").mkdir(parents=True)
            for file in (template / "src").iterdir():
                (original / "src" / file.name).write_bytes(file.read_bytes())
            (original / ".contextual-dendritic-reproduction.json").write_text(json.dumps({
                "schema": "contextual-dendritic-isolated-reproduction-v1",
                "reproduction_id": "original-seed24",
                "source_revision": REVISION,
            }))
            h5_path = original / "results/sim_files/data_Fig_S2_large_imprint_single_dendrite.h5"
            h5_path.parent.mkdir(parents=True)
            with h5py.File(h5_path, "w") as handle:
                group = handle.create_group("fixture-group")
                group.attrs["seed"] = 24
                group.create_dataset("all_imprint_ids", data=np.arange(20))
                group.create_dataset("filename_for_baseline_network", data=b"baseline-state")
            checkpoint_dir = original / "stored_networks/Fig_S2"
            checkpoint_dir.mkdir(parents=True)
            prefix = "stored_imprint_fixture"
            checkpoint_hashes = []
            for index in range(20):
                checkpoint = checkpoint_dir / f"{prefix}_{index}"
                checkpoint.write_bytes(f"checkpoint {index}".encode())
                checkpoint_hashes.append(sha256_file(checkpoint))
            (checkpoint_dir / "baseline-state").write_bytes(b"baseline")
            row = {
                "seed": 24,
                "candidate_group": "fixture-group",
                "candidate_checkpoint_prefix": prefix,
                "candidate": {
                    "h5_path": str(h5_path),
                    "h5_sha256": sha256_file(h5_path),
                    "checkpoint_dir": str(checkpoint_dir),
                    "checkpoint_sha256": checkpoint_hashes,
                    "imprints": 20,
                },
            }
            item = validate_candidate(
                24, original, row,
                source_revision=REVISION,
                source_manifest_sha256=manifest,
            )
            campaign_root = root / "recall-campaign"
            result = import_candidate(
                item,
                template=template,
                root=campaign_root,
                campaign_id="fixture-recall",
                source_revision=REVISION,
                final_report_sha256="f" * 64,
            )
            imported_repo = Path(result["repository"])
            report = json.loads(Path(result["imported_stage_report"]).read_text())
            self.assertTrue(report["completed"])
            self.assertFalse(report["simulation_executed_by_import"])
            self.assertEqual(result["checkpoint_count"], 20)
            self.assertEqual(
                sha256_file(imported_repo / "results/sim_files/data_Fig_S2_large_imprint_single_dendrite.h5"),
                sha256_file(h5_path),
            )
            marker = json.loads((imported_repo / ".contextual-dendritic-reproduction.json").read_text())
            self.assertEqual(marker["reproduction_id"], "fixture-recall-s2-large-s0024")
            with self.assertRaisesRegex(ValueError, "already exists"):
                import_candidate(
                    item,
                    template=template,
                    root=campaign_root,
                    campaign_id="fixture-recall",
                    source_revision=REVISION,
                    final_report_sha256="f" * 64,
                )
            (checkpoint_dir / f"{prefix}_0").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint 0 differs"):
                validate_candidate(
                    24, original, row,
                    source_revision=REVISION,
                    source_manifest_sha256=manifest,
                )

    def test_incomplete_final_gate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fake-final.json"
            path.write_text(json.dumps({
                "schema": "contextual-dendritic-s2-large-imprint-ensemble-comparison-v1",
                "reported_timings": False,
                "complete": False,
                "passed": True,
                "allow_incomplete": False,
                "checks": {str(index): True for index in range(8)},
            }))
            with self.assertRaisesRegex(ValueError, "did not pass"):
                official_rows(path, sha256_file(path))


if __name__ == "__main__":
    unittest.main()
