#!/usr/bin/env python3
"""No-simulation identity tests for the generalized Fig. 8 recall driver."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import h5py

from contextual_dendritic_fig8_ensemble_recall_job import (
    SOURCE_REVISION,
    digest,
    source_tree_digest,
    validate_copy,
)


class EnsembleRecallIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="fig8-recall-identity-")
        self.repo = Path(self.tmp.name) / "paper-repository"
        (self.repo / "src").mkdir(parents=True)
        (self.repo / "src/marker.py").write_text("VALUE = 1\n")
        (self.repo / "results/sim_files").mkdir(parents=True)
        (self.repo / "stored_networks/Fig_8").mkdir(parents=True)
        (self.repo / ".contextual-dendritic-reproduction.json").write_text(json.dumps({
            "schema": "contextual-dendritic-isolated-reproduction-v1",
            "reproduction_id": "synthetic-imprint-job",
            "source_revision": SOURCE_REVISION,
        }))
        h5_path = self.repo / "results/sim_files/data_Fig_8.h5"
        with h5py.File(h5_path, "w") as h5:
            for index in range(5):
                h5.create_group(f"imprint-{index}").create_dataset("all_imprint_ids", data=[index])
        checkpoints = []
        for index in range(5):
            name = f"checkpoint-{index}"
            path = self.repo / "stored_networks/Fig_8" / name
            path.write_bytes(f"saved-{index}".encode())
            checkpoints.append({"name": name, "sha256": digest(path)})
        source_hash, _ = source_tree_digest(self.repo)
        self.imprint = {
            "schema": "contextual-dendritic-fig8-imprint-only-job-v1",
            "completed": True, "simulation_executed": True, "imprint_only": True,
            "seed": 5, "case_id": 0, "source_revision": SOURCE_REVISION,
            "source_manifest_sha256": source_hash,
            "isolation": {"reproduction_id": "synthetic-imprint-job"},
            "h5": {"sha256": digest(h5_path), "groups": 5},
            "checkpoints": checkpoints,
        }
        self.restore = {
            "schema": "contextual-dendritic-fig8-fresh-restore-preflight-v2",
            "passed": True, "seed": 5, "case_id": 0,
            "unique_checkpoints": 5, "h5_unchanged": True,
            "h5_before_sha256": digest(h5_path),
            "h5_after_sha256": digest(h5_path),
            "restored": [
                {"stored_name": item["name"], "checkpoint_sha256": item["sha256"]}
                for item in checkpoints
            ],
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_valid_copy(self) -> None:
        result = validate_copy(self.repo, self.imprint, self.restore)
        self.assertEqual(len(result["imprint_groups"]), 5)
        self.assertEqual(len(result["checkpoints"]), 5)

    def test_changed_checkpoint_rejected(self) -> None:
        (self.repo / "stored_networks/Fig_8/checkpoint-0").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "copied checkpoint differs"):
            validate_copy(self.repo, self.imprint, self.restore)

    def test_wrong_restore_seed_rejected(self) -> None:
        self.restore["seed"] = 6
        with self.assertRaisesRegex(ValueError, "restore preflight did not pass"):
            validate_copy(self.repo, self.imprint, self.restore)

    def test_extra_recall_group_rejected(self) -> None:
        with h5py.File(self.repo / "results/sim_files/data_Fig_8.h5", "a") as h5:
            h5.create_group("recall-extra")
        self.imprint["h5"]["sha256"] = digest(self.repo / "results/sim_files/data_Fig_8.h5")
        with self.assertRaisesRegex(ValueError, "pristine five-imprint cache"):
            validate_copy(self.repo, self.imprint, self.restore)


if __name__ == "__main__":
    unittest.main()
