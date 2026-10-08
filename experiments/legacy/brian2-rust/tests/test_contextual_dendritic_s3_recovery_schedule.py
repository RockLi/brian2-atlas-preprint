"""Pure-data safety checks for an exact missing-cell S3 recovery schedule."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from contextual_dendritic_s3_campaign import (  # noqa: E402
    load_missing_recovery_ids,
    sha256_file,
)


class S3RecoveryScheduleTest(unittest.TestCase):
    def report(self, ids: list[str]) -> dict:
        return {
            "schema": "contextual-dendritic-s3-recurrent-recovered-ensemble-v1",
            "paper_source_revision": "73feb595ede908a368947d932055dc0a4e1b3817",
            "original_campaign_cells_expected": 1000,
            "original_campaign_cells_observed": 1000,
            "resolved_cells": 1000 - len(ids),
            "missing_recovery_ids": ids,
            "complete": False,
            "final_ensemble_passed": None,
        }

    def test_accepts_only_hash_pinned_exact_missing_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recovery.json"
            path.write_text(json.dumps(self.report([
                "recurrent-s001-on", "recurrent-s000-off"
            ])))
            digest = sha256_file(path)
            self.assertEqual(
                load_missing_recovery_ids(path, digest),
                ["recurrent-s000-off", "recurrent-s001-on"],
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_missing_recovery_ids(path, "0" * 64)

    def test_rejects_duplicate_or_ill_formed_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recovery.json"
            for ids in (
                ["recurrent-s001-on", "recurrent-s001-on"],
                ["recurrent-s1000-on"],
                ["ff-s001-on"],
                [["recurrent-s001-on"]],
            ):
                path.write_text(json.dumps(self.report(ids)))
                with self.assertRaisesRegex(ValueError, "unresolved-cell audit"):
                    load_missing_recovery_ids(path, sha256_file(path))


if __name__ == "__main__":
    unittest.main()
