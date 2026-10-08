"""Small, simulation-free regression checks for the recovered Fig. S3 gate."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from contextual_dendritic_s3_recurrent_recovered_ensemble_compare import (  # noqa: E402
    REFERENCE_HDF5_SHA256,
    SOURCE_SHA256,
    merge,
    sha256_file,
    validate_recovery,
)


REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"


def fixture_base() -> dict:
    cells = [
        {
            "id": "recurrent-s001-off",
            "seed": 1,
            "condition": "off",
            "group": "off-group",
            "reference_assembly_size": 20,
            "candidate_assembly_size": 20,
            "reference_tagged_loader_contract": True,
            "candidate_tagged_loader_contract": True,
        },
        {
            "id": "recurrent-s001-on",
            "seed": 1,
            "condition": "on",
            "group": "on-group",
            "reference_assembly_size": 22,
            "candidate_assembly_size": 22,
            "reference_tagged_loader_contract": True,
            "candidate_tagged_loader_contract": False,
        },
    ]
    return {
        "cells": cells,
        "observed_cells": 2,
        "expected_total": 2,
        "complete": True,
        "failures": [],
    }


class RecoveredGateTest(unittest.TestCase):
    def test_missing_sidecar_never_claims_final_gate(self) -> None:
        result = merge(fixture_base(), {}, expected_total=2)
        self.assertEqual(result["resolved_cells"], 1)
        self.assertEqual(result["missing_recovery_ids"], ["recurrent-s001-on"])
        self.assertFalse(result["complete"])
        self.assertIsNone(result["final_ensemble_passed"])

    def test_validated_sidecar_completes_pair_without_overwriting_original(self) -> None:
        recovery = {"recurrent-s001-on": {"candidate_assembly_size": 22}}
        result = merge(fixture_base(), recovery, expected_total=2)
        self.assertTrue(result["complete"])
        self.assertTrue(result["final_ensemble_passed"])
        self.assertEqual(result["loader_compatible_original_cells"], 1)
        self.assertEqual(result["exact_order_recovered_cells"], 1)

    def test_compatible_original_cannot_be_replaced(self) -> None:
        recovery = {"recurrent-s001-off": {"candidate_assembly_size": 19}}
        with self.assertRaisesRegex(ValueError, "must not be overwritten"):
            merge(fixture_base(), recovery, expected_total=2)

    def test_sidecar_identity_and_checksum_are_required(self) -> None:
        row = fixture_base()["cells"][1]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate = root / "candidate.h5"
            candidate.write_bytes(b"synthetic bytes: no simulation")
            sidecar = root / "sidecar.json"
            sidecar.write_text(json.dumps({
                "schema": "contextual-dendritic-s3-recurrent-saved-neuron-order-v2",
                "seed": 1,
                "hdf5_group": "on-group",
                "recurrent_inhibition_enabled": True,
                "paper_source_revision": REVISION,
                "hdf5_sha256": sha256_file(candidate),
            }))
            report = root / "comparison.json"
            report.write_text(json.dumps({
                "schema": "contextual-dendritic-s3-recurrent-sidecar-comparison-v2",
                "reported_timings": False,
                "scientific_identity_valid": True,
                "checks": {"synthetic_comparator_check": True},
                "expected_paper_source_revision": REVISION,
                "reference_hdf5_source_sha256_pinned_not_rehashed": SOURCE_SHA256,
                "group": "on-group",
                "reference_paper_assembly_size": 22,
                "full_1000_cell_ensemble_gate_executed": False,
                "sidecar": str(sidecar),
                "sidecar_sha256": sha256_file(sidecar),
                "candidate_hdf5": str(candidate),
                "candidate_hdf5_sha256": sha256_file(candidate),
                "candidate_paper_assembly_size_with_exact_saved_order": 22,
            }))
            value = validate_recovery(
                "recurrent-s001-on", row, report, REVISION, rehash_candidate=True
            )
            self.assertEqual(value["candidate_assembly_size"], 22)

            report_value = json.loads(report.read_text())
            report_value["reference_hdf5_source_sha256_pinned_not_rehashed"] = REFERENCE_HDF5_SHA256
            report.write_text(json.dumps(report_value))
            with self.assertRaisesRegex(ValueError, "semantic-input HDF5 identity differs"):
                validate_recovery(
                    "recurrent-s001-on", row, report, REVISION, rehash_candidate=True
                )
            report_value["reference_hdf5_source_sha256_pinned_not_rehashed"] = SOURCE_SHA256
            report.write_text(json.dumps(report_value))

            sidecar.write_text(sidecar.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                validate_recovery(
                    "recurrent-s001-on", row, report, REVISION, rehash_candidate=True
                )


if __name__ == "__main__":
    unittest.main()
