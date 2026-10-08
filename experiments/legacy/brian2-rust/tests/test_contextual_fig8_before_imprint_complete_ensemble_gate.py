"""Pure-JSON positive and negative contracts for the 180-condition gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = (Path(__file__).resolve().parents[1] / "tools"
          / "contextual_fig8_before_imprint_complete_ensemble_gate.py")
spec = importlib.util.spec_from_file_location("before_imprint_ensemble_gate", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


class BeforeImprintEnsembleGateTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.plan = self.root / "plan.json"
        self.pilot = self.root / "pilot.json"
        self.pilot_gate = self.root / "pilot-gate.json"
        self.campaign = self.root / "campaign"
        self.output = self.root / "ensemble-gate.json"
        self.seeds = list(range(19)) + [6427]
        rows = [self.row(seed, order, stimulus) for seed in self.seeds
                for order in range(3) for stimulus in range(3)]
        save(self.plan, {"before_imprint_conditions": 180, "seed_count": 20,
                         "source_stimulus_count_per_order": 3,
                         "paper_source_sha256": module.SOURCE_SHA256,
                         "official_hdf_sha256": module.OFFICIAL_HDF_SHA256,
                         "rows": rows})
        save(self.pilot, {"status": "completed", "host": module.HOST,
                          "run_recall_after_imprint": False,
                          "run_durations_seconds": [2.0, 0.1]})
        save(self.pilot_gate, {"passed": True,
                               "source_report_sha256": module.sha256(self.pilot),
                               "published_groups_byte_identical": 212,
                               "published_datasets_byte_identical": 2853})
        for seed in self.seeds:
            groups = [f"g{seed}_{order}_{stimulus}" for order in range(3)
                      for stimulus in range(3)]
            records = [{"order": order, "stimulus": stimulus,
                        "final_checkpoint": f"final_{seed}_{order}",
                        "converted_baseline_checkpoint": f"converted_{seed}_{order}",
                        "new_hdf_group": f"g{seed}_{order}_{stimulus}",
                        "selected_counts": [4, 5],
                        "response_mean_hz_by_area": [2.0, 3.0],
                        "response_active_neurons_by_area": [1.0, 2.0],
                        "background_mean_hz_by_area": [0.1, 0.2]}
                       for order in range(3) for stimulus in range(3)]
            candidate_hash = hashlib.sha256(f"hdf-{seed}".encode()).hexdigest()
            source = {"status": "completed", "host": module.HOST, "seed": seed,
                      "source_sha256": module.SOURCE_SHA256,
                      "official_hdf_sha256": module.OFFICIAL_HDF_SHA256,
                      "condition_plan_sha256": module.sha256(self.plan),
                      "compiled_queue_sha256": module.QUEUE_SHA256,
                      "driver_sha256": module.DRIVER_SHA256,
                      "brian2_version": "2.9.0",
                      "final_checkpoint_sha256": {
                          str(order): self.hash(f"final_{seed}_{order}")
                          for order in range(3)},
                      "converted_baseline_checkpoint_sha256": {
                          str(order): self.hash(f"converted_{seed}_{order}")
                          for order in range(3)},
                      "orders": [0, 1, 2], "stimuli": [0, 1, 2],
                      "run_recall_after_imprint": False,
                      "change_firing_rate": True, "all_recall_sizes": [20],
                      "expected_new_hdf_groups": 9,
                      "max_network_run_segments": 18,
                      "run_durations_seconds": [2.0, 0.1] * 9,
                      "records": records, "new_groups": groups,
                      "performance_authorized": False,
                      "whole_figure8_s7_science_gate_passed": False,
                      "candidate_hdf_sha256": candidate_hash}
            directory = self.campaign / f"seed{seed}-v1"
            source_path = directory / "report-v1.json"
            save(source_path, source)
            save(directory / "hdf-gate-v1.json", {
                "passed": True, "host": module.HOST, "seed": seed,
                "validator_sha256": module.RAW_GATE_SHA256,
                "source_report_sha256": module.sha256(source_path),
                "condition_plan_sha256": module.sha256(self.plan),
                "official_hdf_sha256": module.OFFICIAL_HDF_SHA256,
                "candidate_hdf_sha256": candidate_hash,
                "published_groups_byte_identical": 212,
                "published_datasets_byte_identical": 2853,
                "new_recall_datasets_per_group": 12,
                "new_before_imprint_groups": groups,
                "performance_authorized": False,
                "whole_figure8_s7_science_gate_passed": False})

    @staticmethod
    def hash(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    def row(self, seed: int, order: int, stimulus: int) -> dict:
        return {"seed": seed, "order": order, "stimulus": stimulus,
                "source_final_checkpoint": f"final_{seed}_{order}",
                "source_final_checkpoint_sha256": self.hash(f"final_{seed}_{order}"),
                "converted_baseline_checkpoint": f"converted_{seed}_{order}",
                "converted_baseline_checkpoint_sha256": self.hash(
                    f"converted_{seed}_{order}")}

    def run_gate(self) -> None:
        argv = [str(SCRIPT), "--condition-plan", str(self.plan),
                "--pilot-report", str(self.pilot), "--pilot-gate", str(self.pilot_gate),
                "--campaign-dir", str(self.campaign), "--output", str(self.output)]
        with (patch.object(sys, "argv", argv),
              patch.object(module, "PLAN_SHA256", module.sha256(self.plan)),
              patch.object(module, "PILOT_REPORT_SHA256", module.sha256(self.pilot)),
              patch.object(module, "PILOT_GATE_SHA256", module.sha256(self.pilot_gate))):
            module.main()

    def test_accepts_complete_20_seed_180_condition_fixture(self) -> None:
        self.run_gate()
        result = json.loads(self.output.read_text())
        self.assertTrue(result["passed"])
        self.assertEqual(result["new_unique_before_imprint_hdf_groups"], 180)
        self.assertFalse(result["whole_figure8_s7_science_gate_passed"])
        self.assertFalse(result["performance_authorized"])

    def test_rejects_weak_independent_hdf_proof(self) -> None:
        path = self.campaign / "seed0-v1" / "hdf-gate-v1.json"
        proof = json.loads(path.read_text())
        proof["published_datasets_byte_identical"] = 2852
        save(path, proof)
        with self.assertRaisesRegex(ValueError, "independent raw-HDF gate differs"):
            self.run_gate()
        self.assertFalse(self.output.exists())

    def test_rejects_missing_condition_even_with_rehashed_proof(self) -> None:
        source_path = self.campaign / "seed0-v1" / "report-v1.json"
        source = json.loads(source_path.read_text())
        source["records"][-1]["stimulus"] = 1
        save(source_path, source)
        proof_path = self.campaign / "seed0-v1" / "hdf-gate-v1.json"
        proof = json.loads(proof_path.read_text())
        proof["source_report_sha256"] = module.sha256(source_path)
        save(proof_path, proof)
        with self.assertRaisesRegex(ValueError, "unexpected or duplicate source condition"):
            self.run_gate()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
