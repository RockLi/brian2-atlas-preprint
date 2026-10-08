"""Pure-JSON contract tests; never imports or executes a Brian2 model."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = (Path(__file__).resolve().parents[1] / "tools"
          / "contextual_fig8_combined_cue_complete_ensemble_gate.py")
spec = importlib.util.spec_from_file_location("combined_cue_ensemble_gate", SCRIPT)
assert spec is not None and spec.loader is not None
gate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate_module)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


class CombinedCueEnsembleGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.campaign = self.root / "campaign"
        self.cache = self.root / "map.json"
        self.pilot = self.root / "pilot.json"
        self.pilot_gate = self.root / "pilot-gate.json"
        self.output = self.root / "ensemble-gate.json"
        self.seeds = list(range(19)) + [6427]
        rows = [{"seed": seed, "order": order,
                 "final_checkpoint": f"checkpoint_{seed}_{order}",
                 "final_checkpoint_sha256": hashlib.sha256(
                     f"checkpoint_{seed}_{order}".encode()).hexdigest()}
                for seed in self.seeds for order in range(3)]
        write_json(self.cache, {"seed_count": 20,
                                "final_checkpoint_hashes_verified": 60,
                                "rows": rows})
        pilot = {"status": "completed", "new_hdf_group": "g6427_0",
                 "response_mean_hz_by_area": [2.0, 3.0],
                 "response_active_neurons_by_area": [1.0, 1.0],
                 "background_mean_hz_by_area": [0.1, 0.2],
                 "selected_counts": [4, 5]}
        write_json(self.pilot, pilot)
        write_json(self.pilot_gate, {"passed": True,
                                     "source_report_sha256": gate_module.sha256(self.pilot),
                                     "published_groups_byte_identical": 212,
                                     "published_datasets_byte_identical": 2853})
        for seed in self.seeds:
            groups = [f"g{seed}_{order}" for order in range(3)]
            records = [{"order": order, "checkpoint": f"checkpoint_{seed}_{order}",
                        "new_hdf_group": groups[order], "selected_counts": [4, 5],
                        "response_mean_hz_by_area": [2.0, 3.0],
                        "response_active_neurons_by_area": [1.0, 1.0],
                        "background_mean_hz_by_area": [0.1, 0.2]}
                       for order in range(3)]
            source = {"status": "completed", "host": gate_module.HOST,
                      "seed": seed, "source_sha256": gate_module.SOURCE_SHA256,
                      "official_hdf_sha256": gate_module.OFFICIAL_HDF_SHA256,
                      "cache_map_sha256": gate_module.sha256(self.cache),
                      "driver_sha256": gate_module.DRIVER_SHA256,
                      "compiled_queue_sha256": gate_module.QUEUE_SHA256,
                      "brian2_version": "2.9.0", "expected_new_hdf_groups": 3,
                      "max_network_run_segments": 6,
                      "final_checkpoint_sha256": {
                          str(order): hashlib.sha256(
                              f"checkpoint_{seed}_{order}".encode()).hexdigest()
                          for order in range(3)},
                      "orders": [0, 1, 2], "stimulus": 2, "all_recall_sizes": [20],
                      "run_recall_after_imprint": True, "change_firing_rate": True,
                      "run_durations_seconds": [2.0, 0.1] * 3,
                      "performance_authorized": False,
                      "whole_figure8_s7_science_gate_passed": False,
                      "records": records,
                      "new_groups": groups,
                      "candidate_hdf_sha256": hashlib.sha256(f"hdf-{seed}".encode()).hexdigest()}
            directory = self.campaign / f"seed{seed}-v1"
            source_path = directory / "report-v1.json"
            write_json(source_path, source)
            write_json(directory / "hdf-gate-v1.json", {
                "passed": True, "host": gate_module.HOST, "seed": seed,
                "validator_sha256": gate_module.RAW_GATE_SHA256,
                "source_report_sha256": gate_module.sha256(source_path),
                "official_hdf_sha256": gate_module.OFFICIAL_HDF_SHA256,
                "candidate_hdf_sha256": source["candidate_hdf_sha256"],
                "published_groups_byte_identical": 212,
                "published_datasets_byte_identical": 2853,
                "new_recall_datasets_per_group": 12,
                "performance_authorized": False,
                "whole_figure8_s7_science_gate_passed": False,
                "new_combined_cue_groups": groups})

    def run_gate(self) -> None:
        args = [str(SCRIPT), "--cache-map", str(self.cache), "--pilot-report", str(self.pilot),
                "--pilot-gate", str(self.pilot_gate), "--campaign-dir", str(self.campaign),
                "--output", str(self.output)]
        with (patch.object(sys, "argv", args),
              patch.object(gate_module, "CACHE_MAP_SHA256", gate_module.sha256(self.cache)),
              patch.object(gate_module, "PILOT_REPORT_SHA256", gate_module.sha256(self.pilot)),
              patch.object(gate_module, "PILOT_GATE_SHA256", gate_module.sha256(self.pilot_gate))):
            gate_module.main()

    def test_complete_20_seed_60_condition_contract(self) -> None:
        self.run_gate()
        result = json.loads(self.output.read_text())
        self.assertTrue(result["passed"])
        self.assertEqual(result["order_stimulus_conditions"], 60)
        self.assertFalse(result["whole_figure8_s7_science_gate_passed"])
        self.assertFalse(result["performance_authorized"])

    def test_rejects_changed_independent_raw_hdf_gate(self) -> None:
        path = self.campaign / "seed0-v1" / "hdf-gate-v1.json"
        value = json.loads(path.read_text())
        value["published_datasets_byte_identical"] = 2852
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "independent raw-HDF gate differs"):
            self.run_gate()
        self.assertFalse(self.output.exists())

    def test_rejects_changed_exact_pilot_replication(self) -> None:
        source_path = self.campaign / "seed6427-v1" / "report-v1.json"
        value = json.loads(source_path.read_text())
        value["records"][0]["response_mean_hz_by_area"][0] += 0.001
        write_json(source_path, value)
        gate_path = self.campaign / "seed6427-v1" / "hdf-gate-v1.json"
        proof = json.loads(gate_path.read_text())
        proof["source_report_sha256"] = gate_module.sha256(source_path)
        write_json(gate_path, proof)
        with self.assertRaisesRegex(ValueError, "exact independent pilot control failed"):
            self.run_gate()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
