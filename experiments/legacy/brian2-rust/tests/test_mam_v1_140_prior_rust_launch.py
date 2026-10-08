"""Exact-once admission checks for prior-Rust V1 extraction control."""

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "prior_rust_launcher_fixture", TOOLS / "mam_v1_140_prior_rust_launch.py")
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


class PriorRustLauncherTest(unittest.TestCase):
    def test_completion_verification_rehashes_every_member(self):
        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "rust1729"
            output.mkdir()
            members = {"intent.json": b"{}\n",
                       "selected-cell-counts.npz": b"counts",
                       "v1-four-views.npz": b"views",
                       "analysis.json": (json.dumps({
                           "schema": "b2-mam-prior-rust-v1-140-spectrum-v1",
                           "seed": 1729, "source_identity_sha256": "frozen",
                           "current_raw_rehashed": True,
                           "scientific_acceptance": False,
                           "selected_global_ids": [list(range(140))],
                       }) + "\n").encode()}
            for name, content in members.items():
                (output / name).write_bytes(content)
            catalog = {name: {"bytes": len(content),
                              "sha256": sha(output / name)}
                       for name, content in members.items()}
            (output / "catalog.json").write_text(json.dumps(catalog) + "\n")
            for name in ("launch", "controller", "guard"):
                (root / f"rust1729-{name}-v1.json").write_text("{}\n")
            bound = {name: sha(root / f"rust1729-{name}-v1.json")
                     for name in ("launch", "controller", "guard")}
            bound["catalog"] = sha(output / "catalog.json")
            receipt = {"schema": "b2-mam-prior-rust-v1-140-completion-v1",
                       "seed": 1729, "complete": True, "attempts": 1,
                       "automatic_retry": False, "scientific_acceptance": False,
                       "input_sha256": bound,
                       "output_bytes": sum(len(content) for content in members.values())}
            (root / "rust1729-completion-v1.json").write_text(json.dumps(receipt) + "\n")
            with (patch.object(LAUNCHER, "ROOT", root),
                  patch.object(LAUNCHER, "EXPECTED", {"identity-v3.json": "frozen"})):
                self.assertTrue(LAUNCHER.verify_completion(1729)["valid"])
                (output / "v1-four-views.npz").write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "member differs"):
                    LAUNCHER.verify_completion(1729)

    def test_command_is_seed_specific_and_bounded(self):
        one, two = LAUNCHER.command(1729), LAUNCHER.command(1750)
        self.assertNotEqual(one, two)
        for seed, cmd in ((1729, one), (1750, two)):
            self.assertIn("--unit=b2mpi-analysis-rust%d-v1-140" % seed, cmd)
            self.assertIn("--property=MemoryMax=24576M", cmd)
            self.assertIn("--property=MemorySwapMax=0", cmd)
            self.assertIn("--property=CPUQuota=200%", cmd)
            self.assertIn("--property=RuntimeMaxSec=10800", cmd)
            self.assertIn("--property=TasksMax=64", cmd)
            self.assertIn("--min-free-gib", cmd)
            self.assertEqual(cmd[cmd.index("--seed") + 1], str(seed))

    def test_existing_attempt_rejected_before_mutation(self):
        state = {"phase": "not_started",
                 "destinations": {"launch": True, "controller": False,
                                  "guard": False, "completion": False},
                 "output_exists": False,
                 "unit_state": {"LoadState": "not-found"}}
        with (patch.object(LAUNCHER.os, "geteuid", return_value=0),
              patch.object(LAUNCHER, "probe", return_value=state),
              patch.object(LAUNCHER, "exclusive_json") as publish):
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                LAUNCHER.run(1729)
            publish.assert_not_called()

    def test_finalization_rejected_after_prior_controller(self):
        state = {"phase": "finalization_rejected",
                 "destinations": {"launch": True, "controller": True,
                                  "guard": True, "completion": False}}
        with (patch.object(LAUNCHER, "probe", return_value=state),
              patch.object(LAUNCHER, "exclusive_json") as publish):
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                LAUNCHER.finalize(1729)
            publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
