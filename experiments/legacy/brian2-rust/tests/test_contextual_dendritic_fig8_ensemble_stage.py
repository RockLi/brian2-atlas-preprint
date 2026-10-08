"""No-simulation safety tests for the Fig. 8 two-mode staging tool."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from contextual_dendritic_fig8_ensemble_stage import (  # noqa: E402
    SOURCE_REVISION,
    executable_path,
    independent_copy,
    stage,
    validate_finished_imprint,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Fig8EnsembleStageSafetyTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, dict]:
        repo = root / "original"
        h5 = repo / "results/sim_files/data_Fig_8.h5"
        h5.parent.mkdir(parents=True)
        h5.write_bytes(b"five-imprint-synthetic-cache")
        checkpoints = []
        for index in range(5):
            name = f"stored_imprint_{index}_0"
            data = f"checkpoint-{index}".encode()
            path = repo / "stored_networks/Fig_8" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            checkpoints.append({"name": name, "sha256": digest(data)})
        marker = {
            "schema": "contextual-dendritic-isolated-reproduction-v1",
            "reproduction_id": "synthetic-fig8-s0005-case0",
            "source_revision": SOURCE_REVISION,
        }
        (repo / ".contextual-dendritic-reproduction.json").write_text(json.dumps(marker))
        report = {
            "schema": "contextual-dendritic-fig8-imprint-only-job-v1",
            "completed": True,
            "imprint_only": True,
            "simulation_executed": True,
            "reported_timings": False,
            "source_revision": SOURCE_REVISION,
            "seed": 5,
            "case_id": 0,
            "paper_repo": str(repo),
            "h5": {"sha256": digest(h5.read_bytes()), "groups": 5,
                   "imprint_groups": 5},
            "isolation": {"reproduction_id": marker["reproduction_id"]},
            "checkpoints": checkpoints,
        }
        return repo, report

    def test_finished_imprint_identity_and_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, report = self.make_fixture(Path(directory))
            state = validate_finished_imprint(report, repo, 5)
            self.assertEqual(state["checkpoint_count"], 5)
            with self.assertRaisesRegex(ValueError, "not one completed"):
                validate_finished_imprint(report, repo, 6)
            (repo / "results/sim_files/data_Fig_8.h5").write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_finished_imprint(report, repo, 5)

    def test_two_copies_do_not_hardlink_scientific_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, report = self.make_fixture(root)
            rate = root / "rate"
            active = root / "active"
            independent_copy(repo, rate, report["checkpoints"])
            independent_copy(repo, active, report["checkpoints"])
            relative = Path("results/sim_files/data_Fig_8.h5")
            self.assertNotEqual((rate / relative).stat().st_ino, (active / relative).stat().st_ino)
            (rate / relative).write_bytes(b"rate-recall-mutated")
            self.assertEqual((repo / relative).read_bytes(), b"five-imprint-synthetic-cache")
            self.assertEqual((active / relative).read_bytes(), b"five-imprint-synthetic-cache")

    def test_macos_staging_is_forbidden_before_filesystem_access(self) -> None:
        with patch("contextual_dendritic_fig8_ensemble_stage.platform.system", return_value="Darwin"):
            with self.assertRaisesRegex(RuntimeError, "remote-only"):
                stage(argparse.Namespace())

    def test_virtualenv_python_symlink_is_not_resolved_away(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "system-python"
            executable.write_bytes(b"synthetic")
            venv_link = root / "venv-python"
            venv_link.symlink_to(executable)
            self.assertEqual(executable_path(venv_link), venv_link.absolute())
            self.assertNotEqual(executable_path(venv_link), executable.resolve())


if __name__ == "__main__":
    unittest.main()
