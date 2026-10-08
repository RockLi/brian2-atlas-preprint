"""Lightweight, no-simulation Figure S2 large-recall science-gate tests."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from contextual_dendritic_s2_large_recall_compare import compare  # noqa: E402
from contextual_dendritic_s2_large_recall_extract import SEEDS  # noqa: E402
from contextual_dendritic_s2_large_recall_published_compare import compare as compare_published  # noqa: E402


REFERENCES = [Path(f"/private/tmp/contextual-s2-large-recall-reference-seed{s}-v1.json")
              for s in SEEDS]
EXPORT_DIR = Path("/atlas-storage/0002/brian2-paper-reproduction/contextual-dendritic-gating/reference/repository/results/Fig_S2")


class S2LargeRecallGateTests(unittest.TestCase):
    @unittest.skipUnless(all(path.is_file() for path in REFERENCES),
                         "official compact S2 large-recall reference not on this host")
    def test_reference_derived_candidate_passes_and_silence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            identical: list[Path] = []
            silent: list[Path] = []
            for seed, reference_path in zip(SEEDS, REFERENCES):
                reference = json.loads(reference_path.read_text())
                clone = copy.deepcopy(reference)
                clone["side"] = "candidate"
                same_path = root / f"same-{seed}.json"
                same_path.write_text(json.dumps(clone))
                identical.append(same_path)
                for row in clone["records"].values():
                    row["assembly_mean_hz"] = 0.0
                    row["assembly_active"] = 0
                silent_path = root / f"silent-{seed}.json"
                silent_path.write_text(json.dumps(clone))
                silent.append(silent_path)
            self.assertTrue(compare(REFERENCES, identical)["passed"])
            failed = compare(REFERENCES, silent)
            self.assertFalse(failed["passed"])
            self.assertFalse(failed["checks"]["recalled_assembly_firing_effect"])
            self.assertFalse(failed["checks"]["recalled_assembly_active_effect"])
            if EXPORT_DIR.is_dir():
                published_selftest = compare_published(EXPORT_DIR, identical)
                self.assertTrue(published_selftest["passed"])
                self.assertEqual(len(published_selftest["checks"]), 10)
                published_silent = compare_published(EXPORT_DIR, silent)
                self.assertFalse(published_silent["passed"])


if __name__ == "__main__":
    unittest.main()
