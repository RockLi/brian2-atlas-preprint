"""Correctness-only checks for the Figure S2 recall reader and frozen gate."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from contextual_dendritic_s2_recall_compare import compare  # noqa: E402
from contextual_dendritic_s2_recall_extract import activity  # noqa: E402


REFERENCE = Path("/private/tmp/contextual-s2-independent-recall-reference-v2.json")


class S2RecallGateTests(unittest.TestCase):
    def test_activity_uses_strict_two_second_window_and_first_background(self) -> None:
        times = np.asarray([100.0, 101.0, 1100.0, 2100.0, 1000.0, 2000.0])
        indices = np.asarray([2, 2, 2, 2, 2, 2])
        result = activity(indices, times, np.asarray([2]), 4, 100.0, 2100.0)
        self.assertEqual(result["window_spikes"], 4)
        self.assertEqual(result["assembly_mean_hz"], 2.0)
        self.assertEqual(result["background_mean_hz"], 0.0)
        self.assertEqual(result["assembly_active"], 0)

    @unittest.skipUnless(REFERENCE.is_file(), "reference extraction is not on this host")
    def test_self_comparison_and_effect_failure(self) -> None:
        self.assertTrue(compare(REFERENCE, [REFERENCE])["passed"])
        payload = json.loads(REFERENCE.read_text())
        altered = copy.deepcopy(payload)
        for row in altered["records"].values():
            row["assembly_mean_hz"] = 0.0
            row["assembly_active"] = 0
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "altered.json"
            path.write_text(json.dumps(altered))
            result = compare(REFERENCE, [path])
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["cue-size:assembly_mean_hz:endpoint_gain"])
        self.assertFalse(result["checks"]["cue-rate:assembly_active:endpoint_gain"])


if __name__ == "__main__":
    unittest.main()
