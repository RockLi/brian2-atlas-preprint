"""No-simulation regression for Fig. 5 two-sided mean-difference gates."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from contextual_dendritic_fig5_semantic_compare import bounded_mean_delta  # noqa: E402


class SymmetricDeltaTest(unittest.TestCase):
    def test_rejects_equal_magnitude_large_drift_in_both_directions(self) -> None:
        self.assertFalse(bounded_mean_delta(5.1, 5.0))
        self.assertFalse(bounded_mean_delta(-5.1, 5.0))

    def test_accepts_both_boundary_values(self) -> None:
        self.assertTrue(bounded_mean_delta(5.0, 5.0))
        self.assertTrue(bounded_mean_delta(-5.0, 5.0))

    def test_existing_interim_values_remain_within_two_sided_limits(self) -> None:
        report = Path(__file__).resolve().parents[3] / (
            "brian2-experiments-artifacts/contextual-dendritic-gating-20260921/"
            "full-paper-audit-v1/fig5-interim-v1/"
            "contextual-fig5-partial-semantic-v5.json"
        )
        if not report.is_file():
            self.skipTest("archived Fig. 5 interim report not mounted")
        value = json.loads(report.read_text())
        thresholds = value["thresholds"]
        self.assertTrue(bounded_mean_delta(
            value["assembly_size_comparison"]["mean_delta"],
            thresholds["assembly_size_mean_delta_maximum_neurons"],
        ))
        for name, threshold in (
            ("background_rate", "background_rate_mean_delta_maximum_hz"),
            ("background_active", "background_active_mean_delta_maximum"),
        ):
            self.assertTrue(bounded_mean_delta(
                value["imprint_comparisons"][name]["mean_delta"],
                thresholds[threshold],
            ))


if __name__ == "__main__":
    unittest.main()
