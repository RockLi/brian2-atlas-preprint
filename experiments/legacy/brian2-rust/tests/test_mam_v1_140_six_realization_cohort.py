"""Axis and source-preserving checks for the six-realization V1 archive."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "six_v1_cohort_fixture", TOOLS / "mam_v1_140_six_realization_cohort.py")
cohort = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cohort)


class SixV1CohortTest(unittest.TestCase):
    def test_frequency_axis_must_be_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "views.npz"
            arrays = {"sampled_population_counts": np.zeros((8, 100000), dtype=np.uint8),
                      "frequency_hz": np.fft.rfftfreq(1024, d=.001)}
            arrays.update({"power__" + key: np.zeros(513) for key in cohort.previous.VIEWS})
            arrays.update({"rate__" + key: np.zeros(100000) for key in cohort.previous.RATE_VIEWS})
            np.savez_compressed(path, **arrays)
            cohort.load_views(path)
            arrays["frequency_hz"][1] += .01
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "frequency axis"):
                cohort.load_views(path)

    def test_rehashed_publication_still_must_match_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            arrays = {"rate": np.arange(12).reshape(6, 2)}
            report = {"labels": list(cohort.LABELS), "scientific_acceptance": False}
            (output / "report.json").write_text(json.dumps(report))

            def publish(values):
                np.savez_compressed(output / "aligned-views.npz", **values)
                catalog = {name: {"bytes": (output / name).stat().st_size,
                                  "sha256": cohort.sha(output / name)}
                           for name in ("aligned-views.npz", "report.json")}
                (output / "catalog.json").write_text(json.dumps(catalog))

            publish(arrays)
            with patch.object(cohort, "OUTPUT", output):
                cohort.verify_output(arrays, report)
                publish({"rate": arrays["rate"][::-1]})
                with self.assertRaisesRegex(ValueError, "arrays differ from sources"):
                    cohort.verify_output(arrays, report)


if __name__ == "__main__":
    unittest.main()
