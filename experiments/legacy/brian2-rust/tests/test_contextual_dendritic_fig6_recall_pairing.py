"""No-simulation tests for Fig. 6/S6 recall pairing across different images."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import contextual_dendritic_fig6_semantic_compare as fig6  # noqa: E402

recall_conditions = fig6.recall_conditions


def fixture_file(name: str, figure: str, input_prefix: str) -> h5py.File:
    handle = h5py.File(name, "w", driver="core", backing_store=False)
    contexts = (0, 0, 1, 1) if figure == "Fig_6" else (0, 1, 1, 0)
    for index in range(36):
        group = handle.create_group(f"{35 - index:08x}")
        group.attrs["run_recall_after_imprint"] = True
        if index < 16:
            target = index % 4
            recall_id = [0, 0, 0, -1, contexts[target]]
            key = f"{input_prefix}-visual-{index}"
        elif index < 20:
            target = index - 16
            recall_id = [0, -1, 0, target, contexts[target]]
            key = "empty-visual-input"
        else:
            target = (index - 20) % 4
            recall_id = [0, 0, 0, target, contexts[target]]
            key = f"{input_prefix}-visual-{index - 20}"
        group.attrs["recall_id"] = np.asarray(recall_id)
        group.attrs["all_assembly_inputs_key_recall"] = key
    return handle


class RecallPairingTests(unittest.TestCase):
    def test_assembly_selection_and_metric_both_use_last_two_seconds(self) -> None:
        with h5py.File("memory-imprint-window", "w", driver="core", backing_store=False) as handle:
            initial = handle.create_group("initial")
            initial.attrs["all_imprint_ids"] = np.zeros((40, 5), dtype=int)
            additional = handle.create_group("additional")
            additional.attrs["n_somas"] = 4
            additional.attrs["n_dend_each"] = 1
            additional.attrs["runtime_baseline"] = 0.8
            additional.attrs["runtime_imprint"] = 6.0
            schedule = np.zeros((20, 5), dtype=int)
            schedule[:, 1] = np.arange(20)
            additional.attrs["all_imprint_ids"] = schedule
            imprint_ends = [272800.0 + 800.0 + 6800.0 * index + 6000.0 for index in (4, 9, 14, 19)]
            for area in "ABC":
                additional.create_dataset(f"{area}_weights", data=np.ones((4, 4)))
                additional.create_dataset(
                    f"{area}_spikes_somas_t",
                    data=np.sort(np.r_[np.asarray(imprint_ends) - 5000.0, np.asarray(imprint_ends) - 1000.0]),
                )
                additional.create_dataset(f"{area}_spikes_somas_i", data=np.tile([0, 1], 4))

            selection_rates = []

            def capture_selection(rates: np.ndarray, _weights: np.ndarray) -> np.ndarray:
                selection_rates.append(rates.copy())
                return np.asarray([1])

            with mock.patch.object(fig6, "choose_assembly", side_effect=capture_selection):
                assemblies, metrics = fig6.reconstruct_assemblies(initial, additional)
            self.assertEqual(len(selection_rates), 12)
            self.assertTrue(all(rates[0] == 0.0 for rates in selection_rates))
            self.assertTrue(all(rates[1] == 0.5 for rates in selection_rates))
            self.assertTrue(all(values == [0.5, 0.0, 0.0, 0.0] for area in "ABC" for values in metrics[area]))
            self.assertTrue(all(ids.tolist() == [1] for area in "ABC" for ids in assemblies[area]))

    def test_schedule_slots_ignore_sample_hash_but_validate_reuse(self) -> None:
        for figure in ("Fig_6", "Fig_S6"):
            with self.subTest(figure=figure), fixture_file(
                f"memory-{figure}-left", figure, "left"
            ) as left, fixture_file(
                f"memory-{figure}-right", figure, "right"
            ) as right:
                left_slots = recall_conditions(left, figure)
                right_slots = recall_conditions(right, figure)
                self.assertEqual(
                    sorted(left_slots), sorted(right_slots),
                )
                self.assertEqual(len(left_slots), 12)
                self.assertEqual(sum(map(len, left_slots.values())), 36)
                self.assertNotEqual(
                    left_slots["visual:target=0"][0].attrs["all_assembly_inputs_key_recall"],
                    right_slots["visual:target=0"][0].attrs["all_assembly_inputs_key_recall"],
                )
                self.assertEqual(len(left_slots["visual:target=0"]), 4)
                self.assertEqual(len(left_slots["auditory:target=0"]), 1)
                self.assertEqual(len(left_slots["combined:target=0"]), 4)

    def test_schedule_slot_rejects_wrong_recall_id(self) -> None:
        with fixture_file("memory-fig6-wrong-id", "Fig_6", "same") as handle:
            handle[f"{35 - 20:08x}"].attrs["recall_id"] = np.asarray([0, 0, 0, 3, 0])
            with self.assertRaisesRegex(ValueError, "recall schedule mismatch"):
                recall_conditions(handle, "Fig_6")

    def test_schedule_slot_rejects_broken_visual_combined_cue_reuse(self) -> None:
        with fixture_file("memory-fig6-wrong-cue", "Fig_6", "same") as handle:
            handle[f"{35 - 20:08x}"].attrs["all_assembly_inputs_key_recall"] = "different"
            with self.assertRaisesRegex(ValueError, "visual/combined input reuse failed"):
                recall_conditions(handle, "Fig_6")


if __name__ == "__main__":
    unittest.main()
