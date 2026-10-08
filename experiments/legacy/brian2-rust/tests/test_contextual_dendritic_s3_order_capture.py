"""Low-load, no-Brian2-simulation checks for the Fig. S3 order sidecar hook."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from contextual_dendritic_s3_official_job import (  # noqa: E402
    capture_saved_neuron_order,
    final_saved_capture,
)


class FakeNetwork:
    def __init__(self) -> None:
        self.calls = []
        self.save_calls = 0

    def sort_neurons_by_firing_rate(self, *positional, **keywords):
        self.calls.append((positional, keywords))
        if keywords.get("shuffle_rest") is False:
            self.save_calls += 1
        selected_count = 44 if self.save_calls == 1 else 45
        return [list(range(400))], list(range(selected_count)), [1.0] * 400


def test_capture_observes_original_save_call_without_rerunning_selection():
    network = FakeNetwork()
    captured = capture_saved_neuron_order(network)

    network.sort_neurons_by_firing_rate(shuffle_rest=True, reverse_order=True)
    assert captured == []

    first_result = network.sort_neurons_by_firing_rate(
        shuffle_rest=False, reverse_order=True
    )
    second_result = network.sort_neurons_by_firing_rate(
        shuffle_rest=False, reverse_order=True
    )
    assert len(network.calls) == 3
    assert first_result[1] == list(range(44))
    assert second_result[1] == list(range(45))
    assert captured == [
        {
            "selected_ids": list(range(44)),
            "saved_neuron_ids": list(range(69)),
            "expected_saved_dimension": 69,
        },
        {
            "selected_ids": list(range(45)),
            "saved_neuron_ids": list(range(70)),
            "expected_saved_dimension": 70,
        },
    ]
    assert final_saved_capture(captured, [70, 70]) is captured[-1]


def test_final_saved_capture_rejects_wrong_count_and_shape():
    capture = {"selected_ids": list(range(45)), "saved_neuron_ids": list(range(70))}
    for captures, shape in (([capture], [70, 70]), ([capture, capture], [69, 69])):
        try:
            final_saved_capture(captures, shape)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid saved capture was accepted")
