"""Analytical and lifecycle checks for the research reference."""
import numpy as np
import pytest

from mechanism import evaluate, experiment, patterns, update


def test_single_active_synapse_has_known_prediction_step():
    weights = np.full((2, 3), 0.75)
    update(weights, np.array([0., 1., 0.]), 1., eta=0.2)
    np.testing.assert_allclose(weights[:, 1], [0.85, 0.65], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(weights[:, [0, 2]], 0.75)


@pytest.mark.parametrize("options", [{"gate": 0}, {"eta": 0}])
def test_training_disable_preserves_every_weight(options):
    weights = np.full((2, 3), 0.75)
    update(weights, np.array([0.2, 0.3, 0.5]), 1, **options)
    np.testing.assert_array_equal(weights, 0.75)


def test_eligibility_delay_has_analytic_decay():
    immediate, delayed = np.full((2, 3), 0.75), np.full((2, 3), 0.75)
    activity = np.array([0., 1., 0.])
    update(immediate, activity, 1)
    update(delayed, activity, 1, delay_seconds=1, tau_seconds=0.5)
    np.testing.assert_allclose(delayed - 0.75, (immediate - 0.75) * np.exp(-2), atol=1e-16)


def test_held_out_response_noise_and_read_only_order_independent_probe():
    train, labels = patterns(11, 80)
    heldout, heldout_labels = patterns(11, 80, noise_seed=1011)
    assert not np.array_equal(train, heldout)
    weights = np.full((2, 128), 0.75)
    for activity, label in zip(train, labels):
        update(weights, activity, float(label == 0))
    before = weights.copy()
    first = evaluate(weights, heldout, heldout_labels)
    evaluate(weights, train, labels)
    second = evaluate(weights, heldout, heldout_labels)
    assert first == second
    np.testing.assert_array_equal(weights, before)


def test_acquisition_reversal_and_null_controls():
    report, states = experiment(11)
    conditions = report["conditions"]
    assert conditions["paired"]["A_minus_B"] > 0.7
    assert conditions["reversal"]["A_minus_B"] < -0.7
    assert conditions["frozen"]["A_minus_B"] == 0
    assert conditions["gate_off"]["A_minus_B"] == 0
    for state in states.values():
        assert np.isfinite(state).all()
        assert np.all((state >= 0) & (state <= 1.5))


def test_silent_activity_and_invalid_timing():
    weights = np.full((2, 3), 0.75)
    update(weights, np.zeros(3), 1)
    np.testing.assert_array_equal(weights, 0.75)
    with pytest.raises(ValueError):
        update(weights, np.ones(3), 1, tau_seconds=0)
