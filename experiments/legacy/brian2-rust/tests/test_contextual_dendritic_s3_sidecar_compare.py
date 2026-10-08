"""No-simulation checks for exact Fig. S3 saved-neuron-order validation."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from contextual_dendritic_s3_recurrent_sidecar_compare import (  # noqa: E402
    validate_saved_order,
)


def test_in_memory_order_contract_accepts_valid_and_rejects_duplicate_ids():
    with h5py.File("memory-only-s3-sidecar", "w", driver="core", backing_store=False) as file:
        group = file.create_group("test-group")
        group.attrs["n_somas"] = 400
        group.attrs["seed"] = 19
        group.create_dataset("weights", data=np.zeros((70, 70)))
        sidecar = {
            "schema": "contextual-dendritic-s3-recurrent-saved-neuron-order-v2",
            "hdf5_group": "test-group",
            "seed": 19,
            "saved_weight_shape": [70, 70],
            "selected_ids_at_save": list(range(45)),
            "saved_neuron_ids_in_weight_matrix_order": list(range(70)),
            "paper_model_or_parameters_modified": False,
            "capture_call_count": 2,
            "selected_capture_index": 1,
            "capture_selected_counts": [44, 45],
        }
        ids, checks = validate_saved_order(sidecar, group)
        assert ids.tolist() == list(range(70))
        assert all(checks.values()), checks

        sidecar["saved_neuron_ids_in_weight_matrix_order"][-1] = 0
        _, checks = validate_saved_order(sidecar, group)
        assert checks["ids_unique"] is False
        assert checks["ids_length_exact"] is True
