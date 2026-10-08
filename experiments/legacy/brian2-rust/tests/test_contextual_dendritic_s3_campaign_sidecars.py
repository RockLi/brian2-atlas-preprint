"""No-simulation checks for isolated Fig. S3 order-sidecar campaign wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from contextual_dendritic_s3_campaign import (  # noqa: E402
    command_for,
    completed_cell,
    load_incompatible_cell_ids,
    sha256_file,
    unavailable_worker_cpus,
)


def test_cpu_affinity_preflight_rejects_unavailable_worker_slots():
    pin_succeeds = lambda cpu: cpu in {160, 162}
    assert unavailable_worker_cpus([160, 162], 2, pin_succeeds) == []
    assert unavailable_worker_cpus([192, 193], 2, pin_succeeds) == [192, 193]
    assert unavailable_worker_cpus([160, 192], 1, pin_succeeds) == []


def test_sidecar_command_and_resume_integrity_without_simulation():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        report = root / "report.json"
        sidecar = root / "neuron-order-sidecar.json"
        sidecar.write_text('{"seed":19}\n')
        report.write_text(json.dumps({
            "completed": True,
            "neuron_order_sidecar": {
                "path": str(sidecar.resolve()),
                "sha256": sha256_file(sidecar),
            },
        }) + "\n")
        assert completed_cell(report, True)

        command = command_for(
            {"stage": "recurrent-inhibition", "seed": 19,
             "recurrent_inhibition_enabled": False},
            174, Path("/venv/python"), Path("/driver.py"),
            Path("/paper"), report, "locked-revision", "pilot-cell", sidecar,
        )
        assert command[:3] == ["taskset", "-c", "174"]
        assert "--no-recurrent-inhibition-enabled" in command
        assert command[-2:] == ["--neuron-order-sidecar", str(sidecar)]

        sidecar.write_text('{"seed":20}\n')
        assert not completed_cell(report, True)
        assert completed_cell(report, False)


def test_loader_audit_inclusion_is_exact_and_checksum_pinned():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "strict-audit.json"
        path.write_text(json.dumps({
            "schema": "contextual-dendritic-s3-recurrent-loader-audit-v1",
            "loader_incompatible_ids": ["recurrent-s019-off", "recurrent-s000-on"],
            "loader_incompatible_cells": 2,
            "full_paper_ensemble_gate_executed": False,
        }) + "\n")
        ids, digest = load_incompatible_cell_ids(path)
        assert ids == ["recurrent-s000-on", "recurrent-s019-off"]
        assert digest == sha256_file(path)
