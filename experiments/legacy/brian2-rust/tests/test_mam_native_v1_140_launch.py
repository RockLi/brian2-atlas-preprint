import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import mam_native_v1_140_launch as launch


def test_completion_verification_detects_member_corruption(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    files = {
        "launch": tmp_path / "launch.json",
        "guard": tmp_path / "guard.json",
        "controller": tmp_path / "controller.json",
        "catalog": output / "catalog.json",
        "analysis": output / "analysis.json",
    }
    for name in ("launch", "guard", "controller"):
        files[name].write_text('{"bound":true}\n')
    (output / "intent.json").write_text('{"attempt":1}\n')
    (output / "selected-cell-counts.npz").write_bytes(b"selected")
    (output / "v1-four-views.npz").write_bytes(b"views")
    files["analysis"].write_text('{"source_bound":true}\n')
    catalog = {
        path.name: {"bytes": path.stat().st_size, "sha256": launch.sha(path)}
        for path in output.iterdir()
    }
    files["catalog"].write_text(json.dumps(catalog))
    total = sum(row["bytes"] for row in catalog.values())
    completion = tmp_path / "completion.json"
    completion.write_text(json.dumps({
        "schema": "b2-mam-native-v1-140-completion-v1", "seed": 1730,
        "complete": True, "attempts": 1, "automatic_retry": False,
        "scientific_acceptance": False, "output_bytes": total,
        "input_sha256": {name: launch.sha(path) for name, path in files.items()},
    }))
    mapping = {"output": output, "completion": completion,
               "rejection": tmp_path / "rejection.json", **files}
    monkeypatch.setattr(launch, "paths", lambda seed: mapping)
    assert launch.verify_completion(1730)
    (output / "selected-cell-counts.npz").write_bytes(b"altered")
    assert not launch.verify_completion(1730)


def test_existing_launch_receipt_refuses_second_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(launch.os, "geteuid", lambda: 0)
    monkeypatch.setattr(launch, "paths", lambda seed: {"launch": tmp_path / "launch.json"})
    monkeypatch.setattr(launch, "probe", lambda seed: {
        "phase": "running", "destinations": {"launch": True},
        "output_exists": False, "unit_state": {"LoadState": "loaded"},
        "active_b2mpi_units": [launch.unit(seed)],
    })
    with pytest.raises(ValueError, match="already exists"):
        launch.run(1730)
