#!/usr/bin/env python3
"""Archive only the final two completed Fig. 3 association pipelines.

The earlier eight are already archived on T7. This script pins the full
terminal inventory and refuses to touch any pipeline outside seeds 325/832.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import subprocess


ROOT = Path("/atlas-home/0003/workspace/contextual-dendritic-gating-20260921")
INVENTORY_SHA256 = "94f58cf254c598d50d285c8e811bb0495ba749bbba09d52c4c2928def066627f"
FINAL_IDS = {"fig3-association-s0325", "fig3-association-s0832"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def main() -> None:
    if socket.gethostname() != "hk-prod-model-ae09-94":
        raise SystemExit("approved remote host only")
    inventory_path = ROOT / "fig3-association-inventory-v1/full-campaign-coverage-20260928-final-v1.json"
    if sha256(inventory_path) != INVENTORY_SHA256:
        raise ValueError("final Fig. 3 inventory differs")
    inventory = json.loads(inventory_path.read_text())
    family = inventory["families"]["association"]
    if family["completed"] != 10 or family["pending"] != 0:
        raise ValueError("association family is not fully complete")
    entries = {item["id"]: item for item in family["completed_pipelines"]}
    if not FINAL_IDS <= set(entries):
        raise ValueError("final two association pipelines missing")
    campaign = ROOT / "fig3-full-campaign-v1"
    output = ROOT / "fig3-association-final-archive-v1"
    output.mkdir(exist_ok=True)
    records = []
    for cell_id in sorted(FINAL_IDS):
        item = entries[cell_id]
        cell = campaign / "pipelines" / cell_id
        if sha256(cell / "status.json") != item["status_sha256"]:
            raise ValueError(f"{cell_id}: terminal status changed")
        for stage, expected_hash in item["stage_report_sha256"].items():
            if sha256(cell / "reports" / f"{stage}.json") != expected_hash:
                raise ValueError(f"{cell_id}: stage report changed: {stage}")
        hdf5 = campaign / item["hdf5_relative_path"]
        if sha256(hdf5) != item["hdf5_sha256"]:
            raise ValueError(f"{cell_id}: raw HDF5 changed")
        archive = output / f"{cell_id}.tar.zst"
        partial = output / f"{cell_id}.tar.zst.partial"
        if archive.exists() or partial.exists():
            raise FileExistsError(f"refusing to overwrite {cell_id} archive")
        run("tar", "-I", "zstd -T1 -1", "-cf", str(partial),
            "-C", str(campaign / "pipelines"), cell_id)
        partial.rename(archive)
        run("zstd", "-tq", str(archive))
        members = run("tar", "-I", "zstd", "-tf", str(archive)).splitlines()
        expected_member = f"{cell_id}/paper-repository/results/sim_files/{hdf5.name}"
        if members.count(expected_member) != 1:
            raise ValueError(f"{cell_id}: expected HDF5 not unique in archive")
        records.append({"id": cell_id, "archive": archive.name,
                        "archive_bytes": archive.stat().st_size,
                        "archive_sha256": sha256(archive),
                        "source_hdf5_sha256": item["hdf5_sha256"],
                        "source_status_sha256": item["status_sha256"],
                        "archive_member_count": len(members)})
    report = {"schema": "contextual-dendritic-fig3-final-association-archive-v1",
              "purpose": "archive_only_two_finished_association_pipelines_no_simulation_or_timing",
              "inventory_sha256": INVENTORY_SHA256,
              "records": records,
              "earlier_eight_pipelines_unchanged": True,
              "remote_source_files_moved": False,
              "reported_timings": False,
              "whole_figure_scientific_gate_passed": None,
              "performance_authorized": False}
    path = output / "archive-report-v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"archived": len(records), "total_bytes": sum(
        item["archive_bytes"] for item in records)}))


if __name__ == "__main__":
    main()
