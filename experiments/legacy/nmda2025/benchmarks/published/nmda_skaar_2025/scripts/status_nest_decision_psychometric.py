#!/usr/bin/env python3
"""Print a compact, machine-readable status for a psychometric campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from statistics import median


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign
    captured = datetime.now(timezone.utc)
    manifest = json.loads((campaign / "manifest.json").read_text())
    target_pairs = manifest["protocol"]["total_pairs"]

    statuses = [
        json.loads(path.read_text())
        for path in sorted((campaign / "status").glob("*.json"))
    ]
    pairs = [
        json.loads(path.read_text())
        for path in sorted((campaign / "results").glob("*/pair.json"))
    ]
    completed_by_host = {}
    for host in sorted({pair["host"] for pair in pairs}):
        rows = [pair for pair in pairs if pair["host"] == host]
        completed_by_host[host] = {
            "pairs": len(rows),
            "approximate_inner_median_seconds": median(
                row["models"]["approximate"]["wall_seconds_inner"] for row in rows
            ),
            "exact_inner_median_seconds": median(
                row["models"]["exact"]["wall_seconds_inner"] for row in rows
            ),
            "pair_wall_median_seconds": median(
                row["pair_wall_seconds"] for row in rows
            ),
        }
        assigned = next(
            status["assigned_pairs"] for status in statuses if status["host"] == host
        )
        remaining = assigned - len(rows)
        slots = len(next(status["running"] for status in statuses if status["host"] == host))
        slots = slots or 12
        estimated_remaining = (
            remaining
            * completed_by_host[host]["pair_wall_median_seconds"]
            / slots
        )
        completed_by_host[host].update(
            {
                "assigned_pairs": assigned,
                "remaining_pairs": remaining,
                "estimated_remaining_seconds_at_current_host_median": estimated_remaining,
                "estimated_finish_utc": (captured + timedelta(seconds=estimated_remaining)).isoformat(),
                "estimate_scope": "projection only; median pair wall under current 12-slot contention",
            }
        )
    result = {
        "schema": "nmda-skaar-2025-decision-psychometric-live-status-v1",
        "captured_at_utc": captured.isoformat(),
        "campaign": str(campaign),
        "result_directories": sum(
            path.is_dir() for path in (campaign / "results").iterdir()
        ),
        "target_pairs": target_pairs,
        "failure_files": len(list((campaign / "attempts").glob("**/failure.json"))),
        "completed_by_host": completed_by_host,
        "workers": [
            {
                "host": status["host"],
                "completed": len(status["completed_this_invocation"])
                + len(status["skipped_existing"]),
                "running": len(status["running"]),
                "failed": len(status["failed"]),
                "finished": status["finished"],
                "updated_utc": status.get("updated_utc"),
            }
            for status in statuses
        ],
    }
    result["completion_fraction"] = result["result_directories"] / target_pairs
    result["projected_campaign_finish_utc"] = max(
        value["estimated_finish_utc"] for value in completed_by_host.values()
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
