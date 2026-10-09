#!/usr/bin/env python3
"""Independently audit the fixed 2,000-pair psychometric task manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


COHERENCES = (1, 5, 10, 20, 40)
EXPECTED_COMMIT = "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e"
EXPECTED_SOURCE = "781010ca51948dad030db050dfedaab77826a0ac9a346565c0fa6af888a4a347"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.manifest.read_text())
    protocol = value.get("protocol", {})
    tasks = value.get("tasks", [])
    nodes = protocol.get("nodes", [])
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(
        value.get("schema") == "nmda-skaar-2025-decision-psychometric-manifest-v1",
        "unexpected schema",
    )
    require(protocol.get("upstream_commit") == EXPECTED_COMMIT, "upstream commit mismatch")
    require(
        protocol.get("upstream_source_sha256") == EXPECTED_SOURCE,
        "upstream source hash mismatch",
    )
    require(protocol.get("trials_per_coherence_per_model") == 400, "trial count mismatch")
    require(protocol.get("total_pairs") == 2000, "total pair count mismatch")
    require(protocol.get("total_simulations") == 4000, "simulation count mismatch")
    require(len(tasks) == 2000, f"manifest contains {len(tasks)} tasks")
    require(len(nodes) == 5 and len(set(nodes)) == 5, "expected five unique nodes")

    task_ids = [task.get("task_id") for task in tasks]
    seeds = [task.get("seed") for task in tasks]
    require(len(set(task_ids)) == 2000, "task IDs are not unique")
    require(len(set(seeds)) == 2000, "task seeds are not unique")
    require(
        all(task.get("seed") == task.get("numpy_stimulus_seed") for task in tasks),
        "a paired NEST/NumPy seed differs",
    )
    require(all(isinstance(seed, int) and 0 <= seed < 2**31 for seed in seeds), "seed range invalid")

    coherence_counts = Counter(task.get("coherence_percent") for task in tasks)
    host_counts = Counter(task.get("host") for task in tasks)
    host_coherence_counts: dict[str, Counter] = defaultdict(Counter)
    for task in tasks:
        host_coherence_counts[task.get("host")][task.get("coherence_percent")] += 1
        slot = task.get("slot")
        require(isinstance(slot, int) and 0 <= slot < 12, f"invalid slot in {task.get('task_id')}")
        if isinstance(slot, int):
            require(
                task.get("cpuset") == f"{slot * 8}-{slot * 8 + 7}",
                f"cpuset mismatch in {task.get('task_id')}",
            )
    for coherence in COHERENCES:
        require(coherence_counts[coherence] == 400, f"coherence {coherence} count mismatch")
        indices = {
            task.get("trial_index")
            for task in tasks
            if task.get("coherence_percent") == coherence
        }
        require(indices == set(range(400)), f"coherence {coherence} trial indices mismatch")
    for node in nodes:
        require(host_counts[node] == 400, f"host {node} pair count mismatch")
        for coherence in COHERENCES:
            require(
                host_coherence_counts[node][coherence] == 80,
                f"host {node} coherence {coherence} balance mismatch",
            )

    result = {
        "schema": "nmda-skaar-2025-decision-psychometric-manifest-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "complete": not errors,
        "task_count": len(tasks),
        "simulation_count": 2 * len(tasks),
        "unique_task_ids": len(set(task_ids)),
        "unique_seeds": len(set(seeds)),
        "coherence_counts": {str(key): coherence_counts[key] for key in COHERENCES},
        "host_counts": dict(host_counts),
        "host_coherence_counts": {
            host: {str(key): counts[key] for key in COHERENCES}
            for host, counts in host_coherence_counts.items()
        },
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
