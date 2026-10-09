#!/usr/bin/env python3
"""Create the deterministic task manifest for the full Figure 4 validation.

The upstream paper used 400 trials per coherence and model.  Each task in this
manifest is one exact/approximate pair with identical NEST and NumPy stimulus
seeds.  Seeds are derived from SHA-256 so the schedule does not depend on a
Python or NumPy random-number implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


COHERENCES = (1, 5, 10, 20, 40)
HOSTS = (
    "hk-prod-model-ae02-24",
    "hk-prod-model-ae02-25",
    "hk-prod-model-ae03-33",
    "hk-prod-model-ae05-53",
    "hk-prod-model-ae05-54",
)
UPSTREAM_COMMIT = "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e"
SOURCE_SHA256 = "781010ca51948dad030db050dfedaab77826a0ac9a346565c0fa6af888a4a347"
IMAGE = (
    "nest/nest-simulator@"
    "sha256:72f7598f515f8b4bf9409ee53f9100b5a79be424463992fd3c001223a6060cdf"
)


def task_seed(coherence: int, trial: int) -> int:
    label = f"nmda-skaar-2025-figure4-v1:c={coherence}:trial={trial}".encode()
    # NEST accepts positive 32-bit seeds.  Reserve zero and INT32_MAX.
    return int.from_bytes(hashlib.sha256(label).digest()[:8], "big") % (2**31 - 2) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--slots-per-host", type=int, default=12)
    args = parser.parse_args()
    if args.trials <= 0 or args.slots_per_host <= 0:
        raise SystemExit("--trials and --slots-per-host must be positive")

    tasks = []
    seeds = set()
    per_host_counts = {host: 0 for host in HOSTS}
    # Interleave coherence levels so every host has a representative partial
    # result throughout a resumed campaign.
    for trial in range(args.trials):
        for coherence_index, coherence in enumerate(COHERENCES):
            seed = task_seed(coherence, trial)
            if seed in seeds:
                raise RuntimeError(f"seed collision: {seed}")
            seeds.add(seed)
            # Rotate coherence-to-host assignment by trial. Every host gets
            # exactly one fifth of every coherence level.
            host_index = (trial + coherence_index) % len(HOSTS)
            host = HOSTS[host_index]
            slot = per_host_counts[host] % args.slots_per_host
            tasks.append(
                {
                    "task_id": f"c{coherence:02d}-trial{trial:03d}",
                    "coherence_percent": coherence,
                    "trial_index": trial,
                    "seed": seed,
                    "numpy_stimulus_seed": seed,
                    "host": host,
                    "slot": slot,
                    "cpuset": f"{slot * 8}-{slot * 8 + 7}",
                }
            )
            per_host_counts[host] += 1

    manifest = {
        "schema": "nmda-skaar-2025-decision-psychometric-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "400 matched exact/approximate trials at each published coherence; "
            "paper Figure 4 trial count"
        ),
        "protocol": {
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_source_sha256": SOURCE_SHA256,
            "models": ["iaf_bw_2001", "iaf_bw_2001_exact"],
            "coherences_percent": list(COHERENCES),
            "trials_per_coherence_per_model": args.trials,
            "paired_inputs": True,
            "seed_derivation": (
                "SHA-256 of 'nmda-skaar-2025-figure4-v1:c=<c>:trial=<i>', "
                "first 64 bits big-endian modulo 2^31-2 plus 1"
            ),
            "threads_per_trial": 8,
            "dt_ms": 0.1,
            "biological_duration_ms": 4000.0,
            "image": IMAGE,
            "nodes": list(HOSTS),
            "slots_per_node": args.slots_per_host,
            "total_pairs": len(tasks),
            "total_simulations": len(tasks) * 2,
        },
        "scientific_protocol_difference_from_upstream_batch": (
            "The unchanged upstream run_sim body is used. Unlike the upstream top-level "
            "batch, NumPy is reset before each exact/approximate call so each pair receives "
            "identical time-varying stimulus draws; the NEST seed is also identical within "
            "the pair. This is required for a controlled exact-versus-approximate comparison."
        ),
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pairs": len(tasks),
                "simulations": len(tasks) * 2,
                "unique_seeds": len(seeds),
                "per_host": {
                    host: sum(task["host"] == host for task in tasks) for host in HOSTS
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
