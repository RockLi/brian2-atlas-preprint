#!/usr/bin/env python3
"""Check all Fig. 8 saved-imprint restores in a fresh Python process.

The original five imprints are loaded from cache. Any attempt to advance a
Brian2 network is trapped. This is a correctness preflight, not a simulation
or timing measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--seed", type=int, default=6427)
    parser.add_argument("--case-id", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    os.environ.setdefault("MPLBACKEND", "Agg")
    repo = args.paper_repo.resolve()
    h5_path = repo / "results" / "sim_files" / "data_Fig_8.h5"
    if not h5_path.is_file():
        parser.error(f"missing completed imprint cache: {h5_path}")
    initial_hash = digest(h5_path)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "scripts"))
    os.chdir(repo / "scripts")

    from brian2 import Network  # type: ignore
    import Fig_8  # type: ignore

    original_run = Network.run

    def forbid_simulation(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("cache-only preflight forbids any Brian2 simulation")

    Network.run = forbid_simulation
    result: dict[str, object] = {
        "schema": "contextual-dendritic-fig8-fresh-restore-preflight-v2",
        "purpose": "cache_only_restore_no_simulation_no_performance_measurement",
        "paper_repo": str(repo),
        "h5_before_sha256": initial_hash,
        "seed": args.seed,
        "case_id": args.case_id,
        "passed": False,
    }
    try:
        schedule = Fig_8.setup_result_dict(case_id=args.case_id)
        net = Fig_8.get_network_for_investigation(seed=args.seed)
        net.only_load_results = True
        restored = []
        for order_id, imprint_inputs in enumerate(schedule["all_case_imprint_inputs"]):
            stored_name = None
            for imprint_id, imprint in enumerate(imprint_inputs):
                stored_name = Fig_8.get_simulated_network(
                    net=net,
                    filename_for_stored_network=stored_name,
                    all_assembly_ids_for_areas=[imprint],
                )
                if stored_name is None:
                    raise ValueError(
                        f"imprint order={order_id} index={imprint_id} "
                        "was not found in the existing cache"
                    )
                checkpoint = net.get_path_to_stored_networks(file_name=stored_name)
                net.network.restore(filename=checkpoint)
                restored.append(
                    {
                        "order_id": order_id,
                        "imprint_id": imprint_id,
                        "stored_name": stored_name,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": digest(Path(checkpoint)),
                    }
                )
        result["restored"] = restored
        result["unique_checkpoints"] = len({item["checkpoint"] for item in restored})
        result["passed"] = len(restored) == 5 and result["unique_checkpoints"] == 5
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
    finally:
        Network.run = original_run
        result["h5_after_sha256"] = digest(h5_path)
        result["h5_unchanged"] = result["h5_after_sha256"] == initial_hash
        result["passed"] = bool(result["passed"] and result["h5_unchanged"])
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
