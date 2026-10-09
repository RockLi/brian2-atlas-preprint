#!/usr/bin/env python3
"""Wait for the 1-node measurements, then run the 2/4/5-node campaign."""

import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw/mpi_multinode_rank40_20480_20260919"
ONE = RAW / "formal_1node_measured_v2/report.json"
DRIVER = Path(__file__).with_name("run_mpi_multinode_rank40_20480.py")
STATUS = RAW / "followon_status.json"
EVENTS = "99c9a97cd232fa38a40f150d77dd5a92f2ce3b6b1786f02bb7bb639d5bdaffa5"


def save(payload):
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATUS)


def completed(path):
    try:
        report = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return report.get("summary", {}).get("all_byte_exact") is True


def main():
    state = {"schema": "nmda-skaar-2025-multinode-followon-v1",
             "state": "waiting_for_1node", "completed_topologies": []}
    save(state)
    deadline = time.monotonic() + 8 * 3600
    while not completed(ONE):
        if time.monotonic() >= deadline:
            raise TimeoutError("1-node campaign did not complete within eight hours")
        time.sleep(30)

    for nodes in (2, 4, 5):
        state["state"] = f"running_{nodes}nodes"
        state["started_unix"] = time.time()
        save(state)
        root = RAW / f"formal_{nodes}nodes"
        command = [sys.executable, str(DRIVER), "--nodes", str(nodes),
                   "--warmups", "1", "--repetitions", "5",
                   "--expected-events", EVENTS, "--root", str(root),
                   "--campaign-id", "nmda20480-formal"]
        subprocess.run(command, check=True, cwd=ROOT.parents[2])
        report = json.loads((root / "report.json").read_text())
        if report.get("summary", {}).get("all_byte_exact") is not True:
            raise RuntimeError(f"{nodes}-node campaign did not pass exactness")
        state["completed_topologies"].append(nodes)
        state["last_finished_unix"] = time.time()
        save(state)

    state["state"] = "complete"
    state["finished_unix"] = time.time()
    save(state)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save({"schema": "nmda-skaar-2025-multinode-followon-v1",
              "state": "failed", "error": repr(error),
              "failed_unix": time.time()})
        raise
