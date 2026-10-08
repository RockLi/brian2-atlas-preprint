#!/usr/bin/env python3
"""Acquire seven exact-source-rate Fig. 7 population recalls after a passing pilot.

Approved remote host only. At most two isolated workers run together; each
closed HDF gets the frozen independent gate. This is not a benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback


HOST = "hk-prod-model-ae09-94"
ROOT = Path("/atlas-home/0003/workspace/contextual-dendritic-gating-20260921")
MANIFEST_SHA256 = "2a2247bc830caf64f89657c7fd4fb616a11b1233c561a38703d633fa12fc1d3c"
SUBSET_REPORT_SHA256 = "3f9a27e1e14da1751372a34fdae89c20c816f91a286a23d24981fadd2e995a0a"
SPECIAL_SUBSET_REPORT_SHA256 = "201577bfb4cf92aac8a1aa455cb3177d87718067d493864f03dcd1ef7abf917c"
WORKER_SOURCE_SHA256 = "2a9f3bf3efe1f23c2629dc7bdfb9c503f5e437ab83d52abb83cc93a18a91fbd9"
GATE_SOURCE_SHA256 = "defb9c3d0bb1cbf2e4b542ec9685a6afe58d7421d6ec007003079fe8a49ad8ae"
QUEUE_SHA256 = "b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01"
ZIG_SHA256 = "2317bbb91798556d9d0f38aabdac23db83f0979b25f767259ae474546724087c"
PILOT_INDEX = 1289
REMAINING_INDICES = [1297, 1299, 1311, 1315, 1319, 1333, 1335]
CPU_SLOTS = [179, 180]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".temporary")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def no_open_hdf(lsof_binary: str, path: Path) -> bool:
    checked = subprocess.run([lsof_binary, "--", str(path)],
                             capture_output=True, text=True, check=False)
    require(checked.returncode in (0, 1),
            f"cannot establish closed HDF: {checked.stderr[-500:]}")
    return checked.returncode == 1 and not checked.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-independent-gate", type=Path, required=True)
    parser.add_argument("--pilot-closed-hdf", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    output_root = args.output_root.absolute()
    require(output_root.is_relative_to(ROOT) and not output_root.exists(),
            "campaign output must be a new path on the approved host")
    lsof_binary = shutil.which("lsof")
    require(lsof_binary is not None, "lsof required for closed HDF checks")
    inputs = ROOT / "fig7-population-missing-v1/reference-data"
    manifest_path = ROOT / "fig7-missing-recall-restore-preflight-v1/missing-recall-frozen-inputs-v1.json"
    subset_report_path = inputs / "report.json"
    worker_source = ROOT / "fig7-population-exact-rate-v2/contextual_dendritic_fig7_population_exact_rate_visit.py"
    gate_source = ROOT / "fig7-population-exact-rate-v2/contextual_fig7_population_exact_rate_closed_gate.py"
    special_subset_report = ROOT / "fig7-missing-recall-restore-preflight-v1/reference-data/seed-7433-input-2-official-imprint-subset-v1.json"
    special_subset_path = ROOT / "fig7-missing-recall-restore-preflight-v1/reference-data/seed-7433-input-2-official-imprint-subset-v1.h5"
    queue_path = ROOT / "fig4-spikequeue-recovery-v1/overlay/brian2/synapses/cythonspikequeue.cpython-310-x86_64-linux-gnu.so"
    zig_path = ROOT / "fig7-toolchain-zig-v1/ziglang/zig"
    python_path = ROOT / ".paper-venv-py310/bin/python"
    template = ROOT / "contextual-remote-stage/paper-repository"
    for path, expected in ((manifest_path, MANIFEST_SHA256),
                           (subset_report_path, SUBSET_REPORT_SHA256),
                           (worker_source, WORKER_SOURCE_SHA256),
                           (gate_source, GATE_SOURCE_SHA256),
                           (queue_path, QUEUE_SHA256), (zig_path, ZIG_SHA256)):
        require(path.is_file() and sha256(path) == expected,
                f"frozen input hash differs: {path}")
    require(python_path.is_file() and template.is_dir(),
            "remote Python or paper source missing")
    require(sha256(special_subset_report) == SPECIAL_SUBSET_REPORT_SHA256,
            "visit 1311 frozen subset report differs")
    manifest = json.loads(manifest_path.read_text())
    by_index = {visit["visit_index"]: visit for visit in manifest["visits"]}
    subsets = json.loads(subset_report_path.read_text())
    subset_by_index = {row["visit_index"]: row for row in subsets["subsets"]}
    require([row["visit_index"] for row in subsets["subsets"]]
            == [1289, 1297, 1299, 1315, 1319, 1333, 1335],
            "seven original subset rows differ")
    special = json.loads(special_subset_report.read_text())
    require(special["group"] == by_index[1311]["imprint_group"],
            "visit 1311 imprint group differs")
    subset_by_index[1311] = {
        "imprint_group": special["group"],
        "checkpoint_sha256": by_index[1311]["checkpoint_sha256"],
        "subset_sha256": special["subset_hdf_sha256"],
    }

    def subset_path_for(index: int) -> Path:
        return (special_subset_path if index == 1311
                else inputs / subset_by_index[index]["subset_filename"])

    def subset_report_for(index: int) -> Path:
        return special_subset_report if index == 1311 else subset_report_path
    for index in REMAINING_INDICES:
        visit = by_index[index]
        row = subset_by_index[index]
        subset_path = subset_path_for(index)
        checkpoint_path = (ROOT / "fig7-missing-recall-restore-preflight-v1" /
                           f"stored_imprint_{visit['imprint_group']}_0")
        require(visit["panel"] == "population_maximum"
                and visit["imprint_group"] == row["imprint_group"]
                and visit["checkpoint_sha256"] == row["checkpoint_sha256"]
                and sha256(subset_path) == row["subset_sha256"]
                and sha256(checkpoint_path) == visit["checkpoint_sha256"],
                f"visit {index} subset/checkpoint differs")
    pilot_gate = json.loads(args.pilot_independent_gate.read_text())
    require(pilot_gate["schema"] == "contextual-fig7-population-exact-rate-closed-gate-v2"
            and pilot_gate["visit_index"] == PILOT_INDEX
            and pilot_gate["independent_protocol_and_metrics_gate_passed"] is True
            and pilot_gate["full_fig7_scientific_acceptance"] is False
            and pilot_gate["performance_authorized"] is False,
            "required cross-checkpoint pilot gate has not passed")
    require(pilot_gate["candidate_hdf_sha256"] == sha256(args.pilot_closed_hdf)
            and no_open_hdf(lsof_binary, args.pilot_closed_hdf),
            "pilot gate does not identify a closed HDF")

    output_root.mkdir(parents=True)
    state_path = output_root / "campaign-state.json"
    state = {
        "schema": "contextual-fig7-population-exact-rate-campaign-v2",
        "purpose": "scientific_data_acquisition_not_performance",
        "host": HOST,
        "pilot_independent_gate_sha256": sha256(args.pilot_independent_gate),
        "pilot_closed_hdf_sha256": pilot_gate["candidate_hdf_sha256"],
        "controller_source_sha256": sha256(Path(__file__).resolve()),
        "maximum_simultaneous_workers": 2,
        "source_mode": "explicit_assembly_firing_rate_recall_no_size_override",
        "visits": {str(index): {"status": "pending"}
                   for index in REMAINING_INDICES},
        "all_seven_closed_gates_passed": False,
        "full_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    atomic_json(state_path, state)
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": "fig4-spikequeue-recovery-v1/overlay:fig8-imprint-campaign-tools-v1",
        "MPLBACKEND": "Agg",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "CC": f"{zig_path} cc", "CXX": f"{zig_path} c++",
        "LDSHARED": f"{zig_path} c++ -shared",
        "LDCXXSHARED": f"{zig_path} c++ -shared",
    })
    pending = list(REMAINING_INDICES)
    active: dict[int, tuple[subprocess.Popen, int, object]] = {}
    failed = False
    try:
        while active or (pending and not failed):
            occupied = {cpu for _, cpu, _ in active.values()}
            free_cpus = [cpu for cpu in CPU_SLOTS if cpu not in occupied]
            while pending and free_cpus and not failed:
                index = pending.pop(0)
                cpu = free_cpus.pop(0)
                visit = by_index[index]
                row = subset_by_index[index]
                output = output_root / f"visit-{index}"
                require(not output.exists(), f"visit output already exists: {index}")
                log_path = output_root / f"visit-{index}-launch.log"
                log_stream = log_path.open("x")
                command = ["taskset", "-c", str(cpu), "nice", "-n", "10",
                           str(python_path), str(worker_source),
                           "--visit-index", str(index),
                           "--template-repo", str(template),
                           "--imprint-subset", str(subset_path_for(index)),
                           "--subset-report", str(subset_report_for(index)),
                           "--checkpoint", str(ROOT / "fig7-missing-recall-restore-preflight-v1" /
                                                 f"stored_imprint_{visit['imprint_group']}_0"),
                           "--frozen-manifest", str(manifest_path),
                           "--compiled-queue-extension", str(queue_path),
                           "--compiler-binary", str(zig_path),
                           "--output-root", str(output)]
                worker = subprocess.Popen(command, cwd=ROOT, env=env,
                                          stdin=subprocess.DEVNULL,
                                          stdout=log_stream, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                active[index] = (worker, cpu, log_stream)
                state["visits"][str(index)].update({
                    "status": "running", "worker_pid": worker.pid,
                    "cpu_affinity": cpu, "log_path": str(log_path)})
                atomic_json(state_path, state)
            for index, (worker, cpu, log_stream) in list(active.items()):
                exit_code = worker.poll()
                if exit_code is None:
                    continue
                log_stream.close()
                del active[index]
                visit_state = state["visits"][str(index)]
                visit_state["worker_exit_code"] = exit_code
                if exit_code != 0:
                    visit_state["status"] = "worker_failed"
                    failed = True
                    atomic_json(state_path, state)
                    continue
                output = output_root / f"visit-{index}"
                hdf_path = output / "paper-repository/results/sim_files/data_Fig_7.h5"
                require(hdf_path.is_file() and no_open_hdf(lsof_binary, hdf_path),
                        f"visit {index} HDF missing/open after worker exit")
                gate_path = output_root / f"visit-{index}-closed-gate.json"
                command = [str(python_path), str(gate_source),
                           "--visit-index", str(index),
                           "--frozen-manifest", str(manifest_path),
                           "--subset-report", str(subset_report_for(index)),
                           "--official-imprint-subset", str(subset_path_for(index)),
                           "--worker-source", str(worker_source),
                           "--worker-report", str(output / "report.json"),
                           "--candidate-hdf", str(hdf_path),
                           "--output", str(gate_path)]
                checked = subprocess.run(command, cwd=ROOT, env=env,
                                         capture_output=True, text=True, check=False)
                visit_state["closed_gate_exit_code"] = checked.returncode
                visit_state["closed_gate_output_tail"] = (
                    checked.stdout + checked.stderr)[-1500:]
                if checked.returncode == 0:
                    gate = json.loads(gate_path.read_text())
                    require(gate["independent_protocol_and_metrics_gate_passed"]
                            is True, f"visit {index} closed gate did not pass")
                    visit_state["status"] = "closed_gate_passed"
                    visit_state["closed_hdf_sha256"] = gate["candidate_hdf_sha256"]
                    visit_state["gate_report_sha256"] = sha256(gate_path)
                else:
                    visit_state["status"] = "closed_gate_failed"
                    failed = True
                atomic_json(state_path, state)
            if active or (pending and not failed):
                time.sleep(15)
        require(not failed, "one or more missing population recalls failed")
        require(all(item["status"] == "closed_gate_passed"
                    for item in state["visits"].values()),
                "not all seven exact-rate closed gates passed")
        state["all_seven_closed_gates_passed"] = True
        atomic_json(state_path, state)
    except Exception:
        state["error"] = traceback.format_exc()[-7000:]
        atomic_json(state_path, state)
        raise
    print(json.dumps({"all_seven_closed_gates_passed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
