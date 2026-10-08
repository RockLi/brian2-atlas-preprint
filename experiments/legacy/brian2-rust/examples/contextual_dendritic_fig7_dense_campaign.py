#!/usr/bin/env python3
"""Run the 12 frozen Fig. 7 dense-recall chunks on the approved remote host.

This is scientific-data acquisition, never a benchmark. It refuses to start
unless the earlier two-visit trial's independent closed-HDF gate passed.
Each chunk is independently gated after its worker exits and releases HDF.
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
PLAN_SHA256 = "c010b6351d53f5560214a090f1f616b8b767b77ac04684562da4f3678af26670"
MANIFEST_SHA256 = "2a2247bc830caf64f89657c7fd4fb616a11b1233c561a38703d633fa12fc1d3c"
SUBSET_SHA256 = "f29ee11c0eefb18306db87cc8da5ee070acb97a964523570361d2f64e2288949"
CHECKPOINT_SHA256 = "e6811d7c4e2e9ba5131cd50d0aac64b3da848056aeb5613cd62f4402af8433ad"
BATCH_SOURCE_SHA256 = "7034fd1b6833aa88f3cb9c60dca1116093459e5697268b9ec049cf9d92d9820b"
GATE_SOURCE_SHA256 = "69a298e6aef1a6fc7020100e6d1342805fddac005be236d01694a5b85105ff98"
HELPER_SOURCE_SHA256 = "7be5d238569a6ea436eddcdc6e9657e6189d76015404b0d9e41f906d1e84b887"
QUEUE_SHA256 = "b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01"
ZIG_SHA256 = "2317bbb91798556d9d0f38aabdac23db83f0979b25f767259ae474546724087c"
CPU_SLOTS = [172, 173, 174, 175]


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


def check_hash(path: Path, expected: str) -> None:
    require(path.is_file() and sha256(path) == expected,
            f"missing/changed frozen input: {path}")


def no_open_hdf(lsof_binary: str, path: Path) -> bool:
    result = subprocess.run([lsof_binary, "--", str(path)],
                            capture_output=True, text=True, check=False)
    require(result.returncode in (0, 1),
            f"cannot determine HDF writer state: {result.stderr[-500:]}")
    return result.returncode == 1 and not result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-independent-gate", type=Path, required=True)
    parser.add_argument("--trial-closed-hdf", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    output_root = args.output_root.absolute()
    require(output_root.is_relative_to(ROOT), "campaign output must stay on approved host")
    require(not output_root.exists(), "refusing to overwrite an existing campaign")
    lsof_binary = shutil.which("lsof")
    require(lsof_binary is not None, "lsof required to establish closed HDF")
    inputs = ROOT / "fig7-dense-seed843-acquisition-v1/reference-data"
    manifest_path = ROOT / "fig7-missing-recall-restore-preflight-v1/missing-recall-frozen-inputs-v1.json"
    checkpoint_path = ROOT / "fig7-missing-recall-restore-preflight-v1/stored_imprint_c1937623_0"
    batch_source = inputs / "contextual_dendritic_fig7_dense_recall_batch.py"
    gate_source = inputs / "contextual_fig7_dense_chunk_closed_gate.py"
    helper_source = inputs / "contextual_fig7_dense_recall_closed_gate.py"
    subset_path = inputs / "official-imprint-subset.h5"
    plan_path = inputs / "dense-chunk-plan-v1.json"
    queue_path = ROOT / "fig4-spikequeue-recovery-v1/overlay/brian2/synapses/cythonspikequeue.cpython-310-x86_64-linux-gnu.so"
    zig_path = ROOT / "fig7-toolchain-zig-v1/ziglang/zig"
    python_path = ROOT / ".paper-venv-py310/bin/python"
    template = ROOT / "contextual-remote-stage/paper-repository"
    for path, expected in ((manifest_path, MANIFEST_SHA256),
                           (checkpoint_path, CHECKPOINT_SHA256),
                           (batch_source, BATCH_SOURCE_SHA256),
                           (gate_source, GATE_SOURCE_SHA256),
                           (helper_source, HELPER_SOURCE_SHA256),
                           (subset_path, SUBSET_SHA256),
                           (plan_path, PLAN_SHA256),
                           (queue_path, QUEUE_SHA256), (zig_path, ZIG_SHA256)):
        check_hash(path, expected)
    require(python_path.is_file() and template.is_dir(),
            "remote paper environment/source is missing")
    trial_gate_path = args.trial_independent_gate.resolve(strict=True)
    trial_hdf_path = args.trial_closed_hdf.resolve(strict=True)
    trial_gate = json.loads(trial_gate_path.read_text())
    require(trial_gate["schema"] == "contextual-fig7-dense-recall-closed-gate-v1"
            and trial_gate["visit_indices"] == [593, 595]
            and trial_gate["independent_protocol_and_metrics_gate_passed"] is True
            and trial_gate["full_fig7_scientific_acceptance"] is False
            and trial_gate["performance_authorized"] is False,
            "required two-visit independent gate has not passed")
    require(trial_gate["candidate_hdf_sha256"] == sha256(trial_hdf_path),
            "trial gate does not identify the closed trial HDF")
    require(no_open_hdf(lsof_binary, trial_hdf_path),
            "trial HDF is still open")
    plan = json.loads(plan_path.read_text())
    require(plan["schema"] == "contextual-fig7-dense-seed843-chunk-plan-v1"
            and plan["require_trial_closed_independent_gate_pass_before_launch"] is True
            and plan["maximum_simultaneous_chunks"] == 4
            and plan["planned_cpu_affinities"] == CPU_SLOTS
            and len(plan["chunks"]) == 12,
            "frozen plan shape differs")
    indices = [index for chunk in plan["chunks"] for index in chunk["visit_indices"]]
    require(len(indices) == len(set(indices)) == 120
            and plan["all_122_dense_visits_required_for_panel_coverage"][:2]
            == [593, 595]
            and indices == plan["all_122_dense_visits_required_for_panel_coverage"][2:],
            "planned dense-response coverage differs")

    output_root.mkdir(parents=True)
    state_path = output_root / "campaign-state.json"
    state = {
        "schema": "contextual-fig7-dense-seed843-remote-campaign-v1",
        "purpose": "scientific_data_acquisition_not_performance",
        "host": HOST,
        "plan_sha256": PLAN_SHA256,
        "trial_gate_sha256": sha256(trial_gate_path),
        "trial_closed_hdf_sha256": trial_gate["candidate_hdf_sha256"],
        "orchestrator_source_sha256": sha256(Path(__file__).resolve()),
        "maximum_simultaneous_chunks": 4,
        "chunks": {chunk["chunk_id"]: {"visit_indices": chunk["visit_indices"],
                                       "status": "pending"}
                   for chunk in plan["chunks"]},
        "all_chunks_closed_gates_passed": False,
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
    pending = list(plan["chunks"])
    active: dict[str, tuple[subprocess.Popen, int, object]] = {}
    failed = False
    try:
        while active or (pending and not failed):
            busy_cpus = {cpu for _, cpu, _ in active.values()}
            available_cpus = [cpu for cpu in CPU_SLOTS if cpu not in busy_cpus]
            while pending and available_cpus and not failed:
                chunk = pending.pop(0)
                chunk_id = chunk["chunk_id"]
                cpu = available_cpus.pop(0)
                batch_dir = output_root / chunk_id
                require(not batch_dir.exists(), f"batch root exists: {batch_dir}")
                log_path = output_root / f"{chunk_id}-launch.log"
                log_stream = log_path.open("x")
                command = ["taskset", "-c", str(cpu), "nice", "-n", "10",
                           str(python_path), str(batch_source),
                           "--template-repo", str(template),
                           "--imprint-subset", str(subset_path),
                           "--checkpoint", str(checkpoint_path),
                           "--frozen-manifest", str(manifest_path),
                           "--compiled-queue-extension", str(queue_path),
                           "--compiler-binary", str(zig_path),
                           "--visit-indices", ",".join(map(str, chunk["visit_indices"])),
                           "--output-root", str(batch_dir)]
                worker = subprocess.Popen(command, cwd=ROOT, env=env,
                                          stdin=subprocess.DEVNULL,
                                          stdout=log_stream, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                active[chunk_id] = (worker, cpu, log_stream)
                state["chunks"][chunk_id].update({"status": "running",
                                                  "worker_pid": worker.pid,
                                                  "cpu_affinity": cpu,
                                                  "log_path": str(log_path)})
                atomic_json(state_path, state)
            for chunk_id, (worker, cpu, log_stream) in list(active.items()):
                exit_code = worker.poll()
                if exit_code is None:
                    continue
                log_stream.close()
                del active[chunk_id]
                chunk_state = state["chunks"][chunk_id]
                chunk_state["worker_exit_code"] = exit_code
                batch_dir = output_root / chunk_id
                if exit_code != 0:
                    chunk_state["status"] = "worker_failed"
                    failed = True
                    atomic_json(state_path, state)
                    continue
                hdf_path = batch_dir / "paper-repository/results/sim_files/data_Fig_7.h5"
                require(hdf_path.is_file() and no_open_hdf(lsof_binary, hdf_path),
                        f"completed worker still has open/missing HDF: {chunk_id}")
                gate_report = output_root / f"{chunk_id}-closed-gate.json"
                command = [str(python_path), str(gate_source),
                           "--chunk-plan", str(plan_path),
                           "--chunk-id", chunk_id,
                           "--frozen-manifest", str(manifest_path),
                           "--official-imprint-subset", str(subset_path),
                           "--batch-source", str(batch_source),
                           "--helper-source", str(helper_source),
                           "--batch-report", str(batch_dir / "report.json"),
                           "--candidate-hdf", str(hdf_path),
                           "--output", str(gate_report)]
                checked = subprocess.run(command, cwd=ROOT, env=env,
                                         capture_output=True, text=True, check=False)
                chunk_state["closed_gate_exit_code"] = checked.returncode
                chunk_state["closed_gate_output_tail"] = (
                    checked.stdout + checked.stderr)[-1500:]
                if checked.returncode == 0:
                    gate = json.loads(gate_report.read_text())
                    require(gate["independent_chunk_protocol_and_metrics_gate_passed"]
                            is True, f"chunk gate did not report pass: {chunk_id}")
                    chunk_state["status"] = "closed_gate_passed"
                    chunk_state["closed_hdf_sha256"] = gate["candidate_hdf_sha256"]
                    chunk_state["gate_report_sha256"] = sha256(gate_report)
                else:
                    chunk_state["status"] = "closed_gate_failed"
                    failed = True
                atomic_json(state_path, state)
            if active or (pending and not failed):
                time.sleep(15)
        require(not failed, "one or more frozen ten-visit chunks failed")
        require(all(item["status"] == "closed_gate_passed"
                    for item in state["chunks"].values()),
                "not every frozen chunk passed its independent closed gate")
        state["all_chunks_closed_gates_passed"] = True
        atomic_json(state_path, state)
    except Exception:
        state["error"] = traceback.format_exc()[-7000:]
        atomic_json(state_path, state)
        raise
    print(json.dumps({"all_chunks_closed_gates_passed": True,
                      "chunk_count": len(state["chunks"])}, sort_keys=True))


if __name__ == "__main__":
    main()
