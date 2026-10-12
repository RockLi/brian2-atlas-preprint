"""Capture real closure exits after the original exec session expired.

The immutable full controller re-audits already accepted test directories.
All original test records must exist and remain byte-identical. The previous
outer process exit is unknown and is never replaced with an inferred value.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
OUT = P / "closure"
AUDITOR = P.parent / "training-goal-audit-20261010/audit_completion_v9_modules.py"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")

def execute(name, script):
    command = [sys.executable, str(script)]
    started = time.time()
    with (OUT / (name + ".log")).open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        save(OUT / (name + "-launch.json"), dict(command=command, pid=child.pid,
             started_at=started, script_sha256=sha(script)))
        code = child.wait()
    record = dict(command=command, pid=child.pid, actual_wait=code,
                  wait="subprocess.wait", seconds=time.time()-started,
                  script_sha256=sha(script), log_sha256=sha(OUT / (name + ".log")))
    save(OUT / (name + "-terminal.json"), record)
    print(json.dumps(dict(stage=name, actual_wait=code)), flush=True)
    return code

def main():
    OUT.mkdir(exist_ok=False)
    save(OUT / "original-session-observation.json", dict(
        session_id=1304, last_observed_live_at=1791786508.961629,
        session_handle="missing", original_controller_pid=75738,
        process_observation="ps returned no matching process", original_actual_exit=None,
        original_status=json.loads((P / "status.json").read_text()),
        original_status_sha256=sha(P / "status.json"),
        original_controller_log_sha256=sha(P / "controller.log"),
        original_launch_receipt_sha256=sha(P / "launch-receipt.json"), observed_at=time.time()))
    spec = importlib.util.spec_from_file_location("closure_full_controller", P / "run.py")
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    cfg = controller.configuration()
    paths = []
    for mode in ("cpu", "metal"):
        for index in range(272):
            directory = controller.location(index, mode)
            assert (directory / "acceptance.json").is_file(), (mode, index)
            paths.extend(sorted(p for p in directory.rglob("*") if p.is_file()))
    before = {str(p): sha(p) for p in paths}
    save(OUT / "original-test-records.json", before)
    code = execute("full-controller-reaudit", P / "run.py")
    after = {str(p): sha(p) for p in paths}
    unchanged = before == after
    save(OUT / "test-record-integrity.json", dict(
        files=len(before), unchanged=unchanged,
        changed=[p for p in before if before[p] != after[p]], new_test_processes=0 if unchanged else None))
    assert code == 0 and unchanged, "Controller closure or record integrity failed"
    code = execute("completion-audit", AUDITOR)
    report = AUDITOR.parent / "completion-gates-v9-modules.json"
    result = json.loads(report.read_text()) if report.exists() else {}
    accepted = code == 0 and result.get("complete_goal_proven") is True
    save(OUT / "receipt.json", dict(
        original_outer_exit_unknown=True, controller_actual_wait=0,
        completion_audit_actual_wait=code, original_test_records_unchanged=unchanged,
        cpu_receipt_sha256=sha(P / "cpu-accepted.json"),
        metal_receipt_sha256=sha(P / "metal-accepted.json"),
        completion_report_sha256=sha(report) if report.exists() else None,
        driver_sha256=sha(Path(__file__)), complete_goal_proven=accepted,
        missing_gates=result.get("missing_gates")))
    assert accepted, "Original-scope completion audit did not pass"

if __name__ == "__main__":
    main()
