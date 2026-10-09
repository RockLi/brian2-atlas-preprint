"""Stop only the verified, idle r2 waiter so an approved r3 plan can replace it.
This control has no process-group signals and never targets scientific workers.
"""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, signal, time, traceback
import psutil

ROOT = Path("/atlas-home/0004/workspace/atlas-training-evaluation/20261004-r1")
PID = 23312
CREATED = 1791113681.165096
EXECUTABLE = "/opt/homebrew/Cellar/python@3.12/3.12.14/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python"
SPEC_SHA = "58aaddc787d092d79ac93c35359c5b894320322dcd5bdca914fc7c0795bdd7c8"
FREEZE_SHA = "dbd9575e9e6bd949e3167800500e15cb22b8b0a22b7dbb4ead415de5eec214ae"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def exact_process(process):
    if not process.is_running() or abs(process.create_time() - CREATED) > 1e-6:
        raise RuntimeError("Idle waiter process identity changed")
    ids = process.uids()
    if ids.real != os.getuid() or ids.effective != os.geteuid():
        raise RuntimeError("Waiter user identity changed")
    argv = process.cmdline()
    wanted = [EXECUTABLE, str(ROOT / "tools/run_extended_followup_r2.py"), "--allow-run", "--spec",
              str(ROOT / "protocol/extended-followup-r2.json")]
    if argv != wanted:
        raise RuntimeError("Waiter argv differs from the authorized exact target")
    return dict(pid=process.pid, create_time=process.create_time(), argv=argv,
                real_uid=ids.real, effective_uid=ids.effective)

def preexecution_only(process):
    identity = exact_process(process)
    out = ROOT / "evidence/extended-followup-r2"
    files = sorted(p.name for p in out.iterdir())
    if files != ["barrier-wait.json", "freeze.json"]:
        raise RuntimeError("r2 no longer has only frozen idle-wait evidence")
    if process.children(recursive=True):
        raise RuntimeError("r2 has descendants; cannot replace an idle-only waiter")
    if digest(ROOT / "protocol/extended-followup-r2.json") != SPEC_SHA or digest(out / "freeze.json") != FREEZE_SHA:
        raise RuntimeError("r2 frozen source/plan identity changed")
    spec = json.loads((ROOT / "protocol/extended-followup-r2.json").read_text())
    freeze = json.loads((out / "freeze.json").read_text())
    if digest(ROOT / "tools/run_extended_followup_r2.py") != spec["coordinator_sha256"]:
        raise RuntimeError("Actual idle coordinator source differs from its frozen plan")
    if freeze.get("spec_sha256") != SPEC_SHA or freeze.get("spec") != spec:
        raise RuntimeError("r2 freeze does not bind the exact plan")
    prerequisites = {p: (ROOT/p).exists() for p in spec["prerequisites"]}
    if all(prerequisites.values()):
        raise RuntimeError("Prerequisites became ready; decline idle replacement")
    return dict(identity=identity, output_files=files, prerequisites=prerequisites,
                descendants=[], stages_started=0)

def stop_idle(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    record = dict(schema="extended-r2-to-r3-idle-transition-r1", status="checking",
                  authorization="User explicitly authorized continuing the evaluation; replace only the idle waiter to add isolated ARM64 build, qualification and35fresh E1 slots",
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), signals=[])
    process = None
    suspended = False
    try:
        process = psutil.Process(PID)
        record["before_suspend"] = preexecution_only(process)
        process.suspend()
        suspended = True
        record["signals"].append("SIGSTOP exact PID/create_time")
        stop_deadline = time.monotonic() + 5
        while process.status() != psutil.STATUS_STOPPED:
            exact_process(process)
            if time.monotonic() >= stop_deadline:
                raise RuntimeError("Waiter did not enter STOPPED state; restore it and decline replacement")
            time.sleep(.02)
        record["stopped_state_confirmed"] = True
        record["while_suspended"] = preexecution_only(process)
        process.send_signal(signal.SIGINT)
        record["signals"].append("SIGINT exact PID/create_time")
        process.resume()
        suspended = False
        record["signals"].append("SIGCONT exact PID/create_time")
        until = time.monotonic()+20
        while True:
            try:
                alive = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                alive = False
            if not alive:
                break
            if time.monotonic() >= until:
                raise RuntimeError("Idle waiter did not finish controlled shutdown; do not launch replacement")
            time.sleep(.2)
        oldout = ROOT/"evidence/extended-followup-r2"
        terminal = json.loads((oldout/"terminal.json").read_text())
        rows = terminal.get("records", [])
        if terminal.get("status") != "coordinator_stopped" or terminal.get("elapsed_execution_s") is not None:
            raise RuntimeError("r2 terminal is not preexecution-only")
        if len(rows) != 8 or any(row.get("status") != "not_launched" for row in rows):
            raise RuntimeError("r2 has attempted stages; stop replacement")
        if sorted(p.name for p in oldout.iterdir()) != ["barrier-wait.json", "freeze.json", "terminal.json"]:
            raise RuntimeError("Unexpected r2 phase evidence; stop replacement")
        with (ROOT/"evidence/extended-followup-r1.lock").open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(lock,fcntl.LOCK_UN)
        record.update(status="idle_r2_closed_zero_stages", old_terminal_sha256=digest(oldout/"terminal.json"),
                      old_spec_sha256=SPEC_SHA, old_freeze_sha256=FREEZE_SHA, old_stage_count=8,
                      started_stages=0, family_lock_released=True,
                      scientific_worker_signals_sent=False, replacement_not_started_by_this_control=True)
    except BaseException as error:
        record.update(status="transition_failed", error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if suspended:
            try:
                exact_process(process)
                process.resume()
                record["signals"].append("SIGCONT after aborted check; original waiter restored")
            except Exception as error:
                record["resume_error"]=repr(error)
        record["ended_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
        with (out/"transition.json").open("x") as stream:
            json.dump(record,stream,indent=2);stream.write("\n")
    return record

# Deliberately no auto-entry point. Root invokes stop_idle only after the complete
# replacement sources and frozen stage plan have been checked.
