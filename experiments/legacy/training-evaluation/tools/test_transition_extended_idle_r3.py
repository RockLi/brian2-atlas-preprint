"""Pure fake-process controls: no real process API or signal is invoked."""
import datetime, hashlib, importlib.util, json, os
from pathlib import Path
import signal, sys, tempfile, types

ROOT=Path(__file__).resolve().parents[1]
fake_module=types.ModuleType("psutil")
class Error(Exception):pass
class NoSuchProcess(Error):pass
class ZombieProcess(Error):pass
fake_module.Error=Error
fake_module.NoSuchProcess=NoSuchProcess
fake_module.ZombieProcess=ZombieProcess
fake_module.STATUS_ZOMBIE="zombie"
fake_module.STATUS_STOPPED="stopped"
saved_module=sys.modules.get("psutil")
sys.modules["psutil"]=fake_module
spec=importlib.util.spec_from_file_location("idle_transition_under_test",ROOT/"tools/transition_extended_idle_r3.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def note(n):checks.append(dict(name=n,passed=True))
def fixture(r):
    m.ROOT=r
    (r/"protocol").mkdir(parents=True)
    out=r/"evidence/extended-followup-r2";out.mkdir(parents=True)
    (r/"tools").mkdir()
    source=r/"tools/run_extended_followup_r2.py";source.write_text("fake coordinator source")
    data=dict(prerequisites=["evidence/a1/terminal.json","evidence/audit/terminal.json"],coordinator_sha256=sha(source))
    (r/"protocol/extended-followup-r2.json").write_text(json.dumps(data))
    m.SPEC_SHA=sha(r/"protocol/extended-followup-r2.json")
    (out/"freeze.json").write_text(json.dumps(dict(spec=data,spec_sha256=m.SPEC_SHA)))
    (out/"barrier-wait.json").write_text('{}')
    m.SPEC_SHA=sha(r/"protocol/extended-followup-r2.json");m.FREEZE_SHA=sha(out/"freeze.json")
    return out
class FakeProcess:
    pid=23312
    def __init__(self,r,mode):
        self.r=r;self.mode=mode;self.alive=True;self.stopped=False;self.pending=False;self.events=[]
    def is_running(self):return self.alive
    def create_time(self):return m.CREATED+(1 if self.mode=="pid_reused" else 0)
    def uids(self):return types.SimpleNamespace(real=os.getuid(),effective=os.geteuid()+(1 if self.mode=="wrong_uid" else 0))
    def cmdline(self):
        a=[m.EXECUTABLE,str(self.r/"tools/run_extended_followup_r2.py"),"--allow-run","--spec",str(self.r/"protocol/extended-followup-r2.json")]
        if self.mode=="wrong_argv":a[1]="scientific_worker.py"
        if self.mode=="wrong_executable":a[0]="unexpected-python"
        return a
    def children(self,recursive=False):
        return [object()] if self.mode=="has_child" or (self.mode=="child_after_stop" and self.stopped) else []
    def status(self):
        if self.mode=="exit_status_race" and self.pending:raise NoSuchProcess("fake exit race")
        return "stopped" if self.stopped else "sleeping"
    def suspend(self):
        self.events.append("STOP");self.stopped=True
        if self.mode=="stage_after_stop":(self.r/"evidence/extended-followup-r2/progress.json").write_text("[]")
    def send_signal(self,s):
        assert s==signal.SIGINT
        self.events.append("INT");self.pending=True
    def resume(self):
        self.events.append("CONT");self.stopped=False
        if self.pending:
            rows=[dict(name=str(i),status="not_launched") for i in range(8)]
            elapsed=None
            if self.mode=="active_terminal":rows[0]["status"]="supervision_completed";elapsed=1
            p=self.r/"evidence/extended-followup-r2/terminal.json"
            p.write_text(json.dumps(dict(status="coordinator_stopped",elapsed_execution_s=elapsed,records=rows)))
            self.alive=self.mode=="exit_status_race"

try:
    with tempfile.TemporaryDirectory(prefix="idle-transition-controls-") as td:
        root=Path(td)
        modes=["valid","pid_reused","wrong_uid","wrong_argv","wrong_executable","changed_source","has_child","ready_prerequisites","already_started","stage_after_stop","child_after_stop","active_terminal","missing_process","exit_status_race"]
        for i,mode in enumerate(modes):
            r=root/str(i);out=fixture(r);process=FakeProcess(r,mode)
            def factory(pid,p=process,mode=mode):
                assert pid==23312
                if mode=="missing_process":raise NoSuchProcess("fake missing process")
                return p
            fake_module.Process=factory
            if mode=="ready_prerequisites":
                for rel in ("evidence/a1/terminal.json","evidence/audit/terminal.json"):
                    p=r/rel;p.parent.mkdir(parents=True);p.write_text("{}")
            if mode=="already_started":(out/"barrier.json").write_text("{}")
            if mode=="changed_source":(r/"tools/run_extended_followup_r2.py").write_text("changed fake source")
            folder=r/"evidence/transition"
            caught=None
            try:result=m.stop_idle(folder)
            except Exception as e:caught=e
            receipt=json.loads((folder/"transition.json").read_text())
            if mode in ("valid","exit_status_race"):
                assert caught is None and receipt["status"]=="idle_r2_closed_zero_stages"
                assert receipt["family_lock_released"] and receipt["started_stages"]==0
                assert receipt["stopped_state_confirmed"] is True
                assert process.events==["STOP","INT","CONT"]
            else:
                assert caught is not None and receipt["status"]=="transition_failed"
                if mode in ("stage_after_stop","child_after_stop"):
                    assert process.events==["STOP","CONT"] and not process.stopped and process.alive
                elif mode=="active_terminal":
                    assert process.events==["STOP","INT","CONT"]
                else:assert process.events==[]
            note(mode)
finally:
    if saved_module is None:sys.modules.pop("psutil",None)
    else:sys.modules["psutil"]=saved_module
report=dict(schema="idle-transition-controls-r1",status="passed",check_count=len(checks),checks=checks,
            process_api="fake psutil module only",real_signal_sent=False,real_process_inspected=False,
            model_executed=False,source_sha256=sha(ROOT/"tools/transition_extended_idle_r3.py"),
            test_sha256=sha(Path(__file__)),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
out=ROOT/"evidence/extended-preparation-r3";out.mkdir(parents=True,exist_ok=True)
with (out/"transition-control-tests-final.json").open("x") as stream:json.dump(report,stream,indent=2);stream.write("\n")
print(json.dumps(dict(status="passed",checks=len(checks),real_signal_sent=False)))
