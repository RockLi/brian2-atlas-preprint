"""Stdlib fake-data controls; never invokes compiler, runtime, oracle, or model."""
import copy, hashlib, importlib.util, json, os
from pathlib import Path
import socket, struct, sys, tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import run_arm64_qualification_stage_r2 as m
checks=[]
def note(name): checks.append(dict(name=name,passed=True))
def put(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data,bytes) else data.encode())
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix="arm64-stage-controls-") as directory:
    r=Path(directory)
    src="snapshot/brian2-rust/src/train_main.rs"
    helper="tools/helper.py"
    driver="tools/build_arm64_runtime_r2.py"
    put(r/src,"frozen rust")
    put(r/helper,"frozen helper")
    put(r/driver,"frozen build driver")
    put(r/"runtime/b2-train","old x86 bytes")
    native=struct.pack("<8I",0xFEEDFACF,0x0100000C,0,2,0,0,0,0)
    put(r/m.RUNNER,native);(r/m.RUNNER).chmod(0o755)
    contract=dict(source_identities={src:h(r/src)},helper_identities={helper:h(r/helper)},
        driver_sha256=h(r/driver),original_runtime_sha256=h(r/"runtime/b2-train"))
    put(r/m.CONTRACT,json.dumps(contract))
    producer=dict(schema="runtime-arm64-build-r2",status="built_unqualified",source_integrity_after_exit=True,
        runtime_path=m.RUNNER,runtime_architecture="arm64-only",runtime_sha256=h(r/m.RUNNER),
        original_runtime_sha256=contract["original_runtime_sha256"],
        sources_before=contract["source_identities"],sources_after=contract["source_identities"],
        sources_final=contract["source_identities"])
    put(r/m.PRODUCER,json.dumps(producer))
    contract_sha=h(r/m.CONTRACT);frozen={helper:h(r/helper)}
    def verify():return m.verify_producer(r,expected_contract_sha=contract_sha,frozen=frozen)
    result=verify()
    assert result["runner"]["gate_status"]=="passed"
    note("valid_fake_producer_is_identity_only")
    for field,value in [("status","build_or_preflight_failed"),("source_integrity_after_exit",False),
                        ("runtime_path","runtime/b2-train"),("runtime_architecture","x86_64"),
                        ("runtime_sha256",None),("sources_after",{}),("original_runtime_sha256","0"*64)]:
        bad=copy.deepcopy(producer);bad[field]=value;put(r/m.PRODUCER,json.dumps(bad))
        try:verify()
        except (RuntimeError,ValueError):note("reject_"+field)
        else:raise AssertionError(field)
    put(r/m.PRODUCER,json.dumps(producer))
    for relative in (src,helper,driver,m.RUNNER,"runtime/b2-train",m.CONTRACT):
        p=r/relative;old=p.read_bytes();put(p,old+b"changed")
        try:verify()
        except (RuntimeError,ValueError):note("reject_changed_"+relative)
        else:raise AssertionError(relative)
        put(p,old)
    for stage in m.SLOTS:
        command=m.stage_command(stage,r)
        assert str(r/m.RUNNER) in command
        assert str(r/"runtime/b2-train") not in command
        note("explicit_new_runner_"+stage)
    # Exercise main only with a synthetic producer refusal. No child may launch.
    original_root=m.ROOT;original_verify=m.verify_producer;original_run=m.subprocess.Popen
    original_argv=sys.argv;original_host=m.socket.gethostname
    m.ROOT=r;m.socket.gethostname=lambda:m.HOST
    def reject(*a,**k):raise RuntimeError("synthetic producer rejection")
    def never_run(*a,**k):raise AssertionError("No child allowed in controls")
    m.verify_producer=reject;m.subprocess.Popen=never_run
    try:
        for stage,count in (("dense",4),("metal-prepare",1),("metal-run",2)):
            sys.argv=["control","--allow-run","--stage",stage]
            assert m.main()==1
            path=r/"evidence/arm64-r2/stage-gates"/stage/"terminal.json"
            row=json.loads(path.read_text())
            assert row["child_launched"] is False and row["finite_slots"]==count
            assert len(row["slots"])==count and all(s["status"]=="not_launched" for s in row["slots"])
            note("full_denominator_on_preflight_refusal_"+stage)
            before=path.read_bytes()
            try:m.main()
            except FileExistsError:pass
            else:raise AssertionError("repeat should not overwrite")
            assert path.read_bytes()==before
            note("repeat_preserves_failure_"+stage)
    finally:
        m.ROOT=original_root;m.verify_producer=original_verify;m.subprocess.Popen=original_run
        sys.argv=original_argv;m.socket.gethostname=original_host
report=dict(schema="arm64-qualification-stage-controls-r1",status="passed",checks=checks,check_count=len(checks),
            source_sha256=h(ROOT/"tools/run_arm64_qualification_stage_r2.py"),
            test_source_sha256=h(Path(__file__)),model_executed=False,compiler_executed=False,
            scope="stdlib fake files/producer receipts and explicit launch trap; no real model or producer read")
out=ROOT/"evidence/runtime-arm64-preparation-r2/stage-gate-control-tests-final.json"
with out.open("x") as stream:json.dump(report,stream,indent=2);stream.write("\n")
print(json.dumps(dict(status=report["status"],checks=len(checks),output=str(out))))
