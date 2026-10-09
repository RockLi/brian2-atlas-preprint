"""One-time read-only evidence validation after E1-large terminal; no model calls."""
import datetime, hashlib, json, pathlib, socket, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/e1-large-validation-job-r1'
SOURCE=ROOT/'tools/validate_e1_large_run.py'
EXPECTED='fb1a6d7542b8fe7c8556c3f1513f95d2fbb22dc1bb213171b673c7c585ce2a6d'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def main():
    assert socket.gethostname()=='rock-mac-studio-1.local'
    assert sha(SOURCE)==EXPECTED
    OUT.mkdir(parents=True,exist_ok=False)
    save('launch.json',dict(validator_sha256=EXPECTED,waiter_sha256=sha(pathlib.Path(__file__)),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        role='read-only identity/count/timing validation, no engine or oracle execution',
        raw_data_remains_remote=True,wait_cap_s=21600,validation_cap_s=1200))
    start=time.monotonic(); prior=ROOT/'evidence/remote-e1-large-v1/terminal.json'
    while not prior.exists():
        if time.monotonic()-start>=21600:
            save('terminal.json',dict(status='prerequisite_pending'));return 2
        time.sleep(15)
    assert sha(SOURCE)==EXPECTED
    output=ROOT/'evidence/e1-large-validation-full-r1.json'
    report=ROOT/'evidence/e1-large-validation-full-r1.zh.md'
    command=[sys.executable,str(SOURCE),'--root',str(ROOT),'--run','evidence/remote-e1-large-v1',
             '--mode','full','--output',str(output),'--report',str(report)]
    save('barrier.json',dict(status='released',waited_s=time.monotonic()-start,prerequisite_sha256=sha(prior)))
    begin=time.monotonic()
    with (OUT/'validator.log').open('x') as log:
        try:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
            terminal=dict(status='validator_exited',exit_code=result.returncode)
        except subprocess.TimeoutExpired:
            terminal=dict(status='validator_timeout',exit_code=None)
    terminal.update(elapsed_s=time.monotonic()-begin,validator_sha256=sha(SOURCE))
    if output.exists():
        data=json.loads(output.read_text());terminal.update(output_sha256=sha(output),validation_status=data.get('status'),validation_errors=data.get('errors'))
    if report.exists():terminal['report_sha256']=sha(report)
    save('terminal.json',terminal)
    return 0 if terminal.get('exit_code')==0 else 1
if __name__=='__main__':raise SystemExit(main())
