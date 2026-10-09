"""Bounded one-time read-only A1-v4 audit after its terminal receipt."""
import datetime, hashlib, json, os, pathlib, socket, subprocess, sys, time, traceback
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT/'evidence/a1-v4-validation-job-r1'
SOURCE = ROOT/'tools/validate_a1_v4.py'
EXPECTED = 'e4371d0d1518c1bd2fd8ebb876b08a6c99b63d4abdde04a2c9d48b091c31314d'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def save(name,value):
    path=OUT/name
    tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)
def main():
    assert socket.gethostname()=='rock-mac-studio-1.local'
    assert sha(SOURCE)==EXPECTED
    OUT.mkdir(parents=True,exist_ok=False)
    save('launch.json',dict(validator_sha256=EXPECTED,waiter_sha256=sha(pathlib.Path(__file__)),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        role='read-only evidence audit: no model, oracle, or held-out data reads',
        wait_cap_s=129600,validation_cap_s=3600))
    start=time.monotonic()
    prior=ROOT/'evidence/a1-queue-v4/terminal.json'
    terminal=dict(status='coordinator_error')
    try:
        while not prior.exists():
            if time.monotonic()-start>=129600:
                terminal=dict(status='prerequisite_pending')
                return 2
            time.sleep(15)
        assert sha(SOURCE)==EXPECTED
        save('barrier.json',dict(status='released',waited_s=time.monotonic()-start,
             prerequisite_sha256=sha(prior)))
        output=ROOT/'evidence/a1-v4-validation-r1.json'
        command=[sys.executable,str(SOURCE),'--root',str(ROOT),'--mode','full','--output',str(output)]
        begin=time.monotonic()
        with (OUT/'validator.log').open('x') as log:
            try:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    timeout=3600,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
                terminal=dict(status='validator_exited',exit_code=result.returncode)
            except subprocess.TimeoutExpired:
                terminal=dict(status='validator_timeout',exit_code=None)
        terminal.update(elapsed_s=time.monotonic()-begin,validator_sha256=sha(SOURCE))
        if output.exists():
            data=json.loads(output.read_text())
            terminal.update(output_sha256=sha(output),validation_status=data.get('status'),
                validation_errors=data.get('errors'))
        return 0 if terminal.get('exit_code')==0 else 1
    except Exception as error:
        terminal.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        terminal.update(complete_evaluation=False)
        save('terminal.json',terminal)
if __name__=='__main__': raise SystemExit(main())
