import hashlib
import json
from pathlib import Path
import sys
import time
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_rust_terminal_sync as sync


def fixture(tmp_path):
    output=tmp_path/'output';output.mkdir()
    runtime=dict(schema='b2-mpi-runtime-v0',ranks=32,plan_sha256=sync.PLAN_SHA)
    summary=dict(schema='b2-result-dump-v4',population_count=254,neuron_count=4129924,
                 final_time_seconds=100.5,dump_bytes=8,event_dump_bytes=8,mpi=runtime)
    for name,value in [('summary.json',summary),('mpi-runtime.json',runtime)]:
        (output/name).write_text(json.dumps(value))
    for name in ['results.bin','events.bin']:(output/name).write_bytes(b'fixture!')
    return output


def test_sync_preserves_all_output_bytes_and_never_claims_binary_audit(tmp_path):
    output=fixture(tmp_path);before={p.name:p.read_bytes() for p in output.iterdir()}
    result=sync.sync_outputs(output)
    assert result['file_sync_calls']==4 and result['metadata_readback_verified']
    assert not result['binary_contents_independently_audited']
    assert {p.name:p.read_bytes() for p in output.iterdir()}==before
    assert result['files']['summary.json']['sha256']==hashlib.sha256(before['summary.json']).hexdigest()


@pytest.mark.parametrize('fault,match',[('size','binary size'),('runtime','runtime identity'),('spool','not terminal'),('symlink','')])
def test_invalid_output_refuses_completion(tmp_path,fault,match):
    output=fixture(tmp_path)
    if fault=='size':(output/'events.bin').write_bytes(b'short')
    elif fault=='runtime':
        p=output/'mpi-runtime.json';value=json.loads(p.read_text());value['ranks']=31;p.write_text(json.dumps(value))
    elif fault=='spool':(output/'spike-spool').mkdir()
    else:
        p=output/'events.bin';p.unlink();p.symlink_to(output/'results.bin')
    with pytest.raises((ValueError,OSError),match=match):sync.sync_outputs(output)


def test_sync_error_propagates_without_done_receipt(tmp_path,monkeypatch):
    output=fixture(tmp_path);receipts=tmp_path/'receipts';receipts.mkdir()
    def fail(_):raise OSError('injected sync failure')
    monkeypatch.setattr(sync.os,'fsync',fail)
    with pytest.raises(OSError,match='injected'):
        sync.finish_child(0,0,output,receipts,dict(rank=0),time.monotonic())
    assert not (receipts/'rank0.done.json').exists()


def test_failed_executable_does_not_touch_or_accept_outputs(tmp_path):
    output=fixture(tmp_path);receipts=tmp_path/'receipts';receipts.mkdir()
    with pytest.raises(RuntimeError,match='executable failed'):
        sync.finish_child(7,0,output,receipts,dict(rank=0),time.monotonic())
    assert not (receipts/'rank0.done.json').exists()
    assert json.loads((receipts/'rank0.failed.json').read_text())['returncode']==7


def test_nonleader_has_no_leader_output_claim(tmp_path):
    receipts=tmp_path/'receipts';receipts.mkdir()
    result=sync.finish_child(0,8,tmp_path/'absent',receipts,dict(rank=8),time.monotonic())
    assert result['leader_output_synchronization'] is None and not result['whole_job_accepted']


def test_receipt_sync_failure_keeps_failure_evidence(tmp_path,monkeypatch):
    output=fixture(tmp_path);receipts=tmp_path/'receipts';receipts.mkdir()
    original=sync.os.fsync;calls=0
    def fail_receipt(descriptor):
        nonlocal calls
        calls+=1
        if calls==7:raise OSError('receipt sync failed')
        original(descriptor)
    monkeypatch.setattr(sync.os,'fsync',fail_receipt)
    with pytest.raises(OSError,match='receipt sync failed'):
        sync.finish_child(0,0,output,receipts,dict(rank=0),time.monotonic())
    assert (receipts/'rank0.failed.json').exists()
    assert not json.loads((receipts/'rank0.done.json').read_text())['whole_job_accepted']


def test_real_child_preserves_pmi_channel_and_closes_unrelated_descriptor(monkeypatch):
    import socket,os
    unused,unused_peer=socket.socketpair()
    with unused,unused_peer:
        # The unrelated descriptor is intentionally inheritable, so this test
        # also rejects a broad close_fds=False workaround.
        unused.set_inheritable(True)
        parent,child=socket.socketpair()
        try:
            monkeypatch.setenv('PMI_FD',str(child.fileno()))
            code=f'''import os
os.write(int(os.environ['PMI_FD']),b'PMI channel intact')
try:os.fstat({unused.fileno()})
except OSError:pass
else:raise RuntimeError('unrelated descriptor leaked')
'''
            result=sync.run_child([sys.executable,'-c',code])
            parent.settimeout(1)
            assert result.returncode==0 and parent.recv(64)==b'PMI channel intact'
        finally:parent.close();child.close()


@pytest.mark.parametrize('value',[None,'','-1','0','1','2','3.0','+3',' 3','not-a-fd'])
def test_invalid_pmi_descriptor_rejected(value):
    environment={} if value is None else {'PMI_FD':value}
    with pytest.raises(ValueError):sync.pmi_pass_fds(environment)


def test_closed_pmi_descriptor_rejected():
    import os
    read_fd,write_fd=os.pipe();os.close(read_fd);os.close(write_fd)
    with pytest.raises(OSError):sync.pmi_pass_fds({'PMI_FD':str(read_fd)})
