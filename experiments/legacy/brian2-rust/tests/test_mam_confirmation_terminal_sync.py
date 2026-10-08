import ast
import hashlib
import json
from pathlib import Path
import sys
import time
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_confirmation_terminal_sync as sync

E=Path(__file__).resolve().parents[1]/'mpi-evidence'


def fixture(tmp_path, plan='new-plan'):
    output=tmp_path/'output';output.mkdir()
    runtime=dict(schema='b2-mpi-runtime-v0',ranks=32,plan_sha256=plan)
    summary=dict(schema='b2-result-dump-v4',population_count=254,neuron_count=4129924,
                 final_time_seconds=100.5,dump_bytes=8,event_dump_bytes=8,mpi=runtime)
    for n,v in [('summary.json',summary),('mpi-runtime.json',runtime)]:
        (output/n).write_text(json.dumps(v))
    for n in ['results.bin','events.bin']:(output/n).write_bytes(b'fixture!')
    return output


def test_real_identity_preserves_unsigned_runtime_key_and_pins_paths():
    v=sync.load_identity(E/'confirmation-topology-v1-seed1750/identity.json')
    assert v['replicate']==1750 and v['random_keys']['runtime_input']==16757147634959265529
    assert v['files']['mpi/b2-mpi']['bytes']==8742208
    assert v['output'].startswith('/data/brick2/') and 'replicate1750' in v['analysis']


def test_identity_cannot_relabel_legacy_outputs(tmp_path):
    v=json.loads((E/'confirmation-topology-v1-seed1750/identity.json').read_text())
    v['replicate']=1729;p=tmp_path/'identity.json';p.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='identity digest'):sync.load_identity(p)


def test_wrong_plan_creates_failure_and_never_done(tmp_path):
    out=fixture(tmp_path,'legacy-plan');receipts=tmp_path/'receipts';receipts.mkdir()
    with pytest.raises(ValueError,match='runtime identity'):
        sync.finish_child(0,0,out,receipts,dict(rank=0,plan_sha256='new-plan'),time.monotonic())
    assert (receipts/'rank0.failed.json').exists() and not (receipts/'rank0.done.json').exists()


def test_correct_plan_sync_retains_identity_without_claiming_audit(tmp_path):
    out=fixture(tmp_path);receipts=tmp_path/'receipts';receipts.mkdir()
    before={p.name:p.read_bytes() for p in out.iterdir()}
    v=sync.finish_child(0,0,out,receipts,dict(rank=0,replicate=1750,plan_sha256='new-plan'),time.monotonic())
    assert v['replicate']==1750 and not v['scientific_acceptance'] and not v['whole_job_accepted']
    assert not v['leader_output_synchronization']['binary_contents_independently_audited']
    assert {p.name:p.read_bytes() for p in out.iterdir()}==before


def test_rank_input_rejects_same_size_changed_content_and_symlink(tmp_path):
    p=tmp_path/'input';p.write_bytes(b'new')
    item=dict(bytes=3,sha256=hashlib.sha256(b'old').hexdigest())
    with pytest.raises(ValueError,match='digest'):sync.verify_input(p,item,tmp_path)
    q=tmp_path/'link';q.symlink_to(p)
    with pytest.raises(ValueError,match='location'):sync.verify_input(q,item,tmp_path)


def test_fd_and_durability_helpers_unchanged_from_tested_wrapper():
    def functions(path):
        return {x.name:ast.dump(x) for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef)}
    p=Path(sync.__file__);old=functions(p.with_name('mam_rust_terminal_sync.py'));new=functions(p)
    for name in ['pmi_pass_fds','run_child','parse','stamp','sync_directory','receipt']:
        assert new[name]==old[name]


def test_real_pmi_channel_survives(monkeypatch):
    import os,socket
    with socket.socketpair()[0] as unused:
        unused.set_inheritable(True)
        a,b=socket.socketpair()
        with a,b:
            monkeypatch.setenv('PMI_FD',str(b.fileno()))
            code=f"import os\nos.write(int(os.environ['PMI_FD']),b'PMI')\ntry:os.fstat({unused.fileno()})\nexcept OSError:pass\nelse:raise RuntimeError('unrelated descriptor leaked')"
            r=sync.run_child([sys.executable,'-c',code]);a.settimeout(1)
            assert r.returncode==0 and a.recv(3)==b'PMI'


def test_second_real_identity_and_cross_replica_tampering(tmp_path):
    first=E/'confirmation-topology-v1-seed1750/identity.json'
    second=E/'confirmation-topology-v1-seed1751/identity.json'
    assert sync.load_identity(first)['replicate']==1750
    value=sync.load_identity(second)
    assert value['replicate']==1751
    assert value['random_keys']['runtime_input']==990883333388651797
    for change in ('replicate','output'):
        bad=json.loads(first.read_text())
        bad[change]=value[change]
        target=tmp_path/(change+'.json');target.write_text(json.dumps(bad,indent=2)+'\n')
        with pytest.raises(ValueError,match='identity digest'):sync.load_identity(target)
