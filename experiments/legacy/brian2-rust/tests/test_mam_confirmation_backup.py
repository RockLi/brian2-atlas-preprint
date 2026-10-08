"""1750 backup input bindings and retained protocol integration; no transfer."""
import ast,gzip,hashlib,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_confirmation_backup as backup
import mam_cluster_raw_backup as original
from test_mam_confirmation_terminal import controls
from test_mam_confirmation_raw import bound
from test_mam_launch_confirmation_raw import publication
from test_mam_confirmation_analysis import accepted


def test_pending_raw_gate_neither_connects_nor_creates_output(tmp_path,monkeypatch):
    monkeypatch.setattr(backup,'remote',lambda *a,**k:pytest.fail('unexpected connection'))
    result=backup.run(tmp_path/'evidence',tmp_path/'t7')
    assert not result['ready'] and not result['backup_started'] and list(tmp_path.iterdir())==[]


def test_verified_source_includes_all_four_output_files(accepted,tmp_path,monkeypatch):
    case,digest=accepted
    # Keep the fixture's exact contents, under the fixed campaign basename.
    destination=tmp_path/backup.CASE;case.rename(destination)
    monkeypatch.setattr(backup,'remote',lambda *a,**k:pytest.fail('source gate must be local'))
    proof=backup.source_gate(tmp_path)
    assert proof['completion_sha256']==digest and proof['identity_sha256']==backup.IDENTITY_SHA
    assert [r['path'] for r in proof['catalog']['files']]==['events.bin','mpi-runtime.json','results.bin','summary.json']
    with gzip.open(destination/'terminal/host-0.json.gz','rb') as stream:host=json.load(stream)
    for r in proof['catalog']['files']:
        if r['path'].endswith('.json'):
            raw=host['leader_outputs']['runtime_json' if r['path']=='mpi-runtime.json' else 'summary_json'].encode()
            assert r['bytes']==len(raw) and r['sha256']==hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('fault',['legacy_source','wrong_plan','pending','size','capacity'])
def test_mixed_identity_incomplete_and_oversized_catalog_rejected(controls,publication,fault):
    identity=controls[0];raw=publication[0];raw.update(raw_output_audit_passed=True,audit_guard_passed=True,pending_guard_acceptance=False)
    terminal=publication[3]['terminal'];leader=controls[4][0]['leader_outputs']
    if fault=='legacy_source':identity['output']=original.ROOT
    elif fault=='wrong_plan':raw['plan_sha256']='0'*64
    elif fault=='pending':raw['audit_guard_passed']=False
    elif fault=='size':terminal['output_bytes']['summary.json']+=1
    else:
        for n in ['results.bin','events.bin']:raw['dump_bytes'][n]=terminal['output_bytes'][n]=64*2**30
    with pytest.raises(ValueError):backup.catalog_for(identity,raw,terminal,leader)


def test_reuses_exact_readback_function_and_pinned_helper_bytes():
    assert backup.readback is original.readback
    for name,digest in backup.HELPERS.items():
        assert hashlib.sha256(Path(backup.__file__).with_name(name).read_bytes()).hexdigest()==digest


def test_fixed_destination_preserves_old_backup_and_resource_limits():
    assert backup.BASES[backup.DEST]!=original.BASES[original.DEST]
    assert backup.BASES[backup.DEST]=='/atlas-home/0003/backups/brian2-mpi/confirmation-run-v1-seed1750'
    for node,volume,reserve in [(backup.SOURCE,'/data/brick2','1280'),(backup.DEST,'/','128')]:
        command=backup.command(node,'readback',1800,['python3','readback'])
        assert command[command.index('--volume')+1]==volume and command[command.index('--min-free-gib')+1]==reserve
        assert '--property=MemoryMax=4096M' in command and '--property=MemorySwapMax=0' in command
        assert '--property=CPUQuota=200%' in command and '--property=AllowedCPUs=8-9' in command
    assert backup.TOTAL==7200 and backup.TRANSFER+backup.READBACK<backup.TOTAL


def test_archive_does_not_visit_previously_collected_large_terminal_file(tmp_path):
    case=tmp_path/'case';output=case/'backup24';output.mkdir(parents=True)
    (case/'terminal').mkdir();(case/'terminal/host-0.json.gz').write_bytes(b'x'*(2*2**20))
    (output/'intent.json').write_text('{}\n');t7=tmp_path/'t7'
    backup.archive_backup(output,t7)
    assert (t7/'artifacts'/backup.CASE/'backup24/intent.json').read_bytes()==b'{}\n'
    assert not (t7/'artifacts'/backup.CASE/'terminal').exists()
    (output/'intent.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='archive differs'):backup.archive_backup(output,t7)


def test_terminal_collection_checks_source_and_catalog_without_tokens():
    code=backup.terminal_code(backup.SOURCE,backup.HELPERS,dict(schema=backup.SCHEMA,files=[]))
    ast.parse(code)
    assert 'ready.json' not in code and 'catalog_verified' in code
    assert 'source_sha256' in code and backup.PREFIX in code
