"""Synthetic1751 controls: identity selection does not weaken terminal rejection."""
import copy,json,os,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_confirmation_profile as profile
import mam_confirmation_terminal as terminal
import mam_launch_confirmation_run as launch
import mam_confirmation_backup as backup
import test_mam_confirmation_terminal as previous
from test_mam_confirmation_terminal import test_bad_terminal_evidence_rejected

pytestmark=pytest.mark.skipif(profile.REPLICATE!=1751,reason='explicit1751 subprocess required')
E=Path(__file__).resolve().parents[1]/'mpi-evidence'

@pytest.fixture
def controls(tmp_path,monkeypatch):
    identity=json.loads((E/'confirmation-topology-v1-seed1751/identity.json').read_text())
    protocol=json.loads((E/'confirmation-readiness-v1-seed1751/proposed-protocol.json').read_text())
    admission=json.loads((E/'confirmation-run-v1-seed1750/admission.json').read_text())
    admission.update(replicate=1751,label=identity['label'],identity_sha256=launch.IDENTITY_SHA,
        protocol_sha256=launch.sha(E/'confirmation-readiness-v1-seed1751/proposed-protocol.json'),
        readiness_sha256=launch.READINESS,resources=protocol['resources'],source_catalog=terminal.SOURCE_CATALOG)
    for k in ['model_sha256','instance_sha256','plan_sha256','executable_sha256']:admission[k]=identity[k]
    admission['launch_options']['application']=launch.launch_options(identity,Path('/unused'))['application']
    p=tmp_path/launch.CASE;p.mkdir()
    for n,v in [('identity',identity),('protocol',protocol),('admission',admission)]:
        (p/(n+'.json')).write_text(json.dumps(v,indent=2)+'\n')
    monkeypatch.setattr(previous,'E',tmp_path)
    result=previous.controls.__wrapped__()
    for host in result[-1]:
        for rank in host['ranks'].values():
            for key in ['started_json','done_json']:
                v=json.loads(rank[key]);v['replicate']=1751;rank[key]=json.dumps(v)
    return result


def test_selected_full_terminal_checks_and_backup_location(controls):
    result=terminal.audit(*controls)
    assert result['replicate']==1751 and len(result['ranks'])==32
    assert result['terminal_resource_audit_passed'] and not result['scientific_acceptance']
    assert backup.DEST=='hk-prod-model-ae08-82' and backup.DEST_IP=='192.168.30.82'
    assert backup.ROOT==controls[0]['output'] and backup.BACKUP_DIRECTORY=='backup82'
    p=launch.protocol(controls[0],{'counts_sha256':terminal.COUNTS_SHA})
    assert p==controls[1]
    assert p['maximum_neural_runs']==1 and p['budgets']['simulation_seconds']==64800
    assert p['resources']['memory_bytes_per_service']==256*2**30
    assert p['raw_backup']['minimum_free_bytes']==128*2**30


def test_old1750_rank_receipt_rejected(controls):
    rank=controls[-1][0]['ranks']['0'];value=json.loads(rank['done_json'])
    value['replicate']=1750;rank['done_json']=json.dumps(value)
    with pytest.raises(ValueError,match='rank wrapper identity'):terminal.audit(*controls)
