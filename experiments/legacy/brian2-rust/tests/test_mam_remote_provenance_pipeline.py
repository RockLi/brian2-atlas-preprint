"""Synthetic remote terminal -> full raw gate -> science bundle provenance."""
import ast,gzip,hashlib,io,json,sys,tarfile
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_confirmation_raw as raw
import mam_launch_confirmation_raw as launch
import mam_confirmation_analysis as science
import mam_launch_confirmation_analysis as science_launch
import mam_confirmation_remote_terminal as terminal
from test_mam_confirmation_remote_terminal import controls as remote_controls
from test_mam_confirmation_terminal import controls
from test_mam_launch_confirmation_raw import publication

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

@pytest.fixture
def bound(tmp_path,remote_controls):
    case=tmp_path/'case';(case/'terminal').mkdir(parents=True)
    v,p,a,anchor,hosts=remote_controls
    for name,value in [('identity.json',v),('protocol.json',p),('admission.json',a),(terminal.PROVENANCE_NAME,anchor)]:write(case/name,value)
    files=[]
    for i,h in enumerate(hosts):
        path=case/'terminal'/f'host-{i}.json.gz';path.write_bytes(gzip.compress(json.dumps(h).encode(),mtime=0))
        files.append(dict(host=h['host'],file=path.name,bytes=path.stat().st_size,sha256=sha(path)))
    report=terminal.audit(*remote_controls)
    report.update(input_sha256=dict(admission=sha(case/'admission.json'),identity=sha(case/'identity.json'),protocol=sha(case/'protocol.json'),recovery=terminal.PROVENANCE_SHA),collection_files=files,verifier_sha256=sha(Path(terminal.__file__)),raw_binary_payloads_collected=False,automatic_retry=False)
    write(case/'terminal/report.json',report)
    binding=tmp_path/'binding.json';write(binding,dict(schema='b2-mam-confirmation-remote-raw-binding-v1',identity_sha256=raw.IDENTITY_SHA,case_id=raw.CASE,protocol_sha256=sha(case/'protocol.json'),files={n:sha(case/n) for n in raw.terminal_names(report)}))
    return case,binding,sha(binding)

@pytest.fixture
def accepted(bound,publication):
    case,binding,digest=bound;p,g,c,kw=publication
    _,identity,_,audit=raw.terminal_gate(*bound)
    folder=case/'raw';folder.mkdir();(folder/'binding.json').write_bytes(binding.read_bytes())
    worker_sha=sha(Path(raw.__file__));p.update(binding_sha256=digest,implementation_sha256=worker_sha)
    g['command']=launch.application(digest);c['command']=launch.command(digest)
    kw.update(binding_sha=digest,terminal=audit,identity=identity,raw_source_sha=worker_sha)
    rows={'pending.json':p,'guard.json':g,'controller.json':c,'admission.json':dict(source_catalog={'tools/mam_confirmation_raw.py':dict(sha256=worker_sha)}),'attempt.json':dict(binding_sha256=digest,automatic_retry=False),'intent.json':dict(binding_sha256=digest,automatic_retry=False,attempts=1,total_stage_seconds=7200)}
    for n,v in rows.items():write(folder/n,v)
    report=launch.publish(p,g,c,**kw);report['evidence_sha256']={n:sha(folder/n) for n in [*rows,'binding.json']};write(folder/'report.json',report)
    names=raw.terminal_names(json.loads((case/'terminal/report.json').read_text()))[:5]+['raw/report.json','raw/intent.json']
    completion=dict(schema='b2-mam-confirmation-engineering-completion-v1',case_id=raw.CASE,replicate=1750,identity_sha256=raw.IDENTITY_SHA,protocol_sha256=sha(case/'protocol.json'),terminal_resource_audit_passed=True,raw_output_audit_passed=True,scientific_acceptance=False,performance_cost_acceptance=False,input_sha256={n:sha(case/n) for n in names})
    write(case/'completion.json',completion)
    return case,sha(case/'completion.json')

def test_remote_binding_replays_all_terminal_checks(bound):
    b,v,h,t=raw.terminal_gate(*bound)
    assert b['schema']=='b2-mam-confirmation-remote-raw-binding-v1'
    assert t['remote_terminal_recovery'] and t['local_launcher_returncodes'] is None
    assert len(t['ranks'])==32 and not t['raw_output_audit_passed']

def test_recovered_completion_replays_full_raw_guard(accepted):
    identity,report,terminal_result=science.accepted_raw(*accepted)
    assert report['raw_output_audit_passed'] and identity['replicate']==1750
    assert terminal_result['local_launcher_returncodes'] is None
    assert not report['scientific_acceptance'] and not report['performance_cost_acceptance']

@pytest.mark.parametrize('fault',['anchor','binding_route','terminal_route','mixed_launch','raw_guard','completion_coverage'])
def test_rebinding_cannot_hide_invalid_provenance(accepted,fault):
    case,digest=accepted
    if fault=='anchor':
        path=case/terminal.PROVENANCE_NAME;v=json.loads(path.read_text());v['simulation_restarted']=True;write(path,v)
    elif fault=='binding_route':
        path=case/'raw/binding.json';v=json.loads(path.read_text());v['schema']='b2-mam-confirmation-raw-binding-v1';write(path,v)
    elif fault=='terminal_route':
        path=case/'terminal/report.json';v=json.loads(path.read_text());v['schema']='b2-mam-confirmation-terminal-v1';write(path,v)
    elif fault=='mixed_launch':write(case/'launch.json',{'returncodes':{'controller':0}})
    elif fault=='raw_guard':
        path=case/'raw/guard.json';v=json.loads(path.read_text());v['returncode']=1;write(path,v)
    else:
        path=case/'completion.json';v=json.loads(path.read_text());v['input_sha256'].pop(terminal.PROVENANCE_NAME);write(path,v);digest=sha(path)
    with pytest.raises(ValueError):science.accepted_raw(case,digest)

def test_remote_raw_bundle_preserves_real_sources_and_no_fake_launch(bound):
    case,_,_=bound;payload,catalog,binding,digest,t=launch.prepare(ROOT,case)
    assert binding['schema']=='b2-mam-confirmation-remote-raw-binding-v1'
    assert 'case/launch.json' not in catalog and 'case/'+terminal.PROVENANCE_NAME in catalog
    assert catalog['tools/mam_confirmation_raw.py']['sha256']==launch.REMOTE_RAW_WORKER_SHA
    for code in [launch.stage_code(hashlib.sha256(payload).hexdigest(),catalog),launch.collect_code(catalog)]:
        ast.parse(code);assert len(code.encode())<100000
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        assert sha(Path(raw.__file__))==hashlib.sha256(archive.extractfile('tools/mam_confirmation_raw.py').read()).hexdigest()

def test_remote_science_bundle_keeps_all_provenance_and_fixed_estimators(accepted):
    case,digest=accepted;t7=Path('/atlas-storage/0002/brian2-mpi-20260908')
    payload,catalog,identity,report,t=science_launch.bundle(ROOT,case,t7,digest)
    assert 'case/launch.json' not in catalog and 'case/'+terminal.PROVENANCE_NAME in catalog
    assert all('case/terminal/host-'+str(i)+'.json.gz' in catalog for i in range(4))
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        assert hashlib.sha256(archive.extractfile('case/'+terminal.PROVENANCE_NAME).read()).hexdigest()==terminal.PROVENANCE_SHA
    assert len(science.stages(Path('/data/brick2/example'),identity))==6
    assert not report['scientific_acceptance']


def test_remote_preparation_rejects_mixed_local_receipt_before_packaging(bound):
    case,_,_=bound;write(case/'launch.json',{'returncodes':{'controller':0}})
    with pytest.raises(ValueError,match='mixed'):launch.prepare(ROOT,case)
