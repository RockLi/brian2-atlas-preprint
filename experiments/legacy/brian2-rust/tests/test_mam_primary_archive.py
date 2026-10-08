"""Archive failure/byte-integrity checks; no simulation recordings are read."""
import ast
from datetime import datetime, timedelta
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'tools'))
    import mam_primary_archive as archive
    import mam_launch_primary_archive as launch
    if not hasattr(os,'posix_fadvise'):
        # macOS verifies archive semantics with an explicit cache-release stub;
        # the same tests on Linux use the real syscall.
        monkeypatch.setattr(os,'posix_fadvise',lambda *a:None,raising=False)
        monkeypatch.setattr(os,'POSIX_FADV_DONTNEED',4,raising=False)
    return archive,launch


def sources(tmp_path):
    values={'run/spikes.bin':bytes(range(256))*8192,'control/run.json':b'{"seed":1729}\n','empty':b''}
    files={}
    for n,raw in values.items():
        p=tmp_path/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);files[n]=p
    expected={n:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for n,raw in values.items()}
    return values,files,expected


def test_roundtrip_byte_exact_and_hashes_compressed_file(modules,tmp_path):
    archive,_=modules
    values,files,expected=sources(tmp_path);target=tmp_path/'archive.tar.gz'
    result=archive.write_verified_archive(files,expected,target)
    assert result['verified_members']==len(values)
    assert result['sha256']==hashlib.sha256(target.read_bytes()).hexdigest()
    assert result['bytes']==target.stat().st_size
    with tarfile.open(target) as t:
        assert {m.name:t.extractfile(m).read() for m in t}==values
    assert not target.with_name(target.name+'.pending').exists()
    with pytest.raises(ValueError,match='already exists'):archive.write_verified_archive(files,expected,target)


def test_changed_source_never_publishes_archive(modules,tmp_path):
    archive,_=modules
    _,files,expected=sources(tmp_path);files['empty'].write_bytes(b'changed')
    target=tmp_path/'archive.tar.gz'
    with pytest.raises(ValueError,match='size changed'):archive.write_verified_archive(files,expected,target)
    assert not target.exists() and target.with_name(target.name+'.pending').exists()


def test_catalogued_appledouble_preserves_bytes_and_original_path(modules,tmp_path):
    archive,_=modules
    source=tmp_path/'source';source.mkdir()
    normal=source/'report.json';normal.write_bytes(b'{"seed":1729}\n')
    sidecar=source/'._report.json'
    sidecar.write_bytes(bytes.fromhex('0005160700020000')+b'metadata'*511)
    files,expected={},{}
    for p in [normal,sidecar]:
        name,metadata=archive.member_identity(p,source)
        files[name]=p
        expected[name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),**metadata)
    alias=next(n for n in files if n.startswith('appledouble-metadata/'))
    assert expected[alias]['original_relative_path']=='._report.json'
    target=tmp_path/'archive.tar.gz'
    result=archive.write_verified_archive(files,expected,target)
    assert result['verified_members']==2
    with tarfile.open(target) as t:
        assert t.extractfile(alias).read()==sidecar.read_bytes()
        assert t.extractfile('report.json').read()==normal.read_bytes()


@pytest.mark.parametrize('fault',['header','missing-sibling','oversized','nested','symlink'])
def test_appledouble_alias_does_not_admit_arbitrary_hidden_files(modules,tmp_path,fault):
    archive,_=modules
    parent=tmp_path/'._nested' if fault=='nested' else tmp_path/'normal'
    parent.mkdir();sibling=parent/'report.json';sibling.write_bytes(b'{}')
    p=parent/'._report.json';p.write_bytes(bytes.fromhex('0005160700020000')+bytes(32))
    if fault=='header':p.write_bytes(b'not AppleDouble metadata')
    if fault=='missing-sibling':sibling.unlink()
    if fault=='oversized':p.write_bytes(bytes.fromhex('0005160700020000')+bytes(65536))
    if fault=='symlink':p.unlink();p.symlink_to(sibling)
    with pytest.raises(ValueError,match='metadata|AppleDouble'):
        archive.member_identity(p,tmp_path)


def test_wrong_expected_hash_never_publishes_archive(modules,tmp_path):
    archive,_=modules
    _,files,expected=sources(tmp_path);expected['run/spikes.bin']['sha256']='0'*64
    target=tmp_path/'archive.tar.gz'
    with pytest.raises(ValueError,match='source changed'):archive.write_verified_archive(files,expected,target)
    assert not target.exists()


def test_corrupted_compressed_file_detected_in_independent_pass(modules,tmp_path,monkeypatch):
    archive,_=modules
    _,files,expected=sources(tmp_path);target=tmp_path/'archive.tar.gz'
    original=tarfile.open
    def tamper(*args,**kwargs):
        if kwargs.get('mode')=='r|':
            with target.with_name(target.name+'.pending').open('r+b') as f:f.write(b'xx')
        return original(*args,**kwargs)
    monkeypatch.setattr(tarfile,'open',tamper)
    with pytest.raises((gzip.BadGzipFile,tarfile.ReadError)):
        archive.write_verified_archive(files,expected,target)
    assert not target.exists()


def test_cache_window_flushes_and_releases_actual_byte_ranges(modules,tmp_path,monkeypatch):
    archive,_=modules
    actual=os.posix_fadvise;calls=[]
    def advise(fd,start,size,mode):
        calls.append((start,size));actual(fd,start,size,mode)
    monkeypatch.setattr(os,'posix_fadvise',advise)
    with (tmp_path/'cache').open('xb') as f:
        writer=archive.CachedStream(f,writing=True,window=1024)
        for _ in range(5):writer.write(b'x'*600)
        writer.release()
    assert calls==[(0,1200),(1200,1200),(2400,600)]
    assert (tmp_path/'cache').read_bytes()==b'x'*3000


def test_stream_limit_refuses_before_write(modules,tmp_path):
    archive,_=modules
    with (tmp_path/'limit').open('xb') as f:
        writer=archive.CachedStream(f,writing=True,limit=2)
        with pytest.raises(ValueError,match='byte budget'):writer.write(b'abc')
    assert (tmp_path/'limit').stat().st_size==0


@pytest.mark.parametrize('bad',['../escape','/absolute','._metadata'])
def test_unsafe_archive_names_rejected_before_creation(modules,tmp_path,bad):
    archive,_=modules
    p=tmp_path/'file';p.write_bytes(b'x')
    with pytest.raises(ValueError,match='unsafe archive'):
        archive.write_verified_archive({bad:p},{bad:dict(bytes=1,sha256=hashlib.sha256(b'x').hexdigest())},tmp_path/'archive')
    assert not (tmp_path/'archive.pending').exists()


def test_missing_primary_gate_never_contacts_hosts_or_writes(modules,tmp_path,monkeypatch):
    _,launch=modules
    monkeypatch.setattr(launch,'remote',lambda *a,**k:pytest.fail('network before prerequisite'))
    result=launch.run(tmp_path/'evidence',tmp_path/'t7')
    assert result['ready'] is False and result['archive_started'] is False
    assert not list(tmp_path.iterdir())


def test_postrun_host_inventory_and_guard_are_bounded(modules):
    archive,launch=modules
    package=json.loads((ROOT/'mpi-evidence/primary-run/package.json').read_text())
    for i in range(4):ast.parse(launch.host_code(i,package,'0'*64))
    cmd=launch.guard_command(6000)
    assert '--property=MemoryMax=8192M' in cmd and '--property=AllowedCPUs=8-9' in cmd
    assert '/atlas-home/0003/workspace/brian2-lk-20260906/.venv/bin/python' in cmd
    assert cmd[cmd.index('--timeout')+1]=='6000'
    assert cmd[cmd.index('--file-mib')+1]==str(archive.MAX_BYTES//2**20)
    with pytest.raises(ValueError):launch.guard_command(7200)


@pytest.mark.parametrize('fault',[None,'missing-timing','prior-overbudget','different-report'])
def test_prior_collection_time_is_required_and_bound_to_resource_report(modules,tmp_path,monkeypatch,fault):
    _,launch=modules
    resource=tmp_path/'t7/artifacts/primary-terminal-resources-v1';resource.mkdir(parents=True)
    analysis=tmp_path/'t7/artifacts/primary-postrun-v1';analysis.mkdir()
    (resource/'report.json').write_text('{"fixture":true}\n')
    (resource/'admission.json').write_text(json.dumps(dict(experiment_budget=dict(collection_wall_limit_seconds=7200))))
    (analysis/'report.json').write_text(json.dumps(dict(analysis_complete=True,output_audit_sha256='audited')))
    monkeypatch.setattr(launch,'rust_gate',lambda *a:dict(output_report_sha256='audited'))
    timing=dict(schema='b2-mam-terminal-collection-controller-v1',elapsed_seconds=42.5,
                resource_report_sha256=launch.sha(resource/'report.json'),
                collection_complete=True,shared_collection_budget_seconds=7200)
    if fault=='prior-overbudget':timing['elapsed_seconds']=7200
    if fault=='different-report':timing['resource_report_sha256']='changed'
    if fault!='missing-timing':(resource/'collection-controller.json').write_text(json.dumps(timing))
    if fault is None:
        assert launch.prerequisites(tmp_path/'evidence',tmp_path/'t7')['previous_collection_seconds']==42.5
    elif fault=='missing-timing':
        assert launch.prerequisites(tmp_path/'evidence',tmp_path/'t7') is None
    else:
        with pytest.raises(ValueError):launch.prerequisites(tmp_path/'evidence',tmp_path/'t7')


@pytest.mark.parametrize('fault',[None,'controller','resource','sidecar','finished','expired','clock'])
def test_corrected_attempt_requires_actual_failure_and_charges_elapsed_time(modules,tmp_path,fault):
    _,launch=modules
    prior=tmp_path/'artifacts/primary-archive-v1';prior.mkdir(parents=True)
    recorded=ROOT/'mpi-evidence/primary-archive'
    for name in ['admission.json','controller.json','controller.log','guard.json','catalog.json']:
        (prior/name).write_bytes((recorded/name).read_bytes())
    admission=json.loads((prior/'admission.json').read_text());gate=admission['gate']
    origin=min(datetime.fromisoformat(h['utc']) for h in admission['hosts'])
    now=origin+timedelta(seconds=600)
    if fault=='controller':
        (prior/'controller.json').write_text('{"returncode": 0, "error": null}')
    if fault=='resource':
        guard=json.loads((prior/'guard.json').read_text())
        guard['after']['memory.events']=guard['after']['memory.events'].replace('max 0','max 1')
        (prior/'guard.json').write_text(json.dumps(guard))
    if fault=='sidecar':
        catalog=json.loads((prior/'catalog.json').read_text())
        name=next(n for n in catalog if '/._' in n);catalog[name]['sha256']='0'*64
        (prior/'catalog.json').write_text(json.dumps(catalog))
    if fault=='finished':(prior/'complete.json').write_text('{}')
    if fault=='expired':now=origin+timedelta(seconds=7200)
    if fault=='clock':now=origin-timedelta(seconds=1)
    if fault is not None:
        with pytest.raises(ValueError):launch.recovery_gate(tmp_path,tmp_path,gate,now=now)
    else:
        result=launch.recovery_gate(tmp_path,tmp_path,gate,now=now)
        assert result['previous_collection_seconds']==gate['previous_collection_seconds']+660
        assert result['maximum_attempts']==2 and result['automatic_retry'] is False
        assert result['prior_attempt_sha256']['guard.json']==launch.sha(prior/'guard.json')


def test_corrected_attempt_uses_distinct_paths_and_checks_original_failure(modules):
    archive,launch=modules
    first,second=archive.attempt_paths(1),archive.attempt_paths(2)
    assert all(first[k]!=second[k] for k in first)
    package=json.loads((ROOT/'mpi-evidence/primary-run/package.json').read_text())
    recovery=dict(prior_attempt_sha256={'guard.json':'a'*64,'catalog.json':'b'*64})
    code=launch.host_code(0,package,'c'*64,second,recovery)
    ast.parse(code)
    assert str(first['archive'])+'.pending' in code
    assert str(first['guard']) in code and 'a'*64 in code
    cmd=launch.guard_command(6000,2)
    assert '--unit=b2mpi-archive-primary-v2' in cmd
    assert cmd[cmd.index('--output')+1]==str(second['guard'])
    assert cmd[cmd.index('--control')+1]==str(second['source']/'control')
    assert cmd[cmd.index('--attempt-version')+1]=='2'
    with pytest.raises(ValueError):archive.attempt_paths(3)
