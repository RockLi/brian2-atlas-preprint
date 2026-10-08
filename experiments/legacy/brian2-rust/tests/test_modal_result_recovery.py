"""Storage failures retain verdicts and can only resume the same Modal call."""
import errno,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from modal_gpu_composed import save_result,resume_result


def test_report_survives_payload_disk_failure_then_atomic_retry(tmp_path,monkeypatch):
    report=dict(passed=True,stdout='all checks passed',junit='<testsuites/>',payloads={'case/result.npz':b'complete'})
    original=Path.write_bytes
    def fail(path,data):
        if path.name=='result.npz.partial':raise OSError(errno.ENOSPC,'disk full')
        return original(path,data)
    with monkeypatch.context() as m:
        m.setattr(Path,'write_bytes',fail)
        with pytest.raises(OSError):save_result(report,tmp_path,dict(function_call_id='fc-same-call'))
    saved=json.loads((tmp_path/'report.json').read_text())
    assert saved['passed'] and saved['junit']==report['junit'] and saved['function_call_id']=='fc-same-call'
    assert saved['artifact_status']=='pending'
    assert saved['result_artifacts']['case/result.npz']==dict(bytes=8,sha256=hashlib.sha256(b'complete').hexdigest())
    assert not (tmp_path/'case/result.npz').exists() and 'payloads' in report
    save_result(report,tmp_path,dict(function_call_id='fc-same-call'))
    assert (tmp_path/'case/result.npz').read_bytes()==b'complete'
    assert json.loads((tmp_path/'report.json').read_text())['artifact_status']=='complete'


def test_resume_reads_only_saved_call_and_preserves_failed_verdict(tmp_path,monkeypatch):
    manifest=b'{}\n';(tmp_path/'source-hashes.json').write_bytes(manifest)
    cp=dict(function_call_id='fc-saved-call',requested_gpu='L4',source_manifest_sha256=hashlib.sha256(manifest).hexdigest())
    (tmp_path/'call.json').write_text(json.dumps(cp));calls=[]
    def get(timeout):
        calls.append(('get',timeout));return dict(passed=False,stdout='failed gate',payloads={'failure.npz':b'failed arrays'})
    def from_id(call):calls.append(('from_id',call));return SimpleNamespace(get=get)
    # The stub offers no App, Function, spawn or remote method: recovery must
    # never launch another paid GPU operation, even for a failed result.
    monkeypatch.setitem(sys.modules,'modal',SimpleNamespace(FunctionCall=SimpleNamespace(from_id=from_id)))
    with pytest.raises(SystemExit) as error:resume_result(tmp_path)
    assert error.value.code==1 and calls==[('from_id','fc-saved-call'),('get',0)]
    assert (tmp_path/'failure.npz').read_bytes()==b'failed arrays'
    saved=json.loads((tmp_path/'report.json').read_text());assert not saved['passed'] and saved['recovered_existing_call'] and saved['remote_wall_seconds'] is None
    (tmp_path/'source-hashes.json').write_bytes(b'changed')
    with pytest.raises(ValueError,match='manifest changed'):resume_result(tmp_path)
    assert len(calls)==2


@pytest.mark.parametrize('name',['../outside.npz','/absolute.npz'])
def test_result_paths_cannot_escape_output(tmp_path,name):
    with pytest.raises(ValueError,match='Invalid result path'):
        save_result(dict(passed=True,stdout='',payloads={name:b'data'}),tmp_path,{})
