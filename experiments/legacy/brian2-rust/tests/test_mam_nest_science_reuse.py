import ast
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import mam_nest_science_reuse as reuse


@pytest.fixture
def events():
    case=ROOT/'mpi-evidence/performance-runs-v1/nest-selected-target'
    old=ROOT/'mpi-evidence/primary-native-postrun/run'
    read=lambda p:json.loads(p.read_text())
    admission=read(case/'admission.json')
    return (read(case/'raw/report.json'),read(old/'activity/activity.json'),read(old/'summary.json'),
            admission['identity'],admission['parameters_sha256'])


def test_full_accepted_event_identity_covers_all_ranks(events):
    rows=reuse.match_events(*events)
    assert [r['rank'] for r in rows]==list(range(48))
    assert sum(r['bytes'] for r in rows)==52299346728


@pytest.mark.parametrize('change',['checksum','missing_rank','histogram','parameters','duration'])
def test_reuse_rejects_identity_or_window_changes(events,change):
    raw,activity,summary,identity,parameters=events
    row=raw['hosts'][0]['ranks'][0]
    if change=='checksum':row['event_sha256']='0'*64
    elif change=='missing_rank':raw['hosts'][0]['ranks'].pop()
    elif change=='histogram':row['physical_50ms_bin_counts'][-1]+=1
    elif change=='parameters':activity['parameters_sha256']='0'*64
    elif change=='duration':activity['simulation']['duration_ms']=2500
    with pytest.raises(ValueError):reuse.match_events(*events)


@pytest.mark.parametrize('path',['/tmp/unrelated','/data/brick2/brian2-mpi-region-20260907/../outside','relative'])
def test_remote_manifest_cannot_escape_data_directory(path):
    with pytest.raises(ValueError):reuse.add_pin({},path,'a'*64)


def test_remote_manifest_rejects_conflicting_digest_and_oversized_file():
    pins={};path=reuse.BASE/'artifact'
    reuse.add_pin(pins,path,'a'*64,10)
    with pytest.raises(ValueError):reuse.add_pin(pins,path,'b'*64,10)
    with pytest.raises(ValueError):reuse.add_pin(pins,path,'a'*64,11)
    with pytest.raises(ValueError):reuse.add_pin({},path,'a'*64,129*2**20)
    code=reuse.remote_code(pins);ast.parse(code)
    assert 'signal.alarm(55)' in code and 'RLIMIT_AS' in code and 'RLIMIT_CPU' in code
    assert 'f.read(2**20)' in code and 'POSIX_FADV_DONTNEED' in code
