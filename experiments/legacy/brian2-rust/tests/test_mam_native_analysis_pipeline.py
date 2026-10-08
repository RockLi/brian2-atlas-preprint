"""Native analysis orchestration contracts; never run a simulation or raw replay."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_native_analysis_pipeline as p
import mam_launch_native_analysis as launch


def test_missing_native_collection_never_contacts_remote(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'remote',lambda *a,**k:pytest.fail('remote before prerequisites'))
    monkeypatch.setattr(launch,'bundle',lambda *a,**k:pytest.fail('source read before prerequisites'))
    result=launch.run(tmp_path/'e',tmp_path/'t7',tmp_path/'norm',tmp_path/'generated')
    assert not result['ready'] and not result['analysis_started'] and not list(tmp_path.iterdir())


def test_actual_run_evidence_preserves_fixture_and_ignores_appledouble(tmp_path):
    fixture=tmp_path/'e/primary-native-postrun';fixture.mkdir(parents=True)
    (fixture/'admission.json').write_text('{"fixture":true}')
    out=tmp_path/'out';out.mkdir()
    (out/'admission.json').write_text('{"actual":true}')
    (out/'._admission.json').write_bytes(b'filesystem metadata')
    launch.sync_evidence(out,tmp_path/'e')
    assert (fixture/'admission.json').read_text()=='{"fixture":true}'
    assert (fixture/'run/admission.json').read_text()=='{"actual":true}'
    assert not (fixture/'run/._admission.json').exists()


@pytest.mark.parametrize('debit',[-1,10800])
def test_invalid_preparation_debit_refused_before_access(tmp_path,monkeypatch,debit):
    monkeypatch.setattr(launch,'prerequisites',lambda *a:pytest.fail('read before debit validation'))
    with pytest.raises(ValueError,match='prior preparation'):
        launch.run(tmp_path,tmp_path,tmp_path,tmp_path,debit)


def test_publication_uses_the_same_numpy_python_as_raw_audit():
    code=launch.publication_code('a'*64);ast.parse(code)
    assert repr(launch.PYTHON) in code and 'timeout=40' in code
    assert 'RLIMIT_AS' in code and 'RLIMIT_CPU' in code
    assert "'OPENBLAS_NUM_THREADS':'1'" in code


def test_publication_resume_retains_failed_budget_and_never_repeats_raw(tmp_path,monkeypatch):
    d=tmp_path/'artifacts/native-primary-postrun-v1';d.mkdir(parents=True)
    failure=d/'failure.json';failure.write_text(json.dumps(dict(error='from module import publish',elapsed_seconds=700.)))
    (d/'raw-controller.json').write_text(json.dumps(dict(returncode=0,error=None,command=launch.command('raw',7200))))
    (d/'pending.json').write_text(json.dumps(dict(raw_output_audit_passed=True,passed=False)))
    monkeypatch.setattr(launch.time,'time',lambda:failure.stat().st_mtime+50.)
    monkeypatch.setattr(launch.time,'monotonic',lambda:1000.)
    monkeypatch.setattr(launch,'guard_ok',lambda *a:None)
    monkeypatch.setattr(launch,'guarded',lambda *a:pytest.fail('raw replay during publication recovery'))
    monkeypatch.setattr(launch,'remote',lambda code:dict(active_own_units=[]))
    monkeypatch.setattr(launch,'finish',lambda start,*a:dict(start=start))
    assert launch.resume_publication(tmp_path/'e',tmp_path)==dict(start=250.)
    marker=json.loads((d/'resume-publication.json').read_text())
    assert marker['charged_previous_seconds']==750. and marker['raw_audit_repeated'] is False
    with pytest.raises(ValueError,match='already attempted'):launch.resume_publication(tmp_path/'e',tmp_path)


def test_native_guard_splits_preserve_scratch_budget_and_global_deadline():
    raw=launch.command('raw',7200);science=launch.command('science',1400)
    assert raw[raw.index('--file-mib')+1]=='512'
    assert science[science.index('--file-mib')+1]==str(96*1024)
    assert '--property=RuntimeMaxSec=1405' in science
    for c in [raw,science]:
        assert '--property=MemoryMax=16384M' in c and '--property=MemorySwapMax=0' in c
        assert '--property=AllowedCPUs=8-9' in c and '--property=TasksMax=64' in c
        assert c[c.index('--min-free-gib')+1]=='1280'
        assert not any('mpiexec' in part for part in c)
    assert launch.remaining(0,7200,clock=lambda:1000)==7200
    assert launch.remaining(0,10800,clock=lambda:9000)==1710
    with pytest.raises(ValueError,match='budget exhausted'):launch.remaining(0,7200,clock=lambda:10711)


def test_native_science_uses_shared_deadline_including_previous_raw_audit(tmp_path):
    now=[9000.];timeouts=[]
    def runner(command,**kwargs):
        timeouts.append(kwargs['timeout']);now[0]+=500
        return SimpleNamespace(returncode=0)
    plan=[('activity',2400,['first']),('cell',1800,['second'])]
    p.execute_sequence(plan,tmp_path,started=0,clock=lambda:now[0],runner=runner)
    assert timeouts==[1800,1300]


def test_native_plans_match_cli_flags_and_full_window():
    plan=p.stages(launch.SOURCE,launch.SOURCE/launch.NORMALIZATION)
    assert [r[0] for r in plan]==['activity','cell','correlation','series']
    c=plan[0][2]
    assert c[c.index('--audit-dir')+1]==str(p.RAW)
    assert c[c.index('--max-bytes')+1]==str(96*2**30)
    assert c[c.index('--end-tick')+1]=='1005000'
    for _,_,c in plan:
        assert '--bounded-memory' in c and '--model' not in c
    for _,_,c in plan[1:]:assert '--native-audit' in c and '--baseline' in c


def test_stage_and_preflight_code_parse_with_bounded_archive():
    for code in [launch.preflight_code(),launch.preflight_code(True),launch.stage_code(100,'0'*64),launch.collect_code()]:
        ast.parse(code)
    assert '1536*2**30' in launch.preflight_code(True)
    assert 'm.isfile()' in launch.stage_code(100,'0'*64)
    assert '2*2**30' in launch.stage_code(100,'0'*64)


@pytest.fixture
def output(tmp_path,monkeypatch):
    raw=tmp_path/'raw';out=tmp_path/'out';(out/'activity').mkdir(parents=True);raw.mkdir()
    monkeypatch.setattr(p,'RAW',raw);monkeypatch.setattr(p,'OUTPUT',out)
    results={}
    for rank in range(48):
        r=raw/('node'+p.NODES[rank//8].rsplit('-',1)[-1])/'runs'/p.LABEL;r.mkdir(parents=True,exist_ok=True)
        row=dict(event_bytes=8,event_sha256='a'*64)
        (r/('rank'+str(rank)+'.json')).write_text(json.dumps(row))
        results[str(r/('rank'+str(rank)+'.events.bin'))]=dict(bytes=8,sha256='a'*64)
    (raw/'summary.json').write_text(json.dumps(dict(physical_50ms_bin_counts=[0]*2010,spikes=48,terminal_tick_events=48)))
    baseline=dict(schema='b2-native-mam-activity-v1',simulation=p.IDENTITY,parameters_sha256=p.PARAMETERS,
        neurons=p.NEURONS,population_count=254,bounded_memory=True,observed_spikes=0,result_files=results,
        window=dict(start_tick=5000,end_tick=1005000,seconds=100.,spike_tick_offset=0,endpoint='[start,end)',raster_end_tick=105000))
    (out/'activity/activity.json').write_text(json.dumps(baseline))
    for name in ['activity-arrays.npz','activity-overview.png','spike-rasters.png']:(out/'activity'/name).write_bytes(b'explicit fixture')
    return raw,out,baseline


def catalog(directory):
    rows={f.name:dict(bytes=f.stat().st_size,sha256=p.sha(f)) for f in directory.iterdir() if f.name!='catalog.json'}
    (directory/'catalog.json').write_text(json.dumps(rows))


@pytest.mark.parametrize('fault',[None,'short-window','wrong-raw-hash'])
def test_activity_is_bound_to_complete_native_sources(output,fault):
    raw,out,baseline=output
    if fault=='short-window':baseline['window']['end_tick']=105000
    elif fault=='wrong-raw-hash':baseline['result_files'][next(iter(baseline['result_files']))]['sha256']='b'*64
    (out/'activity/activity.json').write_text(json.dumps(baseline));catalog(out/'activity')
    if fault is None:p.validate_output('activity')
    else:
        with pytest.raises(ValueError):p.validate_output('activity')


def test_cell_identity_does_not_require_nonexistent_condition_field(output):
    raw,out,baseline=output;catalog(out/'activity');(out/'cell').mkdir()
    report=dict(identity=dict(simulator='NEST',**p.IDENTITY),scientific_equivalence=False,bounded_memory=True,
        source_sha256={str(out/'activity/activity.json'):p.sha(out/'activity/activity.json')},
        window=dict(end_tick=1005000),scratch_removed=True)
    (out/'cell/paper-cell-metrics.json').write_text(json.dumps(report));(out/'cell/cell-metrics.npz').write_bytes(b'explicit fixture')
    catalog(out/'cell');p.validate_output('cell')


def test_raw_budget_cannot_ignore_prior_audit_wall(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'RAW',tmp_path)
    report=dict(schema='b2-mam-native-primary-output-audit-v1',passed=True,audit_guard_passed=True,
        raw_output_audit_passed=True,terminal_resource_audit_passed=True,label=p.LABEL,parameters_sha256=p.PARAMETERS,
        retained_prefix_spikes=p.PRIOR_SPIKES,all_first_10500ms_event_prefixes_exact=True,duration_ms=100500,event_bytes=96*2**30,
        audit_guard_sha256='a'*64,shared_analysis_budget_seconds=10800,analysis_wall_seconds_consumed=5000.)
    (tmp_path/'summary.json').write_text(json.dumps(report))
    budget=dict(summary_sha256=p.sha(tmp_path/'summary.json'),raw_guard_sha256='a'*64,
                shared_analysis_budget_seconds=10800,previous_analysis_seconds=5100.)
    assert p.raw_gate(budget)==report
    budget['previous_analysis_seconds']=100.
    with pytest.raises(ValueError,match='budget differs'):p.raw_gate(budget)
