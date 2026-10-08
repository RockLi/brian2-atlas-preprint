import ast
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import mam_launch_rust_performance as launch
from mam_rust_guard23 import arguments


def test_pending_nest_causes_no_deployment_or_admission(tmp_path,monkeypatch):
    def reject(*args,**kwargs):raise AssertionError('no remote operations allowed')
    monkeypatch.setattr(launch,'remote',reject)
    before=list(tmp_path.rglob('*'))
    result=launch.run(SimpleNamespace(evidence=tmp_path,t7=tmp_path/'missing-t7'))
    assert result['ready'] is False and result['launch_started'] is False
    assert list(tmp_path.rglob('*'))==before


def test_time_and_resource_limits_cover_wrapper(tmp_path):
    options=launch.launch_options(tmp_path)
    command=options['application'][2]
    assert command.index('/usr/bin/time')<command.index('mam_rust_terminal_sync.py')
    assert 'B2_THREAD_AFFINITY=required' in command and '"${PMI_RANK}.time"' in command
    assert '--executable '+launch.BASE+'/'+launch.OLD+'/mpi/b2-mpi' in command
    assert options['ranks_per_node']==8 and options['guard_cpu_ids']==launch.CPUS
    assert options['timeout']==64800 and options['guard_memory_mib']==262144
    assert options['guard_file_mib']*2**20*2+2*32*2**20==128*2**30
    assert options['guard_node_overrides'][launch.NODES[0]]['volume']=='/data/brick2'
    shlex.split(command)


def test_four_host_preflight_and_staging_are_parseable():
    package=json.loads((ROOT/'mpi-evidence/primary-run/package.json').read_text())
    admission=json.loads((ROOT/'mpi-evidence/primary-run/admission.json').read_text())
    catalog={'a.py':dict(bytes=1,sha256='0'*64)}
    for i in range(4):
        ast.parse(launch.stage_code(i,{'a.py':'eA=='},catalog))
        code=launch.preflight_code(i,package,catalog,admission['preflight'][i]['selected_cpu_topology'])
        ast.parse(code)
        assert 'all_artifact_hashes_verified=True' in code and 'signal.alarm(40)' in code
        assert launch.LABEL in code


def test_leader_reserve_override_only_changes_reserve():
    args=['guard','--volume','/data/brick2','--min-free-gib','128','--','program']
    result=arguments(args)
    assert result==['guard','--volume','/data/brick2','--min-free-gib','1280','--','program']
    assert args[4]=='128'


@pytest.mark.parametrize('args',[['guard','--volume','/','--min-free-gib','128'],
                                  ['guard','--volume','/data/brick2','--min-free-gib','64'],
                                  ['guard','--volume','/data/brick2','--min-free-gib','128','--min-free-gib','128']])
def test_invalid_leader_reserve_refused(args):
    with pytest.raises(ValueError):arguments(args)


@pytest.fixture
def deferred_case(tmp_path,monkeypatch):
    out=tmp_path/'case';out.mkdir();options=launch.launch_options(tmp_path/'logs')
    catalog={'wrapper.py':dict(bytes=1,sha256='a'*64)}
    prerequisite=dict(completion_sha256='b'*64,terminal_sha256='c'*64,raw_sha256='d'*64)
    def save(name,value):
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
        return launch.sha(path)
    cpu_sha=save('cpu-availability.json',dict(idle=False))
    diagnostics={f'preflight-diagnosis/host-{i}.json':save(f'preflight-diagnosis/host-{i}.json',dict(passed=False)) for i in range(4)}
    proof_sha=save('preflight-deferred.json',dict(neural_runs_started=0,launch_started=False,
        source_staging_complete=True,nest_completion_sha256=prerequisite['completion_sha256'],
        cpu_availability_sha256=cpu_sha,diagnostics=diagnostics))
    monkeypatch.setattr(launch,'DEFERRED_SHA',proof_sha)
    save('failure.json',dict(error_type='CalledProcessError'))
    save('intent.json',dict(case_id=launch.CASE,protocol_sha256=launch.PROTOCOL_SHA,
        nest_prerequisite=prerequisite,source_catalog=catalog,maximum_neural_runs=1,automatic_retry=False,
        launch_options={k:str(v) if isinstance(v,Path) else v for k,v in options.items()}))
    for i,node in enumerate(launch.NODES):
        save(f'stage-{i}.json',dict(host=node,source_catalog=catalog,neural_simulations=0,
             base_resolved=launch.BRICK if i==0 else launch.BASE))
    return out,options,catalog,prerequisite


def test_only_pinned_unlaunched_deferral_can_reuse_staging(deferred_case):
    out,*_=deferred_case;before={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    staged=launch.deferred_gate(*deferred_case)
    assert [s['host'] for s in staged]==launch.NODES
    assert before=={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}


@pytest.mark.parametrize('marker',['admission.json','launch.json','resume-preflight-v1','logs'])
def test_resume_refuses_any_admission_launch_or_previous_attempt(deferred_case,marker):
    out,options,*_=deferred_case
    target=options['output'] if marker=='logs' else out/marker
    target.mkdir()
    with pytest.raises(ValueError):launch.deferred_gate(*deferred_case)


@pytest.mark.parametrize('name',['preflight-deferred.json','cpu-availability.json','preflight-diagnosis/host-0.json'])
def test_resume_refuses_changed_historical_evidence(deferred_case,name):
    out,*_=deferred_case;(out/name).write_text('{}')
    with pytest.raises(ValueError):launch.deferred_gate(*deferred_case)


def test_resume_refuses_changed_source_or_launch_contract(deferred_case):
    out,*_=deferred_case
    path=out/'intent.json';row=json.loads(path.read_text());row['launch_options']['timeout']=999999
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='contract'):launch.deferred_gate(*deferred_case)


def test_resume_refuses_changed_stage_receipt(deferred_case):
    out,*_=deferred_case
    path=out/'stage-2.json';row=json.loads(path.read_text());row['source_catalog']={}
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='staged source'):launch.deferred_gate(*deferred_case)


@pytest.fixture
def window_case(deferred_case,monkeypatch):
    out,*_=deferred_case;directory=out/'resume-preflight-v1';directory.mkdir()
    inputs={}
    for name in ['failure.json','intent.json']+[f'preflight-{i}.json' for i in range(7)]:
        path=directory/name;path.write_text('{}');inputs[str(path.relative_to(out))]=launch.sha(path)
    path=out/'recovery-deferred.json'
    path.write_text(json.dumps(dict(recovery_attempt_terminal=True,admission_created=False,
        neural_runs_started=0,launch_started=False,input_sha256=inputs)))
    monkeypatch.setattr(launch,'RECOVERY_DEFERRED_SHA',launch.sha(path))
    return deferred_case


def test_window_reuses_only_pinned_unlaunched_sources(window_case):
    assert len(launch.deferred_gate(*window_case,resource_window=True))==4


@pytest.mark.parametrize('marker',['admission.json','launch.json','resource-window-v2','logs'])
def test_window_refuses_admission_launch_or_previous_window(window_case,marker):
    out,options,*_=window_case
    (options['output'] if marker=='logs' else out/marker).mkdir()
    with pytest.raises(ValueError):launch.deferred_gate(*window_case,resource_window=True)


@pytest.mark.parametrize('name',['recovery-deferred.json','resume-preflight-v1/preflight-0.json',
                                'resume-preflight-v1/failure.json','preflight-deferred.json'])
def test_window_refuses_changed_prior_evidence(window_case,name):
    out,*_=window_case;(out/name).write_text('{"changed":true}')
    with pytest.raises(ValueError):launch.deferred_gate(*window_case,resource_window=True)


def busy_error(value=56):
    busy=dict.fromkeys(launch.CPUS,0);busy[launch.CPUS[0]]=value
    return dict(error_type='CalledProcessError',stderr='Traceback\nAssertionError: '+repr(busy)+'\n')


def test_cpu_classifier_accepts_only_actual_threshold_violation():
    assert launch.cpu_busy_error(busy_error())
    assert launch.cpu_busy_error(busy_error(50))
    assert launch.cpu_busy_error(dict(error_type='CalledProcessError',
        stderr='AssertionError: '+repr(dict.fromkeys(launch.CPUS,25))))
    for value in [0,49,True,-1,101,float('inf'),float('nan'),'56']:
        assert not launch.cpu_busy_error(busy_error(value))
    for error in [dict(error_type='TimeoutExpired',stderr=busy_error()['stderr']),
                  dict(error_type='CalledProcessError',stderr='AssertionError: model.json'),
                  dict(error_type='CalledProcessError',stderr='AssertionError: {0: 99}'),
                  dict(error_type='CalledProcessError',stderr='AssertionError: {broken')]:
        assert not launch.cpu_busy_error(error)


def fake_window(errors_by_check,durations=None):
    now=[0.0];calls=[];sleeps=[]
    def perform(directory):
        start=now[0];i=len(calls);calls.append(directory)
        now[0]+=durations[i] if durations else 1
        return ['rows'],['extra'],errors_by_check[i],start
    def sleep(seconds):sleeps.append(seconds);now[0]+=seconds
    return perform,lambda:now[0],sleep,calls,sleeps


def test_window_reobserves_cpu_once_then_returns_fresh_preflight(tmp_path):
    perform,clock,sleep,calls,sleeps=fake_window([[busy_error()],[]])
    assert launch.bounded_window(tmp_path,perform,clock=clock,sleeper=sleep)==(['rows'],['extra'],31)
    assert [p.name for p in calls]==['check-1','check-2'] and sleeps==[30]
    assert not (tmp_path/'admission.json').exists()


def test_window_never_repeats_non_cpu_failure(tmp_path):
    perform,clock,sleep,calls,sleeps=fake_window([[dict(error_type='TimeoutExpired',stderr='')]])
    with pytest.raises(ValueError,match='non-CPU'):
        launch.bounded_window(tmp_path,perform,clock=clock,sleeper=sleep)
    assert len(calls)==1 and not sleeps


def test_window_stops_after_three_cpu_checks(tmp_path):
    perform,clock,sleep,calls,sleeps=fake_window([[busy_error()]]*3)
    with pytest.raises(ValueError,match='three checks'):
        launch.bounded_window(tmp_path,perform,clock=clock,sleeper=sleep)
    assert len(calls)==3 and sleeps==[30,30]


@pytest.mark.parametrize('duration,reason',[(60,'stale'),(600,'elapsed budget')])
def test_window_refuses_stale_or_over_budget_success(tmp_path,duration,reason):
    perform,clock,sleep,calls,sleeps=fake_window([[]],[duration])
    with pytest.raises(ValueError,match=reason):
        launch.bounded_window(tmp_path,perform,clock=clock,sleeper=sleep)
    assert len(calls)==1 and not sleeps
