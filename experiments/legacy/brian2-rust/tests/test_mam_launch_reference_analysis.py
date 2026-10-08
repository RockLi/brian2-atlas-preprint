"""Reference controller boundaries; no network, simulation, or event replay."""
import ast
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_launch_native_analysis as launch
import mam_launch_native_full_reference as reference


@pytest.mark.parametrize('seed',[1730,1731])
def test_reference_missing_collection_is_read_only(tmp_path,monkeypatch,seed):
    monkeypatch.setattr(reference,'protocol_gate',lambda evidence,value:{'seed':value})
    monkeypatch.setattr(launch,'remote',lambda *a,**k:pytest.fail('network before complete collection'))
    monkeypatch.setattr(launch,'bundle',lambda *a,**k:pytest.fail('bundle before complete collection'))
    result=launch.run(tmp_path/'e',tmp_path/'t7',tmp_path/'norm',tmp_path/'generated',reference_seed=seed)
    assert result['ready'] is False and result['analysis_started'] is False
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('seed',[1730,1731])
def test_reference_commands_and_remote_code_keep_identity(seed):
    paths=launch.analysis_paths(seed)
    for phase in ['raw','science']:
        command=launch.command(phase,1200,reference_seed=seed)
        assert command[command.index('--reference-seed')+1]==str(seed)
        assert '--unit='+paths['unit_prefix']+phase+'-v1' in command
        assert '--property=MemoryMax=16384M' in command
        assert '--property=AllowedCPUs=8-9' in command
        assert not any('native-primary-postrun-v1' in arg for arg in command)
    codes=[launch.preflight_code(reference_seed=seed),launch.preflight_code(True,reference_seed=seed),
           launch.stage_code(100,'a'*64,reference_seed=seed),launch.collect_code(reference_seed=seed),
           launch.publication_code('b'*64,reference_seed=seed)]
    for code in codes:
        ast.parse(code)
        assert 'native-primary-postrun-v1' not in code
    assert 'reference_seed='+str(seed) in codes[-1]
    assert 'fc/fc.json' in codes[-2] and 'lags/lags.json' in codes[-2]


def terminal_guard(seed,phase,timeout):
    command=launch.command(phase,timeout,reference_seed=seed)
    facts={'memory.events':'max 0\noom 0\noom_kill 0\noom_group_kill 0',
           'memory.max':str(16*2**30),'memory.swap.max':'0','memory.peak':'1000',
           'pids.max':'64','cpuset.cpus.effective':'8-9','cpu.max':'200000 100000'}
    return dict(admitted=True,returncode=0,error=None,host=launch.NODE,uid=1000,
        cgroup='/system.slice/'+launch.analysis_paths(seed)['unit_prefix']+phase+'-v1.service',
        wall_seconds=20,before=facts,after=facts,command=command[command.index('--')+1:],
        file_limit_bytes=512*2**20 if phase=='raw' else launch.MAX_RAW,
        data_volume='/data/brick2',data_device=2,root_device=1,
        minimum_free_bytes=1280*2**30,minimum_observed_free_bytes=1300*2**30)


@pytest.mark.parametrize('phase',['raw','science'])
@pytest.mark.parametrize('fault',[None,'wrong-seed','wrong-cgroup','oom','overspent'])
def test_reference_terminal_guard_rejects_cross_run_or_resource_failure(tmp_path,phase,fault):
    guard=copy.deepcopy(terminal_guard(1730,phase,1200))
    if fault=='wrong-seed':guard['command'][-1]='1731'
    elif fault=='wrong-cgroup':guard['cgroup']=guard['cgroup'].replace('1730','1731')
    elif fault=='oom':guard['after']['memory.events']='max 0\noom 1\noom_kill 0\noom_group_kill 0'
    elif fault=='overspent':guard['wall_seconds']=1201
    path=tmp_path/'guard.json';path.write_text(json.dumps(guard))
    if fault is None:assert launch.guard_ok(path,phase,1200,reference_seed=1730)==guard
    else:
        with pytest.raises(ValueError):launch.guard_ok(path,phase,1200,reference_seed=1730)


@pytest.mark.parametrize('missing',[None,'fc','lags'])
def test_finish_keeps_shared_budget_and_requires_all_six_stages(tmp_path,monkeypatch,missing):
    out=tmp_path/'out';out.mkdir();evidence=tmp_path/'e'
    for name in ['pending.json','summary.json','raw-guard.json']:(out/name).write_text('{}')
    seed=1730;required=['activity','cell','correlation','series','fc','lags']
    report=dict(analysis_complete=True,total_accounted_seconds=1000,seed=seed,
        label=launch.analysis_spec(seed)['label'],protocol_sha256=reference.PROTOCOL_SHA,
        required_stages=required,catalogs={name:'a'*64 for name in required},interarea_analysis_complete=True)
    if missing:del report['catalogs'][missing]
    (out/'report.json').write_text(json.dumps(report))
    remote_calls=[];guards=[]
    def remote(code,**kwargs):
        remote_calls.append((code,kwargs))
        if 'import publish' in code:return dict(passed=True,summary_sha256=launch.sha(out/'summary.json'))
        return {}
    monkeypatch.setattr(launch,'remote',remote)
    monkeypatch.setattr(launch,'collect',lambda destination,**kwargs:None)
    monkeypatch.setattr(launch.time,'monotonic',lambda:1000.)
    monkeypatch.setattr(launch,'guarded',lambda *args,**kwargs:guards.append((args,kwargs)))
    if missing:
        with pytest.raises(ValueError,match='complete reference science'):launch.finish(0,out,evidence,reference_seed=seed)
        assert not (out/'controller-complete.json').exists()
    else:
        result=launch.finish(0,out,evidence,reference_seed=seed)
        assert result['interarea_analysis_complete'] and not result['scientific_acceptance']
        assert (evidence/launch.analysis_paths(seed)['evidence_subdir']/'controller-complete.json').exists()
        assert not (evidence/'primary-native-postrun').exists()
    budget=json.loads(remote_calls[-1][1]['payload'])
    assert budget['previous_analysis_seconds']==1090 and budget['seed']==seed
    assert budget['protocol_sha256']==reference.PROTOCOL_SHA
    assert guards==[(('science',9710,out),{'reference_seed':seed})]


def test_interarea_pin_validation_rejects_changed_bytes_and_symlink(tmp_path,monkeypatch):
    p=tmp_path/'reference';p.write_bytes(b'reference')
    monkeypatch.setattr(launch,'INTERAREA_INPUTS',{'reference':('reference',9,launch.sha(p))})
    assert launch.interarea_files(tmp_path)=={'interarea/reference':p}
    p.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='reference changed'):launch.interarea_files(tmp_path)
    p.unlink();(tmp_path/'target').write_bytes(b'reference');p.symlink_to(tmp_path/'target')
    with pytest.raises(ValueError,match='reference changed'):launch.interarea_files(tmp_path)
