import ast,json,sys,copy
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import mam_launch_rust_recovery as recovery
from test_mam_rust_benchmark_terminal import controls


def test_recovery_requires_actual_failed_target_fixed_wrapper_and_nest():
    protocol,nest,package=recovery.prerequisites(ROOT/'mpi-evidence')
    assert protocol['maximum_corrected_target_launches']==1
    assert protocol['target_duration_ms']==100500 and package['build']['executable_sha256']==recovery.old.EXE_SHA
    assert nest['completion_sha256']=='15f6ffc2bae7ec859d7b1b503b40c06cebde02d33397ff1011de6081ed8bd641'


def test_pending_or_modified_recovery_protocol_does_not_touch_remote(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'remote',lambda *a:pytest.fail('remote before gates'))
    with pytest.raises(FileNotFoundError):recovery.run(tmp_path,tmp_path/'t7','prepare')
    assert list(tmp_path.iterdir())==[]


def test_new_commands_keep_old_case_and_deployed_source_separate(tmp_path):
    target=recovery.options(tmp_path);smoke=recovery.options(tmp_path,smoke=True)
    assert target['output']!=smoke['output']
    assert target['timeout']==64800 and target['guard_memory_mib']==262144
    assert smoke['timeout']==180 and smoke['guard_memory_mib']==1024 and smoke['guard_file_mib']==8
    for opts in [target,smoke]:
        assert opts['guard_cpu_ids']==recovery.ALTERNATE_CPUS and opts['ranks_per_node']==8
        assert recovery.SOURCE in opts['guard_script'] and recovery.old.SOURCE not in opts['guard_script']
        assert opts['guard_node_overrides'][recovery.old.NODES[0]]['volume']=='/data/brick2'
    assert recovery.LABEL in target['application'][2] and recovery.old.LABEL not in target['application'][2]
    assert 'B2_THREAD_AFFINITY=required' in target['application'][2]


def test_remote_templates_use_the_new_source_and_keep_full_checks():
    package=json.loads((ROOT/'mpi-evidence/primary-run/package.json').read_text())
    for i,t in enumerate(recovery.candidate_topologies(ROOT/'mpi-evidence')):
        stage=recovery.old.stage_code(i,{}, {},source=recovery.SOURCE)
        code=recovery.old.preflight_code(i,package,{},t,cpu_ids=recovery.ALTERNATE_CPUS,
                                         source=recovery.SOURCE,label=recovery.LABEL)
        ast.parse(stage);ast.parse(code)
        assert recovery.SOURCE in stage and recovery.SOURCE in code and recovery.LABEL in code
        assert 'max(busy.values())<50' in code and 'signal.alarm(40)' in code
        assert 'digest(path)==item' in code


@pytest.fixture
def smoke_fixture(tmp_path,controls,monkeypatch):
    case=tmp_path/'case';out=case/'smoke';out.mkdir(parents=True)
    (case/'prepared.json').write_text('{}')
    opts=recovery.options(tmp_path,smoke=True);opts['output'].mkdir(parents=True)
    probe=dict(schema='b2-mpi-cluster-probe-v1',ok=True,ranks=32,sum=496,
               hosts=[n for n in recovery.old.NODES for _ in range(8)])
    (opts['output']/'controller.log').write_text(json.dumps(probe,separators=(',',':'))+'\n')
    prefix=controls[1]['resource_guard']['unit_prefix'];hosts=[]
    for index,host in enumerate(controls[2]):
        for g in host['guards'].values():
            g['wall_seconds']=1.;g['file_limit_bytes']=8*2**20
            for phase in ['before','after']:
                g[phase]['memory.max']=str(1024*2**20)
                g[phase]['cpuset.cpus.effective']=','.join(map(str,recovery.ALTERNATE_CPUS))
        ranks=[dict(rank=r,host=host['host'],cpu=recovery.ALTERNATE_CPUS[r%8],pmi_fd=r+3,child_returncode=0,
                    time_record=host['ranks'][str(r)]['time_record'].replace('15:00.00','0:00.20')) for r in range(index*8,(index+1)*8)]
        hosts.append(dict(host=host['host'],active_units=[],ranks=ranks,guards=host['guards']))
    def remote(node,code):ast.parse(code);return hosts[recovery.old.NODES.index(node)]
    monkeypatch.setattr(recovery,'remote',remote);monkeypatch.setattr(recovery,'backup',lambda *a:None)
    result=dict(resource_guard=dict(unit_prefix=prefix),error=None,wall_seconds=5.,returncodes=dict(controller=0,**{f'proxy-{i}':0 for i in range(4)}))
    return case,tmp_path,opts,result,dict(smoke_catalog={}),hosts


def test_all_32_ranks_and_five_guards_required_for_smoke_success(smoke_fixture):
    *args,hosts=smoke_fixture
    report=recovery.collect_smoke(*args)
    assert report['passed'] and report['neural_simulations']==0 and len(report['guards'])==5


@pytest.mark.parametrize('mutate',[
    lambda h:h[3]['ranks'].pop(),
    lambda h:h[0]['ranks'][0].update(cpu=0),
    lambda h:h[0]['ranks'][0].update(child_returncode=1),
    lambda h:h[0]['guards'].pop('controller'),
    lambda h:h[0]['ranks'][0].update(time_record=h[0]['ranks'][0]['time_record'].replace('0:00.20','15:00.00')),
    lambda h:h[0]['guards']['proxy-0']['after'].update({'memory.events':'max 1\noom 0\noom_kill 0\noom_group_kill 0'}),
])
def test_smoke_missing_failed_or_wrong_binding_not_accepted(smoke_fixture,mutate):
    *args,hosts=smoke_fixture;mutate(hosts)
    with pytest.raises(ValueError):recovery.collect_smoke(*args)
    assert not (args[0]/'smoke/completion.json').exists()
