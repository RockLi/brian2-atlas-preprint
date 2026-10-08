"""Synthetic science publication and real bounded reference bundle tests."""
import ast,copy,hashlib,io,json,sys,tarfile
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_launch_confirmation_analysis as launch
from test_mam_confirmation_terminal import controls
from test_mam_confirmation_raw import bound
from test_mam_launch_confirmation_raw import publication
from test_mam_confirmation_analysis import accepted

T7=Path('/atlas-storage/0002/brian2-mpi-20260908')


def test_missing_raw_acceptance_has_no_remote_actions(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'remote',lambda *a,**k:pytest.fail('unexpected network'))
    r=launch.run(tmp_path/'evidence',tmp_path/'t7')
    assert not r['ready'] and not r['analysis_started'] and list(tmp_path.iterdir())==[]


def test_full_source_and_reference_bundle_is_bounded_and_verified(accepted):
    case,digest=accepted;root=Path(__file__).resolve().parents[1]
    payload,catalog,identity,raw,terminal=launch.bundle(root,case,T7,digest)
    assert raw['raw_output_audit_passed'] and terminal['terminal_resource_audit_passed'] and identity['replicate']==1750
    assert len(payload)<32*2**20 and sum(r['bytes'] for r in catalog.values())<64*2**20
    with tarfile.open(fileobj=io.BytesIO(payload)) as t:
        assert set(t.getnames())==set(catalog)|{'catalog.json'}
        for name,item in catalog.items():
            data=t.extractfile(name).read()
            assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
    assert {n:catalog[n] for n in launch.NORMS}==launch.NORMS
    assert 'interarea/fc-reference/reconstructed.npz' in catalog and 'interarea/lag-reference/report.json' in catalog
    for code in [launch.preflight_code(),launch.stage_code(hashlib.sha256(payload).hexdigest(),catalog),launch.collect_code(catalog)]:
        ast.parse(code);assert len(code.encode())<100000


@pytest.fixture
def science(controls):
    g=copy.deepcopy(controls[4][0]['guards']['proxy-0']);completion='b'*64;previous=20.;limit=10600
    g.update(cgroup='/system.slice/'+launch.UNIT+'.service',command=launch.application(completion,previous),
        root_volume_allowed=False,file_limit_bytes=512*2**20)
    for phase in ['before','after']:
        g[phase].update({'memory.max':str(16*2**30),'cpuset.cpus.effective':'8-9','cpu.max':'200000 100000'})
    catalog={'fixture':dict(bytes=1,sha256='c'*64)}
    pending=dict(schema='b2-mam-confirmation-six-metrics-pending-v1',replicate=1750,identity_sha256=launch.IDENTITY_SHA,
        completion_sha256=completion,model_sha256=controls[0]['model_sha256'],runtime_random_key=controls[0]['random_keys']['runtime_input'],raw_report_sha256='e'*64,source_catalog_sha256=hashlib.sha256((json.dumps(catalog,indent=2)+'\n').encode()).hexdigest(),
        full_descriptive_analysis_complete=True,analysis_guard_passed=False,pending_guard_acceptance=True,
        scientific_acceptance=False,performance_cost_acceptance=False,formal_equivalence_acceptance=False,
        stages=[dict(stage=n,state=s) for n in launch.REQUIRED for s in ['started','complete']],
        catalogs={n:'d'*64 for n in launch.REQUIRED},previous_seconds=previous,shared_budget_seconds=10800,
        science_seconds=850.,total_accounted_seconds=870.)
    for index,(name,cap,command) in enumerate(launch.stages(launch.SOURCE,controls[0])):
        pending['stages'][index*2].update(command=[launch.PYTHON,*command[1:]],timeout_seconds=cap)
    controller=dict(returncode=0,error=None,command=launch.command(completion,previous,limit))
    return pending,g,controller,dict(completion_sha=completion,identity=controls[0],raw_report_sha='e'*64,previous=previous,limit=limit,catalog=catalog,elapsed=1000.)


def test_science_guard_and_complete_six_stages_do_not_confer_equivalence(science):
    p,g,c,k=science;r=launch.publish(p,g,c,**k)
    assert r['analysis_guard_passed'] and not r['pending_guard_acceptance'] and not r['scientific_acceptance']
    assert not r['performance_cost_acceptance'] and not r['formal_equivalence_acceptance']
    assert r['resource_accounting']['peak_bytes']==1024 and not r['automatic_retry']


@pytest.mark.parametrize('fault',['oom','timeout','missing_stage','identity','equivalence','elapsed','source','model','stage_command'])
def test_incomplete_science_cannot_publish(science,fault):
    p,g,c,k=science
    if fault=='oom':g['after']['memory.events']='max 0\noom 1\noom_kill 0\noom_group_kill 0'
    elif fault=='timeout':c.update(returncode=None,error='observation timeout')
    elif fault=='missing_stage':p['stages'].pop()
    elif fault=='identity':p['identity_sha256']='0'*64
    elif fault=='equivalence':p['scientific_acceptance']=True
    elif fault=='elapsed':k['elapsed']=10801
    elif fault=='model':p['runtime_random_key']=1750
    elif fault=='stage_command':p['stages'][0]['command'].append('different')
    else:p['source_catalog_sha256']='0'*64
    with pytest.raises(ValueError):launch.publish(p,g,c,**k)


def test_guard_reserves_collection_time_without_resetting_science_budget():
    with pytest.raises(ValueError):launch.command('a'*64,600,10000)
    with pytest.raises(ValueError):launch.command('a'*64,20,10601)
    cmd=launch.command('a'*64,20,10600)
    assert '--property=RuntimeMaxSec=10605' in cmd and '--property=MemoryMax=16384M' in cmd
    assert '--property=MemorySwapMax=0' in cmd and '--property=CPUQuota=200%' in cmd
    assert cmd[cmd.index('--timeout')+1]=='10600' and cmd[cmd.index('--min-free-gib')+1]=='1280'
