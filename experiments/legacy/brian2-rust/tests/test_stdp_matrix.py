"""The matrix preserves gates, physical allocation and full artifact boundaries."""
from pathlib import Path
import pytest

@pytest.fixture
def matrix(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    import modal_stdp_matrix
    return modal_stdp_matrix

@pytest.mark.parametrize('names,repeats,backends',[
    ([],5,None),(['lower-drive','lower-drive'],5,None),(['unknown'],5,None),
    (['lower-drive'],True,None),(['lower-drive'],6,None),(['lower-drive'],5,['cuda']),
    (['lower-drive'],5,['cpu-f32','cpu-f32']),(['lower-drive'],5,['cpu-f32','bad']),
])
def test_invalid_requests_do_not_start_workers(matrix,monkeypatch,names,repeats,backends):
    import modal_stdp_precompiled
    monkeypatch.setattr(modal_stdp_precompiled,'compare',lambda *a,**kw:pytest.fail('worker started'))
    with pytest.raises(ValueError):matrix.compare_matrix(names,repeats,backends)

@pytest.mark.parametrize('status',['passed-matched-numeric-gates','completed-with-gate-failures','failed'])
def test_case_boundaries_and_failed_gates_are_preserved(matrix,monkeypatch,status):
    import modal_stdp_precompiled
    calls=[]
    def compare(n,t,k,repeats,**kw):
        calls.append((n,t,k,repeats,kw))
        return dict(nvidia_smi='A100, same-uuid, driver',payloads={'bootstrap.npz':b'full data'},status=status if len(calls)==1 else 'passed-matched-numeric-gates')
    monkeypatch.setattr(modal_stdp_precompiled,'compare',compare)
    r=matrix.compare_matrix(['lower-drive','long-delays'],3,['rust-f64','cpu-f32','cuda'])
    assert r['status']==status and len(r['cases'])==2 and len(calls)==2
    assert r['payloads']=={'lower-drive/bootstrap.npz':b'full data','long-delays/bootstrap.npz':b'full data'}
    for call in calls:
        assert call[:4]==(1024,256,8,3)
        assert call[4]['continue_on_gate_failure'] is True
    assert [c[4]['workload']['delay_span'] for c in calls]==[8,16]
    assert [c[4]['workload']['post_delay'] for c in calls]==[3,16]


@pytest.mark.parametrize('initial',['GPU-1',None])
def test_changed_gpu_rejected(matrix,monkeypatch,initial):
    import modal_stdp_precompiled
    devices=iter([initial,'GPU-2'])
    monkeypatch.setattr(modal_stdp_precompiled,'compare',lambda *a,**kw:dict(nvidia_smi=next(devices),payloads={},status='passed-matched-numeric-gates'))
    with pytest.raises(RuntimeError,match='identity'):matrix.compare_matrix(['lower-drive','long-delays'])


def test_artifact_path_rejected(matrix,monkeypatch):
    import modal_stdp_precompiled
    monkeypatch.setattr(modal_stdp_precompiled,'compare',lambda *a,**kw:dict(nvidia_smi='GPU-1',payloads={'../escape':b'data'},status='passed-matched-numeric-gates'))
    with pytest.raises(ValueError,match='filename'):matrix.compare_matrix(['lower-drive'])


def test_unfinished_status_never_passes(matrix,monkeypatch):
    import modal_stdp_precompiled
    monkeypatch.setattr(modal_stdp_precompiled,'compare',lambda *a,**kw:dict(nvidia_smi='GPU-1',payloads={},status='running'))
    with pytest.raises(RuntimeError,match='status'):matrix.compare_matrix(['lower-drive'])


def test_random_cases_preserve_sizes_and_graph_seed(matrix,monkeypatch):
    import modal_stdp_precompiled
    calls=[]
    def compare(n,t,k,repeats,**kw):
        calls.append((n,t,k,kw))
        return dict(nvidia_smi='same GPU',payloads={},status='passed-matched-numeric-gates')
    monkeypatch.setattr(modal_stdp_precompiled,'compare',compare)
    matrix.compare_matrix(['random-sparse','random-larger'],5,['rust-f64','cpu-f32','cuda'])
    assert [c[:3] for c in calls]==[(1024,256,8),(4096,256,32)]
    for _,_,_,opts in calls:
        assert opts['workload']['topology_kind']=='random-fixed-outdegree'
        assert opts['workload']['topology_seed']==42
        assert opts['continue_on_gate_failure'] is True


def test_too_many_cases_rejected_before_modal_allocation(matrix,monkeypatch,tmp_path,capsys):
    import sys
    monkeypatch.setattr(sys,'argv',['matrix','--case','random-sparse','--case','random-larger',
        '--case','lower-drive','--output',str(tmp_path/'never-created')])
    with pytest.raises(SystemExit) as error:matrix.main()
    assert error.value.code==2 and 'one or two' in capsys.readouterr().err
    assert not (tmp_path/'never-created').exists()
