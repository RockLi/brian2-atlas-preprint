"""New admission boundaries and resource overrides; no actual SSH in tests."""
from pathlib import Path
import importlib.util
import shlex
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bounded_scale_launcher',HERE/'tools/mpi_teleport_launch.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
spec=importlib.util.spec_from_file_location('scale_layout',HERE/'prepare.py')
layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)


def arguments(tmp_path):
    return dict(nodes=['node-a'],ips=['192.0.2.1'],ranks_per_node=8,
        remote_base='/atlas-home/0003/scaling-test',application=['true'],output=tmp_path/'launch',
        login='root',guard_script='/atlas-home/0003/scaling-test/guard.py',guard_volume='/',
        guard_allow_root_volume=True,guard_memory_mib=665*1024,guard_cpu_count=8,
        guard_cpu_percent=800,guard_cpu_ids=[1,13,25,37,49,61,73,85])


@pytest.mark.parametrize('memory',[0,63,786433,True,768.0,'786432'])
def test_override_memory_rejected_before_spawn(tmp_path,monkeypatch,memory):
    args=arguments(tmp_path)
    args['guard_node_overrides']={'node-a':dict(volume='/',remote_base='/atlas-home/0003/scaling-test',
        script='/atlas-home/0003/scaling-test/guard.py',memory_mib=memory)}
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda *a,**k:pytest.fail('invalid allocation spawned a process'))
    with pytest.raises(ValueError):launcher.launch(**args)


@pytest.mark.parametrize('memory',[63,786433,True,786432.0,'680960'])
def test_global_memory_rejected_before_spawn(tmp_path,monkeypatch,memory):
    args=arguments(tmp_path);args['guard_memory_mib']=memory
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda *a,**k:pytest.fail('invalid allocation spawned a process'))
    with pytest.raises(ValueError):launcher.launch(**args)


@pytest.mark.parametrize('override',[None,64,665*1024,768*1024])
def test_accepted_cap_matches_both_systemd_and_guard(tmp_path,monkeypatch,override):
    args=arguments(tmp_path)
    if override is not None:
        args['guard_node_overrides']={'node-a':dict(volume='/',remote_base='/atlas-home/0003/scaling-test',
            script='/atlas-home/0003/scaling-test/guard.py',memory_mib=override)}
    captured=[]
    def stop_before_ssh(command,**kwargs):
        captured.append(command);raise OSError('dry-run boundary')
    monkeypatch.setattr(launcher.subprocess,'Popen',stop_before_ssh)
    with pytest.raises(OSError,match='dry-run boundary'):launcher.launch(**args)
    command=shlex.split(captured[0][-1]);memory=override if override is not None else 665*1024
    assert '--property=MemoryMax='+str(memory)+'M' in command
    assert command[command.index('--memory-mib')+1]==str(memory)
    assert '--property=MemorySwapMax=0' in command and '--uid=rock' in command


@pytest.mark.parametrize('hosts',[3,6,12,24,30])
def test_repeated_workload_and_incoming_edge_domains(hosts):
    sizes,owners,e_pools=layout.layout(hosts)
    assert sum(sizes)==86_000_000*(hosts//3)
    assert sum(sizes[:e_pools])==sum(sizes)*4//5
    assert sorted(owners)==list(range(hosts*8))
    e_block=[3_621_053]*12+[3_621_052]*7
    assert sizes[:e_pools]==e_block*(hosts//3)
    assert sizes[e_pools:]==[3_440_000]*(5*hosts//3)
    counts=layout.edge_counts(sizes);p=len(sizes)
    assert max(sum(counts[s*p+t] for s in range(p)) for t in range(p))<4_290_000_000
    nodes=[sum(n for n,owner in zip(sizes,owners) if owner//8==h) for h in range(hosts)]
    assert max(nodes)/min(nodes)<1.007
