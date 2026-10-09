"""Exercise admission and actual cgroup checks in the isolated policy guard."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('policy48_guard',Path(__file__).with_name('guard-host-reserve48-v3.py'))
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)


def environment(monkeypatch,tmp_path,available_gib=704,changes=None,uid=1000):
    requested=655
    values={'memory.max':str(requested*2**30),'memory.swap.max':'0',
            'memory.current':'0','memory.peak':'0','memory.events':'oom 0\noom_kill 0',
            'cpu.max':'800000 100000','cpu.stat':'usage_usec 1','pids.max':'64',
            'cpuset.cpus.effective':'1,13,25,37,49,61,73,85'}
    values.update(changes or {})
    original_read=Path.read_text;original_exists=Path.exists
    def read(path,*args,**kwargs):
        name=str(path)
        if name=='/proc/self/cgroup':return '0::/system.slice/policy48-test.service\n'
        if name=='/proc/meminfo':return f'MemAvailable: {int(available_gib*2**20)} kB\nMemTotal: {1024*2**20} kB\n'
        if name.startswith('/sys/fs/cgroup/'):return values[path.name]
        return original_read(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(Path,'exists',lambda path:True if str(path).startswith('/sys/fs/cgroup/') else original_exists(path))
    monkeypatch.setattr(guard.os,'getuid',lambda:uid)
    monkeypatch.setattr(guard.os,'statvfs',lambda _:SimpleNamespace(f_bavail=512*2**30,f_frsize=1))
    calls=[];limits=[]
    monkeypatch.setattr(guard.resource,'setrlimit',lambda kind,value:limits.append((kind,value)))
    class Process:
        returncode=0
        def poll(self):return 0
    def spawn(command,**kwargs):
        calls.append(command);kwargs['preexec_fn']();return Process()
    monkeypatch.setattr(guard.subprocess,'Popen',spawn)
    args=SimpleNamespace(output=tmp_path/'guard.json',volume=Path('/'),allow_root_volume=True,
                         memory_mib=requested*1024,cpu_percent=800,file_mib=64*1024,
                         min_free_gib=128,timeout=1800,command=['true'])
    return args,calls,limits


@pytest.mark.parametrize('available_gib',[703,704])
def test_positive_admission_retains_all_kernel_and_file_limits(monkeypatch,tmp_path,available_gib):
    args,calls,limits=environment(monkeypatch,tmp_path,available_gib)
    assert guard.run(args)==0
    record=json.loads(args.output.read_text())
    assert record['admitted'] and record['reserved_host_memory_bytes']==48*2**30
    assert record['before']['memory.max']==str(655*2**30)
    assert record['before']['memory.swap.max']=='0'
    assert record['minimum_free_bytes']==128*2**30
    assert record['file_limit_bytes']==64*2**30
    assert limits==[(guard.resource.RLIMIT_FSIZE,(64*2**30,64*2**30))]
    assert calls==[['true']]


def test_insufficient_available_memory_starts_no_child(monkeypatch,tmp_path):
    args,calls,_=environment(monkeypatch,tmp_path,702)
    with pytest.raises(RuntimeError,match='memory headroom'):guard.run(args)
    assert not calls and not json.loads(args.output.read_text())['admitted']


@pytest.mark.parametrize('changes',[
    {'memory.max':'max'}, {'memory.max':str(656*2**30)},
    {'memory.swap.max':'1'}, {'cpu.max':'max 100000'},
    {'cpu.max':'900000 100000'}, {'pids.max':'max'}, {'pids.max':'65'},
])
def test_missing_or_excessive_cgroup_limits_start_no_child(monkeypatch,tmp_path,changes):
    args,calls,_=environment(monkeypatch,tmp_path,changes=changes)
    with pytest.raises(RuntimeError):guard.run(args)
    assert not calls and not json.loads(args.output.read_text())['admitted']


def test_child_cannot_run_as_root(monkeypatch,tmp_path):
    args,calls,_=environment(monkeypatch,tmp_path,uid=0)
    with pytest.raises(RuntimeError,match='unprivileged'):guard.run(args)
    assert not calls
