"""Synthetic full-geometry controls, not simulator-generated evidence."""
import copy
import ast
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from mam_benchmark_terminal import audit, WORKLOAD_SHA256
from mam_native_primary_resources import parameter_geometry
from mam_nest_reference import virtual_process_capacity
import mam_collect_benchmark_terminal as collector

PARAMETERS='ee1a642c94b9d6e808627c54f78162e7f5d87f4140ba56496c5dfee3ed900d3e'


@pytest.fixture
def fixture():
    p=dict(schema='b2-official-mam-parameters-v1',total_neurons=4129924,
           total_recurrent_synapses=24126516728,N_scaling=1.,K_scaling=1.,
           populations=[dict(name='P'+str(i),count=1 if i<253 else 4129924-253) for i in range(254)],
           projections=[dict(count=1 if i<8343 else 24126516728-8343) for i in range(8344)])
    populations,edges=parameter_geometry(p)
    sources={name:dict(bytes=10,sha256=str(i+1)*64) for i,name in enumerate(
        ['mam_nest_benchmark.py','mam_benchmark_recording.py','mam_benchmark_event_io.py'])}
    sources['mam_benchmark_workload_v1.json']=dict(bytes=10,sha256=WORKLOAD_SHA256)
    identity=dict(ranks=2,threads=96,seed=1729,dt_ms=.1,duration_ms=100500,nest_version='3.10.0')
    specs=[dict(host=h,role=role,memory_bytes=2**31,pids_max=128,cpu_ids=[0],cpu_quota_cores=1,
                volume='/home/rock',allow_root_volume=True,file_limit_bytes=2**30,
                minimum_free_bytes=2**30,reserved_host_memory_bytes=2**30)
           for h,role in [('h0','controller'),('h0','proxy-0'),('h1','proxy-1')]]
    a=dict(schema='b2-mam-benchmark-admission-v1',admitted=True,workload_sha256=WORKLOAD_SHA256,
           parameters_sha256=PARAMETERS,identity=identity,limits=dict(chunk_ms=50,automatic_retry=False,
           max_chunk_spikes=10,max_spikes_per_rank=10,total_event_bytes=160,wall_seconds=100),
           placements=[dict(rank=i,host='h'+str(i),cpu_ids=[0]) for i in range(2)],
           guards=specs,nodes=['h0','h1'],ranks_per_node=1,source_catalog=sources,runtime_catalog_sha256='f'*64)
    launch=dict(schema='b2-teleport-hydra-launch-v0',error=None,nodes=['h0','h1'],ranks_per_node=1,
                remote_timeout_seconds=100,returncodes={'controller':0,'proxy-0':0,'proxy-1':0},
                resource_guard=dict(unit_prefix='b2mpi-0123456789ab'),wall_seconds=50.)
    collected=[]
    for rank in range(2):
        host='h'+str(rank);guards={}
        for spec in [x for x in specs if x['host']==host]:
            before={'memory.max':str(2**31),'memory.swap.max':'0','pids.max':'128',
                    'cpuset.cpus.effective':'0','cpu.max':'100000 100000',
                    'memory.events':'max 0\noom 0\noom_kill 0\noom_group_kill 0\n',
                    'cpu.stat':'usage_usec 0\nuser_usec 0\nsystem_usec 0\n'}
            after={**before,'memory.peak':'1024','cpu.stat':'usage_usec 5000000\nuser_usec 3000000\nsystem_usec 2000000\n'}
            guards[spec['role']]=dict(schema='b2-mpi-resource-guard-v1',host=host,
                cgroup='/system.slice/b2mpi-0123456789ab-'+spec['role']+'.service',admitted=True,
                uid=1000,returncode=0,wall_seconds=40.,before=before,after=after,
                data_volume='/home/rock',root_volume_allowed=True,data_device=1,root_device=1,
                file_limit_bytes=2**30,minimum_free_bytes=2**30,data_free_bytes=4*2**30,
                minimum_observed_free_bytes=3*2**30,reserved_host_memory_bytes=2**30,
                host_memory_bytes={'MemAvailable':4*2**30})
        local=[n//2 if rank==0 else n-n//2 for n in edges]
        chunks=[dict(end_ms=(i+1)*50,spikes=2 if i==0 else 0,simulation_seconds=.001,
                     output_seconds=.001,monitor_progress_seconds=.001) for i in range(2010)]
        row=dict(schema='b2-native-nest-mam-benchmark-events-v2',rank=rank,host=host,**identity,
                 parameters_sha256=PARAMETERS,workload_sha256=WORKLOAD_SHA256,
                 producer_sha256=sources['mam_nest_benchmark.py']['sha256'],
                 source_sha256={k:sources[k]['sha256'] for k in ['mam_benchmark_recording.py','mam_benchmark_event_io.py']},
                 populations=populations,N_scaling=1.,K_scaling=1.,
                 connection_index_capacity=virtual_process_capacity(p,2,96),
                 projection_local_counts=local,local_recurrent_edges=sum(local),
                 local_external_connections=2064962,local_recording_connections=2064962,
                 local_total_connections=sum(local)+4129924,allowed_cpus=[0],
                 local_spikes=2,event_bytes=16,event_sha256='e'*64,
                 durable_event_receipt=dict(schema='b2-mam-durable-event-stream-v1',complete=True,
                     records=2,bytes=16,durable_bytes=16,sha256='e'*64,maximum_bytes=80,
                     buffer_bytes=65536,data_sync_interval_bytes=1048576,data_sync_calls=1,
                     cache_release_calls=1,directory_synced=True,io_backend='LinuxIO'),
                 event_readback_verified=True,chunks=chunks,simulation_seconds=2.01,event_output_seconds=2.01,
                 monitor_progress_seconds=2.01,recording_finish_seconds=.1,event_verification_seconds=.1,
                 recording_wall_seconds=10.,wall_seconds=12.,rank_pre_report_wall_seconds=15.,peak_rss_kib=1)
        raw=json.dumps(row)
        done=dict(event='rank_complete',rank=rank,event_readback_verified=True,
                  report=dict(bytes=len(raw.encode()),sha256=hashlib.sha256(raw.encode()).hexdigest()),
                  rank_wall_through_report_seconds=16.)
        collected.append(dict(host=host,active_own_units=[],failure_files=[],source_catalog=sources,
                              runtime_catalog_sha256='f'*64,guards=guards,
                              ranks={str(rank):dict(report_json=raw,done_json=json.dumps(done),
                                  initial_rank_cpu_ids=[0],observed_event_file_bytes=16)}))
    return a,launch,collected,p


def run(f):
    return audit(*f,parameters_sha256=PARAMETERS)


def change_report(f, modify, rehash=True):
    e=f[2][0]['ranks']['0'];row=json.loads(e['report_json']);modify(row)
    e['report_json']=json.dumps(row)
    if rehash:
        done=json.loads(e['done_json'])
        done['report']=dict(bytes=len(e['report_json'].encode()),sha256=hashlib.sha256(e['report_json'].encode()).hexdigest())
        e['done_json']=json.dumps(done)


def test_all_rank_controls_and_accounting_preserve_remaining_gates(fixture):
    result=run(fixture)
    assert result['terminal_resource_audit_passed'] and result['construction_ledger_passed']
    assert result['rank_completion_report_hashes_passed']
    assert result['reported_event_bytes']==32 and result['reported_spikes']==4
    assert not result['raw_output_audit_passed'] and not result['performance_cost_acceptance']
    assert result['accounting']['measured_cgroup_core_hours']==15/3600
    assert result['accounting']['worker_capacity_core_hours']==100/3600
    assert result['accounting']['shared_participating_node_hours']==100/3600
    assert result['accounting']['monetary_cost'] is None
    assert result['accounting']['exclusive_allocated_node_hours'] is None


@pytest.mark.parametrize('mutation', [
    lambda f:f[1]['returncodes'].pop('proxy-1'),
    lambda f:f[1]['returncodes'].update({'proxy-1':1}),
    lambda f:f[1].update(wall_seconds=101),
    lambda f:f[2][1]['ranks'].clear(),
    lambda f:f[2][0]['active_own_units'].append('still-running'),
    lambda f:f[2][0]['failure_files'].append('rank0.failed.json'),
    lambda f:f[2][0]['guards']['proxy-0']['after'].update({'memory.events':'max 1\noom 0\noom_kill 0\noom_group_kill 0'}),
    lambda f:f[2][0]['guards']['proxy-0']['after'].update({'memory.max':'100'}),
    lambda f:f[2][0]['guards']['proxy-0'].update(minimum_observed_free_bytes=0),
    lambda f:f[2][0]['guards']['proxy-0']['after'].update({'cpu.stat':'usage_usec -1\nuser_usec 0\nsystem_usec 0'}),
    lambda f:f[2][0]['ranks']['0'].update(observed_event_file_bytes=8),
    lambda f:f[2][0].update(runtime_catalog_sha256='b'*64),
    lambda f:f[0].update(workload_sha256='0'*64),
])
def test_missing_failed_or_out_of_budget_controls_are_rejected(fixture,mutation):
    mutation(fixture)
    with pytest.raises(ValueError):run(fixture)


@pytest.mark.parametrize('modify', [
    lambda r:r.update(seed=1730),
    lambda r:r.update(producer_sha256='0'*64),
    lambda r:r['projection_local_counts'].__setitem__(0,1),
    lambda r:r['chunks'].pop(),
    lambda r:r['chunks'][0].update(spikes=1),
    lambda r:r['durable_event_receipt'].update(directory_synced=False),
    lambda r:r['durable_event_receipt'].update(durable_bytes=8),
    lambda r:r.update(event_readback_verified=False),
    lambda r:r.update(rank_pre_report_wall_seconds=17.),
    lambda r:r.update(recording_wall_seconds=5.),
    lambda r:r.update(allowed_cpus=[1]),
])
def test_consistent_report_hash_does_not_bypass_semantic_checks(fixture,modify):
    change_report(fixture,modify)
    with pytest.raises(ValueError):run(fixture)


def test_report_tampering_without_new_terminal_digest_rejected(fixture):
    change_report(fixture,lambda r:r.update(local_spikes=99),rehash=False)
    with pytest.raises(ValueError,match='report bytes'):run(fixture)


def test_duplicate_json_key_cannot_override_identity(fixture):
    entry=fixture[2][0]['ranks']['0']
    entry['report_json']=entry['report_json'][:-1]+',"rank":1}'
    with pytest.raises(ValueError,match='duplicate JSON'):run(fixture)


def collection_args(tmp_path, fixture, monkeypatch):
    a,launch,collected,parameters=fixture
    a['host_paths']=[dict(base='/atlas-home/0003/test',project='project',run='runs/test',
                         guards='guards',affinity='affinity/test',runtime_catalog='runtime/catalog.json') for _ in range(2)]
    values={'admission':a,'launch':launch,'parameters':parameters}
    monkeypatch.setattr(collector,'pinned',lambda path,expected,cap:values[path.name])
    return SimpleNamespace(admission=Path('admission'),admission_sha256='a'*64,
                           launch=Path('launch'),launch_sha256='b'*64,parameters=Path('parameters'),
                           output=tmp_path/'controls',wall_seconds=120)


def test_collector_pipeline_with_synthetic_transport_is_not_raw_acceptance(tmp_path,fixture,monkeypatch):
    args=collection_args(tmp_path,fixture,monkeypatch);calls=[]
    def transport(command,**kw):
        index=len(calls);calls.append(command)
        kw['stdout'].write(gzip.compress(json.dumps(fixture[2][index]).encode()))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(collector.subprocess,'run',transport)
    result=collector.run(args)
    assert len(calls)==2 and all(command[:2]==['tsh','ssh'] for command in calls)
    assert result['terminal_resource_audit_passed'] and not result['raw_output_audit_passed']
    assert not result['raw_events_collected'] and not result['automatic_retry']
    assert (args.output/'host-0.json.gz').exists() and (args.output/'report.json').exists()


def test_nonterminal_launch_refused_before_any_transport_or_output(tmp_path,fixture,monkeypatch):
    args=collection_args(tmp_path,fixture,monkeypatch)
    fixture[1]['returncodes']['proxy-1']=None
    def forbidden(*a,**kw):
        pytest.fail('network must not run')
    monkeypatch.setattr(collector.subprocess,'run',forbidden)
    with pytest.raises(ValueError,match='terminal'):collector.run(args)
    assert not args.output.exists()


def test_transport_failure_retained_without_retry(tmp_path,fixture,monkeypatch):
    args=collection_args(tmp_path,fixture,monkeypatch);calls=[]
    def failure(command,**kw):
        calls.append(command);kw['stdout'].write(b'partial evidence');return SimpleNamespace(returncode=1)
    monkeypatch.setattr(collector.subprocess,'run',failure)
    with pytest.raises(ValueError,match='no retry'):collector.run(args)
    assert len(calls)==1 and (args.output/'host-0.json.gz').read_bytes()==b'partial evidence'
    failed=json.loads((args.output/'failure.json').read_text())
    assert failed['completed_hosts']==0 and failed['automatic_retry'] is False
    assert not (args.output/'report.json').exists()


def test_generated_host_read_is_bounded_and_does_not_open_event_bytes(tmp_path,fixture,monkeypatch):
    collection_args(tmp_path,fixture,monkeypatch)
    code=collector.host_code(fixture[0],1,'b2mpi-0123456789ab')
    ast.parse(code)
    assert 'signal.alarm(40)' in code and 'RLIMIT_AS' in code
    assert 'observed_event_file_bytes=event.stat().st_size' in code
    assert 'event.open(' not in code and 'read(event' not in code


def test_pinned_input_digest_and_size_are_checked(tmp_path):
    p=tmp_path/'input.json';p.write_text('{"value":1}')
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    assert collector.pinned(p,sha,100)==dict(value=1)
    with pytest.raises(ValueError):collector.pinned(p,'0'*64,100)
    with pytest.raises(ValueError):collector.pinned(p,sha,1)


def test_extended_transfer_deadline_keeps_remote_guard_and_audit(tmp_path,fixture,monkeypatch):
    args=collection_args(tmp_path,fixture,monkeypatch)
    args.host_timeout_seconds=120;args.wall_seconds=900;calls=[]
    def transport(command,**kw):
        assert kw['timeout']==120 and 'signal.alarm(115)' in command[-1]
        index=len(calls);calls.append(command)
        kw['stdout'].write(gzip.compress(json.dumps(fixture[2][index]).encode()))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(collector.subprocess,'run',transport)
    result=collector.run(args)
    assert result['host_timeout_seconds']==120 and result['terminal_resource_audit_passed']
    assert not result['raw_output_audit_passed'] and not result['automatic_retry']


@pytest.mark.parametrize('deadline',[0,44,121,True])
def test_transfer_deadline_outside_budget_refused_before_output(tmp_path,fixture,monkeypatch,deadline):
    args=collection_args(tmp_path,fixture,monkeypatch);args.host_timeout_seconds=deadline
    monkeypatch.setattr(collector.subprocess,'run',lambda *a,**kw:pytest.fail('no network before valid budget'))
    with pytest.raises(ValueError,match='host transfer deadline'):collector.run(args)
    assert not args.output.exists()


def test_short_duration_requires_explicit_tuning_purpose(fixture):
    fixture[0]['identity']['duration_ms']=2500
    with pytest.raises(ValueError,match='explicit target'):run(fixture)
    fixture[0]['run_purpose']='performance_tuning'
    for rank in range(2):
        entry=fixture[2][rank]['ranks'][str(rank)];row=json.loads(entry['report_json'])
        row.update(duration_ms=2500,chunks=row['chunks'][:50],simulation_seconds=.05,
                   event_output_seconds=.05,monitor_progress_seconds=.05)
        entry['report_json']=json.dumps(row);done=json.loads(entry['done_json'])
        done['report']=dict(bytes=len(entry['report_json'].encode()),sha256=hashlib.sha256(entry['report_json'].encode()).hexdigest())
        entry['done_json']=json.dumps(done)
    result=run(fixture)
    assert result['run_purpose']=='performance_tuning' and result['duration_ms']==2500
    assert not result['scientific_acceptance'] and not result['performance_cost_acceptance']
