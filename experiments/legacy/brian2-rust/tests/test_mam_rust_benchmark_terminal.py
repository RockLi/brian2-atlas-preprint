"""Synthetic controls exercise failure gates; they are not simulation evidence."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_collect_rust_benchmark_terminal as collector
from mam_rust_benchmark_terminal import audit,CPUS,MODEL,PLAN,NODES,PROTOCOL_SHA,EXE_SHA
from mam_launch_rust_performance import CASE,LABEL,PACKAGE_SHA,launch_options


@pytest.fixture
def controls():
    sources={name:dict(bytes=10,sha256=str(i+1)*64) for i,name in enumerate(
        ['mam_rust_terminal_sync.py','mpi_resource_guard.py','mam_rust_guard23.py'])}
    sources['protocol.json']=dict(bytes=10,sha256=PROTOCOL_SHA)
    resources=dict(nodes=NODES,ranks_per_node=8,cpu_ids=CPUS,memory_bytes_per_service=256*2**30,
        cpu_quota_cores_per_service=8,zero_swap=True,pids_max=64,file_limit_bytes=65504*2**20,
        minimum_free_bytes_by_host={h:(1280 if i==0 else 128)*2**30 for i,h in enumerate(NODES)},
        total_final_output_bytes=128*2**30,spool_bytes=96*2**30,wall_seconds=64800)
    app=launch_options(Path('/unused'))['application']
    admission=dict(schema='b2-mam-rust-performance-admission-v1',admitted=True,case_id=CASE,label=LABEL,
        protocol_sha256=PROTOCOL_SHA,model_sha256=MODEL,plan_sha256=PLAN,executable_sha256=EXE_SHA,
        package_sha256=PACKAGE_SHA,identity=dict(ranks=32,threads=1,seed=1729,duration_ms=100500,dt_ms=.1),
        resources=resources,source_catalog=sources,launch_options=dict(application=app),
        preflight=[dict(host=h,all_artifact_hashes_verified=True,source_catalog_verified=True,
            active_units=[],memory=dict(MemAvailable=600*2**30),free_bytes=2200*2**30) for h in NODES])
    prefix='b2mpi-0123456789ab'
    launch=dict(schema='b2-teleport-hydra-launch-v0',error=None,nodes=NODES,ranks_per_node=8,
        remote_timeout_seconds=64800,returncodes={r:0 for r in ['controller']+['proxy-'+str(i) for i in range(4)]},
        resource_guard=dict(unit_prefix=prefix),wall_seconds=1000.,
        commands=dict(controller=['tsh','ssh','root@'+NODES[0],shlex.join(['mpiexec','-n','32',*app])]))
    runtime=dict(schema='b2-mpi-runtime-v0',ranks=32,plan_sha256=PLAN,exchange_calls_per_rank=1005000,
        processor_names=[h for h in NODES for _ in range(8)],rank_cpu_ids=CPUS*4,
        rank_work=[x for r in range(32) for x in [1 if r<31 else 4129924-31,1,2]],
        spike_history_records=32,spike_history_storage='bounded-disk-spool',spike_spool_bytes=256,
        spike_spool_maximum_bytes=96*2**30,spike_spool_maximum_population_bytes=16*2**30,
        rank_stage_columns=['initialization_local','initialization_wait','simulation_local','simulation_wait','result_collection_and_reporting'],
        rank_stage_seconds=[1.]*160)
    summary=dict(schema='b2-result-dump-v4',population_count=254,neuron_count=4129924,
        final_time_seconds=100.5,mpi=runtime,dump_bytes=256,event_dump_bytes=256,spike_count=32,synaptic_events=64)
    output=dict(summary_json=json.dumps(summary),runtime_json=json.dumps(runtime),
        binary_file_bytes={'results.bin':256,'events.bin':256},spool_present=False)
    sync=dict(file_sync_calls=4,output_and_parent_directories_synced=True,metadata_readback_verified=True,
        binary_contents_independently_audited=False,elapsed_seconds=.1,
        files={name:dict(bytes=256) for name in ['results.bin','events.bin']})
    for name,key in [('summary.json','summary_json'),('mpi-runtime.json','runtime_json')]:
        raw=output[key].encode();sync['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    hosts=[]
    for index,host in enumerate(NODES):
        guards={}
        for role in (['controller'] if index==0 else [])+['proxy-'+str(index)]:
            before={'memory.max':str(256*2**30),'memory.swap.max':'0','pids.max':'64',
                'cpuset.cpus.effective':','.join(map(str,CPUS)),'cpu.max':'800000 100000',
                'memory.events':'max 0\noom 0\noom_kill 0\noom_group_kill 0\n',
                'cpu.stat':'usage_usec 0\nuser_usec 0\nsystem_usec 0\n'}
            after={**before,'memory.peak':'1024','cpu.stat':'usage_usec 5000000\nuser_usec 3000000\nsystem_usec 2000000\n'}
            reserve=(1280 if index==0 else 128)*2**30
            guards[role]=dict(schema='b2-mpi-resource-guard-v1',host=host,
                cgroup='/system.slice/'+prefix+'-'+role+'.service',admitted=True,uid=1000,returncode=0,
                wall_seconds=900.,before=before,after=after,data_volume='/data/brick2' if index==0 else '/',
                root_volume_allowed=True,data_device=2 if index==0 else 1,root_device=1,
                file_limit_bytes=65504*2**20,minimum_free_bytes=reserve,data_free_bytes=reserve+100*2**30,
                minimum_observed_free_bytes=reserve+99*2**30,reserved_host_memory_bytes=64*2**30,
                host_memory_bytes=dict(MemAvailable=600*2**30))
        ranks={}
        for rank in range(index*8,(index+1)*8):
            common=dict(rank=rank,ranks=32,host=host,executable_sha256=EXE_SHA,plan_sha256=PLAN,
                wrapper_sha256=sources['mam_rust_terminal_sync.py']['sha256'])
            done=dict(**common,event='rank_wrapper_complete',child_returncode=0,whole_job_accepted=False,
                scientific_acceptance=False,wall_through_child_and_output_sync_seconds=899.,
                leader_output_synchronization=sync if rank==0 else None)
            ranks[str(rank)]=dict(started_json=json.dumps(dict(**common,event='rank_wrapper_started')),
                done_json=json.dumps(done),time_record='User time (seconds): 1.0\nSystem time (seconds): 0.5\n'
                'Maximum resident set size (kbytes): 1024\nElapsed (wall clock) time (h:mm:ss or m:ss): 15:00.00\nExit status: 0\n')
        hosts.append(dict(host=host,active_own_units=[],failure_files=[],source_catalog=sources,
            artifact_catalog_verified=True,mpi_runtime_verified=True,guards=guards,ranks=ranks,
            leader_outputs=output if index==0 else None))
    return admission,launch,hosts


def edit_done(controls,modify,rank=0):
    entry=controls[2][rank//8]['ranks'][str(rank)]
    done=json.loads(entry['done_json']);modify(done);entry['done_json']=json.dumps(done)


def test_complete_controls_keep_raw_science_and_cost_unaccepted(controls):
    result=audit(*controls)
    assert len(result['ranks'])==32 and len(result['guards'])==5
    assert result['terminal_resource_audit_passed'] and result['metadata_sync_audit_passed']
    assert not result['raw_output_audit_passed'] and not result['scientific_acceptance']
    assert not result['performance_cost_acceptance']
    assert result['accounting']['measured_cgroup_core_hours']==25/3600
    assert result['accounting']['worker_capacity_core_hours']==32000/3600
    assert result['accounting']['monetary_cost'] is None


@pytest.mark.parametrize('clock,wrapper,accepted',[
    ('10:46:41',38801.93172843009,True),
    ('10:46:41',38801.999999,True),
    ('10:46:41',38802.,False),
    ('10:46:41',38802.01,False),
    ('59:59.99',3599.999,True),
    ('59:59.99',3600.,False),
    ('1:00:00',3600.999,True),
    ('1:00:00',3601.,False),
    ('15:00.00',900.009,True),
    ('15:00.00',900.02,False),
    ('10:46:41.0',38801.,False),
    ('60:00.00',3600.,False),
])
def test_elapsed_truncation_bounds_are_format_specific(controls,clock,wrapper,accepted):
    controls[1]['wall_seconds']=40000.
    entry=controls[2][0]['ranks']['0']
    entry['time_record']=entry['time_record'].replace('15:00.00',clock)
    edit_done(controls,lambda d:d.update(wall_through_child_and_output_sync_seconds=wrapper))
    if not accepted:
        with pytest.raises(ValueError):audit(*controls)
    else:
        row=audit(*controls)['ranks'][0]
        assert row['elapsed_seconds']<=wrapper<row['elapsed_upper_bound_exclusive_seconds']
        assert row['elapsed_quantization_seconds']==(1. if clock.count(':')==2 else .01)


def test_quantization_does_not_allow_wrapper_beyond_launch_wall(controls):
    controls[1]['wall_seconds']=38801.5
    entry=controls[2][0]['ranks']['0']
    entry['time_record']=entry['time_record'].replace('15:00.00','10:46:41')
    edit_done(controls,lambda d:d.update(wall_through_child_and_output_sync_seconds=38801.9))
    with pytest.raises(ValueError,match='timing/RSS'):audit(*controls)


@pytest.mark.parametrize('mutate',[
    lambda f:f[2][3]['ranks'].pop('31'),
    lambda f:f[2][0]['failure_files'].append('rank0.failed.json'),
    lambda f:edit_done(f,lambda d:d.update(child_returncode=1)),
    lambda f:f[1]['returncodes'].update({'proxy-3':None}),
    lambda f:f[2][0]['guards']['proxy-0']['after'].update({'memory.events':'max 1\noom 0\noom_kill 0\noom_group_kill 0'}),
    lambda f:f[2][0]['guards']['proxy-0'].update(minimum_observed_free_bytes=1279*2**30),
    lambda f:f[2][0]['guards']['proxy-0'].update(file_limit_bytes=128*2**30),
    lambda f:f[2][0]['leader_outputs'].update(summary_json='{}'),
    lambda f:edit_done(f,lambda d:d['leader_output_synchronization'].update(metadata_readback_verified=False)),
    lambda f:f[1]['commands']['controller'].__setitem__(-1,'mpiexec -n 32 unwrapped-program'),
    lambda f:f[2][1].update(leader_outputs={}),
    lambda f:edit_done(f,lambda d:d.update(leader_output_synchronization={}),rank=8),
])
def test_missing_failed_or_out_of_budget_controls_rejected(controls,mutate):
    mutate(controls)
    with pytest.raises(ValueError):audit(*controls)


def test_rehashed_metadata_still_requires_runtime_semantics(controls):
    out=controls[2][0]['leader_outputs'];runtime=json.loads(out['runtime_json']);runtime['ranks']=31
    summary=json.loads(out['summary_json']);summary['mpi']=runtime
    out['runtime_json']=json.dumps(runtime);out['summary_json']=json.dumps(summary)
    def rehash(done):
        for name,key in [('summary.json','summary_json'),('mpi-runtime.json','runtime_json')]:
            raw=out[key].encode()
            done['leader_output_synchronization']['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    edit_done(controls,rehash)
    with pytest.raises(ValueError,match='runtime'):audit(*controls)


def args_for(tmp_path,controls,monkeypatch):
    values=dict(admission=controls[0],launch=controls[1],package=dict(catalog={},mpi_runtime={}))
    monkeypatch.setattr(collector,'pinned',lambda path,expected,cap:values[path.name])
    return SimpleNamespace(admission=Path('admission'),admission_sha256='a'*64,launch=Path('launch'),
        launch_sha256='b'*64,package=Path('package'),output=tmp_path/'terminal',wall_seconds=900)


def test_collection_uses_four_bounded_reads_and_keeps_raw_gate(tmp_path,controls,monkeypatch):
    args=args_for(tmp_path,controls,monkeypatch);calls=[]
    def transport(command,**kw):
        index=len(calls);calls.append(command)
        remote=shlex.split(command[-1]);ast.parse(remote[-1])
        assert remote[:5]==['taskset','-c','8,9','python3','-c'] and kw['timeout']==45
        kw['stdout'].write(gzip.compress(json.dumps(controls[2][index]).encode()))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(collector.subprocess,'run',transport)
    result=collector.run(args)
    assert len(calls)==4 and result['terminal_resource_audit_passed']
    assert not result['raw_binary_payloads_collected'] and not result['automatic_retry']
    assert (args.output/'report.json').exists()


def test_nonterminal_launch_stops_before_network_and_output(tmp_path,controls,monkeypatch):
    args=args_for(tmp_path,controls,monkeypatch);controls[1]['returncodes']['proxy-0']=None
    monkeypatch.setattr(collector.subprocess,'run',lambda *a,**k:pytest.fail('unexpected transport'))
    with pytest.raises(ValueError,match='terminal'):collector.run(args)
    assert not args.output.exists()


def test_transport_failure_retains_evidence_without_retry(tmp_path,controls,monkeypatch):
    args=args_for(tmp_path,controls,monkeypatch);calls=[]
    def transport(command,**kw):
        calls.append(command);kw['stdout'].write(b'partial controls');return SimpleNamespace(returncode=1)
    monkeypatch.setattr(collector.subprocess,'run',transport)
    with pytest.raises(ValueError,match='no retry'):collector.run(args)
    assert len(calls)==1 and (args.output/'host-0.json.gz').read_bytes()==b'partial controls'
    failure=json.loads((args.output/'failure.json').read_text())
    assert failure['completed_hosts']==0 and not failure['automatic_retry']
