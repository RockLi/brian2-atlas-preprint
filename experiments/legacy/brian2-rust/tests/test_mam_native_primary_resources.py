"""Synthetic full-coverage controls: these contain no simulation observations."""
import ast
import gzip
import json
import zlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mam_native_primary_resources as m
import mam_collect_native_primary_resources as collector


@pytest.fixture
def inputs():
    populations=[dict(name='fixture'+str(i),count=m.NEURONS//254+(i<m.NEURONS%254)) for i in range(254)]
    projections=[dict(count=m.EDGES//8344+(i<m.EDGES%8344)) for i in range(8344)]
    p=dict(schema='b2-official-mam-parameters-v1',total_neurons=m.NEURONS,
           total_recurrent_synapses=m.EDGES,N_scaling=1.,K_scaling=1.,populations=populations,projections=projections)
    geometry,counts=m.parameter_geometry(p)
    layout=dict(schema='b2-mam-nest-affinity-v1',ranks_per_host=8,threads_per_rank=4,
                hosts=[dict(host=n,rank_cpu_ids=[m.CPUS[i:i+4] for i in range(0,32,4)]) for n in m.NODES])
    options=m.launch_options(Path('/atlas-storage/0002/fixture'));options['output']=str(options['output'])
    admission=dict(schema='b2-mam-native-primary-admission-v1',label=m.LABEL,admitted=True,
        parameters_sha256=m.PARAMETERS,layout_sha256=m.LAYOUT_SHA,
        experiment_budget=dict(runs=1,automatic_retry=False,wall_seconds=54000,collection_seconds=7200,analysis_stage_seconds=10800),
        prerequisite={k:'a'*64 for k in ['resource_report_sha256','output_report_sha256','output_guard_sha256']},
        rust_leader_terminal=dict(rust_primary_active_units=[]),launch_options=options,
        preflight=[dict(host=n,available_memory_bytes=600*m.GIB,free_bytes=200*m.GIB,active_own_units=[],
            source_catalog_verified=True,runtime_catalog_verified=True,mpi_verified=True,selected_topology_verified=True,
            cpu_busy_percent={str(c):1. for c in m.CPUS}) for n in m.NODES])
    prefix='b2mpi-abcdef012345'
    launch=dict(schema='b2-teleport-hydra-launch-v0',error=None,nodes=m.NODES[:],ips=m.IPS[:],ranks_per_node=8,
        remote_timeout_seconds=54000,wall_seconds=200.,returncodes={r:0 for r in ['controller']+['proxy-'+str(i) for i in range(6)]},
        resource_guard=dict(script=m.PROJECT+'/mpi_resource_guard.py',volume='/',allow_root_volume=True,memory_mib=262144,
            cpu_percent=3200,cpu_count=32,cpu_ids=m.CPUS,file_mib=3072,unit_prefix=prefix,minimum_free_gib=128))
    chunks=[dict(end_ms=(i+1)*50,spikes=0,simulation_seconds=.01,output_seconds=.001) for i in range(2010)]
    hosts=[]
    for index,host in enumerate(m.NODES):
        guards={}
        for role in (['controller'] if index==0 else [])+['proxy-'+str(index)]:
            before={'memory.max':str(256*m.GIB),'memory.swap.max':'0','pids.max':'64',
                'cpuset.cpus.effective':','.join(str(c) for c in m.CPUS),'cpu.max':'3200000 100000',
                'memory.events':'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0',
                'cpu.stat':'usage_usec 100\nuser_usec 80\nsystem_usec 20'}
            after=dict(before,**{'memory.peak':str(m.GIB), 'cpu.stat':'usage_usec 3600000100\nuser_usec 3000000080\nsystem_usec 600000020'})
            guards[role]=dict(schema='b2-mpi-resource-guard-v1',host=host,uid=1000,admitted=True,returncode=0,
                cgroup='/system.slice/'+prefix+'-'+role+'.service',wall_seconds=199.,before=before,after=after,
                data_volume='/',root_volume_allowed=True,data_device=1,root_device=1,file_limit_bytes=3*m.GIB,
                minimum_free_bytes=128*m.GIB,disk_check_interval_seconds=5,data_free_bytes=200*m.GIB,
                minimum_observed_free_bytes=150*m.GIB,reserved_host_memory_bytes=64*m.GIB,
                host_memory_bytes={'MemAvailable':600*m.GIB})
        reports={};affinity={}
        for rank in range(index*8,(index+1)*8):
            local=[n//48+(rank<n%48) for n in counts]
            device=m.NEURONS//48+(rank<m.NEURONS%48)
            cpus,env=m.binding(layout,rank,host,m.CPUS)
            affinity[str(rank)]=dict(rank=rank,ranks=48,host=host,initial_rank_cpu_ids=cpus,
                openmp_environment=env,layout_sha256=m.LAYOUT_SHA)
            reports[str(rank)]=dict(schema='b2-native-nest-mam-v1',host=host,rank=rank,ranks=48,threads=4,
                seed=1729,dt_ms=.1,duration_ms=100500,parameters_sha256=m.PARAMETERS,nest_version='3.10.0',allowed_cpus=[cpus[0]],
                populations=geometry,N_scaling=1.,K_scaling=1.,connection_index_capacity=m.virtual_process_capacity(p,48,4),
                projection_local_counts=local,local_recurrent_edges=sum(local),local_external_connections=device,
                local_recording_connections=device,local_total_connections=sum(local)+2*device,local_spikes=0,event_bytes=0,
                event_sha256='b'*64,chunks=chunks,min_delay_ms=.1,max_delay_ms=30.,wall_seconds=90.,peak_rss_kib=1000,
                neuron_seconds=1.,recurrent_seconds=2.,device_seconds=1.,simulation_seconds=sum(c['simulation_seconds'] for c in chunks),
                phase_memory=dict(neurons_created=dict(rss_kib=100,peak_rss_kib=100,user_seconds=1.,system_seconds=.1),
                    after_100500ms=dict(rss_kib=900,peak_rss_kib=1000,user_seconds=10.,system_seconds=1.)))
        hosts.append(dict(host=host,guards=guards,rank_reports=reports,affinity=affinity,active_own_units=[]))
    return admission,launch,hosts,p,layout


def test_global_ledger_and_measured_resource_accounting(inputs):
    r=m.audit(*inputs)
    assert r['terminal_resource_audit_passed'] and r['construction_ledger_passed']
    assert len(r['guards'])==7 and len(r['ranks'])==48 and r['recurrent_edges']==24126516728
    assert r['accounting']['measured_cgroup_core_hours_including_controller']==7.
    assert r['accounting']['worker_allowed_capacity_core_hours']==192*200/3600
    assert r['accounting']['exclusive_allocated_node_hours'] is None
    assert r['accounting']['monetary_cost'] is None
    assert not r['raw_output_audit_passed'] and not r['scientific_acceptance']
    assert not r['performance_cost_acceptance'] and not r['all_actual_openmp_workers_observed']


def test_explicit_diagnostic_seed_keeps_all_resource_checks_and_refuses_mixed_identity(inputs):
    a,l,h,p,layout=inputs
    label='nest-mam-full-reference-v1-metastable-seed1730-100500ms'
    a['label']=label
    a['launch_options']=m.launch_options(Path('/atlas-storage/0002/fixture'),label=label,seed=1730)
    a['launch_options']['output']=str(a['launch_options']['output'])
    for host in h:
        for row in host['rank_reports'].values(): row['seed']=1730
    with pytest.raises(ValueError): m.audit(*inputs)
    result=m.audit(*inputs,label=label,seed=1730)
    assert result['label']==label and result['terminal_resource_audit_passed']
    h[5]['rank_reports']['47']['seed']=1729
    with pytest.raises(ValueError): m.audit(*inputs,label=label,seed=1730)


def test_new_control_reader_uses_retained_source_and_new_output_only():
    label='nest-mam-full-reference-v1-metastable-seed1730-100500ms'
    code=collector.host_code(0,'b2mpi-abcdef012345',{},run_label=label)
    ast.parse(code)
    assert repr(m.LABEL)+"+'/parameters.json'" in code
    assert "'runs/'+"+repr(label) in code and "'affinity/'+"+repr(label) in code
    assert "'runs/'+"+repr(m.LABEL) not in code


@pytest.mark.parametrize('fault',['live','failed-launch','short-duration','wrong-family','missing-guard',
    'memory-pressure','oom','quota','swap','old-cpus','disk-reserve','guard-wall','missing-rank',
    'wrong-host','wrong-worker-env','main-thread-cpus','projection-total','device-total','chunk-total',
    'short-chunks','rank-cpu-regression','rank-rss','source-unverified','busy-cpu','weakened-budget'])
def test_bad_full_native_controls_fail_closed(inputs,fault):
    a,l,h,p,layout=inputs;g=h[0]['guards']['proxy-0'];r=h[0]['rank_reports']['0'];receipt=h[0]['affinity']['0']
    if fault=='live':h[0]['active_own_units']=['fixture.service']
    elif fault=='failed-launch':l['returncodes']['proxy-5']=1
    elif fault=='short-duration':r['duration_ms']=10500
    elif fault=='wrong-family':g['cgroup']=g['cgroup'].replace('abcdef','123456')
    elif fault=='missing-guard':del h[0]['guards']['controller']
    elif fault=='memory-pressure':g['after']['memory.events']=g['after']['memory.events'].replace('max 0','max 1')
    elif fault=='oom':g['after']['memory.events']=g['after']['memory.events'].replace('oom 0','oom 1')
    elif fault=='quota':g['after']['cpu.max']='6400000 100000'
    elif fault=='swap':g['after']['memory.swap.max']='1024'
    elif fault=='old-cpus':g['after']['cpuset.cpus.effective']='0-31'
    elif fault=='disk-reserve':g['minimum_observed_free_bytes']=127*m.GIB
    elif fault=='guard-wall':g['wall_seconds']=54001.
    elif fault=='missing-rank':del h[5]['rank_reports']['47']
    elif fault=='wrong-host':r['host']=m.NODES[1]
    elif fault=='wrong-worker-env':receipt['openmp_environment']['OMP_PLACES']='cores'
    elif fault=='main-thread-cpus':r['allowed_cpus']=[0,1,2,3]
    elif fault=='projection-total':
        r['projection_local_counts'][0]+=1;r['local_recurrent_edges']+=1;r['local_total_connections']+=1
    elif fault=='device-total':r['local_external_connections']+=1;r['local_total_connections']+=1
    elif fault=='chunk-total':r['local_spikes']=1;r['event_bytes']=8
    elif fault=='short-chunks':r['chunks']=r['chunks'][:-1]
    elif fault=='rank-cpu-regression':r['phase_memory']['after_100500ms']['user_seconds']=0
    elif fault=='rank-rss':r['peak_rss_kib']=1
    elif fault=='source-unverified':a['preflight'][5]['source_catalog_verified']=False
    elif fault=='busy-cpu':a['preflight'][5]['cpu_busy_percent']['0']=90.
    else:a['experiment_budget']['automatic_retry']=True
    with pytest.raises((ValueError,KeyError)):m.audit(a,l,h,p,layout)


def test_cpuset_ranges_are_semantic():
    assert m.cpu_set('0-3,12-15')==[0,1,2,3,12,13,14,15]
    for text in ['0-3,3','1-0','0-99999','0,a']:
        with pytest.raises(ValueError):m.cpu_set(text)


def test_missing_terminal_is_no_network_no_output(tmp_path,monkeypatch):
    monkeypatch.setattr(collector.subprocess,'run',lambda *a,**k:pytest.fail('network before readiness'))
    output=tmp_path/'output'
    r=collector.run(tmp_path,output,tmp_path/'absent-parameters')
    assert not r['ready'] and not r['collection_started'] and not output.exists()


def test_all_six_generated_control_readers_are_bounded_and_parse():
    for i in range(6):
        code=collector.host_code(i,'b2mpi-abcdef012345',{})
        ast.parse(code)
        assert 'RLIMIT_AS,(512*2**20,512*2**20)' in code
        assert 'total<=32*2**20' in code and '.events.bin' not in code
        assert m.NODES[i] in code


@pytest.mark.parametrize('fault',[None,'truncated','crc','trailing','oversized','invalid-json'])
def test_compressed_controls_require_bounded_complete_valid_payload(monkeypatch,fault):
    monkeypatch.setattr(collector,'CAP',128)
    raw=b'{"rank":47,"count":100500}'
    if fault=='oversized':raw=b'"'+b'x'*128+b'"'
    if fault=='invalid-json':raw=b'invalid'
    packed=gzip.compress(raw,mtime=0)
    if fault=='truncated':packed=packed[:-1]
    if fault=='crc':packed=packed[:-8]+bytes([packed[-8]^1])+packed[-7:]
    if fault=='trailing':packed+=gzip.compress(b'{}',mtime=0)
    if fault is None:assert collector.decode_control(packed)==raw
    else:
        with pytest.raises((ValueError,zlib.error)):collector.decode_control(packed)


def test_recovery_charges_failure_and_diagnosis_and_refuses_recursive_attempt(tmp_path,monkeypatch):
    gp=tmp_path/'guard.json'
    gp.write_text(json.dumps(dict(returncode=1,stop_reason=None,wall_seconds=45.,
        command=['mam_collect_native_primary_resources.py','--output',str(tmp_path)])))
    (tmp_path/'failure.json').write_text(json.dumps(dict(elapsed_seconds=44.,automatic_retry=False)))
    monkeypatch.setattr(collector.time,'time',lambda:gp.stat().st_mtime+120.)
    result=collector.recovery_cost(tmp_path,gp)
    assert result['failed_attempt_seconds']==45. and result['charged_prior_seconds']==165.
    (tmp_path/'recovery.json').write_text('{}')
    with pytest.raises(ValueError,match='only one'):collector.recovery_cost(tmp_path,gp)


def test_proc_rss_and_rusage_peak_are_separate_observations(inputs):
    a,l,h,p,layout=inputs
    r=h[0]['rank_reports']['0'];last=r['phase_memory']['after_100500ms']
    last['rss_kib']=1004
    result=m.audit(a,l,h,p,layout)
    assert result['ranks'][0]['final_proc_rss_bytes']==1004*1024
    assert result['ranks'][0]['peak_rss_bytes']==1000*1024
    assert result['ranks'][0]['final_proc_rss_minus_rusage_peak_bytes']==4096
    last['rss_kib']=256*m.GIB//1024+1
    with pytest.raises(ValueError,match='RSS'):m.audit(a,l,h,p,layout)
    last['rss_kib']=1004;last['peak_rss_kib']=1001
    with pytest.raises(ValueError,match='RSS'):m.audit(a,l,h,p,layout)
