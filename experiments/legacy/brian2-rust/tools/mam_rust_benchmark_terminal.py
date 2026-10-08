"""Validate the new Rust target's collected terminal controls, not raw events."""
import hashlib
import math
from pathlib import Path
import re
import shlex

from mam_benchmark_terminal import guard,parse_report,require,number,integer
from mam_primary_resources import CPUS,MODEL,PLAN,NODES,time_record
from mam_launch_performance_tuning import PROTOCOL_SHA
from mam_launch_rust_performance import CASE,LABEL,BRICK,BASE,PACKAGE_SHA,launch_options
from mam_rust_terminal_sync import EXE_SHA,parse as parse_metadata
from mam_rust_cpu_placement import admitted_cpus


def elapsed_quantization(text):
    """GNU time %E truncates to whole seconds at one hour, centiseconds below.

    The displayed value is a lower bound, not an exact duration. Keep the
    wrapper's monotonic duration and the launch deadline independently checked.
    """
    values=re.findall(r'^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)\s*$',
                      text,re.MULTILINE)
    require(len(values)==1,'missing/duplicate GNU elapsed field')
    value=values[0]
    if re.fullmatch(r'[1-9][0-9]*:[0-5][0-9]:[0-5][0-9]',value):return 1.0
    require(re.fullmatch(r'(?:[0-9]|[1-5][0-9]):[0-5][0-9]\.[0-9]{2}',value) is not None,
            'noncanonical GNU elapsed format')
    return .01


def target_contract(admission):
    """Recognize only the historical target or the one bounded PMI correction."""
    from mam_launch_rust_recovery import CASE as NEW_CASE,LABEL as NEW_LABEL,SOURCE as NEW_SOURCE,PROTOCOL_SHA as NEW_PROTOCOL
    from mam_launch_rust_performance import SOURCE
    if admission.get('case_id')==NEW_CASE:
        from mam_rust_cpu_placement import ALTERNATE_CPUS
        require(admission.get('base_protocol_sha256')==PROTOCOL_SHA
                and admission.get('transport_only') is not True,'recovery target identity')
        require(admitted_cpus(admission)==ALTERNATE_CPUS,'recovery CPU placement required')
        require(admission['source_catalog']['mam_rust_terminal_sync.py']['sha256']==
                'e66a7f95398696389f749fa8ab60646923b3585f4b9f214cde1afddc2e6f3bfb','fixed PMI wrapper required')
        result=dict(case_id=NEW_CASE,label=NEW_LABEL,source=NEW_SOURCE,protocol_sha256=NEW_PROTOCOL)
    else:result=dict(case_id=CASE,label=LABEL,source=SOURCE,protocol_sha256=PROTOCOL_SHA)
    require(all(admission[k]==result[k] for k in ['case_id','label','protocol_sha256']),'unknown or mixed target contract')
    return result


def audit(admission,launch,collected):
    contract=target_contract(admission)
    require(admission['schema']=='b2-mam-rust-performance-admission-v1' and admission['admitted'] is True
            and admission['case_id']==contract['case_id'] and admission['label']==contract['label']
            and admission['protocol_sha256']==contract['protocol_sha256'],'exact Rust target admission required')
    require(admission['model_sha256']==MODEL and admission['plan_sha256']==PLAN
            and admission['executable_sha256']==EXE_SHA and admission['package_sha256']==PACKAGE_SHA,
            'frozen Rust artifact identity')
    require(admission['identity']==dict(ranks=32,threads=1,seed=1729,duration_ms=100500,dt_ms=.1),
            'full Rust target condition')
    resources=admission['resources']
    cpu_ids=admitted_cpus(admission)
    expected=dict(nodes=NODES,ranks_per_node=8,cpu_ids=cpu_ids,memory_bytes_per_service=256*2**30,
        cpu_quota_cores_per_service=8,zero_swap=True,pids_max=64,file_limit_bytes=65504*2**20,
        minimum_free_bytes_by_host={h:(1280 if i==0 else 128)*2**30 for i,h in enumerate(NODES)},
        total_final_output_bytes=128*2**30,spool_bytes=96*2**30,wall_seconds=64800)
    require(resources==expected,'Rust resource contract changed')
    sources=admission['source_catalog']
    require(set(sources)=={'mam_rust_terminal_sync.py','mpi_resource_guard.py','mam_rust_guard23.py','protocol.json'},
            'Rust source coverage')
    require(sources['protocol.json']['sha256']==contract['protocol_sha256'],'source protocol pin')
    for item in sources.values():
        integer(item['bytes'],'source size',1)
        require(re.fullmatch('[0-9a-f]{64}',item['sha256']) is not None,'source digest')
    require([p['host'] for p in admission['preflight']]==NODES,'fresh admission host coverage')
    for index,row in enumerate(admission['preflight']):
        require(row['all_artifact_hashes_verified'] is True and row['source_catalog_verified'] is True
                and row['active_units']==[] and row['memory']['MemAvailable'] >= (576 if index==0 else 320)*2**30
                and row['free_bytes'] >= (2048 if index==0 else 192)*2**30,'start admission incomplete')
    roles=['controller']+['proxy-'+str(i) for i in range(4)]
    require(launch['schema']=='b2-teleport-hydra-launch-v0' and not launch.get('error')
            and launch['nodes']==NODES and launch['ranks_per_node']==8 and launch['remote_timeout_seconds']==64800
            and set(launch['returncodes'])==set(roles)
            and all(type(v) is int and v==0 for v in launch['returncodes'].values()),'Rust launcher not terminal success')
    prefix=launch['resource_guard']['unit_prefix']
    require(re.fullmatch('b2mpi-[0-9a-f]{12}',prefix) is not None,'service identity')
    expected_app=launch_options(Path('/unused'),source=contract['source'],label=contract['label'])['application']
    require(admission['launch_options']['application']==expected_app
            and shlex.split(launch['commands']['controller'][-1])[-3:]==expected_app,'wrapper/time command differs')
    wall=number(launch['wall_seconds'],'launch wall',True);require(wall<=64800,'launch wall ceiling')
    require([h['host'] for h in collected]==NODES,'terminal source host coverage')
    guard_rows=[];rank_rows=[];leader_sync=None
    for index,host in enumerate(collected):
        require(host['active_own_units']==[] and host['failure_files']==[],'live or failed Rust evidence')
        require(host['source_catalog']==sources and host['artifact_catalog_verified'] is True
                and host['mpi_runtime_verified'] is True,'post-run source/runtime verification')
        host_roles=(['controller'] if index==0 else [])+['proxy-'+str(index)]
        require(set(host['guards'])==set(host_roles),'Rust guard coverage')
        for role in host_roles:
            spec=dict(host=host['host'],role=role,memory_bytes=256*2**30,pids_max=64,cpu_ids=cpu_ids,
                cpu_quota_cores=8,volume='/data/brick2' if index==0 else '/',allow_root_volume=True,
                file_limit_bytes=65504*2**20,minimum_free_bytes=(1280 if index==0 else 128)*2**30,
                reserved_host_memory_bytes=64*2**30)
            g=host['guards'][role]
            require((g['data_device']!=g['root_device']) if index==0 else (g['data_device']==g['root_device']),
                    'Rust storage device differs')
            guard_rows.append(guard(g,spec,prefix,64800))
        ids=list(range(index*8,(index+1)*8))
        require(set(host['ranks'])=={str(r) for r in ids},'Rust rank coverage')
        for rank in ids:
            item=host['ranks'][str(rank)]
            started=parse_report(item['started_json']);done=parse_report(item['done_json'])
            common=dict(rank=rank,ranks=32,host=host['host'],executable_sha256=EXE_SHA,plan_sha256=PLAN,
                        wrapper_sha256=sources['mam_rust_terminal_sync.py']['sha256'])
            require(all(started[k]==v and done[k]==v for k,v in common.items()),'rank wrapper identity')
            require(started['event']=='rank_wrapper_started' and done['event']=='rank_wrapper_complete'
                    and type(done['child_returncode']) is int and done['child_returncode']==0
                    and done['whole_job_accepted'] is False and done['scientific_acceptance'] is False,
                    'wrapper terminal status')
            timing=time_record(item['time_record'])
            elapsed=number(done['wall_through_child_and_output_sync_seconds'],'wrapper wall',True)
            quantum=elapsed_quantization(item['time_record'])
            upper=timing['elapsed_seconds']+quantum
            require(elapsed<upper and timing['elapsed_seconds']<=wall+.02 and elapsed<=wall+.02
                    and timing['peak_rss_bytes']<=256*2**30,'rank timing/RSS outside job')
            if rank==0:leader_sync=done['leader_output_synchronization']
            else:require(done['leader_output_synchronization'] is None,'nonleader claimed output sync')
            rank_rows.append(dict(rank=rank,host=host['host'],**timing,
                elapsed_quantization_seconds=quantum,elapsed_upper_bound_exclusive_seconds=upper,
                wrapper_wall_seconds=elapsed,done_sha256=hashlib.sha256(item['done_json'].encode()).hexdigest()))
        if index:
            require(host['leader_outputs'] is None,'nonleader supplied leader output')
    output=collected[0]['leader_outputs']
    require(output['spool_present'] is False and leader_sync is not None,'leader spool or sync incomplete')
    require(leader_sync['file_sync_calls']==4 and leader_sync['output_and_parent_directories_synced'] is True
            and leader_sync['metadata_readback_verified'] is True
            and leader_sync['binary_contents_independently_audited'] is False,'leader durability receipt')
    number(leader_sync['elapsed_seconds'],'sync wall')
    sizes={}
    for name,key in [('summary.json','summary_json'),('mpi-runtime.json','runtime_json')]:
        raw=output[key].encode();require(0<len(raw)<=32*2**20,'metadata size')
        sizes[name]=len(raw)
        require(leader_sync['files'][name]==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),
                'collected metadata differs from sync receipt')
    require(set(leader_sync['files'])=={'results.bin','events.bin','summary.json','mpi-runtime.json'},'synced file coverage')
    for name in ['results.bin','events.bin']:
        sizes[name]=integer(output['binary_file_bytes'][name],'binary size',1)
        require(sizes[name]<=resources['file_limit_bytes'] and leader_sync['files'][name]==dict(bytes=sizes[name]),
                'binary stat differs from terminal receipt')
    require(sum(sizes.values())<=128*2**30,'combined final output exceeded')
    summary=parse_metadata(output['summary_json']);runtime=parse_metadata(output['runtime_json'])
    require(summary['schema']=='b2-result-dump-v4' and summary['population_count']==254
            and summary['neuron_count']==4129924 and summary['final_time_seconds']==100.5
            and summary['mpi']==runtime and summary['dump_bytes']==sizes['results.bin']
            and summary['event_dump_bytes']==sizes['events.bin'],'full output metadata differs')
    require(runtime['schema']=='b2-mpi-runtime-v0' and runtime['ranks']==32 and runtime['plan_sha256']==PLAN
            and runtime['exchange_calls_per_rank']==1005000
            and runtime['processor_names']==[h for h in NODES for _ in range(8)],'runtime full observation/placement')
    require(len(runtime['rank_cpu_ids'])==32 and all(sorted(runtime['rank_cpu_ids'][i*8:(i+1)*8])==cpu_ids for i in range(4)),
            'runtime CPU coverage')
    work=runtime['rank_work'];require(len(work)==96 and all(type(x) is int and x>=0 for x in work),'rank work fields')
    require(sum(work[::3])==4129924 and sum(work[1::3])==summary['spike_count']==runtime['spike_history_records']
            and sum(work[2::3])==summary['synaptic_events'],'reported work totals differ')
    require(runtime['spike_history_storage']=='bounded-disk-spool'
            and runtime['spike_spool_bytes']==8*summary['spike_count']<=96*2**30
            and runtime['spike_spool_maximum_bytes']==96*2**30
            and runtime['spike_spool_maximum_population_bytes']==16*2**30,'spool report ceiling')
    require(runtime['rank_stage_columns']==['initialization_local','initialization_wait','simulation_local','simulation_wait','result_collection_and_reporting']
            and len(runtime['rank_stage_seconds'])==160
            and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in runtime['rank_stage_seconds']),
            'rank phase times incomplete')
    return dict(schema='b2-mam-rust-benchmark-terminal-v1',case_id=contract['case_id'],label=contract['label'],
        terminal_resource_audit_passed=True,metadata_sync_audit_passed=True,raw_output_audit_passed=False,
        scientific_acceptance=False,performance_cost_acceptance=False,reported_spikes=summary['spike_count'],
        reported_synaptic_events=summary['synaptic_events'],output_bytes=sizes,launch_wall_seconds=wall,
        guards=guard_rows,ranks=rank_rows,
        accounting=dict(measured_cgroup_core_hours=sum(g['measured_cpu_seconds'] for g in guard_rows)/3600,
            measured_rank_user_system_core_hours=sum(r['user_seconds']+r['system_seconds'] for r in rank_rows)/3600,
            worker_capacity_core_hours=wall*32/3600,shared_participating_node_hours=wall*4/3600,
            sum_worker_host_peaks_bytes=sum(g['peak_bytes'] for g in guard_rows if g['role'].startswith('proxy-')),
            exclusive_allocated_node_hours=None,monetary_cost=None),
        limitations=['No independent binary payload scan or scientific acceptance is provided by terminal controls.',
            'Measured cgroup CPU includes controller/proxies and wrapper work; GNU time surrounds each complete wrapper.',
            'Sum of host peaks is not a simultaneous peak. Shared node-hours are not exclusive allocation or monetary cost.',
            'Final reported CPU binding does not independently sample each worker throughout execution.'])
