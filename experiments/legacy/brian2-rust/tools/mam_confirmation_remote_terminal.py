"""Recover remote terminal evidence after loss of the local observer.

Does not fabricate a Teleport exit code or claim local end-to-end wall time.
All model, resource, rank, fsync, duration and work checks remain required.
"""
import hashlib,json,math,re
from pathlib import Path
from mam_confirmation_terminal import (context_gate,require,number,integer,guard,
    parse_report,parse_metadata,time_record,elapsed_quantization,IDENTITY_SHA,
    ALTERNATE_CPUS,NODES,launch_options)

PROVENANCE_NAME='observation-control-loss-20260912T1241Z.json'
PROVENANCE_SHA='36a6ccd37cdec1af47faeba6aee5a869e83351bd7adb996ec8a5a3dd89fc3f8d'
PREFIX='b2mpi-4314b740d2a3'


def provenance_gate(provenance):
    from mam_confirmation_profile import REPLICATE
    require(REPLICATE==1750,'historical remote recovery cannot accept a new replica')
    raw=(json.dumps(provenance,indent=2)+'\n').encode()
    require(hashlib.sha256(raw).hexdigest()==PROVENANCE_SHA,'exact live recovery observation required')
    require(provenance['schema']=='b2-mam-observation-control-loss-v1'
        and provenance['old_exec_session']==17974
        and provenance['local_launcher_present'] is False
        and provenance['local_transport_returncodes_unavailable'] is True
        and provenance['simulation_restarted'] is False
        and provenance['neural_runs_started']==0
        and provenance['terminal_status_proven'] is False,'observer-loss identity differs')
    require([h['host'] for h in provenance['remote_hosts']]==NODES,'live host coverage')
    return PREFIX


def recovered_timing(identity,provenance,collected):
    prefix=provenance_gate(provenance)
    require([h['host'] for h in collected]==NODES,'terminal host coverage')
    remote_wall=None;windows=[]
    for index,(anchor,host) in enumerate(zip(provenance['remote_hosts'],collected,strict=True)):
        expected=(['controller'] if index==0 else [])+['proxy-'+str(index)]
        require([u['role'] for u in anchor['units']]==expected
                and set(host['guards'])==set(expected),'remote guard role coverage')
        for unit in anchor['units']:
            role=unit['role'];props=unit['properties'];g=host['guards'][role]
            name=prefix+'-'+role+'.service';initial=parse_report(unit['guard_text'])
            require(props['Id']==name and props['ActiveState']=='active'
                    and props['ControlGroup']==initial['cgroup']=='/system.slice/'+name
                    and re.fullmatch('[0-9a-f]{32}',props['InvocationID']) is not None,
                    'original live service identity differs')
            require('returncode' not in initial and 'after' not in initial
                    and initial['admitted'] is True,'anchor must precede terminal result')
            require(all(k in g and g[k]==v for k,v in initial.items()),
                    'final guard differs from captured live guard')
            require(type(g.get('returncode')) is int and g['returncode']==0
                    and not g.get('error') and 'after' in g,'remote child not successful terminal')
            duration=number(g['wall_seconds'],'remote guard duration',True)
            require(duration<=64800,'remote duration exceeded')
            observed=anchor['monotonic_seconds']-int(props['ExecMainStartTimestampMonotonic'])/1e6
            require(observed>0 and duration>=observed,'terminal record predates live observation')
            # A final guard must extend its captured CPU counter history.
            for phase in ['before','after']:
                require(type(g[phase]) is dict,'guard counters missing')
            if role=='controller':
                app=launch_options(identity,Path('/unused'))['application']
                require(g['command'][-3:]==app,'remote application differs from admission')
                remote_wall=duration
            windows.append(duration)
    require(remote_wall is not None and max(windows)<=remote_wall+.02,
            'proxy duration exceeds controller scope')
    return prefix,remote_wall


def audit(identity,protocol,admission,provenance,collected):
    contract=context_gate(identity,protocol,admission)
    MODEL,PLAN,EXE_SHA=[identity[k] for k in ['model_sha256','plan_sha256','executable_sha256']]
    resources=admission['resources'];cpu_ids=ALTERNATE_CPUS;sources=admission['source_catalog']
    require([p['host'] for p in admission['preflight']]==NODES,'fresh admission host coverage')
    for index,row in enumerate(admission['preflight']):
        require(row['all_artifact_hashes_verified'] is True and row['source_catalog_verified'] is True
                and row['active_units']==[] and row['memory']['MemAvailable'] >= (576 if index==0 else 320)*2**30
                and row['free_bytes'] >= (2048 if index==0 else 192)*2**30,'start admission incomplete')
    prefix,wall=recovered_timing(identity,provenance,collected)
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
                        wrapper_sha256=sources['mam_confirmation_terminal_sync.py']['sha256'],
                        replicate=1750,identity_sha256=IDENTITY_SHA,model_sha256=MODEL,instance_sha256=identity['instance_sha256'])
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
    return dict(schema='b2-mam-confirmation-remote-terminal-v1',replicate=1750,identity_sha256=IDENTITY_SHA,case_id=contract['case_id'],label=contract['label'],
        terminal_resource_audit_passed=True,metadata_sync_audit_passed=True,raw_output_audit_passed=False,
        scientific_acceptance=False,performance_cost_acceptance=False,reported_spikes=summary['spike_count'],
        reported_synaptic_events=summary['synaptic_events'],output_bytes=sizes,remote_mpi_wall_seconds=wall,
        local_launcher_returncodes=None,local_launcher_wall_seconds=None,
        remote_terminal_recovery=True,provenance_sha256=PROVENANCE_SHA,
        guards=guard_rows,ranks=rank_rows,
        accounting=dict(wall_scope='remote controller guard including MPI startup, execution and output synchronization; local transport overhead unavailable',
            measured_cgroup_core_hours=sum(g['measured_cpu_seconds'] for g in guard_rows)/3600,
            measured_rank_user_system_core_hours=sum(r['user_seconds']+r['system_seconds'] for r in rank_rows)/3600,
            worker_capacity_core_hours=wall*32/3600,shared_participating_node_hours=wall*4/3600,
            sum_worker_host_peaks_bytes=sum(g['peak_bytes'] for g in guard_rows if g['role'].startswith('proxy-')),
            exclusive_allocated_node_hours=None,monetary_cost=None),
        limitations=['Local launcher was lost; Teleport exit codes and local end-to-end wall time are unavailable.',
            'Remote controller guard duration is reported separately and is not interchangeable with previous local-launcher wall measurements.',
            'No independent binary payload scan or scientific acceptance is provided by terminal controls.',
            'Measured cgroup CPU includes controller/proxies and wrapper work; GNU time surrounds each complete wrapper.',
            'Sum of host peaks is not a simultaneous peak. Shared node-hours are not exclusive allocation or monetary cost.',
            'Final reported CPU binding does not independently sample each worker throughout execution.'])
