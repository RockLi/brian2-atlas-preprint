"""Terminal controls for the pinned confirmation run, retaining all legacy resource checks.

Synthetic tests of this verifier do not constitute simulation or science evidence.
"""
from mam_confirmation_profile import REPLICATE,selected,worker_environment,IDENTITY_SHA_1751
import hashlib,json,math,re,shlex
from pathlib import Path
from mam_benchmark_terminal import guard,parse_report,require,number,integer
from mam_primary_resources import time_record
from mam_rust_benchmark_terminal import elapsed_quantization
from mam_confirmation_terminal_sync import parse as parse_metadata,IDENTITY_SHA
from mam_confirmation_identity import NODES
from mam_launch_confirmation_run import LABEL
from mam_launch_confirmation_run import CASE,SOURCE,protocol as expected_protocol,launch_options
from mam_rust_cpu_placement import ALTERNATE_CPUS
IDENTITY_SHA=selected(IDENTITY_SHA,IDENTITY_SHA_1751)

ADMISSION_SHA='e0b126783c32d89d49a67afb8e82686597095aeac5259131a103121d6050d874'
COUNTS_SHA='3127f493e290427179e2ffdcb81198674d76afe4da124b5af07f1c8a465b952d'
SOURCE_CATALOG={'mam_confirmation_terminal_sync.py': {'bytes': 9871, 'sha256': '86a40c78b75494884293fd13858944dd42b763054d3961b959c615b6ffb7cfb7'}, 'mpi_resource_guard.py': {'bytes': 6287, 'sha256': '630fc35f0b38729fc202925b09b4eea55cc965cfeee396ace6085a3a98004014'}, 'mam_rust_guard23.py': {'bytes': 668, 'sha256': '4ff7fcc2cd12f0b9448fb616c649b17c30ae3ed59f0cb345b05a03c2eca5fcfe'}, 'protocol.json': {'bytes': 5073, 'sha256': 'f1582a50dcf5d5766c94fd8edb0c8fa495cdcdd12112441f34b799a38e1f925c'}, 'identity.json': {'bytes': 236927, 'sha256': '9a5a3258ca7b33b3d5296bc74398ba19e3455f3e3e61ffcb872f7a6b374165de'}}
from mam_confirmation_profile import ADMISSION_SHA_1751,COUNTS_SHA_1751,SOURCE_CATALOG_1751
ADMISSION_SHA=selected(ADMISSION_SHA,ADMISSION_SHA_1751)
COUNTS_SHA=selected(COUNTS_SHA,COUNTS_SHA_1751)
SOURCE_CATALOG=selected(SOURCE_CATALOG,SOURCE_CATALOG_1751)

def context_gate(identity,protocol,admission):
    raw=(json.dumps(identity,indent=2)+'\n').encode()
    require(hashlib.sha256(raw).hexdigest()==IDENTITY_SHA,'pinned confirmation identity required')
    require(protocol==expected_protocol(identity,dict(counts_sha256=COUNTS_SHA)),'single-run protocol changed')
    digest=hashlib.sha256((json.dumps(protocol,indent=2)+'\n').encode()).hexdigest()
    require(admission['schema']=='b2-mam-confirmation-admission-v1' and admission['admitted'] is True
        and admission['replicate']==REPLICATE and admission['label']==LABEL
        and admission['identity_sha256']==IDENTITY_SHA and admission['protocol_sha256']==digest,
        'exact confirmation admission required')
    for key in ['model_sha256','instance_sha256','plan_sha256','executable_sha256']:
        require(admission[key]==identity[key],'confirmation artifact identity differs')
    require(admission['resources']==protocol['resources'],'confirmation resource contract changed')
    require(admission['source_catalog']==SOURCE_CATALOG,'confirmation frozen source catalog differs')
    return dict(case_id=CASE,label=LABEL,source=SOURCE,protocol_sha256=digest)


def audit(identity,protocol,admission,launch,collected):
    contract=context_gate(identity,protocol,admission)
    MODEL,PLAN,EXE_SHA=[identity[k] for k in ['model_sha256','plan_sha256','executable_sha256']]
    resources=admission['resources'];cpu_ids=ALTERNATE_CPUS;sources=admission['source_catalog']
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
    expected_app=launch_options(identity,Path('/unused'))['application']
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
                        wrapper_sha256=sources['mam_confirmation_terminal_sync.py']['sha256'],
                        replicate=REPLICATE,identity_sha256=IDENTITY_SHA,model_sha256=MODEL,instance_sha256=identity['instance_sha256'])
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
    return dict(schema='b2-mam-confirmation-terminal-v1',replicate=REPLICATE,identity_sha256=IDENTITY_SHA,case_id=contract['case_id'],label=contract['label'],
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
