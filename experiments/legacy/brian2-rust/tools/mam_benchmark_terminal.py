"""Audit collected terminal controls against a separately hash-pinned admission.

No network access, no launches, and no raw-event acceptance. The caller must bind
admission, parameters, collected bytes and launch controls to their source files.
"""
import hashlib
import json
import math
import re

from mam_native_primary_resources import parameter_geometry, cpu_set
from mam_nest_reference import virtual_process_capacity
from mam_nest_benchmark import WORKLOAD_SHA256


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, message, minimum=0):
    require(type(value) is int and value >= minimum, message)
    return value


def number(value, message, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), message)
    return value


def counters(text):
    result={}
    for line in text.splitlines():
        key,value=line.split()
        require(key not in result, 'duplicate kernel counter')
        result[key]=integer(int(value), 'negative kernel counter')
    return result


def parse_report(raw):
    require(isinstance(raw,str) and len(raw.encode())<=8*2**20, 'bounded raw rank report required')
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result, 'duplicate JSON key')
            result[key]=value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON constant '+value)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)


def guard(g, spec, prefix, wall_limit):
    require(g['schema']=='b2-mpi-resource-guard-v1' and g['host']==spec['host']
            and g['cgroup']=='/system.slice/'+prefix+'-'+spec['role']+'.service'
            and g['admitted'] is True and g['uid']==1000
            and type(g.get('returncode')) is int and g['returncode']==0
            and not g.get('error'), 'guard identity or terminal failure')
    wall=number(g['wall_seconds'],'guard wall',True)
    require(wall<=wall_limit, 'guard wall exceeded')
    for phase in ['before','after']:
        row=g[phase]
        require(row['memory.max']==str(spec['memory_bytes']) and row['memory.swap.max']=='0'
                and row['pids.max']==str(spec['pids_max'])
                and cpu_set(row['cpuset.cpus.effective'])==spec['cpu_ids'], 'guard resource limits changed')
        quota,period=map(int,row['cpu.max'].split())
        require(period>0 and quota==period*spec['cpu_quota_cores'], 'CPU quota changed')
        events=counters(row['memory.events'])
        require(all(events[k]==0 for k in ['max','oom','oom_kill','oom_group_kill']), 'memory pressure is failure')
    peak=integer(int(g['after']['memory.peak']),'peak',1)
    require(peak<=spec['memory_bytes'], 'memory peak exceeded')
    before,after=[counters(g[k]['cpu.stat']) for k in ['before','after']]
    require(all(after[k]>=before[k] for k in ['usage_usec','user_usec','system_usec']), 'CPU counter regression')
    require(g['data_volume']==spec['volume'] and g['root_volume_allowed']==spec['allow_root_volume']
            and (spec['allow_root_volume'] or g['data_device']!=g['root_device'])
            and g['file_limit_bytes']==spec['file_limit_bytes']
            and g['minimum_free_bytes']==spec['minimum_free_bytes']
            and number(g['data_free_bytes'],'initial disk')>=spec['minimum_free_bytes']
            and number(g['minimum_observed_free_bytes'],'minimum disk')>=spec['minimum_free_bytes'],
            'disk identity or reserve failure')
    reserve=number(g['reserved_host_memory_bytes'],'host reserve')
    require(reserve>=spec['reserved_host_memory_bytes']
            and g['host_memory_bytes']['MemAvailable']>=spec['memory_bytes']+reserve,
            'guard host memory admission failed')
    return dict(host=spec['host'],role=spec['role'],peak_bytes=peak,wall_seconds=wall,
                measured_cpu_seconds=(after['usage_usec']-before['usage_usec'])/1e6)


def audit(admission, launch, collected, parameters, *, parameters_sha256):
    """Validate all controls, retaining raw bytes so completion hashes are checked."""
    require(admission['schema']=='b2-mam-benchmark-admission-v1' and admission['admitted'] is True,
            'explicit benchmark admission required')
    require(admission['workload_sha256']==WORKLOAD_SHA256
            and admission['parameters_sha256']==parameters_sha256
            =='ee1a642c94b9d6e808627c54f78162e7f5d87f4140ba56496c5dfee3ed900d3e', 'workload or parameter identity')
    identity=admission['identity'];limits=admission['limits'];placements=admission['placements']
    n=integer(identity['ranks'],'rank count',1);threads=integer(identity['threads'],'threads',1)
    require(n<=256 and len(placements)==n and [p['rank'] for p in placements]==list(range(n)), 'rank placement coverage')
    purpose=admission.get('run_purpose','target')
    require((purpose,identity['duration_ms']) in [('target',100500),('performance_tuning',2500)]
            and identity['dt_ms']==.1 and identity['nest_version']=='3.10.0',
            'explicit target or finite performance-tuning observation required')
    integer(identity['seed'],'seed',1)
    require(limits['chunk_ms']==50 and limits['automatic_retry'] is False, 'fixed chunk/no-retry contract')
    for key in ['max_chunk_spikes','max_spikes_per_rank','total_event_bytes']:
        integer(limits[key],'positive event ceiling',1)
    wall_limit=number(limits['wall_seconds'],'wall limit',True)
    populations,expected_projections=parameter_geometry(parameters)
    capacity=virtual_process_capacity(parameters,n,threads)
    specs=admission['guards'];roles=[s['role'] for s in specs]
    hosts=admission['nodes'];rph=admission['ranks_per_node']
    require(len(set(hosts))==len(hosts) and n==len(hosts)*rph and hosts
            and roles==['controller']+['proxy-'+str(i) for i in range(len(hosts))], 'guard role or host coverage')
    require([s['host'] for s in specs]==[hosts[0]]+hosts, 'guard host assignment')
    require([p['host'] for p in placements]==[h for h in hosts for _ in range(rph)], 'rank host assignment')
    for spec in specs:
        require(spec['cpu_ids']==sorted(set(spec['cpu_ids'])) and spec['cpu_ids']
                and all(type(c) is int and 0<=c<4096 for c in spec['cpu_ids']), 'invalid guard CPU set')
        integer(spec['cpu_quota_cores'],'CPU quota',1)
        require(spec['cpu_quota_cores']<=len(spec['cpu_ids']), 'quota exceeds CPU set')
        for key in ['memory_bytes','pids_max','file_limit_bytes','minimum_free_bytes','reserved_host_memory_bytes']:
            integer(spec[key],'resource limit',1)
    require(launch['schema']=='b2-teleport-hydra-launch-v0' and not launch.get('error')
            and launch['nodes']==hosts and launch['ranks_per_node']==rph
            and launch['remote_timeout_seconds']==wall_limit
            and set(launch['returncodes'])==set(roles)
            and all(type(x) is int and x==0 for x in launch['returncodes'].values()), 'launcher not complete or mismatched')
    prefix=launch['resource_guard']['unit_prefix']
    require(re.fullmatch('b2mpi-[0-9a-f]{12}',prefix) is not None, 'invalid service family')
    wall=number(launch['wall_seconds'],'launch wall',True);require(wall<=wall_limit,'launch wall exceeded')
    require([h['host'] for h in collected]==hosts, 'collected host coverage')
    sources=admission['source_catalog']
    required_sources={'mam_nest_benchmark.py','mam_benchmark_recording.py','mam_benchmark_event_io.py','mam_benchmark_workload_v1.json'}
    require(set(sources)==required_sources, 'source catalog coverage')
    for v in sources.values():
        integer(v['bytes'],'source bytes',1)
        require(re.fullmatch('[0-9a-f]{64}',v['sha256']) is not None,'invalid source digest')
    require(sources['mam_benchmark_workload_v1.json']['sha256']==WORKLOAD_SHA256,'workload source digest')
    guards=[];rank_rows=[];projections=[0]*len(expected_projections);external=recording=spikes=event_bytes=0
    for host_index,host in enumerate(collected):
        require(host['active_own_units']==[] and host['failure_files']==[], 'live or failed rank evidence')
        require(host['source_catalog']==sources
                and host['runtime_catalog_sha256']==admission['runtime_catalog_sha256'], 'deployed source/runtime identity')
        host_specs=[s for s in specs if s['host']==host['host']]
        require(set(host['guards'])=={s['role'] for s in host_specs}, 'missing guard')
        for spec in host_specs:guards.append(guard(host['guards'][spec['role']],spec,prefix,wall_limit))
        rank_ids=list(range(host_index*rph,(host_index+1)*rph))
        require(set(host['ranks'])=={str(i) for i in rank_ids}, 'rank report coverage')
        proxy=specs[host_index+1]
        for rank in rank_ids:
            entry=host['ranks'][str(rank)];raw=entry['report_json'];row=parse_report(raw);done=parse_report(entry['done_json'])
            require(done['event']=='rank_complete' and done['rank']==rank
                    and done['event_readback_verified'] is True
                    and done['report']==dict(bytes=len(raw.encode()),sha256=hashlib.sha256(raw.encode()).hexdigest()),
                    'rank completion does not bind report bytes')
            require(row['schema']=='b2-native-nest-mam-benchmark-events-v2' and row['rank']==rank
                    and row['host']==host['host'] and all(row[k]==v for k,v in identity.items())
                    and row['parameters_sha256']==parameters_sha256 and row['workload_sha256']==WORKLOAD_SHA256,
                    'rank simulation identity')
            require(row['producer_sha256']==sources['mam_nest_benchmark.py']['sha256']
                    and row['source_sha256']=={k:sources[k]['sha256'] for k in ['mam_benchmark_recording.py','mam_benchmark_event_io.py']},
                    'rank source identity')
            require(row['populations']==populations and row['N_scaling']==row['K_scaling']==1.
                    and row['connection_index_capacity']==capacity, 'rank geometry or VP capacity')
            local=row['projection_local_counts'];require(len(local)==len(projections),'projection coverage')
            for i,value in enumerate(local):projections[i]+=integer(value,'negative local projection')
            require(sum(local)==row['local_recurrent_edges'], 'local recurrent total differs')
            ex=integer(row['local_external_connections'],'external');rec=integer(row['local_recording_connections'],'recording')
            require(row['local_total_connections']==sum(local)+ex+rec,'local connection total differs')
            external+=ex;recording+=rec
            cpus=placements[rank]['cpu_ids']
            require(cpus==sorted(set(cpus)) and cpus and set(cpus)<=set(proxy['cpu_ids'])
                    and row['allowed_cpus'] and set(row['allowed_cpus'])<=set(cpus)
                    and entry['initial_rank_cpu_ids']==cpus, 'rank affinity outside admission')
            count=integer(row['local_spikes'],'spike count')
            require(count<=limits['max_spikes_per_rank'] and row['event_bytes']==8*count
                    and entry['observed_event_file_bytes']==row['event_bytes'], 'event byte/count ceiling')
            require(re.fullmatch('[0-9a-f]{64}',row['event_sha256']) is not None,'event digest')
            receipt=row['durable_event_receipt'];syncs=max(1,(row['event_bytes']+1048575)//1048576)
            require(receipt['schema']=='b2-mam-durable-event-stream-v1' and receipt['complete'] is True
                    and receipt['records']==count and receipt['bytes']==receipt['durable_bytes']==row['event_bytes']
                    and receipt['sha256']==row['event_sha256'] and receipt['maximum_bytes']==limits['max_spikes_per_rank']*8
                    and receipt['buffer_bytes']==65536 and receipt['data_sync_interval_bytes']==1048576
                    and receipt['data_sync_calls']==receipt['cache_release_calls']==syncs
                    and receipt['directory_synced'] is True and receipt['io_backend']=='LinuxIO'
                    and row['event_readback_verified'] is True, 'durability receipt mismatch')
            chunks=row['chunks'];require(len(chunks)==identity['duration_ms']//50,'complete admitted time blocks required')
            total=0;parts={k:0. for k in ['simulation_seconds','output_seconds','monitor_progress_seconds']}
            for i,chunk in enumerate(chunks):
                c=integer(chunk['spikes'],'chunk spikes');require(c<=limits['max_chunk_spikes'] and chunk['end_ms']==(i+1)*50,'chunk bounds')
                total+=c
                for k in parts:parts[k]+=number(chunk[k],'chunk time')
            require(total==count,'chunk spike sum')
            for key,source in [('simulation_seconds','simulation_seconds'),('event_output_seconds','output_seconds'),('monitor_progress_seconds','monitor_progress_seconds')]:
                require(math.isclose(number(row[key],'rank time'),parts[source],rel_tol=1e-12,abs_tol=1e-8),'component time total')
            recording_wall=number(row['recording_wall_seconds'],'recording wall',True)
            components=sum(parts.values())+number(row['recording_finish_seconds'],'finish')+number(row['event_verification_seconds'],'verify')
            require(components<=recording_wall+1e-6, 'recording times exceed wall')
            pre=number(row['rank_pre_report_wall_seconds'],'pre-report wall',True)
            terminal=number(done['rank_wall_through_report_seconds'],'terminal rank wall',True)
            require(recording_wall<=row['wall_seconds']<=pre<=terminal<=wall, 'rank timing endpoint order')
            require(0<integer(row['peak_rss_kib'],'rank RSS',1)*1024<=proxy['memory_bytes'],'rank RSS ceiling')
            spikes+=count;event_bytes+=row['event_bytes']
            rank_rows.append(dict(rank=rank,host=host['host'],spikes=count,event_bytes=row['event_bytes'],
                                  rank_wall_through_report_seconds=terminal,report_sha256=done['report']['sha256']))
    require(projections==expected_projections and external==recording==4129924,'global construction ledger differs')
    require(event_bytes<=limits['total_event_bytes'],'total event byte ceiling')
    workers=[g for g in guards if g['role'].startswith('proxy-')]
    return dict(schema='b2-mam-benchmark-terminal-controls-v1',run_purpose=purpose,
                duration_ms=identity['duration_ms'],terminal_resource_audit_passed=True,
                construction_ledger_passed=True,rank_completion_report_hashes_passed=True,
                raw_output_audit_passed=False,scientific_acceptance=False,nest_statistical_acceptance=False,
                performance_cost_acceptance=False,reported_spikes=spikes,reported_event_bytes=event_bytes,
                ranks=rank_rows,guards=guards,launch_wall_seconds=wall,
                accounting=dict(measured_cgroup_core_hours=sum(g['measured_cpu_seconds'] for g in guards)/3600,
                    worker_capacity_core_hours=wall*sum(s['cpu_quota_cores'] for s in specs if s['role'].startswith('proxy-'))/3600,
                    shared_participating_node_hours=wall*len(hosts)/3600,
                    sum_worker_host_peaks_bytes=sum(g['peak_bytes'] for g in workers),
                    exclusive_allocated_node_hours=None,monetary_cost=None),
                limitations=['Independent raw record scanning remains required; reported sync/readback is not collector raw validation.',
                    'Sum of host peaks is an upper bound, not simultaneous cluster peak.',
                    'Affinity receipts do not independently observe every OpenMP worker.',
                    'Shared participating node-hours are not exclusive allocation or a monetary bill.'])
