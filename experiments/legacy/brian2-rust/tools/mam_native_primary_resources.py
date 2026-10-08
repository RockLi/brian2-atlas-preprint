"""Native primary terminal control audit; recording bytes remain a separate gate."""
import math
from pathlib import Path
import re

from mam_launch_native_primary import LABEL, BASE, PROJECT, NODES, IPS, PARAMETERS, LAYOUT_SHA, CPUS, launch_options
from mam_primary_resources import require, number, counters, GIB
from mam_native_rank_audit import identity
from mam_nest_reference import virtual_process_capacity
from mam_nest_rank_affinity import binding

NEURONS = 4129924
EDGES = 24126516728


def terminal_launch(admission, launch, *, label=LABEL, seed=1729):
    require(admission['schema'] == 'b2-mam-native-primary-admission-v1'
            and admission['label'] == label and admission['admitted'] is True
            and admission['parameters_sha256'] == PARAMETERS
            and admission['layout_sha256'] == LAYOUT_SHA, 'native admission identity')
    require(admission['experiment_budget'] == dict(runs=1, automatic_retry=False,
            wall_seconds=54000, collection_seconds=7200, analysis_stage_seconds=10800),
            'native experiment budget changed')
    options = admission['launch_options']
    expected = launch_options(Path(options['output']), label=label, seed=seed)
    expected['output'] = str(expected['output'])
    require(options == expected, 'native admitted command or limits changed')
    require(admission['rust_leader_terminal'] == dict(rust_primary_active_units=[]),
            'missing Rust terminal admission')
    require(set(admission['prerequisite']) == {'resource_report_sha256',
            'output_report_sha256', 'output_guard_sha256'}
            and all(re.fullmatch('[0-9a-f]{64}', v) for v in admission['prerequisite'].values()),
            'missing pinned Rust prerequisite hashes')
    require([r['host'] for r in admission['preflight']] == NODES, 'native admission host coverage')
    for index, row in enumerate(admission['preflight']):
        require(row['available_memory_bytes'] >= (576 if index == 0 else 320)*GIB
                and row['free_bytes'] >= 192*GIB and row['active_own_units'] == []
                and all(row[k] is True for k in ['source_catalog_verified',
                    'runtime_catalog_verified','mpi_verified','selected_topology_verified']),
                'native original admission failed')
        busy = row['cpu_busy_percent']
        require(set(map(int, busy)) == set(CPUS), 'CPU load coverage differs')
        values = [number(v, 'invalid CPU load') for v in busy.values()]
        require(max(values) < 50 and sum(values)/32 < 25, 'admission CPU load failed')
    roles = {'controller'} | {'proxy-'+str(i) for i in range(6)}
    require(launch['schema'] == 'b2-teleport-hydra-launch-v0'
            and launch.get('error') is None and set(launch['returncodes']) == roles
            and all(type(v) is int and v == 0 for v in launch['returncodes'].values()),
            'native launcher not terminal success')
    require(launch['nodes'] == NODES and launch['ips'] == IPS
            and launch['ranks_per_node'] == 8 and launch['remote_timeout_seconds'] == 54000,
            'native launch placement or deadline differs')
    require(number(launch['wall_seconds'], 'launch wall', positive=True) <= 54000, 'native wall budget exceeded')
    guard = launch['resource_guard'];prefix = guard['unit_prefix']
    require(re.fullmatch('b2mpi-[0-9a-f]{12}', prefix), 'invalid native guard family')
    require(guard == dict(script=PROJECT+'/mpi_resource_guard.py', volume='/',
            allow_root_volume=True, memory_mib=262144, cpu_percent=3200, cpu_count=32,
            cpu_ids=CPUS, file_mib=3072, unit_prefix=prefix, minimum_free_gib=128),
            'native launch guard limits differ')
    return prefix


def cpu_set(text):
    result = []
    for part in text.split(','):
        pair = part.split('-')
        require(len(pair) in (1,2) and all(x.isdigit() for x in pair), 'invalid cpuset')
        lo, hi = int(pair[0]), int(pair[-1])
        require(0 <= lo <= hi < 4096, 'invalid CPU interval')
        result.extend(range(lo, hi+1))
    require(len(set(result)) == len(result), 'duplicate CPU')
    return sorted(result)


def audit_guard(g, host, role, prefix):
    require(g['schema'] == 'b2-mpi-resource-guard-v1' and g['host'] == host
            and g['cgroup'] == '/system.slice/'+prefix+'-'+role+'.service'
            and g['admitted'] is True and g['uid'] == 1000
            and type(g.get('returncode')) is int and g['returncode'] == 0
            and not g.get('error'), 'native guard not terminal success/identity')
    wall = number(g['wall_seconds'], 'guard wall', positive=True)
    require(wall <= 54000, 'native guard wall exceeded')
    for phase in ['before','after']:
        row = g[phase]
        require(row['memory.max'] == str(256*GIB) and row['memory.swap.max'] == '0'
                and row['pids.max'] == '64' and cpu_set(row['cpuset.cpus.effective']) == CPUS,
                'native guard limits changed')
        quota, period = map(int, row['cpu.max'].split())
        require(period > 0 and quota == period*32, 'native CPU quota changed')
        events = counters(row['memory.events'])
        require(all(events[k] == 0 for k in ['max','oom','oom_kill','oom_group_kill']),
                'native memory pressure is not acceptance')
    peak = int(g['after']['memory.peak'])
    require(0 < peak <= 256*GIB, 'native memory peak outside budget')
    before, after = [counters(g[phase]['cpu.stat']) for phase in ['before','after']]
    require(all(after[k] >= before[k] for k in ['usage_usec','user_usec','system_usec']), 'CPU counter regression')
    cpu = (after['usage_usec']-before['usage_usec'])/1e6
    require(g['data_volume'] == '/' and g['root_volume_allowed'] is True
            and g['data_device'] == g['root_device'] and g['file_limit_bytes'] == 3*GIB
            and g['minimum_free_bytes'] == 128*GIB and g['disk_check_interval_seconds'] == 5
            and number(g['data_free_bytes'], 'disk admission') >= 128*GIB
            and number(g['minimum_observed_free_bytes'], 'disk minimum') >= 128*GIB,
            'native storage reserve or file limit failed')
    require(number(g['reserved_host_memory_bytes'], 'host memory reserve') >= 64*GIB
            and g['host_memory_bytes']['MemAvailable'] >= 256*GIB+g['reserved_host_memory_bytes'],
            'native guard memory admission failed')
    return dict(host=host, role=role, peak_bytes=peak, wall_seconds=wall,
                measured_cpu_seconds=cpu, minimum_observed_free_bytes=g['minimum_observed_free_bytes'])


def parameter_geometry(parameters):
    require(parameters['schema'] == 'b2-official-mam-parameters-v1'
            and parameters['total_neurons'] == NEURONS
            and parameters['total_recurrent_synapses'] == EDGES
            and parameters['N_scaling'] == parameters['K_scaling'] == 1., 'native full parameter geometry')
    require(len(parameters['populations']) == 254 and len(parameters['projections']) == 8344,
            'native population/projection coverage')
    offset = 0;populations = []
    for row in parameters['populations']:
        count = row['count'];require(type(count) is int and count > 0, 'invalid population count')
        populations.append(dict(name=row['name'], count=count, first_gid=offset+1,
                                cell_start=offset, cell_end=offset+count))
        offset += count
    counts = [q['count'] for q in parameters['projections']]
    require(offset == NEURONS and all(type(n) is int and n > 0 for n in counts)
            and sum(counts) == EDGES, 'native parameter count totals')
    return populations, counts


def audit(admission, launch, collected, parameters, layout, *, label=LABEL, seed=1729):
    """Caller must hash-pin parameter/layout bytes and collected source controls.

    No raw recording is opened here. Reported spike counts and construction
    ledgers are checked, but physical event bins and exact prefixes are separate.
    """
    prefix = terminal_launch(admission, launch, label=label, seed=seed)
    populations, expected = parameter_geometry(parameters)
    require([h['host'] for h in layout['hosts']] == NODES, 'native layout host coverage')
    require([h['host'] for h in collected] == NODES, 'native collected host coverage')
    capacity = virtual_process_capacity(parameters, 48, 4)
    guards = [];ranks = [];totals = [0]*len(expected);devices = [0,0,0]
    for index, host in enumerate(collected):
        roles = (['controller'] if index == 0 else [])+['proxy-'+str(index)]
        require(set(host['guards']) == set(roles) and host['active_own_units'] == [],
                'missing guard or native services still live')
        for role in roles:guards.append(audit_guard(host['guards'][role], host['host'], role, prefix))
        rank_ids = {str(r) for r in range(index*8,(index+1)*8)}
        require(set(host['rank_reports']) == set(host['affinity']) == rank_ids, 'native rank/receipt coverage')
        for rank in range(index*8,(index+1)*8):
            row = host['rank_reports'][str(rank)];receipt = host['affinity'][str(rank)]
            cpus, env = binding(layout, rank, host['host'], CPUS)
            require(cpus == CPUS[(rank%8)*4:(rank%8+1)*4], 'native rank CPU group differs')
            require(receipt['rank'] == rank and receipt['ranks'] == 48 and receipt['host'] == host['host']
                    and receipt['initial_rank_cpu_ids'] == cpus and receipt['openmp_environment'] == env
                    and receipt['layout_sha256'] == LAYOUT_SHA, 'native affinity receipt differs')
            identity(row, rank, 100500, NEURONS, PARAMETERS, [cpus[0]], expected_seed=seed)
            require(row['host'] == host['host'] and row['populations'] == populations
                    and row['N_scaling'] == row['K_scaling'] == 1.
                    and row['connection_index_capacity'] == capacity, 'native rank construction identity differs')
            require(len(row['projection_local_counts']) == len(expected), 'native local projection coverage')
            totals = [a+b for a,b in zip(totals,row['projection_local_counts'],strict=True)]
            for j,key in enumerate(['local_external_connections','local_recording_connections','local_total_connections']):
                require(type(row[key]) is int and row[key] >= 0, 'invalid device count')
                devices[j] += row[key]
            require(number(row['min_delay_ms'], 'minimum delay', positive=True) >= .1
                    and number(row['max_delay_ms'], 'maximum delay', positive=True) >= row['min_delay_ms'],
                    'invalid native delay range')
            require(re.fullmatch('[0-9a-f]{64}', row['event_sha256']), 'invalid native event hash')
            wall = number(row['wall_seconds'], 'rank wall', positive=True)
            require(wall <= launch['wall_seconds'] and 0 < row['peak_rss_kib']*1024 <= 256*GIB,
                    'rank exceeds wall or RSS limit')
            spike_total = 0;simulation_parts = [];output_parts = []
            for i, chunk in enumerate(row['chunks']):
                require(chunk['end_ms'] == (i+1)*50 and type(chunk['spikes']) is int
                        and 0 <= chunk['spikes'] <= 2000000, 'native chunk duration/quota differs')
                spike_total += chunk['spikes']
                simulation_parts.append(number(chunk['simulation_seconds'], 'chunk simulation time'))
                output_parts.append(number(chunk['output_seconds'], 'chunk output time'))
            simulation, output_seconds = sum(simulation_parts), sum(output_parts)
            require(spike_total == row['local_spikes'] and math.isclose(row['simulation_seconds'], simulation, rel_tol=1e-12, abs_tol=1e-9),
                    'native chunk total differs')
            build = sum(number(row[k], 'rank construction time') for k in
                        ['neuron_seconds','recurrent_seconds','device_seconds'])
            require(build+simulation+output_seconds <= wall+1e-6, 'rank component times exceed wall')
            first = row['phase_memory']['neurons_created'];last = row['phase_memory']['after_100500ms']
            for phase in [first,last]:
                for key in ['user_seconds','system_seconds']:number(phase[key], 'rank CPU time')
                # /proc VmRSS and getrusage ru_maxrss are separate asynchronous
                # kernel observations. Do not impose a cross-source ordering.
                require(type(phase['rss_kib']) is int and 0 < phase['rss_kib']*1024 <= 256*GIB
                        and type(phase['peak_rss_kib']) is int
                        and 0 < phase['peak_rss_kib'] <= row['peak_rss_kib'], 'rank RSS snapshot differs')
            require(all(last[k] >= first[k] for k in ['user_seconds','system_seconds','peak_rss_kib']),
                    'rank resource counters regressed')
            ranks.append(dict(rank=rank, host=host['host'], peak_rss_bytes=row['peak_rss_kib']*1024,
                final_proc_rss_bytes=last['rss_kib']*1024,
                final_proc_rss_minus_rusage_peak_bytes=(last['rss_kib']-last['peak_rss_kib'])*1024,
                reported_wall_seconds=wall, simulation_seconds=simulation, spikes=spike_total,
                event_bytes=row['event_bytes'], cpu_seconds_through_final_chunk=last['user_seconds']+last['system_seconds']))
    require(totals == expected and devices == [NEURONS,NEURONS,EDGES+2*NEURONS],
            'native global projection or device totals differ')
    wall = launch['wall_seconds']
    return dict(schema='b2-mam-native-primary-resources-v1', label=label,
        terminal_resource_audit_passed=True, construction_ledger_passed=True,
        raw_output_audit_passed=False, scientific_acceptance=False, performance_cost_acceptance=False,
        all_actual_openmp_workers_observed=False, neurons=NEURONS, recurrent_edges=EDGES, projections=len(expected),
        reported_spikes=sum(r['spikes'] for r in ranks), reported_event_bytes=sum(r['event_bytes'] for r in ranks),
        guards=guards, ranks=ranks, launch_wall_seconds=wall,
        accounting=dict(participating_node_hours=6*wall/3600,
            worker_allowed_capacity_core_hours=192*wall/3600,
            measured_cgroup_core_hours_including_controller=sum(g['measured_cpu_seconds'] for g in guards)/3600,
            rank_cpu_core_hours_through_final_chunk=sum(r['cpu_seconds_through_final_chunk'] for r in ranks)/3600,
            exclusive_allocated_node_hours=None, monetary_cost=None),
        limitations=['Reported recorder SHA/counts are not independent raw-byte validation.',
            '/proc VmRSS is an approximate asynchronous observation; no ordering against getrusage ru_maxrss is asserted. Both are retained; cgroup caps/events provide job resource enforcement.',
            'Affinity receipts and main-thread reports do not observe every OpenMP worker.',
            'Rank CPU snapshots precede final hash/JSON/teardown; use terminal cgroup CPU for whole-job accounting.',
            'Node-hours describe shared-host participation, not exclusive allocation or a bill.',
            'Allowed capacity is not measured consumption. Peak RSS values are not simultaneous/integrated memory.',
            'Scientific equivalence and tuned repeated speed/cost acceptance remain open.'])
