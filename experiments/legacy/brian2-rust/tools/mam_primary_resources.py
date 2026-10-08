"""Terminal resource accounting for the admitted 100.5 s Rust primary run.

This checks small collected control records only. It does not validate spike
files, numerical results, scientific equivalence, or a performance advantage.
"""
import argparse
import json
import math
from pathlib import Path
import re

LABEL = 'full32-n1-k1-spool-primary-v1-100500ms'
NODES = ['hk-prod-model-ae02-23', 'hk-prod-model-ae08-81',
         'hk-prod-model-ae08-83', 'hk-prod-model-ae07-71']
CPUS = [0, 12, 24, 36, 48, 60, 72, 84]
PREFIX = 'b2mpi-098ee34e845b'
PLAN = '240c3a14ab8366811e5837d6451a3ab2d849220f17a7564a06d366fc459f49bd'
MODEL = '9526a75e00e7cd4c3457691610fce4072c2014c6e6c6b53ea8a62f588dcac6bd'
GIB = 2**30


def require(value, message):
    if not value:
        raise ValueError(message)


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), name)
    return value


def counters(text):
    result = {}
    for line in text.splitlines():
        key, value = line.split()
        require(key not in result and value.isdigit(), 'invalid/duplicate counter')
        result[key] = int(value)
    return result


def time_record(text):
    """Require one complete GNU time -v record, including successful exit."""
    def one(label, pattern):
        values = re.findall(r'^\s*' + re.escape(label) + r': ' + pattern + r'\s*$',
                            text, re.MULTILINE)
        require(len(values) == 1, 'missing/duplicate time field: ' + label)
        return values[0]
    require(one('Exit status', r'(\d+)') == '0', 'rank failed')
    user = number(float(one('User time (seconds)', r'(\S+)')), 'user time')
    system = number(float(one('System time (seconds)', r'(\S+)')), 'system time')
    rss = int(one('Maximum resident set size (kbytes)', r'(\d+)')) * 1024
    require(rss > 0, 'missing rank RSS')
    elapsed = one('Elapsed (wall clock) time (h:mm:ss or m:ss)', r'(\S+)')
    parts = elapsed.split(':')
    require(len(parts) in (2, 3), 'invalid elapsed time')
    values = [number(float(v), 'elapsed component') for v in parts]
    require(values[-1] < 60 and (len(values) == 2 or values[-2] < 60),
            'invalid elapsed clock')
    wall = sum(v * 60**i for i, v in enumerate(reversed(values)))
    number(wall, 'rank wall', positive=True)
    return dict(user_seconds=user, system_seconds=system,
                elapsed_seconds=wall, peak_rss_bytes=rss)


def audit(admission, launch, collected, runtime):
    """Fail closed on missing/live/failed records; never infer terminal success."""
    require(admission['label'] == LABEL and admission['model_sha256'] == MODEL,
            'wrong primary admission')
    require(admission['condition'] == dict(chi=1.9, seed=1729,
            duration_seconds=100.5, transient_seconds=.5, observation_seconds=100.,
            neurons=4129924, synapses=24126516728), 'wrong condition')
    budget = admission['experiment_budget']
    require(budget['simulation_wall_limit_seconds'] == 64800
            and budget['primary_simulation_runs'] == 1
            and budget['automatic_retry'] is False, 'wrong experiment budget')
    resources = admission['resources']
    require(resources['nodes'] == NODES and resources['ranks'] == 32
            and resources['ranks_per_node'] == 8 and resources['cpu_ids'] == CPUS,
            'wrong admitted layout')
    require(resources['memory_mib_per_proxy'] == 262144
            and resources['zero_swap'] is True
            and resources['cpu_percent_per_proxy'] == 800
            and resources['file_limit_mib'] == 131072
            and resources['data_minimum_runtime_reserve_bytes'] == 128*GIB,
            'wrong admitted limits')
    require([p['host'] for p in admission['preflight']] == NODES,
            'incomplete admission hosts')
    for index, row in enumerate(admission['preflight']):
        require(row['memory']['MemAvailable'] >= (576 if index == 0 else 320)*GIB
                and row['free_bytes'] >= (1280 if index == 0 else 192)*GIB
                and row['all_artifact_hashes_verified'] is True,
                'missing original resource/artifact admission')
    roles = ['controller'] + ['proxy-' + str(i) for i in range(4)]
    require(launch.get('error') is None and set(launch['returncodes']) == set(roles)
            and all(type(v) is int and v == 0 for v in launch['returncodes'].values()),
            'launcher is not terminal success')
    require(launch['nodes'] == NODES and launch['ranks_per_node'] == 8
            and launch['resource_guard']['unit_prefix'] == PREFIX,
            'wrong launch identity')
    wall = number(launch['wall_seconds'], 'launch wall', positive=True)
    require(wall <= 64800, 'launch exceeded admitted wall budget')
    require([n['host'] for n in collected] == NODES, 'missing/duplicate/wrong host')
    require(runtime['ranks'] == 32 and runtime['plan_sha256'] == PLAN
            and runtime['exchange_calls_per_rank'] == 1005000,
            'runtime identity/duration mismatch')
    require(runtime['processor_names'] == [n for n in NODES for _ in range(8)],
            'runtime rank placement mismatch')
    require(len(runtime['rank_cpu_ids']) == 32, 'missing rank affinity')
    guards = []
    ranks = []
    for index, host in enumerate(collected):
        expected_roles = (['controller'] if index == 0 else []) + ['proxy-' + str(index)]
        require(set(host['guards']) == set(expected_roles), 'missing/extra guard')
        require(sorted(runtime['rank_cpu_ids'][index*8:(index+1)*8]) == CPUS,
                'missing/duplicate CPU in host affinity')
        for role in expected_roles:
            g = host['guards'][role]
            require(g['host'] == host['host'] and g['cgroup'] ==
                    '/system.slice/' + PREFIX + '-' + role + '.service',
                    'guard identity mismatch')
            require(g['schema'] == 'b2-mpi-resource-guard-v1'
                    and g['admitted'] is True and g['uid'] == 1000
                    and type(g.get('returncode')) is int and g['returncode'] == 0
                    and not g.get('error') and 'after' in g, 'guard not terminal success')
            gwall = number(g['wall_seconds'], 'guard wall', positive=True)
            require(gwall <= 64800, 'guard exceeded wall budget')
            for phase in ['before', 'after']:
                facts = g[phase]
                require(facts['memory.max'] == str(256*GIB)
                        and facts['memory.swap.max'] == '0'
                        and facts['pids.max'] == '64'
                        and facts['cpuset.cpus.effective'] == ','.join(map(str, CPUS)),
                        'guard limits changed/missing')
                quota, period = map(int, facts['cpu.max'].split())
                require(period > 0 and quota == 8*period, 'wrong CPU quota')
                events = counters(facts['memory.events'])
                require(all(events[k] == 0 for k in ['max', 'oom', 'oom_kill',
                                                     'oom_group_kill']),
                        'memory pressure/failure is not resource acceptance')
            peak = int(g['after']['memory.peak'])
            require(0 < peak <= 256*GIB, 'peak outside admitted memory')
            cpu0, cpu1 = (counters(g[t]['cpu.stat']) for t in ['before', 'after'])
            require(all(cpu1[k] >= cpu0[k] for k in ['usage_usec', 'user_usec',
                                                    'system_usec']),
                    'CPU counters regressed')
            cpu = (cpu1['usage_usec'] - cpu0['usage_usec']) / 1e6
            expected_volume = '/data/brick2' if index == 0 else '/'
            require(g['data_volume'] == expected_volume
                    and ((g['data_device'] != g['root_device']) if index == 0 else
                         (g['data_device'] == g['root_device'])), 'wrong storage device')
            require(g['file_limit_bytes'] == 128*GIB
                    and g['minimum_free_bytes'] == 128*GIB
                    and g['disk_check_interval_seconds'] == 5
                    and g['data_free_bytes'] >= 128*GIB
                    and number(g['minimum_observed_free_bytes'], 'disk minimum') >= 128*GIB,
                    'disk quota/reserve failed')
            require(g['host_memory_bytes']['MemAvailable'] >=
                    256*GIB + g['reserved_host_memory_bytes'], 'guard admission failed')
            guards.append(dict(host=host['host'], role=role, peak_bytes=peak,
                               wall_seconds=gwall, measured_cpu_seconds=cpu,
                               minimum_observed_free_bytes=g['minimum_observed_free_bytes']))
        expected_ranks = {str(r) for r in range(index*8, (index+1)*8)}
        require(set(host['rank_time']) == expected_ranks, 'missing/extra rank time')
        for rank in range(index*8, (index+1)*8):
            row = time_record(host['rank_time'][str(rank)])
            require(row['elapsed_seconds'] <= 64800 and row['peak_rss_bytes'] <= 256*GIB,
                    'rank exceeds admitted budget')
            ranks.append(dict(rank=rank, host=host['host'], **row))
    return dict(schema='b2-mam-primary-resources-v1', label=LABEL,
                terminal_resource_audit_passed=True, raw_output_audit_passed=False,
                scientific_acceptance=False, performance_cost_acceptance=False,
                launch_wall_seconds=wall, guards=guards, ranks=ranks,
                accounting=dict(
                    participating_node_hours=4*wall/3600,
                    worker_allowed_capacity_core_hours=32*wall/3600,
                    measured_cgroup_core_hours_including_controller=
                        sum(g['measured_cpu_seconds'] for g in guards)/3600,
                    measured_rank_user_system_core_hours=
                        sum(r['user_seconds'] + r['system_seconds'] for r in ranks)/3600,
                    exclusive_allocated_node_hours=None, monetary_cost=None),
                limitations=[
                    'Participating node-hours are shared-host elapsed time, not exclusive allocation or a bill.',
                    'Allowed-capacity core-hours are a quota envelope, not measured CPU consumption.',
                    'Cgroup CPU time includes MPI proxies/controller; rank GNU time counts the engine processes.',
                    'Per-host and per-rank peaks are not simultaneous total or integrated memory-time.',
                    'Affinity is final runtime metadata; live process samples are separate evidence.',
                    'Raw output, numerical/scientific agreement and tuned repeated speed/cost comparisons remain required.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['admission', 'launch', 'collected', 'runtime', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    paths = [args.admission, args.launch, args.collected, args.runtime]
    require(all(p.stat().st_size <= 32*2**20 for p in paths), 'control file exceeds 32 MiB')
    report = audit(*(json.loads(p.read_text()) for p in paths))
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
