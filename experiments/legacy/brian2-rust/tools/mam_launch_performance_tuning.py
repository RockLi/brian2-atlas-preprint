"""Launch exactly one frozen full-geometry tuning case, with fresh admission.

No target run, retry, replacement seed, or automatic next case is supported.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import mam_launch_native_primary as old
from mam_benchmark_rank_affinity import binding

PROTOCOL_SHA = '01097672acd0c68dff73ce6e79e591d1c6d3f8f5aeabddb067b240c84fc11043'
SOURCES = ['mam_nest_benchmark.py', 'mam_benchmark_recording.py',
           'mam_benchmark_event_io.py', 'mam_benchmark_workload_v1.json']


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def options_for(protocol, case_id, output):
    old.check(case_id in ['nest48x4-tuning', 'nest96x2-tuning'], 'only two registered tuning cases')
    case = next(c for c in protocol['cases'] if c['id'] == case_id)
    old.check((case['ranks'], case['threads'], case['duration_ms'], case['wall_seconds'])
              in [(48, 4, 2500, 1800), (96, 2, 2500, 1800)], 'unregistered tuning geometry')
    label = f"nest-mam-perf-v1-n{case['ranks']}-t{case['threads']}-seed1729-2500ms"
    project = protocol['native']['project']
    options = old.launch_options(output)
    app = options['application']
    app = [x.replace(old.PROJECT, project).replace(old.LABEL, label)
           .replace('mam_nest_rank_affinity.py', 'mam_benchmark_rank_affinity.py')
           .replace('mam_nest_reference.py', 'mam_nest_benchmark.py') for x in app]
    replacements = {'--layout': project+'/'+case['layout']+'-layout.json',
                    '--layout-sha256': protocol['native']['layouts'][case['layout']]['sha256'],
                    '--ranks': str(case['ranks']), '--threads': str(case['threads']),
                    '--duration-ms': '2500', '--max-spikes-per-rank': str(case['max_spikes_per_rank'])}
    for flag, value in replacements.items(): app[app.index(flag)+1] = value
    app[app.index('OMP_NUM_THREADS=4')] = 'OMP_NUM_THREADS='+str(case['threads'])
    options.update(application=app, ranks_per_node=case['ranks']//6,
                   timeout=case['wall_seconds'], guard_script=project+'/mpi_resource_guard.py')
    return case, label, options


def admission_for(protocol, case, label, layout, options, checks):
    rph = options['ranks_per_node']
    placements = [dict(rank=rank, host=host['host'],
                       cpu_ids=binding(layout, rank, host['host'], old.CPUS)[0])
                  for i, host in enumerate(layout['hosts']) for rank in range(i*rph, (i+1)*rph)]
    guards = [dict(host=host, role=role, memory_bytes=256*2**30, pids_max=64,
                   cpu_ids=old.CPUS, cpu_quota_cores=32, volume='/', allow_root_volume=True,
                   file_limit_bytes=3*2**30, minimum_free_bytes=128*2**30,
                   reserved_host_memory_bytes=64*2**30)
              for host, role in [(old.NODES[0], 'controller')]+
              [(host, 'proxy-'+str(i)) for i, host in enumerate(old.NODES)]]
    return dict(schema='b2-mam-benchmark-admission-v1', admitted=True, label=label,
        case_id=case['id'], run_purpose=case['purpose'], protocol_sha256=PROTOCOL_SHA,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        identity=dict(ranks=case['ranks'], threads=case['threads'], seed=1729,
                      duration_ms=2500, dt_ms=.1, nest_version='3.10.0'),
        nodes=old.NODES, ranks_per_node=rph, placements=placements, guards=guards,
        workload_sha256=protocol['workload_sha256'], parameters_sha256=old.PARAMETERS,
        source_catalog={name: protocol['sources'][name] for name in SOURCES},
        staged_source_catalog=protocol['sources'],
        layout_sha256=protocol['native']['layouts'][case['layout']]['sha256'],
        runtime_catalog_sha256=protocol['native']['runtime_catalog_sha256'],
        limits=dict(chunk_ms=50, automatic_retry=False, max_chunk_spikes=2000000,
                    max_spikes_per_rank=case['max_spikes_per_rank'],
                    total_event_bytes=case['total_event_bytes'], wall_seconds=1800),
        host_paths=[dict(base=old.BASE, project=Path(protocol['native']['project']).name,
                         run='runs/'+label, affinity='affinity/'+label, guards='guards',
                         runtime_catalog='nest-runtime-v1/catalog.json') for _ in old.NODES],
        preflight=checks, launch_options={k:str(v) if isinstance(v, Path) else v for k,v in options.items()},
        scientific_acceptance=False, performance_cost_acceptance=False)


def run(args):
    phase = args.evidence/'performance-protocol-v1'
    old.check(old.sha(phase/'protocol.json') == PROTOCOL_SHA, 'frozen protocol differs')
    protocol = old.read(phase/'protocol.json')
    case, label, options = options_for(protocol, args.case, args.t7/'logs'/args.case)
    stage = old.read(phase/'staging.json')
    old.check(stage['all_hosts_staged'] and [r['host'] for r in stage['rows']] == old.NODES,
              'six-host staging incomplete')
    for name, row in protocol['sources'].items():
        path = Path(__file__).parent/name
        old.check(path.stat().st_size == row['bytes'] and old.sha(path) == row['sha256'], 'local source differs: '+name)
    layout_path = phase/(case['layout']+'-layout.json')
    old.check(old.sha(layout_path) == protocol['native']['layouts'][case['layout']]['sha256'], 'layout differs')
    layout = old.read(layout_path)
    if args.case == 'nest96x2-tuning':
        prior = args.evidence/'performance-runs-v1'/'nest48x4-tuning'
        completion = old.read(prior/'completion.json')
        old.check(completion['protocol_sha256'] == PROTOCOL_SHA and completion['case_id'] == 'nest48x4-tuning'
                  and completion['terminal_resource_audit_passed'] is True
                  and completion['raw_output_audit_passed'] is True, 'first tuning not independently accepted')
        for name, digest in completion['input_sha256'].items():
            old.check(old.sha(prior/name) == digest, 'prior completion input differs')
    old.check(Path('/Volumes/T7').is_mount() and args.t7.resolve().is_relative_to(Path('/Volumes/T7')),
              'T7 must be mounted')
    stat = os.statvfs(args.t7)
    old.check(stat.f_bavail*stat.f_frsize >= 512*2**30, 'T7 start reserve')
    output = args.evidence/'performance-runs-v1'/args.case
    output.parent.mkdir(exist_ok=True)
    output.mkdir(exist_ok=False)  # One admission attempt, never overwrite/retry.
    old.check(not options['output'].exists(), 'run logs already exist')
    write(output/'intent.json', dict(case_id=args.case, protocol_sha256=PROTOCOL_SHA,
        maximum_neural_runs=1, automatic_retry=False, launch_options={k:str(v) if isinstance(v, Path) else v for k,v in options.items()}))
    inventory = old.read(args.evidence/'primary-native-layout/inventory.json')['nodes']
    started = time.monotonic()
    leader_code = '''import os,json,subprocess,datetime
from pathlib import Path
assert os.uname().nodename=='hk-prod-model-ae02-23'
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5)
assert not units.strip(),units
p=Path('/data/brick2');s=os.statvfs(p);free=s.f_bavail*s.f_frsize
assert os.stat(p).st_dev!=os.stat('/').st_dev and free>=2048*2**30
print(json.dumps(dict(host=os.uname().nodename,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=free,active_own_units=[])))'''
    def check_host(index):
        staged = stage['rows'][index]['result']
        code = old.preflight_code(index, staged, layout['hosts'][index]['selected_topology'],
                                  inventory[index]['mpi_sha256'], output_label=label)
        code = code.replace(old.PROJECT, protocol['native']['project'])
        code = 'import resource,signal\nsignal.alarm(40)\nresource.setrlimit(resource.RLIMIT_AS,(768*2**20,768*2**20))\nresource.setrlimit(resource.RLIMIT_CPU,(30,30))\n'+code
        return old.remote(old.NODES[index], code)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
            leader = pool.submit(old.remote, 'hk-prod-model-ae02-23', leader_code)
            futures = [pool.submit(check_host, i) for i in range(6)]
            checks = dict(node23=leader.result(), native=[f.result() for f in futures])
        old.check(time.monotonic()-started < 60, 'preflight stale')
        admission = admission_for(protocol, case, label, layout, options, checks)
        admission['launcher_sources'] = {n: old.sha(Path(__file__).parent/n) for n in
            ['mam_launch_performance_tuning.py', 'mam_launch_native_primary.py', 'mpi_teleport_launch.py']}
        write(output/'admission.json', admission)
        mirror = args.t7/'artifacts'/'performance-protocol-v1'
        mirror.mkdir(exist_ok=True)
        write(mirror/(args.case+'-admission.json'), admission)
        old.check(time.monotonic()-started < 60, 'preflight expired before launch')
        print(json.dumps(dict(event='admitted', case_id=args.case, nodes=old.NODES,
                              wall_limit_seconds=1800, admission_sha256=old.sha(output/'admission.json'))), flush=True)
        from mpi_teleport_launch import launch
        result = launch(**options)
        write(output/'launch.json', result)
        print(json.dumps(dict(event='launcher_terminal', case_id=args.case,
                              wall_seconds=result['wall_seconds'], audits_required=True)), flush=True)
    except BaseException as error:
        if (options['output']/'launch.json').exists() and not (output/'launch.json').exists():
            write(output/'launch.json', old.read(options['output']/'launch.json'))
        write(output/'failure.json', dict(error_type=type(error).__name__, error=str(error), automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--t7', type=Path, required=True)
    parser.add_argument('--case', choices=['nest48x4-tuning', 'nest96x2-tuning'], required=True)
    run(parser.parse_args())
