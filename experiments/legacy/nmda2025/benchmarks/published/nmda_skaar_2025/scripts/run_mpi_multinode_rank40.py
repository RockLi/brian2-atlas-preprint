#!/usr/bin/env python3
"""Replay one topology of the frozen 40-rank NMDA project over Teleport.

The same preinstalled absolute runtime path must exist on every selected
host.  This script varies placement only; total ranks, executable, instance,
scientific parameters, and result hashes stay fixed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import time

ALL = [
    ('hk-prod-model-ae02-24', '192.168.20.24'),
    ('hk-prod-model-ae02-25', '192.168.20.25'),
    ('hk-prod-model-ae03-33', '192.168.20.33'),
    ('hk-prod-model-ae05-53', '192.168.20.53'),
    ('hk-prod-model-ae05-54', '192.168.20.54'),
]
CONFIG = {
    1: (40, list(range(0, 40, 2)) + list(range(48, 88, 2))),
    2: (20, [0,5,10,15,20,25,30,35,40,45,48,53,58,63,68,73,78,83,88,93]),
    4: (10, [0,10,20,30,40,48,58,68,78,88]),
    5: (8, [0,12,24,36,48,60,72,84]),
}
BASE = '/atlas-home/0003/workspace/nmda2025-multinode-20260918/runtime-v1'
REMOTE_PARENT = '/atlas-home/0003/workspace/nmda2025-multinode-20260918'
EXPECTED_RESULT = '1bb982959d7bd8b4c7cb6c82d2e31a116e8ad73015ea63f270f1f04651308b85'
EXPECTED_EVENTS = '9add17471f307e7e8b046f952e5eab029eb10a959bca8b46b73cc2ef616f5609'

def command(args, *, check=True):
    return subprocess.run(args, check=check, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT).stdout

def remote(host, script, *, login='rock'):
    return command(['tsh','ssh',f'{login}@{host}',script])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--nodes',type=int,choices=CONFIG,required=True)
    p.add_argument('--repetitions',type=int,default=5)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--campaign-id',required=True)
    a=p.parse_args()
    if a.root.exists(): p.error('--root must be new')
    a.root.mkdir(parents=True)
    pairs=ALL[:a.nodes]; nodes=[x[0] for x in pairs]; ips=[x[1] for x in pairs]
    ranks_per_node,cpus=CONFIG[a.nodes]
    preflight={}
    for host in nodes:
        out=remote(host,"hostname; cat /proc/loadavg; systemctl list-units --type=service --state=running 'b2mpi-*' --no-legend",login='root')
        lines=out.splitlines()
        if any('b2mpi-' in line for line in lines[2:]):
            raise RuntimeError(f'busy node {host}: {out}')
        preflight[host]=out
    report={'schema':'nmda-skaar-2025-multinode-fixed-rank-v1','nodes':nodes,'ips':ips,
            'total_ranks':40,'ranks_per_node':ranks_per_node,'cpu_ids':cpus,
            'excluded_warmups':1,'measured_repetitions':a.repetitions,
            'expected_result_sha256':EXPECTED_RESULT,'expected_events_sha256':EXPECTED_EVENTS,
            'preflight':preflight,'runs':[]}
    (a.root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    repository=Path(__file__).resolve().parents[4]
    module_path=repository/'brian2-rust/tools/mpi_teleport_launch.py'
    spec=importlib.util.spec_from_file_location('mpi_teleport_launch',module_path)
    launch_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch_module)
    count=1+a.repetitions
    for index in range(count):
        kind='warmup' if index==0 else 'measured'
        number=0 if index==0 else index
        label=f'{a.campaign_id}-{a.nodes}nodes-{ranks_per_node}rpn-{kind}-{number}'
        local=a.root/label
        remote_output=f'{REMOTE_PARENT}/{label}'
        started=time.time()
        launch=launch_module.launch(
            nodes=nodes,ips=ips,ranks_per_node=ranks_per_node,remote_base=BASE,
            mpi_prefix=BASE+'/mpi',
            application=[BASE+'/project/b2-mpi',BASE+'/project/instance.bin',remote_output],
            output=local,interface='bond0',timeout=300,tsh='tsh',login='root',
            guard_script=BASE+'/mpi_resource_guard.py',guard_memory_mib=32768,
            guard_cpu_percent=100*ranks_per_node,guard_volume='/',guard_allow_root_volume=True,
            guard_file_mib=512,guard_cpu_count=ranks_per_node,guard_cpu_ids=cpus,
            guard_min_free_gib=20)
        runtime=json.loads(remote(nodes[0],f'cat {remote_output}/mpi-runtime.json'))
        summary=json.loads(remote(nodes[0],f'cat {remote_output}/summary.json'))
        hashes={name:digest for digest,name in
                (line.split() for line in remote(nodes[0],f'cd {remote_output} && sha256sum results.bin events.bin').splitlines())}
        exact=(hashes.get('results.bin')==EXPECTED_RESULT and hashes.get('events.bin')==EXPECTED_EVENTS)
        stage=runtime['rank_stage_seconds']; exchange=runtime['spike_exchange_seconds']
        row={'kind':kind,'repetition':number,'label':label,'remote_output':remote_output,
             'started_unix':started,'finished_unix':time.time(),'launch':launch,
             'result_sha256':hashes.get('results.bin'),'events_sha256':hashes.get('events.bin'),
             'byte_exact':exact,'simulation_seconds':max(stage[2::5]),
             'initialization_seconds':max(stage[0::5]),
             'collection_seconds':max(stage[4::5]),
             'exchange_rank_seconds':{'min':min(exchange),'median':statistics.median(exchange),'max':max(exchange)},
             'rank_peak_rss_sum_bytes':sum(runtime['rank_peak_rss_bytes']),
             'rank_peak_rss_max_bytes':max(runtime['rank_peak_rss_bytes']),
             'processor_names':runtime['processor_names'],'rank_cpu_ids':runtime['rank_cpu_ids'],
             'summary':summary}
        report['runs'].append(row)
        (local/'mpi-runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
        (local/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (a.root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'topology':f'{a.nodes}x{ranks_per_node}','kind':kind,'repetition':number,
                          'exact':exact,'simulation_seconds':row['simulation_seconds'],
                          'exchange_median_seconds':row['exchange_rank_seconds']['median'],
                          'wall_seconds':launch['wall_seconds']}),flush=True)
        if not exact: raise RuntimeError('result mismatch')
    measured=[x for x in report['runs'] if x['kind']=='measured']
    def stats(field):
        values=[x[field] for x in measured]
        return {'median':statistics.median(values),'min':min(values),'max':max(values)}
    report['summary']={'all_byte_exact':all(x['byte_exact'] for x in report['runs']),
                       'simulation_seconds':stats('simulation_seconds'),
                       'launch_wall_seconds':{'median':statistics.median(x['launch']['wall_seconds'] for x in measured),
                                              'min':min(x['launch']['wall_seconds'] for x in measured),
                                              'max':max(x['launch']['wall_seconds'] for x in measured)}}
    (a.root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2),flush=True)

if __name__=='__main__': main()
