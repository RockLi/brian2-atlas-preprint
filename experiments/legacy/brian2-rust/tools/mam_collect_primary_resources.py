"""Collect only terminal primary control records through Teleport, onto T7.

No raw simulation data is read. No service is started/stopped. A missing or live
launcher record produces no collection; failed audits retain their inputs.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from mam_primary_resources import LABEL, NODES, PREFIX, audit, require

BASE = '/atlas-home/0003/workspace/brian2-mpi-primary-20260909'


def collect_host(index):
    # Fixed own-job paths, bounded per-file reads. node23 BASE resolves to brick2.
    code = f'''from pathlib import Path
import os,json,hashlib
b=Path({BASE!r});host={NODES[index]!r};index={index}
assert os.uname().nodename==host
assert b.resolve()==Path('/data/brick2/brian2-mpi-region-20260907/primary-host-v1') if index==0 else b.resolve().is_relative_to(Path('/home/rock'))
sources={{}}
def read(relative,cap):
 p=b/relative
 with p.open('rb') as f: raw=f.read(cap+1)
 assert len(raw)<=cap,relative
 sources[relative]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 return raw.decode()
roles=(['controller'] if index==0 else [])+['proxy-'+str(index)]
guards={{r:json.loads(read('guards/'+{PREFIX!r}+'-'+r+'.json',1024*1024)) for r in roles}}
assert all(g.get('returncode')==0 and 'after' in g and not g.get('error') for g in guards.values())
rank_time={{str(r):read('metrics/'+{LABEL!r}+'-rank'+str(r)+'.time',65536) for r in range(index*8,(index+1)*8)}}
runtime=json.loads(read('runs/'+{LABEL!r}+'/mpi-runtime.json',32*1024*1024)) if index==0 else None
print(json.dumps(dict(host=host,guards=guards,rank_time=rank_time,source_files=sources,runtime=runtime)))'''
    result = subprocess.run(['tsh', 'ssh', 'rock@' + NODES[index],
                             shlex.join(['python3', '-c', code])],
                            capture_output=True, check=True, timeout=45)
    require(len(result.stdout) <= 40*2**20, 'oversized control response')
    return json.loads(result.stdout)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True,
                        help='Existing primary-run evidence directory')
    parser.add_argument('--output', type=Path, required=True,
                        help='New collection directory on mounted /Volumes/T7')
    args = parser.parse_args()
    launch_path = args.evidence/'launch.json'
    if not launch_path.exists():
        print(json.dumps(dict(ready=False, reason='terminal launch.json not yet present',
                              collection_started=False)))
        return 2
    require(launch_path.stat().st_size < 2**20, 'oversized launch file')
    launch = json.loads(launch_path.read_text())
    roles = {'controller'} | {'proxy-' + str(i) for i in range(4)}
    require(launch.get('error') is None and set(launch['returncodes']) == roles
            and all(type(v) is int and v == 0 for v in launch['returncodes'].values())
            and launch['resource_guard']['unit_prefix'] == PREFIX
            and launch['nodes'] == NODES, 'launcher not terminal primary success')
    volume = Path('/Volumes/T7')
    require(volume.is_mount() and args.output.resolve().is_relative_to(volume.resolve()),
            'collection must use mounted T7')
    stat = os.statvfs(volume)
    require(stat.f_bavail*stat.f_frsize >= 128*2**30, 'T7 reserve below 128 GiB')
    admission = json.loads((args.evidence/'admission.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    # One bounded attempt per host; no retry and no simulation mutation.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        collected = list(pool.map(collect_host, range(4)))
    runtime = collected[0].pop('runtime')
    for row in collected[1:]:
        require(row.pop('runtime') is None, 'unexpected runtime source')
    inputs = dict(admission=admission, launch=launch, collected=collected, runtime=runtime)
    for name, value in inputs.items():
        (args.output/(name+'.json')).write_text(json.dumps(value, indent=2)+'\n')
    provenance = {name: hashlib.sha256((args.output/(name+'.json')).read_bytes()).hexdigest()
                  for name in inputs}
    (args.output/'input-sha256.json').write_text(json.dumps(provenance, indent=2)+'\n')
    report = audit(admission, launch, collected, runtime)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    elapsed = time.monotonic()-started
    # The later bulk archive consumes the remainder of this SAME collection
    # budget. Timing is separate from the six stable resource-audit inputs.
    timing = dict(schema='b2-mam-terminal-collection-controller-v1',
                  elapsed_seconds=elapsed, shared_collection_budget_seconds=7200,
                  resource_report_sha256=hashlib.sha256((args.output/'report.json').read_bytes()).hexdigest(),
                  collection_complete=elapsed < 7200)
    (args.output/'collection-controller.json').write_text(json.dumps(timing,indent=2)+'\n')
    require(timing['collection_complete'], 'terminal collection exhausted the shared collection budget')
    print(json.dumps(dict(ready=True, terminal_resource_audit_passed=True,
                         output=str(args.output), accounting=report['accounting'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
