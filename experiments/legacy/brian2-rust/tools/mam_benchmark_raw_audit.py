"""Independent bounded raw-event scan for the frozen performance campaign."""
import argparse
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np


def scan_rank(report, progress, path, *, release_cache=True, duration_ms=2500):
    def need(ok, message):
        if not ok: raise ValueError(message)
    rank, ranks, duration = report['rank'], report['ranks'], report['duration_ms']
    need(duration_ms in (2500,100500) and duration==duration_ms
         and ranks in ((48,96) if duration_ms==2500 else (48,))
         and 0 <= rank < ranks, 'admitted recording identity')
    maximum_bytes = 64*2**20 if duration_ms==2500 else 2*2**30
    chunk_count = duration_ms//50
    need(report['schema'] == 'b2-native-nest-mam-benchmark-events-v2', 'benchmark report required')
    need(path.is_file() and not path.is_symlink() and path.stat().st_size == report['event_bytes']
         and report['event_bytes'] <= maximum_bytes, 'event size or path')
    need(len(progress) == len(report['chunks']) == chunk_count, 'complete chunk coverage')
    digest = hashlib.sha256(); total = terminal = 0
    histogram = np.zeros(chunk_count, dtype=np.int64)
    with path.open('rb', buffering=0) as stream:
        before = os.fstat(stream.fileno())
        for index, (chunk, row) in enumerate(zip(report['chunks'], progress, strict=True)):
            count = chunk['spikes']; end = (index+1)*50
            need(type(count) is int and 0 <= count <= 2000000, 'chunk count ceiling')
            total += count
            need(chunk['end_ms'] == row['end_ms'] == end and row['event'] == 'chunk_complete'
                 and row['rank'] == rank and row['spikes'] == count and row['total_spikes'] == total
                 and row['event_bytes'] == 8*total, 'progress count ledger')
            need(row['memory'] == report['phase_memory'][f'after_{end}ms'], 'progress memory ledger')
            need(all(row[k] == chunk[k] for k in ['simulation_seconds','output_seconds']), 'progress time ledger')
            left = count*8
            while left:
                offset = stream.tell(); size = min(left, 65536); raw = stream.read(size)
                need(len(raw) == size, 'truncated event chunk')
                digest.update(raw); left -= size
                block = np.frombuffer(raw, dtype=[('tick','<u4'),('cell','<u4')])
                need(np.all(block['tick'] <= end*10) and np.all(block['cell'] < 4129924), 'event physical domain')
                need(np.all((block['cell'].astype(np.uint64)+1) % ranks == rank), 'event rank ownership')
                histogram += np.bincount(block['tick'][block['tick'] < duration*10]//500, minlength=chunk_count)
                terminal += int(np.count_nonzero(block['tick'] == duration*10))
                if release_cache: os.posix_fadvise(stream.fileno(), offset, size, os.POSIX_FADV_DONTNEED)
        need(not stream.read(1), 'trailing event bytes')
        after = os.fstat(stream.fileno())
        need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
             (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'event changed during scan')
    need(total == report['local_spikes'] and 8*total == report['event_bytes'], 'event total')
    need(int(histogram.sum())+terminal == total and digest.hexdigest() == report['event_sha256'], 'event checksum or histogram')
    return dict(rank=rank, spikes=total, event_bytes=8*total, event_sha256=digest.hexdigest(),
                physical_50ms_bin_counts=histogram.tolist(), terminal_tick_events=terminal,
                raw_event_stream_verified=True, linux_cache_release=release_cache, maximum_buffer_bytes=65536)


def host_code(admission, terminal, index, *, duration_ms=2500):
    if duration_ms not in (2500,100500) or admission['identity']['duration_ms']!=duration_ms:
        raise ValueError('raw scan must match admitted duration')
    wall_seconds,cpu_seconds = (120,100) if duration_ms==2500 else (900,600)
    host = admission['nodes'][index]; paths = admission['host_paths'][index]
    ranks = terminal['ranks'][index*admission['ranks_per_node']:(index+1)*admission['ranks_per_node']]
    code = f'''import os,resource,signal,time,hashlib,json,subprocess
from pathlib import Path
signal.alarm({wall_seconds})
resource.setrlimit(resource.RLIMIT_AS,(1024*2**20,1024*2**20))
resource.setrlimit(resource.RLIMIT_CPU,({cpu_seconds},{cpu_seconds}))
import numpy as np
'''+inspect.getsource(scan_rank)+f'''
start=time.monotonic();host={host!r};paths={paths!r};ranks={ranks!r}
assert os.uname().nodename==host
base=Path(paths['base']);run=base/paths['run']
assert base.resolve()==base and base.is_relative_to(Path('/home/rock')) and run.resolve().is_relative_to(base)
units=subprocess.check_output(['systemctl','list-units','--type=service','--state=running','--no-legend','b2mpi-*'],text=True,timeout=5)
assert not units.strip(),units
def read(path,cap):
 assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(base)
 with path.open('rb') as f:
  raw=f.read(cap+1);assert len(raw)<=cap
  os.posix_fadvise(f.fileno(),0,len(raw),os.POSIX_FADV_DONTNEED)
 return raw
results=[]
for item in ranks:
 rank=item['rank'];raw=read(run/f'rank{{rank}}.json',8*2**20)
 assert hashlib.sha256(raw).hexdigest()==item['report_sha256']
 report=json.loads(raw);progress_raw=read(run/f'rank{{rank}}.progress.jsonl',2*2**20)
 progress=[json.loads(line) for line in progress_raw.splitlines()]
 event=run/f'rank{{rank}}.events.bin';assert event.resolve().is_relative_to(base)
 row=scan_rank(report,progress,event,duration_ms={duration_ms})
 row.update(report_sha256=item['report_sha256'],progress_sha256=hashlib.sha256(progress_raw).hexdigest())
 results.append(row)
print(json.dumps(dict(host=host,ranks=results,wall_seconds=time.monotonic()-start,
 peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,active_own_units=[],
 memory_limit_bytes=1024*2**20,cpu_limit_seconds={cpu_seconds},wall_limit_seconds={wall_seconds})))
'''
    return code


def run(args):
    from mam_launch_performance_tuning import write, PROTOCOL_SHA
    from mam_launch_native_primary import read, sha, check
    admission = read(args.case/'admission.json'); terminal = read(args.case/'terminal/report.json')
    duration = admission['identity']['duration_ms']
    target = admission['run_purpose']=='target'
    if target:
        check(sha(args.case/'admission.json')=='3b9af4c512babb202ccd0dc55096e11d3f2c237fe99a34390f153f9a4dfc5a0e'
              and admission['case_id']=='nest-selected-target' and duration==100500,
              'exact selected full-target admission required')
    else:
        check(admission['run_purpose']=='performance_tuning' and duration==2500,
              'registered performance observation required')
    check(admission['protocol_sha256'] == PROTOCOL_SHA
          and terminal['terminal_resource_audit_passed'] is True
          and terminal['input_sha256']['admission'] == sha(args.case/'admission.json')
          and terminal['input_sha256']['launch'] == sha(args.case/'launch.json'), 'terminal prerequisite')
    for receipt in terminal['collection_files']:
        check(sha(args.case/'terminal'/receipt['file']) == receipt['sha256'], 'terminal source differs')
    host_seconds = 900 if target else 120
    wall_budget = 5700 if target else 780
    out = args.case/'raw'; out.mkdir(exist_ok=False); started = time.monotonic(); rows = []
    write(out/'intent.json', dict(protocol_sha256=PROTOCOL_SHA, case_id=admission['case_id'],
        terminal_sha256=sha(args.case/'terminal/report.json'), implementation_sha256=sha(Path(__file__)),
        maximum_hosts=6, maximum_wall_seconds=wall_budget, neural_simulations=0, automatic_retry=False,
        duration_ms=duration,maximum_event_bytes_per_rank=2*2**30 if target else 64*2**20))
    try:
        for index, host in enumerate(admission['nodes']):
            check(wall_budget-(time.monotonic()-started)>host_seconds+5,'raw audit budget exhausted before host read')
            code = host_code(admission, terminal, index, duration_ms=duration)
            base = admission['host_paths'][index]['base']
            command = ['env', 'PYTHONPATH='+base+'/nest-runtime-v1/site',
                       'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1',
                       'taskset','-c','8,9','/usr/bin/python3','-c',code]
            with (out/f'host-{index}.json').open('xb') as stdout, (out/f'host-{index}.stderr').open('xb') as stderr:
                result = subprocess.run(['tsh','ssh','rock@'+host,shlex.join(command)],
                                        stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,timeout=host_seconds+5)
            check(result.returncode == 0, 'raw scan failed; retained, no retry')
            row = read(out/f'host-{index}.json')
            check(row['host'] == host and row['wall_seconds'] <= host_seconds and row['peak_rss_kib']*1024 <= 1024*2**20,
                  'raw scan host resources')
            rows.append(row)
        flat = [r for h in rows for r in h['ranks']]
        check([r['rank'] for r in flat] == list(range(admission['identity']['ranks']))
              and all(r['raw_event_stream_verified'] for r in flat), 'raw rank coverage')
        check(sum(r['spikes'] for r in flat) == terminal['reported_spikes']
              and sum(r['event_bytes'] for r in flat) == terminal['reported_event_bytes'], 'raw totals')
        check(time.monotonic()-started<=wall_budget,'raw audit total wall exceeded')
        report = dict(raw_output_audit_passed=True, terminal_resource_audit_passed=True,duration_ms=duration,
                      case_id=admission['case_id'], protocol_sha256=PROTOCOL_SHA, hosts=rows,
                      wall_seconds=time.monotonic()-started, scientific_acceptance=False,
                      performance_cost_acceptance=False, retained_prefix_compared=False)
        write(out/'report.json', report)
        write(args.case/'completion.json', dict(case_id=admission['case_id'], protocol_sha256=PROTOCOL_SHA,
            terminal_resource_audit_passed=True, raw_output_audit_passed=True,
            launch_wall_seconds=terminal['launch_wall_seconds'], scientific_acceptance=False,
            performance_cost_acceptance=False, input_sha256={name:sha(args.case/name) for name in
                ['admission.json','launch.json','terminal/report.json','raw/report.json','raw/intent.json']}))
        print(json.dumps(dict(case_id=admission['case_id'],raw_output_audit_passed=True,wall_seconds=report['wall_seconds'])))
    except BaseException as error:
        write(out/'failure.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    run(parser.parse_args())
