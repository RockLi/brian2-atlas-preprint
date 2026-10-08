"""Collect accepted replicate1750 derived arrays once, with bounded T7 output."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from compare_mam_confirmation_lags import accepted_science, require
from mam_launch_confirmation_analysis import OUTPUT, NODE
from mam_launch_confirmation_run import CASE, sha, write

SPECS = {'series': ('time-series.npz', 'time-series.json'),
         'fc': ('fc.npz', 'fc.json'), 'lags': ('lags.npz', 'lags.json')}


def run(evidence, t7):
    started = time.monotonic()
    case = evidence / CASE
    _, science = accepted_science(case)
    catalogs = {stage: json.loads((case/'science'/stage/'catalog.json').read_text())
                for stage in SPECS}
    sizes = [catalogs[s][names[0]]['bytes'] for s, names in SPECS.items()]
    require(all(type(n) is int and 0 < n <= 256*2**20 for n in sizes)
            and sum(sizes) <= 512*2**20, 'derived transfer size budget exceeded')
    require(Path('/Volumes/T7').is_mount() and
            t7.resolve().is_relative_to(Path('/Volumes/T7')), 'mounted T7 required')
    fs = os.statvfs(t7)
    require(fs.f_bavail*fs.f_frsize >= 128*2**30+sum(sizes), 'T7 reserve insufficient')
    output = t7/'artifacts'/CASE/'arrays'
    require(not output.exists(), 'arrays already attempted; no automatic retry')
    output.mkdir()
    write(output/'intent.json', dict(science_report_sha256=sha(case/'science/report.json'),
          attempts=1, automatic_retry=False, wall_limit_seconds=600,
          total_file_bytes_limit=512*2**20, minimum_free_gib=128))
    try:
        for stage, (name, report_name) in SPECS.items():
            root = output/stage
            root.mkdir()
            row = catalogs[stage][name]
            source = OUTPUT/stage/name
            code = f'''import os,sys,signal,resource,hashlib
from pathlib import Path
signal.alarm(150)
resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20))
resource.setrlimit(resource.RLIMIT_CPU,(60,60))
p=Path({str(source)!r})
assert not p.is_symlink() and p.resolve()==p and p.stat().st_size=={row['bytes']!r}
with p.open('rb') as f:
 assert hashlib.file_digest(f,'sha256').hexdigest()=={row['sha256']!r}
 f.seek(0);left={row['bytes']!r}
 while left:
  data=f.read(min(left,2**20));assert data
  sys.stdout.buffer.write(data);left-=len(data)
 assert not f.read(1)
sys.stdout.buffer.flush()
'''
            remaining = 600-(time.monotonic()-started)
            require(remaining > 0, 'shared transfer deadline exceeded')
            cmd = ['tsh', 'ssh', 'rock@'+NODE,
                   shlex.join(['taskset', '-c', '8,9', 'python3', '-c', code])]
            # The remote writer emits exactly the admitted size. Preserve any
            # partial output and stderr on failure; never rename it as complete.
            pending = root/(name+'.partial')
            with pending.open('xb') as dst, (root/'transfer.stderr').open('xb') as err:
                result = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=dst,
                                        stderr=err, timeout=min(160, remaining))
                dst.flush(); os.fsync(dst.fileno())
            require(result.returncode == 0 and pending.stat().st_size == row['bytes']
                    and sha(pending) == row['sha256'], 'derived transfer/readback failed')
            pending.rename(root/name)
            for control in ['catalog.json', report_name]:
                src = case/'science'/stage/control
                require(src.stat().st_size < 2**20, 'oversized derived control')
                with (root/control).open('xb') as dst:
                    dst.write(src.read_bytes()); dst.flush(); os.fsync(dst.fileno())
            require(sha(root/'catalog.json') == science['catalogs'][stage]
                    and sha(root/report_name) == catalogs[stage][report_name]['sha256'],
                    'derived controls changed')
        elapsed = time.monotonic()-started
        require(elapsed <= 600, 'shared transfer deadline exceeded')
        write(output/'complete.json', dict(derived_arrays_verified=True,
              scientific_acceptance=False, elapsed_seconds=elapsed,
              files={s+'/'+SPECS[s][0]: catalogs[s][SPECS[s][0]] for s in SPECS}))
        return output
    except BaseException as exc:
        write(output/'failure.json', dict(error=type(exc).__name__+': '+str(exc),
              elapsed_seconds=time.monotonic()-started, automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--t7', type=Path, required=True)
    args = parser.parse_args()
    print(run(args.evidence, args.t7))
