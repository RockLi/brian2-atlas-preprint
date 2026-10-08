"""Small fresh supervisor: measure native RSS without inheriting large frontend RSS."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    metrics=Path(sys.argv[1])
    command=sys.argv[2:]
    started=time.perf_counter()
    with metrics.with_suffix('.stdout').open('w+') as out, metrics.with_suffix('.stderr').open('w+') as err:
        process=subprocess.Popen(command,stdout=out,stderr=err)
        _,status,usage=os.wait4(process.pid,0)
        process.returncode=os.waitstatus_to_exitcode(status)
        data={'native_wall_seconds':time.perf_counter()-started,
              'native_peak_rss_bytes':int(usage.ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
              'exit_code':process.returncode,'pid':process.pid}
        metrics.write_text(json.dumps(data,indent=2)+'\n')
        out.seek(0);err.seek(0)
        sys.stdout.write(out.read());sys.stderr.write(err.read())
    raise SystemExit(process.returncode if process.returncode>=0 else 128-process.returncode)


if __name__=='__main__':main()
