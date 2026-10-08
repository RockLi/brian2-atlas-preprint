"""Measure one native child from a fresh, small parent process.

On Linux, fork/exec child rusage can include the large Python caller's inherited
RSS before exec. A separate stdlib-only supervisor avoids that accounting floor.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--usage', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--cwd')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a native command is required')
    started = time.perf_counter()
    with args.log.open('wb') as log:
        process = subprocess.Popen(command, cwd=args.cwd, stdout=log,
                                   stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
    args.usage.write_text(json.dumps({
        'returncode': process.returncode,
        'wall_seconds': time.perf_counter() - started,
        'peak_rss_bytes': int(usage.ru_maxrss) * (1 if sys.platform == 'darwin' else 1024),
        'measurement': 'wait4 from fresh stdlib-only supervisor',
    }) + '\n')
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


if __name__ == '__main__':
    sys.exit(main())
