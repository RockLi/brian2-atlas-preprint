"""Record reproducibility metadata without collecting host names or device IDs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def command(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return dict(command=list(map(str, args)), exit_code=result.returncode,
                stdout=result.stdout.strip(), stderr=result.stderr.strip())


def main(output, rustc):
    sources = [HERE / name for name in
               ('experiment.py', 'spiking.py', 'PROTOCOL.md', 'study.py',
                'report.py', 'native_measure.py', 'record_environment.py')]
    sources += [ROOT / 'brian2-rust/python/brian2_rust' / name
                for name in ('device.py', 'export.py')]
    record = dict(
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Host snapshot during the running CPU v1 study; not an isolated benchmark.',
        platform=platform.platform(), machine=platform.machine(), python=sys.version,
        git_head=command(['git', 'rev-parse', 'HEAD']),
        git_branch=command(['git', 'branch', '--show-current']),
        git_status=command(['git', 'status', '--porcelain']),
        rustc=command([str(rustc), '-Vv']),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sources},
        measurement_notes=[
            'Other workloads were active; per-case wall times include shared-host contention.',
            'Generated outputs and Rust target are on external T7 storage.',
            'Native peak RSS is measured by a fresh small wait4 supervisor.',
            'Frontend and native peaks are separate; their maxima are not a process-tree peak.',
        ],
    )
    if platform.system()=='Darwin':
        record['hardware']=command(['/usr/sbin/sysctl', 'hw.model', 'hw.ncpu',
                                    'hw.memsize', 'machdep.cpu.brand_string'])
    if shutil.which('clang++'):
        record['cpp_compiler']=command(['clang++', '--version'])
    # Package metadata avoids importing/reconfiguring a second Brian runtime.
    from importlib.metadata import version
    record['packages']={name:version(name) for name in ('numpy','Cython','sympy')}
    output.write_text(json.dumps(record,indent=2)+'\n')
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rustc',type=Path,required=True)
    args=parser.parse_args()
    main(args.output,args.rustc)
